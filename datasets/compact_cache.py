"""Stream a single semantic PLY into independently recoverable compressed patch shards."""
import argparse
from collections import defaultdict
import fcntl
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
from torch.utils.data import Dataset

from .ply_stream import chunks,header,inspect,values
from .stpls3d_loader import CLASSES,SOURCE
from .storage_budget import budget,require_space
from .voxelization import dense,IGNORE

SIDE=32
RES=.3
ZONES=("central","A1","A2","B1","B2","val","test")
ROW=np.dtype([("patch","i4"),("slot","u2"),("z","f8"),("rgb","f4",3),("y","i1")])
RECORD=np.dtype([("slot","<u2"),("height","u1"),("rgb","u1",3),("label","u1")])
PER_SHARD=32


def sha(p):
    with p.open("rb") as f:
        digest=hashlib.sha256()
        for block in iter(lambda: f.read(1024*1024), b""):
            digest.update(block)
        return digest.hexdigest()


def atomic_json(p,data):
    tmp=p.with_suffix(".writing")
    tmp.write_text(json.dumps(data,separators=(",",":"),allow_nan=False))
    tmp.replace(p)


def zoning(meta):
    shape=np.floor((np.array(meta["maximum"])-meta["origin"])/RES/SIDE).astype(int)+1
    axis=int(np.argmax(shape[:2]))
    n=int(shape[axis])-6
    if n<7: raise ValueError("Need at least 13 horizontal patch columns for seven buffered spatial bands")
    weights=np.array([.4,.1,.1,.1,.1,.1,.1])
    widths=1+np.floor((n-7)*weights).astype(int)
    for i in np.argsort(-((n-7)*weights%1),kind="stable")[:n-sum(widths)]: widths[i]+=1
    bands=[]
    start=0
    for w in widths:
        bands.append([start,start+int(w)])
        start+=int(w)+1
    return shape.tolist(),axis,bands


def grid(xyz,meta):
    v=(xyz-np.asarray(meta["origin"]))/RES
    cells=np.floor(v).astype(np.int64)
    tiles=cells//SIDE
    key=np.ravel_multi_index(tiles.T,meta["shape"])
    return key,tiles,cells,v


def select(path,h,meta,cfg):
    # Bottom-k hashes per class (only in training) and per zone. Bound state by caps.
    train,val,test=cfg["limits"]
    sizes=[train//2]+[train//8]*4+[val,test]
    sizes[0]+=train-sum(sizes[:5])
    if cfg.get("scene_role"):
        sizes=[sum(cfg["limits"])]
    if min(sizes)<1: raise ValueError("Need train cap >=8 and positive evaluation caps")
    pools=defaultdict(set)
    def priority(i):
        return hashlib.blake2b(f"{cfg['seed']}/{i}".encode(),digest_size=8).digest()
    for pts in chunks(path,h,cfg["chunk"],"select"):
        xyz,y=values(pts,cfg["field"],meta["mapping"],cfg["ignore"])
        key,tiles,_,_=grid(xyz,meta)
        col=tiles[:,meta["axis"]]
        for zone,(lo,hi) in enumerate(meta["bands"]):
            use=(col>=lo)&(col<hi)&(y>=0)
            if not use.any(): continue
            keys=key[use]
            stratify=(cfg["scene_role"]!="val") if cfg.get("scene_role") else zone<5
            categories=np.unique(keys*len(CLASSES)+y[use]) if stratify else np.unique(keys)
            for cat in (range(len(CLASSES)) if stratify else [-1]):
                new=categories[categories%len(CLASSES)==cat]//len(CLASSES) if cat>=0 else categories
                pool=pools[zone,cat]
                pool.update(new.tolist())
                if len(pool)>sizes[zone]:
                    pools[zone,cat]=set(sorted(pool,key=priority)[:sizes[zone]])
    selected={}
    for zone,cap in enumerate(sizes):
        lists=[sorted(v,key=priority) for (z,_),v in pools.items() if z==zone and v]
        if not lists: raise ValueError(f"No labeled patches in spatial band {ZONES[zone]}")
        lists.sort(key=len)
        order={}
        for j in range(max(map(len,lists))):
            for group in lists:
                if j<len(group): order[group[j]]=None
            if len(order)>=cap: break
        for k in list(order)[:cap]: selected[k]=zone
    return sorted(selected),[selected[k] for k in sorted(selected)]


def records(path,h,meta,cfg,keys,phase):
    keys=np.asarray(keys)
    for pts in chunks(path,h,cfg["chunk"],phase):
        xyz,y=values(pts,cfg["field"],meta["mapping"],cfg["ignore"])
        key,_,cells,v=grid(xyz,meta)
        pos=np.searchsorted(keys,key)
        use=keys[np.minimum(pos,len(keys)-1)]==key
        if not use.any(): continue
        result=np.empty(int(use.sum()),ROW)
        result["patch"]=pos[use]
        small=cells[use]%SIDE
        result["slot"]=small[:,0]+SIDE*(small[:,1]+SIDE*small[:,2])
        result["z"]=(v[use,2]-((cells[use,2]//SIDE)*SIDE))/SIDE
        result["y"]=y[use]
        result["rgb"]=np.column_stack([pts[k][use] for k in ("red","green","blue")]) if meta["rgb"] else 0
        yield result


def spool_groups(path,h,meta,cfg,keys,pending,work):
    for r in records(path,h,meta,cfg,keys,"bucket"):
        group=r["patch"]//PER_SHARD
        order=np.argsort(group,kind="stable")
        starts=np.r_[0,np.flatnonzero(np.diff(group[order]))+1,len(order)]
        for a,b in zip(starts[:-1],starts[1:]):
            i=int(group[order[a]])
            if i in pending:
                with (work/f"{i}.bin").open("ab") as f: r[order[a:b]].tofile(f)


def batches(p,chunk):
    with p.open("rb") as f:
        while len(x:=np.fromfile(f,ROW,chunk)): yield x


def make_shard(out,group,rows,keys,zones,meta):
    start=group*PER_SHARD
    count=min(PER_SHARD,len(keys)-start)
    slots=SIDE**3
    n=np.zeros(count*slots,dtype=np.int64)
    sums=np.zeros((count*slots,4),dtype=np.float64)
    votes=np.zeros((count*slots,len(CLASSES)),dtype=np.int64)
    for r in rows:
        ids=(r["patch"].astype("int64")-start)*slots+r["slot"]
        np.add.at(n,ids,1)
        np.add.at(sums[:,0],ids,r["z"])
        for j in range(3): np.add.at(sums[:,j+1],ids,r["rgb"][:,j])
        ok=r["y"]>=0
        np.add.at(votes,(ids[ok],r["y"][ok]),1)
    name=f"shard-{group:04}.zip"
    tmp=out/(name+".writing")
    index=[]
    with zipfile.ZipFile(tmp,"w",zipfile.ZIP_DEFLATED,compresslevel=2) as archive:
        for j in range(count):
            k=int(keys[start+j])
            ids=np.flatnonzero(n[j*slots:(j+1)*slots])+j*slots
            actual=sums[ids]/n[ids,None]
            actual[:,1:]/=255
            quant=np.rint(actual*255).clip(0,255).astype("u1")
            err=np.abs(actual-quant/255).max(axis=0)
            if err.max()>0.5/255+1e-7: raise ValueError("Quantization error exceeds half an LSB")
            target=votes[ids].argmax(1)
            target[votes[ids].sum(1)==0]=255
            data=np.empty(len(ids),RECORD)
            data["slot"]=ids-j*slots
            data["height"]=quant[:,0]
            data["rgb"]=quant[:,1:]
            data["label"]=target
            member=f"{k}.bin"
            archive.writestr(member,data.tobytes())
            hist=np.bincount(target[target!=255],minlength=len(CLASSES)).tolist()
            zone=meta.get("scene_role") or ZONES[zones[start+j]]
            index.append(dict(id=member[:-4],tile=list(map(int,np.unravel_index(k,meta["shape"]))),
                              split=zone if zone in ("val","test") else "train",region=zone,shard=name,
                              member=member,hist=hist,occupied=len(data),quant_error=err.tolist(),
                              feature_min=quant.min(0).tolist(),feature_max=quant.max(0).tolist()))
        archive.writestr("index.json",json.dumps(index))
    tmp.replace(out/name)


def check_shard(path):
    with zipfile.ZipFile(path) as z:
        bad=z.testzip()
        if bad: raise ValueError(f"Corrupt shard member {bad}")
        return json.loads(z.read("index.json"))


class CompactPatches(Dataset):
    def __init__(self,root,rows,variant):
        self.root,self.rows,self.variant=Path(root),rows,variant

    def __len__(self): return len(self.rows)

    def __getitem__(self,i):
        r=self.rows[i]
        # Each worker opens its own file description. Reads one deflated member and checks CRC.
        with zipfile.ZipFile(self.root/r["shard"]) as z:
            rec=np.frombuffer(z.read(r["member"]),RECORD)
        features=np.column_stack([rec["height"],rec["rgb"]]).T.astype("float32")/255
        y=rec["label"].astype("int64")
        y[y==255]=IGNORE
        return dense(rec["slot"],features,y,SIDE,self.variant)


def spatial_clients(rows):
    groups={k:[i for i,r in enumerate(rows) if r["region"]==k] for k in ZONES[:5]}
    central,a,b,c,d=(groups[k] for k in ZONES[:5])
    if not all([central,a,b,c,d]): raise ValueError("Empty spatial pool")
    def imbalance(pair):
        h=np.array([np.sum([rows[i]["hist"] for i in ids],axis=0) for ids in pair])
        return np.abs(h[0]/h[0].sum()-h[1]/h[1].sum()).sum()+abs(len(pair[0])-len(pair[1]))/sum(map(len,pair))
    iid=min(([a+c,b+d],[a+d,b+c]),key=imbalance)
    return central,{"IID":iid,"Non-IID":[a+b,c+d]}


def build(path,out,cfg,quota_free=None):
    path,out=Path(path),Path(out)
    h=header(path)
    plan=budget(out,quota_free)
    caps=sum(cfg["limits"])
    # Deflate worst-case overhead plus ZIP headers, manifest, plan, hashes and previews.
    estimate=min(h.count,caps*SIDE**3)*RECORD.itemsize+caps*4096+2_000_000
    spool=h.count*ROW.itemsize
    direct=spool>cfg["work_bytes"]
    stage=0 if direct else spool
    if estimate>cfg["cache_bytes"]:
        raise ValueError(f"Conservative cache bound {estimate:,} exceeds configured cap; reduce patch limits")
    out.mkdir(parents=True,exist_ok=True)
    lock=out/".compact.lock"
    if not lock.exists() and any(out.iterdir()):
        raise ValueError("New cache requires an empty directory")
    with lock.open("a") as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        signature=dict(source_sha256=sha(path),config=cfg,format="compact-v1")
        planfile=out/"plan.json"
        complete=out/"manifest.json"
        if complete.exists() and planfile.exists():
            stored=json.loads(planfile.read_text())
            if stored["signature"]!=signature: raise ValueError("Resume source or configuration mismatch")
            manifest=json.loads(complete.read_text())
            if any(sha(out/name)!=expected for name,expected in manifest["shards"].items()):
                raise ValueError("Completed shard integrity check failed")
            return manifest
        require_space(plan,estimate+stage+100_000_000)
        if planfile.exists():
            saved=json.loads(planfile.read_text())
            if saved["signature"]!=signature: raise ValueError("Resume source or configuration mismatch")
            meta,keys,zones=saved["meta"],saved["keys"],saved["zones"]
        else:
            meta=inspect(path,h,cfg["chunk"],cfg["field"],cfg["schema"],cfg["ignore"])
            meta["shape"],meta["axis"],meta["bands"]=zoning(meta)
            if cfg.get("scene_role"):
                meta["scene_role"]=cfg["scene_role"]
                meta["bands"]=[[0,meta["shape"][meta["axis"]]]]
            keys,zones=select(path,h,meta,cfg)
            atomic_json(planfile,dict(signature=signature,meta=meta,keys=keys,zones=zones,
                                      budget=plan,cache_bound=estimate,workspace_bound=stage))
        groups=range((len(keys)+PER_SHARD-1)//PER_SHARD)
        pending={g for g in groups if not (out/f"shard-{g:04}.zip").exists()}
        # The workspace contains only this cache's derived buckets. Replay an interrupted bucket pass.
        work=out/".buckets"
        work.mkdir(exist_ok=True)
        for g in groups:
            (work/f"{g}.bin").unlink(missing_ok=True)
        if pending and not direct: spool_groups(path,h,meta,cfg,keys,pending,work)
        for g in sorted(pending):
            if direct:
                sub=keys[g*PER_SHARD:(g+1)*PER_SHARD]
                def local_records():
                    for r in records(path,h,meta,cfg,sub,f"shard {g}"):
                        r["patch"]+=g*PER_SHARD
                        yield r
                batch=local_records()
            else:
                batch=batches(work/f"{g}.bin",cfg["chunk"])
            make_shard(out,g,batch,keys,zones,meta)
            (work/f"{g}.bin").unlink(missing_ok=True)
            used=sum(p.stat().st_size for p in out.iterdir() if p.is_file())
            if used>cfg["cache_bytes"]: raise RuntimeError("Cache cap exceeded")
            print(f"shard {g+1}/{len(groups)}: {used:,} cache bytes",flush=True)
            atomic_json(out/"progress.json",dict(shards_committed=len(list(out.glob("shard-*.zip"))),bytes=used))
        rows=[]
        hashes={}
        for g in groups:
            p=out/f"shard-{g:04}.zip"
            rows.extend(check_shard(p))
            hashes[p.name]=sha(p)
        manifest=dict(format="compact-v1",dataset=cfg["dataset"],classes=CLASSES,side=SIDE,resolution=RES,
                      rows=rows,shards=hashes,mapping_source=SOURCE,sources=[dict(file=str(path.resolve()),
                      sha256=signature["source_sha256"],rgb=meta["rgb"],raw_counts=meta["raw_counts"])])
        before=sum(p.stat().st_size for p in out.iterdir() if p.is_file())
        if before+len(json.dumps(manifest).encode())>cfg["cache_bytes"]:
            raise ValueError("Manifest would exceed the cache budget")
        atomic_json(out/"manifest.json",manifest)
        return manifest


def args_parser():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ply",type=Path)
    ap.add_argument("--out",required=True,type=Path)
    ap.add_argument("--limits",nargs=3,type=int,default=[5000,500,500],metavar=("TRAIN","VAL","TEST"))
    ap.add_argument("--chunk",type=int,default=100000)
    ap.add_argument("--cache-bytes",type=int,default=4_000_000_000)
    ap.add_argument("--work-bytes",type=int,default=1_000_000_000)
    ap.add_argument("--quota-free-bytes",type=int)
    ap.add_argument("--schema",choices=["raw","benchmark"],default="raw")
    ap.add_argument("--field",default="class")
    ap.add_argument("--ignore",type=int,nargs="*",default=[])
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--dataset",choices=["STPLS3D","fixture"],default="STPLS3D")
    return ap


if __name__=="__main__":
    opt=vars(args_parser().parse_args())
    path,out,quota=(opt.pop(k) for k in ("ply","out","quota_free_bytes"))
    build(path,out,opt,quota)
