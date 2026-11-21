"""Prepare four independent GTv3 scene caches and a read-only external WMSC index."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from torch.utils.data import ConcatDataset

from .compact_cache import CompactPatches, atomic_json, build, sha
from .ply_stream import header,chunks
from .stpls3d_loader import CLASSES,RAW_TO_CLASS,SOURCE

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/synthetic_to_real"


def read(p):
    return json.loads(p.read_text())


def table(path,rows):
    with path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def manifest_rows(root):
    m=read(root/"manifest.json")
    if m["classes"]!=CLASSES or m["side"]!=32 or m["resolution"]!=.3 or not m["sources"][0]["rgb"]:
        raise ValueError(f"Incompatible manifest {root}")
    if any(sha(root/k)!=v for k,v in m["shards"].items()):
        raise ValueError(f"Corrupt shard in {root}")
    return m["rows"]


def distribution(rows):
    h=np.asarray([r["hist"] for r in rows])
    return h.sum(0),(h>0).sum(0)


def class_rows(role,scene,rows):
    total,present=distribution(rows)
    return [dict(split=role,scene=scene,semantic_class=c,patches=len(rows),voxels=int(n),patches_with_class=int(k))
            for c,n,k in zip(CLASSES,total,present)]


def band_split(rows):
    tiles=np.array([r["tile"] for r in rows])
    axis=int(np.ptp(tiles[:,:2],axis=0).argmax())
    cut=int(np.median(tiles[:,axis]))
    return [r["id"] for r in rows if r["tile"][axis]<cut],[r["id"] for r in rows if r["tile"][axis]>cut],dict(axis=axis,excluded_column=cut)


def prepare(out,cache,quota_free):
    acquisition=read(out/"acquisition.json")
    sources=acquisition["selected"]
    if len(sources)!=4 or len({s["sha256"] for s in sources})!=4: raise ValueError("Four unique sources required")
    candidates=[]
    for s in sources:
        p=Path(s["file"])
        if "/Synthetic_v3/" not in s["member"] or sha(p)!=s["sha256"]: raise ValueError("Source mismatch")
        h=header(p)
        counts=np.zeros(len(RAW_TO_CLASS),dtype=np.int64)
        for block in chunks(p,h,200000,"raw IDs"):
            ids=block["class"].astype("int64")
            if (ids<0).any() or (ids>=len(counts)).any() or (ids!=block["class"]).any(): raise ValueError("Unrecognized raw class")
            counts+=np.bincount(ids,minlength=len(counts))
        common=np.zeros(len(CLASSES),dtype=np.int64)
        np.add.at(common,RAW_TO_CLASS,counts)
        candidates.append(dict(source=s,raw_counts=counts.tolist(),common=common.tolist(),scene=p.stem))
    # Maximize class coverage, then the least represented class, for both selected scenes.
    order=sorted(candidates,key=lambda s:(sum(n>0 for n in s["common"]),min(s["common"])/sum(s["common"])),reverse=True)
    scene_meta={}
    dists=[]
    roles=("val","central","drone_A","drone_B")
    tr_rows={}
    for role,s,cap in zip(roles,order,(384,512,256,256)):
        path=Path(s["source"]["file"])
        dest=cache/s["scene"]
        cfg=dict(scene_role=role,seed=42,chunk=150000,limits=[cap,0,0],cache_bytes=1_000_000_000,
                 work_bytes=1_000_000_000,field="class",schema="raw",ignore=[],dataset="STPLS3D")
        m=build(path,dest,cfg,quota_free)
        meta=read(dest/"plan.json")["meta"]
        if meta["mapping"]!=RAW_TO_CLASS or not meta["rgb"]: raise ValueError("Synthetic channels or mapping mismatch")
        rows=m["rows"]
        tr_rows[role]=rows
        scene_meta[role]=dict(scene=s["scene"],source=s["source"]["member"],source_sha256=s["source"]["sha256"],
                              cache=str(dest.resolve()),manifest_sha256=sha(dest/"manifest.json"),raw_counts=s["raw_counts"],
                              patch_ids=[r["id"] for r in rows])
        dists.extend(class_rows("cached_"+role,s["scene"],rows))
    bands=[]
    local=[]
    for role in ("drone_A","drone_B"):
        rows=tr_rows[role]
        left,right,boundary=band_split(rows)
        kept=set(left+right)
        scene_meta[role].update(patch_ids=[r["id"] for r in rows if r["id"] in kept],iid_buffer=boundary)
        idx={r["id"]:i+len(local) for i,r in enumerate(r for r in rows if r["id"] in kept)}
        local.extend(r for r in rows if r["id"] in kept)
        bands.extend([[idx[k] for k in side] for side in (left,right)])
    a,b,c,d=bands
    if not all(bands): raise ValueError("Empty client band")
    def imbalance(pair):
        h=[distribution([local[i] for i in ids])[0] for ids in pair]
        return np.abs(h[0]/h[0].sum()-h[1]/h[1].sum()).sum()+abs(len(pair[0])-len(pair[1]))/len(local)
    pairs={"Non-IID":[a+b,c+d],"IID":min(([a+c,b+d],[a+d,b+c]),key=imbalance)}
    for case,pair in pairs.items():
        if sorted(pair[0]+pair[1])!=list(range(len(local))): raise ValueError("Invalid client partition")
        for client,ids in zip(("A","B"),pair): dists.extend(class_rows(case+"_"+client,"local_pool",[local[i] for i in ids]))
    for role in ("val","central"):
        if min(distribution(tr_rows[role])[0])==0: raise ValueError(f"{role} lacks a class; reconsider only synthetic coverage")
    wroot=ROOT/"data/wmsc_compact"
    w=manifest_rows(wroot)
    previous=read(wroot/"plan.json")
    if previous["meta"]["mapping"]!=RAW_TO_CLASS or previous["signature"]["source_sha256"] in {s["source"]["sha256"] for s in order}:
        raise ValueError("External test mapping or scene identity mismatch")
    if len(w)!=1683 or len({r["id"] for r in w})!=len(w): raise ValueError("Unexpected external test index")
    manifest=dict(scene_assignment=scene_meta,client_indices=pairs,seed=42,classes=CLASSES,official_mapping=SOURCE,
                  raw_to_shared_id=RAW_TO_CLASS,synthetic_version=3,external=dict(cache=str(wroot),
                  manifest_sha256=sha(wroot/"manifest.json"),patch_ids=[r["id"] for r in w],source_sha256=previous["signature"]["source_sha256"]))
    table(out/"synthetic_class_distribution.csv",dists)
    table(out/"wmsc_class_distribution.csv",class_rows("external_test","WMSC",w))
    atomic_json(out/"dataset_manifest.json",manifest)


def scene_dataset(info):
    root=Path(info["cache"])
    if sha(root/"manifest.json")!=info["manifest_sha256"]: raise ValueError("Manifest changed")
    rows={r["id"]:r for r in manifest_rows(root)}
    return CompactPatches(root,[rows[i] for i in info["patch_ids"]],"B")


def training_sets(m):
    s=m["scene_assignment"]
    return scene_dataset(s["central"]),scene_dataset(s["val"]),ConcatDataset([scene_dataset(s[k]) for k in ("drone_A","drone_B")])


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out",type=Path,default=OUT)
    ap.add_argument("--cache",type=Path,default=ROOT/"data/s2r_cache")
    ap.add_argument("--quota-free-bytes",type=int)
    args=ap.parse_args()
    prepare(args.out,args.cache,args.quota_free_bytes)
