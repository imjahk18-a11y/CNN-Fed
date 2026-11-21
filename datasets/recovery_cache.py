"""Full-Z cache: independently checksummed Zstandard frames, one per occupied XY patch."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import zstandard as zstd

from .compact_cache import atomic_json, sha
from .fullz import SIDE, RESOLUTION, CLASSES, common, representatives, xyz
from .ply_stream import header

BLOCK=250_000
TILES=32
FORMAT="fullz-rgb16-frames-v1"


def point_table(path):
    h=header(path)
    if not h.fmt.startswith("binary") or any(h.dtype[c].kind not in "fiu" for c in "xyz"):
        raise ValueError("Unsupported STPLS3D XYZ format")
    for c in ("red","green","blue","class"):
        if h.dtype[c]!=np.uint8:raise ValueError(f"Unexpected {c} dtype {h.dtype[c]}")
    return np.memmap(path,mode="r",offset=h.offset,dtype=h.dtype,shape=(h.count,))


def inspect_raw(pts):
    lo=np.full(3,np.inf);hi=-lo.copy();hist=np.zeros(256,np.int64)
    for i in range(0,len(pts),BLOCK):
        rows=pts[i:i+BLOCK];p=xyz(rows)
        if not np.isfinite(p).all():raise ValueError("Nonfinite XYZ")
        lo=np.minimum(lo,p.min(0));hi=np.maximum(hi,p.max(0))
        hist+=np.bincount(rows["class"],minlength=256)
    ids=np.flatnonzero(hist)
    common(ids)
    return lo,hi,hist


def encode(cell,raw,rgb):
    # Byte shuffle the lossless float32 payload for compression.
    return b"".join((cell.astype("<u4").tobytes(),raw.tobytes(),common(raw).astype("u1").tobytes(),
                     np.ascontiguousarray(rgb,dtype="<f4").view("u1").reshape(len(rgb),12).T.tobytes()))


def decode(buf,n,height):
    if len(buf)!=18*n:raise IOError("Frame length mismatch")
    key=np.frombuffer(buf,"<u4",n).astype(np.int64)
    z=key%height
    xy=key//height
    coords=np.stack((xy//SIDE,xy%SIDE,z),axis=1).astype(np.int32)
    raw=np.frombuffer(buf,"u1",n,4*n)
    y=np.frombuffer(buf,"u1",n,5*n)
    rgb=np.frombuffer(buf,"u1",12*n,6*n).reshape(12,n).T.copy().view("<f4").reshape(n,3)
    return coords,rgb,raw,y


def process_bucket(bucket,pts,root,origin,shape,n_y):
    indices=np.fromfile(bucket,dtype="<u4")
    rows=pts[indices]
    p=xyz(rows)
    cells=np.floor((p-origin)/RESOLUTION).astype(np.int64)
    tile=(cells[:,0]//SIDE)*n_y+cells[:,1]//SIDE
    slot=((cells[:,0]%SIDE)*SIDE+cells[:,1]%SIDE)*shape[2]+cells[:,2]
    key=tile*(SIDE*SIDE*shape[2])+slot
    sort=np.argsort(key,kind="stable")
    key=key[sort]
    bounds=np.r_[0,np.flatnonzero(np.diff(key))+1,len(key)]
    vkey=key[bounds[:-1]]
    group=np.repeat(np.arange(len(vkey)),np.diff(bounds))
    delta=p-(origin+(cells+.5)*RESOLUTION)
    dist=np.einsum("ij,ij->i",delta,delta)[sort]
    colors=np.stack([rows[c][sort] for c in ("red","green","blue")],axis=1)
    raw,rgb=representatives(group,rows["class"][sort],colors,dist)
    vtile,slot=np.divmod(vkey,SIDE*SIDE*shape[2])
    offsets=np.r_[0,np.flatnonzero(np.diff(vtile))+1,len(vtile)]
    filename=bucket.stem+".zst"
    file=root/filename
    compressor=zstd.ZstdCompressor(level=1,write_checksum=True)
    patches=[]
    with file.with_suffix(".writing").open("wb") as f:
        for a,b in zip(offsets[:-1],offsets[1:]):
            tx,ty=divmod(int(vtile[a]),int(n_y))
            payload=compressor.compress(encode(slot[a:b],raw[a:b],rgb[a:b]))
            patches.append(dict(x=tx*SIDE,y=ty*SIDE,n=int(b-a),offset=f.tell(),size=len(payload),
                                hist=np.bincount(common(raw[a:b]),minlength=6).tolist()))
            f.write(payload)
    file.with_suffix(".writing").replace(file)
    return dict(file=filename,bytes=file.stat().st_size,sha256=sha(file),patches=patches),rgb.sum(0,dtype=np.float64),rgb.min(0),rgb.max(0)


def build(path,root,record,space):
    root.mkdir(parents=True,exist_ok=True)
    if (root/"index.json").is_file():
        index=json.loads((root/"index.json").read_text())
        if index["source_sha256"]!=record["sha256"]:raise IOError("Cache source mismatch")
        return index
    pts=point_table(path)
    lo,hi,rawhist=inspect_raw(pts)
    origin=np.floor(lo/RESOLUTION)*RESOLUTION
    shape=np.floor((hi-origin)/RESOLUTION).astype(np.int64)+1
    if SIDE*SIDE*int(shape[2])>2**32 or len(pts)>=2**32:raise ValueError("Coordinate/index capacity exceeded")
    n_y=(shape[1]+SIDE-1)//SIDE
    # Worst-case scratch index 4 bytes/point, payload 18 bytes/occupied voxel.
    space(23*len(pts)+32*2**20)
    shards=[];sums=np.zeros(3);rgblo=np.full(3,np.inf);rgbhi=-rgblo.copy()
    with TemporaryDirectory(prefix="buckets-",dir=root) as work:
        work=Path(work)
        for i in range(0,len(pts),BLOCK):
            pos=np.floor((xyz(pts[i:i+BLOCK])-origin)/RESOLUTION).astype(np.int64)
            tile=(pos[:,0]//SIDE)*n_y+pos[:,1]//SIDE
            bucket=tile//TILES;order=np.argsort(bucket,kind="stable")
            groups=bucket[order];bounds=np.r_[0,np.flatnonzero(np.diff(groups))+1,len(groups)]
            for a,b in zip(bounds[:-1],bounds[1:]):
                with (work/f"{groups[a]:06d}.idx").open("ab") as f:
                    (order[a:b]+i).astype("<u4").tofile(f)
        jobs=sorted(work.glob("*.idx"))
        with ThreadPoolExecutor(max_workers=4) as workers:
            for shard,total,low,high in workers.map(lambda p:process_bucket(p,pts,root,origin,shape,n_y),jobs):
                shards.append(shard);sums+=total;rgblo=np.minimum(rgblo,low);rgbhi=np.maximum(rgbhi,high)
    patches=[p for shard in shards for p in shard["patches"]]
    h=np.array([p["hist"] for p in patches]);n=int(h.sum())
    index=dict(format=FORMAT,id=record["id"],version=record["version"],member=record["member"],
        source_sha256=record["sha256"],source_crc32=record["crc32"],points=len(pts),resolution=RESOLUTION,
        patch_xy=[SIDE,SIDE],stride=SIDE,full_z=True,rgb="most frequent joint width-16 bin, mean original RGB",
        label="raw majority then Common 6; ties by nearest center then PLY order",classes=CLASSES,
        origin=origin.tolist(),shape=shape.tolist(),xyz_min=lo.tolist(),xyz_max=hi.tolist(),raw_point_hist=rawhist.tolist(),
        voxels=n,patches=len(patches),voxel_hist=h.sum(0).tolist(),class_patch_hist=(h>0).sum(0).tolist(),
        rgb_sum=sums.tolist(),rgb_min=rgblo.tolist(),rgb_max=rgbhi.tolist(),cache_bytes=sum(x["bytes"] for x in shards),shards=shards)
    atomic_json(root/"index.json",index)
    return index


class Frames:
    def __init__(self,root):
        self.root=Path(root)
        self.meta=json.loads((self.root/"index.json").read_text())
        if self.meta["format"]!=FORMAT:raise ValueError("Unsupported frame format")
        self.rows=[(s["file"],p) for s in self.meta["shards"] for p in s["patches"]]

    def read(self,i):
        file,p=self.rows[i]
        with (self.root/file).open("rb") as f:
            f.seek(p["offset"]);data=f.read(p["size"])
        if len(data)!=p["size"]:raise IOError("Truncated frame")
        return decode(zstd.ZstdDecompressor().decompress(data,max_output_size=18*p["n"]),p["n"],self.meta["shape"][2])


def verify_cache(root,record,deep,raw=None):
    if sha(root/"index.json")!=record["index_sha256"]:raise IOError("Index checksum mismatch")
    ds=Frames(root);m=ds.meta
    if m["source_sha256"]!=record["sha256"] or m["id"]!=record["id"]:raise IOError("Source identity mismatch")
    for shard in m["shards"]:
        f=root/shard["file"]
        if f.stat().st_size!=shard["bytes"] or sha(f)!=shard["sha256"]:raise IOError(f"Damaged shard {f}")
    if not deep:return
    rng=np.random.default_rng(918)
    checks=set(rng.choice(len(ds.rows),min(96,len(ds.rows)),replace=False).tolist())
    origin=np.array(m["origin"]);shape=m["shape"];samples={}
    tiles=set();total=0;hist=np.zeros(6,np.int64)
    for i,(_,p) in enumerate(ds.rows):
        coords,rgb,r,y=ds.read(i)
        xy=(p["x"],p["y"])
        if xy in tiles or np.any(coords[:,:2]>=SIDE) or not np.all((rgb>=0)&(rgb<=255)):raise IOError("Invalid decoded patch")
        tiles.add(xy);np.testing.assert_array_equal(common(r),y)
        coords[:,:2]+=xy
        keys=np.ravel_multi_index(coords.T,shape)
        if np.any(np.diff(keys)<=0):raise IOError("Duplicate or unordered coordinates")
        patch_hist=np.bincount(y,minlength=6)
        np.testing.assert_array_equal(patch_hist,p["hist"]);hist+=patch_hist;total+=len(r)
        if i in checks:
            j=rng.integers(len(coords));samples[int(keys[j])]=(int(r[j]),rgb[j].copy())
    np.testing.assert_array_equal(hist,m["voxel_hist"])
    if total!=m["voxels"] or len(tiles)!=m["patches"]:raise IOError("Scene count mismatch")
    wanted=np.array(sorted(samples)); votes=defaultdict(list);pts=point_table(raw)
    for i in range(0,len(pts),BLOCK):
        rows=pts[i:i+BLOCK];pos=xyz(rows)
        cell=np.floor((pos-origin)/RESOLUTION).astype(np.int64)
        keys=np.ravel_multi_index(cell.T,shape)
        j=np.searchsorted(wanted,keys).clip(max=len(wanted)-1)
        for k in np.flatnonzero(wanted[j]==keys):
            d=pos[k]-(origin+(cell[k]+.5)*RESOLUTION)
            votes[int(keys[k])].append((float(d@d),i+k,int(rows["class"][k]),tuple(int(rows[c][k]) for c in ("red","green","blue"))))
    if set(votes)!=set(samples):raise IOError("Sampled voxel missing from PLY")
    for key,rows in votes.items():
        def winner(values):
            counts=Counter(values);maximum=max(counts.values())
            return min((r[0],r[1],v) for r,v in zip(rows,values) if counts[v]==maximum)[2]
        label=winner([r[2] for r in rows]);bins=[tuple(x//16 for x in r[3]) for r in rows];color=winner(bins)
        mean=np.mean([r[3] for r,b in zip(rows,bins) if b==color],axis=0).astype(np.float32)
        if label!=samples[key][0]:raise IOError("Raw semantic vote mismatch")
        np.testing.assert_array_equal(mean,samples[key][1])
    return dict(checked_shards=len(m["shards"]),decoded_voxels=total,source_matched_samples=len(samples))
