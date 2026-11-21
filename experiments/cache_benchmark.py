"""Compare lossless encodings of identical quantized sparse patch records on a bounded sample."""
import argparse
import csv
import json
from pathlib import Path
import tempfile
import time
import zipfile
import numpy as np


def benchmark(cache,out):
    import zarr
    from numcodecs import Blosc
    rows=json.loads((cache/"manifest.json").read_text())["rows"]
    rng=np.random.default_rng(42)
    selected=rng.choice(len(rows),min(128,len(rows)),replace=False)
    blobs=[]
    for idx in selected:
        r=rows[idx]
        with zipfile.ZipFile(cache/r["shard"]) as src: blobs.append(src.read(r["member"]))
    offsets=np.r_[0,np.cumsum(list(map(len,blobs)))]
    seq=rng.permutation(len(blobs))
    results=[]
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        for name in ("numpy_mmap","zarr_zstd","zip_shards"):
            folder=Path(tmp)/name
            folder.mkdir()
            start=time.perf_counter()
            if name=="zip_shards":
                for start_i in range(0,len(blobs),32):
                    with zipfile.ZipFile(folder/f"{start_i//32}.zip","w",zipfile.ZIP_DEFLATED,compresslevel=2) as z:
                        for i,b in enumerate(blobs[start_i:start_i+32],start_i): z.writestr(str(i),b)
            else:
                np.save(folder/"offsets.npy",offsets)
                if name=="zarr_zstd":
                    array=zarr.open_array(str(folder/"records"),mode="w",shape=(int(offsets[-1]),),chunks=(131072,),
                                          dtype="u1",compressor=Blosc(cname="zstd",clevel=3))
                else:
                    array=np.lib.format.open_memmap(folder/"records.npy",mode="w+",dtype="u1",shape=(int(offsets[-1]),))
                array[:]=np.frombuffer(b"".join(blobs),dtype="u1")
                if name=="numpy_mmap": array.flush()
            write=time.perf_counter()-start
            if name!="zip_shards":
                ix=np.load(folder/"offsets.npy",mmap_mode="r")
                array=(zarr.open_array(str(folder/"records"),mode="r") if name=="zarr_zstd" else
                       np.load(folder/"records.npy",mmap_mode="r"))
            timings=[]
            for _ in range(3):
                start=time.perf_counter()
                for i in seq:
                    if name=="zip_shards":
                        with zipfile.ZipFile(folder/f"{i//32}.zip") as z: b=z.read(str(i))
                    else: b=array[ix[i]:ix[i+1]].tobytes()
                    if b!=blobs[i]:raise ValueError("Mismatch in stored patch")
                timings.append(time.perf_counter()-start)
            files=[p for p in folder.rglob("*") if p.is_file()]
            results.append(dict(format=name,patches=len(seq),files=len(files),bytes=sum(p.stat().st_size for p in files),
                                write_s=write,random_ms_per_patch=1000*np.median(timings)/len(seq)))
    with out.open("x") as f:
        w=csv.DictWriter(f,fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("cache",type=Path)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    benchmark(a.cache,a.out)
