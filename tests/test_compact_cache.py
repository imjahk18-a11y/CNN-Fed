from pathlib import Path
import struct
from unittest import mock
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
from datasets import compact_cache as cc
from datasets.ply_stream import header,chunks
from datasets.storage_budget import budget,require_space


def write_ply(p,encoding):
    rows=[]
    for i in range(48):
        x=i*9.6+.1
        rows.extend([(x,0.,0.,0,0,0,0),(x+.02,0.,.01,255,0,0,5),
                     (x+.05,0.,.02,255,0,0,8),(x,.51,0.,1,2,3,-100)])
    with p.open("wb") as f:
        f.write(("\n".join(["ply",f"format {encoding} 1.0",f"element vertex {len(rows)}",
                            *["property double "+k for k in "xyz"],
                            *["property uchar "+k for k in ["red","green","blue"]],"property int class","end_header"])+"\n").encode())
        for r in rows:
            if encoding=="ascii": f.write((" ".join(map(str,r))+"\n").encode())
            else: f.write(struct.pack(("<" if encoding=="binary_little_endian" else ">")+"dddBBBi",*r))
    return rows


@pytest.mark.parametrize("fmt",["ascii","binary_little_endian","binary_big_endian"])
def test_chunks(tmp_path,fmt):
    p=tmp_path/"cloud.ply"
    r=write_ply(p,fmt)
    h=header(p)
    seq=list(chunks(p,h,3,"test"))
    assert sum(map(len,seq))==len(r) and max(map(len,seq))<=3
    np.testing.assert_allclose(np.concatenate(seq)["x"],np.array(r)[:,0])


def test_recovery_and_equivalence(tmp_path):
    p,out=tmp_path/"cloud.ply",tmp_path/"cache"
    write_ply(p,"binary_big_endian")
    cfg=dict(seed=42,chunk=3,limits=[100,100,100],cache_bytes=4_000_000,work_bytes=0,
             schema="raw",field="class",ignore=[-100],dataset="fixture")
    def interrupt_after_shard(*args):
        original(*args)
        raise InterruptedError("Intentional test interruption")
    original=cc.make_shard
    with mock.patch.object(cc,"make_shard",side_effect=interrupt_after_shard),pytest.raises(InterruptedError):
        cc.build(p,out,cfg,30_000_000_000)
    first=out/"shard-0000.zip"
    timestamp=first.stat().st_mtime_ns
    m=cc.build(p,out,cfg,30_000_000_000)
    assert timestamp==first.stat().st_mtime_ns
    train=[r for r in m["rows"] if r["split"]=="train"]
    center,pairs=cc.spatial_clients(train)
    for a,b in pairs.values(): assert sorted(center+a+b)==list(range(len(train)))
    ds=cc.CompactPatches(out,m["rows"],"B")
    x,y=ds[0]
    assert x.shape==(5,32,32,32) and y[0,0,0]==3
    assert x[2,0,0,0]==pytest.approx(2/3,abs=.5/255)
    assert x[0,0,1,0]==1 and y[0,1,0]==-100
    bx,by=next(iter(DataLoader(ds,batch_size=2,num_workers=2,timeout=8)))
    torch.testing.assert_close(bx[0],x)
    with pytest.raises(ValueError,match="mismatch"):cc.build(p,out,{**cfg,"seed":8},30_000_000_000)
    with first.open("ab") as f:f.write(b"corruption")
    with pytest.raises(ValueError,match="integrity"):cc.build(p,out,cfg,30_000_000_000)


def test_budget(tmp_path):
    b=budget(tmp_path,10_000_000_000)
    assert b["reserve"]==5_000_000_000
    with pytest.raises(ValueError):require_space(b,10_000_000_000)
