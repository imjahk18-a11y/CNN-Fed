"""Selective download of the four smallest GTv3 semantic scenes, with bounded previews."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile

import numpy as np
import requests
from .compact_cache import atomic_json,sha
from .fetch_member import RangeZip
from .ply_stream import header
from .storage_budget import budget,require_space
from .stpls3d_loader import RAW_TO_CLASS


def preview(src,z,member):
    if member.compress_type!=zipfile.ZIP_STORED: raise ValueError("Preview requires a stored ZIP member")
    with z.open(member):
        start=src.tell()
        prefix=src.read(2048)
    with tempfile.TemporaryDirectory() as temp:
        p=Path(temp)/"header"
        p.write_bytes(prefix)
        h=header(p)
    if h.fmt=="ascii" or not {"x","y","z","red","green","blue","class"}<=set(h.dtype.names):
        raise ValueError(f"Incompatible PLY in {member.filename}: {h.dtype.names}")
    counts=Counter()
    n=min(256,h.count)
    for offset in np.linspace(0,h.count-n,8,dtype=int):
        src.seek(start+h.offset+int(offset)*h.dtype.itemsize)
        labels=np.frombuffer(src.read(n*h.dtype.itemsize),h.dtype)["class"]
        if np.any(labels!=labels.astype(int)): raise ValueError("Noninteger class ID")
        counts.update(map(int,labels))
    if set(counts)-set(range(len(RAW_TO_CLASS))): raise ValueError(f"Unexpected IDs {counts}")
    return dict(member=member.filename,bytes=member.file_size,compressed_bytes=member.compress_size,
                crc32=f"{member.CRC:08x}",point_count=h.count,properties=h.dtype.descr,
                sampled_points=sum(counts.values()),sampled_raw_ids=sorted(counts),
                sampled_classes=sorted({RAW_TO_CLASS[k] for k in counts}))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("file_id")
    ap.add_argument("--consent-scan-warning",action="store_true")
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--report",type=Path,required=True)
    a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    budget_now=budget(a.out)
    if a.report.exists(): raise FileExistsError(a.report)
    with requests.Session() as session:
        src=RangeZip("https://drive.usercontent.google.com/download?export=download&id="+a.file_id,
                     session,a.consent_scan_warning)
        with zipfile.ZipFile(src) as z:
            by_version={v:sorted((m for m in z.infolist() if f"/Synthetic_v{v}/" in m.filename and
                                 m.filename.endswith(".ply") and "__MACOSX" not in m.filename),key=lambda m:m.file_size)
                        for v in (1,2,3)}
            selected=by_version[3][:4]
            if len(selected)!=4: raise ValueError("Four distinct v3 scenes required")
            estimate=sum(m.file_size for m in selected)+4_000_000_000+1_100_000_000+100_000_000
            require_space(budget_now,estimate)
            info=dict(quota=budget_now,peak_estimate=estimate,archive_bytes=src.size,
                      candidates={v:[dict(member=m.filename,bytes=m.file_size) for m in group] for v,group in by_version.items()},
                      selected=[preview(src,z,m) for m in selected])
            atomic_json(a.report,info)
            for m,record in zip(selected,info["selected"]):
                dest=a.out/Path(m.filename).name
                if dest.exists(): raise FileExistsError(dest)
                tmp=dest.with_suffix(".partial")
                digest=hashlib.sha256()
                with z.open(m) as member,src.member_stream(m.compress_size),tmp.open("xb") as f:
                    while chunk:=member.read(1024**2):
                        f.write(chunk)
                        digest.update(chunk)
                src.pending=None
                if tmp.stat().st_size!=m.file_size: raise ValueError("Truncated scene")
                tmp.rename(dest)
                record.update(file=str(dest.resolve()),sha256=digest.hexdigest())
                atomic_json(a.report,info)
                print(f"verified {m.filename} {m.file_size:,}",flush=True)


if __name__=="__main__":
    main()
