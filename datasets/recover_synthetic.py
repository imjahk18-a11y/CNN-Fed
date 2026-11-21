"""Sequential, resumable STPLS3D acquisition. Run from the project root."""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import zipfile
import zlib

import requests

from .compact_cache import atomic_json, sha
from .fetch_member import RangeZip
from .storage_budget import budget

OUT=Path("results/stpls3d_full_data_recovery")
DATA=Path("data/full_recovery")
STAGE=DATA/"downloads"
CACHE=DATA/"cache"
JOURNAL=OUT/"stream_state.json"
URL="https://drive.usercontent.google.com/download?export=download&id=1U8dduxCdhAXbCpfGLDMCSRHpdb5R7cXb"
RESERVE=8*2**30


def space(needed=0):
    b=budget(Path("."))
    if b["available"]-needed < RESERVE:
        raise RuntimeError(f"8 GiB reserve: available={b['available']}, next stage={needed}")
    return b["quota_free"]


def persist(state):
    atomic_json(JOURNAL,state)
    rows=list(state["scenes"].values())
    path=OUT/"download_manifest.csv"
    tmp=path.with_suffix(".writing")
    fields=("id","version","member","bytes","existing","status","crc32","sha256","raw_removed","error")
    with tmp.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(rows)
    tmp.replace(path)


def change(state,r,status,**data):
    r.update(data,status=status)
    r.setdefault("history",[]).append([status,time.time()])
    persist(state)


def initialize():
    if JOURNAL.exists():
        return json.loads(JOURNAL.read_text())
    backup=OUT/"original_storage_preflight"
    backup.mkdir(exist_ok=True)
    for p in OUT.iterdir():
        if p.is_file() and p.suffix in (".json",".csv",".md") and not (backup/p.name).exists():
            shutil.copy2(p,backup/p.name)
    inventory=list(csv.DictReader((backup/"download_manifest.csv").open()))
    assert len(inventory)==63
    records={}
    for row in inventory:
        member=row["member"]
        v=int(row["version"]); filename=Path(member).name; ident=f"v{v}_{Path(filename).stem}"
        assert Path(member).parts==("STPLS3D",f"Synthetic_v{v}",filename)
        records[ident]=dict(id=ident,version=v,member=member,bytes=int(row["expected_bytes"]),
            existing=row["local_path"],status="pending",crc32=row["expected_crc32"],sha256=row["sha256"],raw_removed=False,error="")
    state=dict(quota_before=space(),scenes=records)
    assert len(records)==63 and len([r for r in records.values() if r["existing"]])==4
    persist(state)
    return state


def digest(path):
    h=hashlib.sha256();crc=0
    with path.open("rb") as f:
        while part:=f.read(4*2**20):
            h.update(part);crc=zlib.crc32(part,crc)
    return h.hexdigest(),f"{crc:08x}"


def fetch(state,r,remote,z,path):
    member=z.getinfo(r["member"])
    if member.compress_type!=zipfile.ZIP_STORED or member.file_size!=r["bytes"] or member.compress_size!=member.file_size:
        raise ValueError("Unexpected archive member size or compression")
    crc=f"{member.CRC:08x}"
    if r["crc32"] and r["crc32"]!=crc:raise ValueError("Archive member CRC changed")
    r["crc32"]=crc
    if not path.exists():
        temp=path.with_suffix(".partial")
        offset=temp.stat().st_size if temp.exists() else 0
        space(r["bytes"]-offset)
        change(state,r,"downloading")
        if remote is not None:
            with z.open(member):
                start=remote.tell()
        if not 0<=offset<=r["bytes"]:raise ValueError("Invalid partial download")
        last=offset
        if remote is None:
            # A local copy of the same official ZIP can finish an interrupted stream.
            with z.open(member) as src,temp.open("ab") as dst:
                src.seek(offset)
                while chunk:=src.read(2**20):
                    dst.write(chunk)
            offset=temp.stat().st_size
        while offset < r["bytes"]:
            stop=min(offset+32*2**20,r["bytes"])
            with remote.request(f"bytes={start+offset}-{start+stop-1}") as response:
                if response.status_code==200 and "text/html" in response.headers.get("Content-Type",""):
                    preview=next(response.iter_content(8192),b"").lower()
                    if b"quota exceeded" in preview or b"too many users" in preview:
                        raise RuntimeError(f"Drive download quota exceeded at member byte {offset}; source file retained if partial")
                remote.check(response,start+offset,start+stop)
                with temp.open("ab") as f:
                    for chunk in response.iter_content(2**20):
                        f.write(chunk)
                    if f.tell()!=stop:raise IOError("Short range; resume partial")
            offset=stop
            if offset-last>=64*2**20 or offset==r["bytes"]:
                last=offset;space(r["bytes"]-offset)
                print(r["id"],"download",f"{100*offset/r['bytes']:.0f}%",flush=True)
        if temp.stat().st_size!=r["bytes"]:raise IOError("Short download; resume partial")
        temp.rename(path)
    change(state,r,"downloaded")


def owned_path(record, path):
    return (not record["existing"] and path == STAGE/(record["id"]+".ply")
            and not path.is_symlink() and any(h[0]=="downloading" for h in record.get("history",[])))


def run(args):
    OUT.mkdir(parents=True,exist_ok=True);STAGE.mkdir(parents=True,exist_ok=True);CACHE.mkdir(exist_ok=True)
    state=initialize()
    records=list(state["scenes"].values())
    if args.only:
        records=[r for r in records if r["id"] in args.only]
        if len(records)!=len(set(args.only)):raise ValueError("Unknown scene ID")
    # Exercise each of v1/v2 before proceeding to the other scenes.
    pilots={"v1_Nevada":0,"v2_5_points_GTv2":1}
    records.sort(key=lambda r:(pilots.get(r["id"],2),r["version"],r["bytes"]))
    remote=z=None
    with requests.Session() as sess:
        for i,r in enumerate(records,1):
            print(f"[{i}/{len(records)}] {r['id']} {r['status']}",flush=True)
            path=Path(r["existing"]) if r["existing"] else STAGE/(r["id"]+".ply")
            root=CACHE/r["id"]
            try:
                if not r["existing"] and path.exists() and not owned_path(r,path):
                    raise FileExistsError(f"Untracked staged source: {path}")
                if r["status"]=="failed" and root.exists() and (root/"index.json").exists():
                    # Keep the failed cache for inspection. Rebuild only the affected scene.
                    root.rename(root.with_name(root.name+f".failed-{time.time_ns()}"))
                if r["status"] in ("raw_removed","cache_verified"):
                    from .recovery_cache import verify_cache
                    verify_cache(root,r,deep=False)
                else:
                    retries=3
                    for attempt in range(retries):
                        try:
                            if not r["existing"] and not (path.exists() and r["crc32"]):
                                if z is None:
                                    remote=None if args.archive else RangeZip(URL,sess,True)
                                    z=zipfile.ZipFile(args.archive or remote)
                                    for entry in state["scenes"].values():
                                        m=z.getinfo(entry["member"])
                                        expected=f"{m.CRC:08x}"
                                        if m.file_size!=entry["bytes"] or (entry["crc32"] and entry["crc32"]!=expected):
                                            raise ValueError("Source ZIP inventory changed")
                                        entry["crc32"]=expected
                                    persist(state)
                                fetch(state,r,remote,z,path)
                            elif not r["existing"]:
                                change(state,r,"downloaded")
                            if path.stat().st_size!=r["bytes"]:raise ValueError("Raw size mismatch")
                            signature,crc=digest(path)
                            if crc!=r["crc32"] or (r["sha256"] and signature!=r["sha256"]):
                                if not r["existing"] and attempt+1<retries:
                                    # Retain corrupt bytes for inspection; never replace a verified PLY.
                                    path.rename(path.with_suffix(f".bad-{time.time_ns()}"))
                                raise IOError("Raw checksum mismatch")
                            break
                        except (requests.RequestException,OSError) as e:
                            if attempt+1==retries:raise
                            remote=z=None
                            print("retry",r["id"],type(e).__name__,flush=True);time.sleep(2**attempt)
                    change(state,r,"raw_verified",sha256=signature,error="")
                    if args.raw_only:return
                    space(4*r["bytes"]+128*2**20)
                    change(state,r,"caching")
                    from .recovery_cache import build,verify_cache
                    index=build(path,root,r,space)
                    change(state,r,"cached",index_sha256=sha(root/"index.json"))
                    proof=verify_cache(root,r,deep=True,raw=path)
                    change(state,r,"cache_verified",audit=proof,cache_bytes=index["cache_bytes"])
                if not r["existing"] and r["status"]!="raw_removed":
                    # Only validated job-owned input, at its specific path. A crash after unlink is resumable.
                    if path.exists():
                        if not owned_path(r,path) or sha(path)!=r["sha256"]:
                            raise ValueError("Staged source identity changed before deletion")
                        path.unlink()
                    change(state,r,"raw_removed",raw_removed=True)
                r["quota_after"]=space();persist(state)
            except Exception as e:
                change(state,r,"failed",error=f"{type(e).__name__}: {e}")
                raise
    print("Completed requested scenes",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--archive",type=Path,help="Local official STPLS3D.zip, read in place")
    p.add_argument("--only",nargs="+");p.add_argument("--raw-only",action="store_true")
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/".stream.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        run(p.parse_args())
