"""Compare the existing FL endpoints using two A6000 client processes and a CPU server."""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import socket
import time

import numpy as np
import torch

from datasets.compact_cache import atomic_json,sha
from datasets.fullz import Scene
from experiments.distributed_ai import CACHE,SCENES,startfile
from federated.fl_server import run_server
from federated.uav_client import serve_client

OUT=Path("results/jetson_phase5_virtual")
CONFIG=Path("configs/jetson_phase5_virtual.json")
VAL=Path("results/distributed_ai_phase3/bundles/validation/patches.npz")


def prior():
    root=Path("results")
    return {str(p):sha(p) for p in sorted(root.rglob("*")) if p.is_file() and OUT not in p.parents}


def prepare(config):
    OUT.mkdir(exist_ok=True)
    split=json.loads(Path(config["split"]).read_text())
    for idx,refs in enumerate(split["NonIID"],1):
        role=f"client{idx}"
        dest=OUT/"data"/role
        dest.mkdir(parents=True,exist_ok=True)
        sid=refs[0][0]
        assert {s for s,_ in refs}=={sid} and len(refs)==len(set(map(tuple,refs)))==96
        scene=Scene(CACHE/SCENES[sid])
        manifest=dest/"manifest.json"
        if manifest.exists():
            info=json.loads(manifest.read_text())
            assert info["split_sha"]==sha(Path(config["split"])) and [p["ref"] for p in info["patches"]]==refs
            assert all(sha(dest/p["file"])==p["sha256"] for p in info["patches"])
            continue
        files=[]
        for j,(s,i) in enumerate(refs):
            c,x,y=scene[i]
            name=f"patch_{j:03}.npz"
            np.savez_compressed(dest/name,coords=c,rgb=x,labels=y)
            files.append(dict(file=name,ref=[s,i],sha256=sha(dest/name)))
        atomic_json(manifest,dict(role=role,scene=sid,patches=files,source_sha=scene.meta["sha256"],
                                  split_sha=sha(Path(config["split"]))))
    before=prior()
    old=OUT/"prior_sha256.json"
    if old.exists(): assert json.loads(old.read_text())==before
    else: atomic_json(old,before)
    return before


class CommittedInterruption(Exception): pass


def server_entry(listener,token,dest,ck,mode,rounds,cfg,interrupt):
    # spawn transfers the descriptor but does not retain the Python socket timeout.
    listener.settimeout(90)
    def committed(r):
        if interrupt and r==1: raise CommittedInterruption()
    try:
        run_server(listener,token,ck,VAL,dest, cfg["seed"],mode,"cpu",rounds,0,
                   edge_batch=cfg["batch_size"],after_commit=committed)
        assert not torch.cuda.is_initialized()
    except CommittedInterruption:
        atomic_json(dest/"interruption.json",dict(committed=1,pid=os.getpid(),time=time.time()))
        raise SystemExit(75)
    finally: listener.close()


def run(config,pct,mode,kind):
    n=config["rounds"] if kind=="clean" else config["smoke_rounds"]
    dest=OUT/"runs"/f"{kind}_p{pct}_{mode}"
    if (dest/"verified.json").exists():return
    dest.mkdir(parents=True,exist_ok=True)
    ck=startfile(config["seed"],pct)
    ctx=mp.get_context("spawn")
    token=os.urandom(16); clients=[]; servers=[]
    with socket.socket() as listener:
        listener.bind(("127.0.0.1",0)); listener.listen(4); listener.settimeout(90)
        address=listener.getsockname()
        def start_server(inject):
            proc=ctx.Process(target=server_entry,args=(listener,token,dest,ck,mode,n,config,inject))
            proc.start(); servers.append(proc); return proc
        try:
            server=start_server(kind=="recovery" and not (dest/"server.pt").exists())
            for idx in (0,1):
                role=f"client{idx+1}"
                proc=ctx.Process(target=serve_client,args=(*address,idx,OUT/"data"/role/"manifest.json",dest/role,
                           token,"cuda:0",False,kind=="recovery" and idx==1,config["batch_size"],
                           config["allocator_cap_gib"][str(config["comparison_budget_gib"])]))
                proc.start(); clients.append(proc)
            deadline=time.monotonic()+600
            while server.is_alive() and time.monotonic()<deadline:
                server.join(0.5)
                if any(c.exitcode not in (0,None) for c in clients):raise RuntimeError("CUDA client failed")
                if server.exitcode==75:
                    server=start_server(False)
            if server.exitcode!=0: raise RuntimeError(f"Ground server exit {server.exitcode}")
            for c in clients:
                c.join(20)
                if c.exitcode!=0: raise RuntimeError(f"Client exit {c.exitcode}")
        finally:
            for proc in clients+servers:
                if proc.is_alive(): proc.terminate(); proc.join()
    done=json.loads((dest/"done.json").read_text())
    rows=done["rows"]
    assert [r["round"] for r in rows]==list(range(n+1))
    assert all(r["aggregation_device"]=="cpu" for r in rows[1:])
    for i,p in enumerate(clients,1):
        for r in range(1,n+1):
            m=json.loads((dest/f"client{i}"/f"metrics_{r:02}.json").read_text())
            assert m["gpu"]=="NVIDIA RTX A6000" and m["inference_voxels"]>0 and m["steps"]==96 and m["checkpoint_sha"]==sha(ck)
    assert len({clients[0].pid,clients[1].pid,server.pid})==3
    if kind=="recovery":
        expected=OUT/"runs"/f"smoke_p{pct}_{mode}"/"server.pt"
        a,b=(torch.load(path,map_location="cpu",weights_only=True)["state"] for path in (expected,dest/"server.pt"))
        assert all(torch.equal(a[k],b[k]) for k in a)
        assert (dest/"client2/replay_02.json").exists() and sum(r.get("retries",0) for r in rows)==1
        assert (dest/"interruption.json").exists()
    atomic_json(dest/"verified.json",dict(kind=kind,pruning=pct,codec=mode,rounds=n,
        client_pids=[c.pid for c in clients],server_pids=[p.pid for p in servers],start_sha=sha(ck)))
    print(kind,pct,mode,n,round(rows[-1]["miou"],5),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--prepare-only",action="store_true")
    args=parser.parse_args()
    cfg=json.loads(CONFIG.read_text())
    before=prepare(cfg)
    if args.prepare_only:return
    for pct in cfg["pruning_percent"]:
        for kind in ("smoke","clean"):
            for mode in cfg["codecs"]: run(cfg,pct,mode,kind)
        if pct==0:
            for mode in cfg["codecs"]: run(cfg,pct,mode,"recovery")
    assert prior()==before
    atomic_json(OUT/"fl_verified.json",dict(preserved_files=len(before),config_sha=sha(CONFIG)))


if __name__=="__main__": main()
