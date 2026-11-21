"""CUDA-only synthetic experiments reusing the Phase 1–4 pipeline."""
import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import Subset

from experiments.run_experiments import ROOT, dump, load_data, write_csv
from federated.server import partition, simulate
from models.voxel_3dcnn import VoxelUNet
from pruning.structured_pruning import prune
from training.evaluate import evaluate, profile
from training.train import load, save, seed_all, train

DEVICE = "cuda:0"
CPU_RUN = ROOT/"results/synthetic_smoke_seed42"
DEFAULT_OUT = ROOT/"results/gpu_experiments"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def environment():
    cmd = ["nvidia-smi","--query-gpu=index,uuid,name,driver_version,memory.total,memory.free,utilization.gpu",
           "--format=csv,noheader,nounits"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    env = dict(timestamp=datetime.now(timezone.utc).isoformat(), host=platform.node(), executable=sys.executable,
               python=platform.python_version(), torch=str(torch.__version__), torch_cuda=torch.version.cuda,
               available=torch.cuda.is_available(), count=torch.cuda.device_count(), target=DEVICE,
               visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), nvidia_smi=result.stdout.strip(),
               nvidia_smi_error=result.stderr.strip(), gpu_name=None, total_bytes=None, free_bytes=None,
               tensor_check=False)
    try:
        env["gpu_name"] = torch.cuda.get_device_name(0)
        env["free_bytes"],env["total_bytes"] = torch.cuda.mem_get_info(0)
        x = torch.arange(8,device=DEVICE)
        env["tensor_sum"] = x.square().sum().item()
        torch.cuda.synchronize(0)
        env["tensor_check"] = x.device == torch.device(DEVICE) and env["tensor_sum"] == 140
    except (AssertionError,RuntimeError) as exc:
        env["error"] = str(exc)
    return env


@contextmanager
def telemetry(out):
    with out.open("x") as stream:
        stream.write("timestamp,gpu_uuid,utilization_pct,device_used_mib\n")
        stream.flush()
        proc = subprocess.Popen(["nvidia-smi","--query-gpu=timestamp,uuid,utilization.gpu,memory.used",
                                 "--format=csv,noheader,nounits","-lms","100"],stdout=stream)
        try:
            yield
        finally:
            proc.terminate()
            proc.wait()


def get_data(cfg):
    data,classes,side,name,_ = load_data(SimpleNamespace(mode="smoke",data=None,variant="A"),cfg)
    central,pairs = partition(data["train"].rows,cfg["seed"])
    split = {k:[r["id"] for r in d.rows] for k,d in data.items()}
    split.update(central_indices=central,client_indices=pairs)
    old = json.loads((CPU_RUN/"splits.json").read_text())
    if any(v != old[k] for k,v in split.items()):
        raise RuntimeError("Synthetic data assignment differs from CPU reference")
    return data,Subset(data["train"],central),pairs,dict(dataset=name,classes=classes,side=side,variant="A",device=DEVICE),split


def smoke(out,cfg):
    out.mkdir(parents=True,exist_ok=False)
    report = out/"gpu_smoke_report.md"
    env = environment()
    dump(out/"gpu_environment.json",env)
    cfg.update(pretrain_epochs=2, finetune_epochs=2, rounds=2, pruning_pct=20)
    dump(out/"smoke_config.json",cfg)
    if not env["tensor_check"]:
        report.write_text("# CUDA smoke test\n\nCUDA tensor preflight failed. "+env.get("error","")+"\n")
        raise RuntimeError("CUDA preflight failed")
    data,central,pairs,meta,split = get_data(cfg)
    dump(out/"splits.json",split)
    ck = out/"smoke_checkpoints"
    ck.mkdir()
    with telemetry(out/"smoke_telemetry.csv"):
        net = VoxelUNet(2,len(meta["classes"]),cfg["widths"]).to(DEVICE)
        logs = train(net,central,DEVICE,cfg,2,cfg["pretrain_lr"],cfg["seed"],data["val"],ck/"best.pt",meta)
        net.cpu()
        best,_ = load(ck/"best.pt",DEVICE)
        example = data["val"][0][0].unsqueeze(0).to(DEVICE)
        small,keep = prune(best,.2,example)
        base_profile = profile(best,example,DEVICE,50)
        best.cpu()
        ft = train(small,central,DEVICE,cfg,2,cfg["finetune_lr"],cfg["seed"])
        save(ck/"ft.pt",small,{**meta,"keep":keep})
        small.cpu()
        small,_ = load(ck/"ft.pt",DEVICE)
        small_profile = profile(small,example,DEVICE,50)
        if any(small_profile[k] >= base_profile[k] for k in ("parameters","macs")):
            raise RuntimeError("Pruning did not reduce model size and computation")
        small.cpu()
        history = simulate(small,data["train"],pairs["IID"],data["val"],DEVICE,cfg,cfg["seed"])
        save(ck/"global.pt",small,meta)
        loaded,_ = load(ck/"global.pt",DEVICE)
        final = evaluate(loaded,data["test"],DEVICE,cfg["batch"])
    dump(out/"smoke_metrics.json",dict(pretrain=logs,finetune=ft,baseline=base_profile,pruned=small_profile,
                                       rounds=history,test=final))
    peak = max(r["peak_training_mb"] for r in logs+ft+history)
    report.write_text(f"# 합성 CUDA smoke test\n\n{env['gpu_name']} · {DEVICE} · PyTorch {env['torch']}.\n\n"
                      f"사전학습 {len(logs)} epochs, 20% 프루닝, 미세조정 {len(ft)} epochs, 두 클라이언트 IID {len(history)} rounds 완료. "
                      f"모든 배치에서 모델·입력·레이블·출력·손실·gradient의 CUDA 위치를 검사했다. "
                      f"FedAvg 집계는 CPU에서 수행했다. 최종 체크포인트 재로딩 후 test mIoU {final['miou']:.6f}. "
                      f"학습 peak {peak:.2f} MB.\n")
    dump(out/"smoke_pass.json",dict(device=DEVICE,torch=env["torch"],gpu_name=env["gpu_name"],
                                    global_sha256=sha(ck/"global.pt")))


def compare(out,cfg,milestones,fl_epoch):
    proof = json.loads((out/"smoke_pass.json").read_text())
    env = environment()
    if (not env["tensor_check"] or env["torch"] != proof["torch"] or env["gpu_name"] != proof["gpu_name"]
            or sha(out/"smoke_checkpoints/global.pt") != proof["global_sha256"]):
        raise RuntimeError("GPU smoke receipt differs from the current environment or checkpoint")
    data,central,pairs,meta,split = get_data(cfg)
    if split != json.loads((out/"splits.json").read_text()):
        raise RuntimeError("Split mismatch")
    root = out/"comparison_checkpoints"
    root.mkdir()
    source = CPU_RUN/"checkpoints/E1_best.pt"
    dump(out/"comparison_config.json",dict(settings=cfg,milestones=milestones,fl_start_epoch=fl_epoch,
                                          baseline=str(source),baseline_sha256=sha(source),**meta))
    example = data["val"][0][0].unsqueeze(0).to(DEVICE)
    central_rows,fl_rows = [],[]
    ft_logs = {}
    with telemetry(out/"comparison_telemetry.csv"):
        for pct in (0,20,40):
            net,_ = load(source,DEVICE)
            if pct:
                small,keep = prune(net,pct/100,example)
                net.cpu()
                net = small
                dump(root/f"pruning_{pct}.json",keep)
            prof = profile(net,example,DEVICE,cfg["latency_repeats"])
            def record(epoch,seconds,peak,allocated):
                metrics = evaluate(net,data["test"],DEVICE,cfg["batch"])
                central_rows.append(dict(dataset=meta["dataset"],device=DEVICE,pruning_pct=pct,ft_epochs=epoch,
                                         **metrics,**prof,ft_time_s=seconds,peak_training_mb=peak,allocated_mb=allocated,
                                         communication_mb=0))
            record(0,0.,None,None)
            fl_source = source
            if pct:
                fl_source = root/f"pruning_{pct}_ep{fl_epoch}.pt"
                elapsed,peak = 0.,0.
                def checkpoint(model,row):
                    nonlocal elapsed,peak
                    elapsed += row["seconds"]
                    peak = max(peak,row["peak_training_mb"])
                    if row["epoch"] in milestones:
                        save(root/f"pruning_{pct}_ep{row['epoch']}.pt",model,{**meta,"pruning_pct":pct,"epoch":row["epoch"]})
                        record(row["epoch"],elapsed,peak,row["allocated_mb"])
                ft_logs[pct] = train(net,central,DEVICE,cfg,max(milestones),cfg["finetune_lr"],cfg["seed"],epoch_callback=checkpoint)
            net.cpu()
            for case,pair in pairs.items():
                global_model,_ = load(fl_source)
                rounds = simulate(global_model,data["train"],pair,data["val"],DEVICE,cfg,cfg["seed"])
                model_file = root/f"global_{pct}_{case}.pt"
                save(model_file,global_model,{**meta,"source_sha256":sha(fl_source),"partition":case})
                restored,_ = load(model_file,DEVICE)
                final = evaluate(restored,data["test"],DEVICE,cfg["batch"])
                final_prof = profile(restored,example,DEVICE,cfg["latency_repeats"])
                restored.cpu()
                for r in rounds:
                    last = r["round"] == cfg["rounds"]
                    fl_rows.append(dict(dataset=meta["dataset"],pruning_pct=pct,partition=case,
                                        initial_ft_epoch=fl_epoch if pct else 0,**r,eval_split="val",
                                        final_test_miou=final["miou"] if last else None,
                                        final_test_macro_f1=final["macro_f1"] if last else None,
                                        **{k:v if last else None for k,v in final_prof.items()}))
            write_csv(out/"gpu_pruning_comparison.csv",central_rows)
            write_csv(out/"gpu_federated_results.csv",fl_rows)
            dump(out/"comparison_training_logs.json",ft_logs)
        # Attribution probes use validation only and never select the FL starting checkpoint.
        original, _ = load(source,DEVICE)
        diagnostics = []
        def probe(label,model,epochs=0,lr=None):
            scores = evaluate(model,data["val"],DEVICE,cfg["batch"])
            diagnostics.append(dict(condition=label,epochs=epochs,lr=lr,widths=model.spec["widths"],**scores))
        probe("original",original)
        for width,layer in enumerate(("encoder","bridge","decoder")):
            widths = original.spec["widths"].copy()
            widths[width] -= int(widths[width]*.4+.5)
            ablated,_ = prune(original,.4,example,target_widths=widths)
            probe("only_"+layer,ablated)
            ablated.cpu()
        target,_ = prune(original,.4,example)
        wanted = target.spec["widths"]
        probe("one_shot",target)
        for rate in (cfg["finetune_lr"], cfg["finetune_lr"]*10):
            trial,_ = prune(original,.4,example)
            def check_lr(model,row):
                if row["epoch"] in milestones:
                    probe("one_shot_ft",model,row["epoch"],rate)
            train(trial,central,DEVICE,cfg,max(milestones),rate,cfg["seed"],epoch_callback=check_lr)
            trial.cpu()
        first,_ = prune(original,.2,example)
        original.cpu()
        early = max(milestones)//2
        if early:
            train(first,central,DEVICE,cfg,early,cfg["finetune_lr"],cfg["seed"])
            last,_ = prune(first,.4,example,target_widths=wanted)
            train(last,central,DEVICE,cfg,max(milestones)-early,cfg["finetune_lr"],cfg["seed"]+early)
            probe("gradual_20_to_40",last,max(milestones),cfg["finetune_lr"])
        write_csv(out/"gpu_pruning_diagnostics.csv",diagnostics)
    from experiments.gpu_report import report
    report(out,central_rows,fl_rows,milestones,fl_epoch)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage",choices=["smoke","compare"])
    ap.add_argument("--out",type=Path,default=DEFAULT_OUT)
    ap.add_argument("--ft-epochs",nargs="+",type=int,default=[5,10,20])
    ap.add_argument("--fl-ft-epoch",type=int,default=10)
    ap.add_argument("--ft-lr",type=float,default=.0001)
    a = ap.parse_args()
    if min(a.ft_epochs)<1 or a.fl_ft_epoch not in a.ft_epochs or a.ft_lr<=0:
        ap.error("Positive milestones and LR required; include the FL checkpoint epoch in the milestones")
    cfg=json.loads((ROOT/"configs/default.json").read_text())
    cfg.update(batch=3,amp=False,verify_cuda=True,latency_repeats=100,finetune_lr=a.ft_lr)
    seed_all(cfg["seed"])
    torch.set_num_threads(2)
    if a.stage=="smoke":
        smoke(a.out,cfg)
    else:
        compare(a.out,cfg,sorted(set(a.ft_epochs)),a.fl_ft_epoch)


if __name__=="__main__":
    main()
