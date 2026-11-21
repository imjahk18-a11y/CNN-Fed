"""Run STPLS3D experiments or a separate synthetic integration smoke test."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import torch
from torch.utils.data import Subset

from datasets.stpls3d_loader import CLASSES, VoxelPatches
from datasets.compact_cache import CompactPatches, spatial_clients
from datasets.synthetic_smoke import SyntheticSmoke
from federated.server import partition, simulate
from models.voxel_3dcnn import VoxelUNet
from pruning.structured_pruning import prune
from training.evaluate import evaluate, profile
from training.train import load, save, seed_all, train

ROOT = Path(__file__).resolve().parents[1]


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def write_csv(path, rows):
    with path.open("w",newline="") as f:
        writer = csv.DictWriter(f,fieldnames=rows[0].keys())
        writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v) if isinstance(v,list) else v for k,v in row.items()})


def load_data(args, cfg):
    if args.mode == "smoke":
        if args.data:
            raise ValueError("Smoke mode uses its own generated patches")
        sets = {key:SyntheticSmoke(key,n,32,args.variant,cfg["seed"])
                for key,n in (("train",25),("val",5),("test",7))}
        return sets, SyntheticSmoke.classes, 32, "synthetic_smoke", None
    if not args.data:
        raise ValueError("STPLS3D mode requires --data pointing to a prepared manifest directory")
    source = args.data/"manifest.json"
    m = json.loads(source.read_text())
    if m["dataset"] != "STPLS3D" or m["classes"] != CLASSES:
        raise ValueError("Expected prepared STPLS3D semantic manifest")
    if args.variant == "B" and not all(s["rgb"] for s in m["sources"]):
        raise ValueError("RGB missing from one or more sites")
    if m.get("format") == "compact-v1":
        sets = {s:CompactPatches(args.data,[r for r in m["rows"] if r["split"]==s],args.variant)
                for s in ("train","val","test")}
        return sets,m["classes"],m["side"],m["dataset"],hashlib.sha256(source.read_bytes()).hexdigest()
    sets = {s:VoxelPatches(args.data,[r for r in m["rows"] if r["split"] == s],m["side"],args.variant)
            for s in ("train","val","test")}
    if not all(len(d) for d in sets.values()):
        raise ValueError("Empty split in manifest")
    return sets, m["classes"], m["side"], "STPLS3D", hashlib.sha256(source.read_bytes()).hexdigest()


def plot(out, results, rounds, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for metric, ylabel in (("miou","Test mIoU"),("model_mb","Model size (MB)"),
                           ("latency_p50_ms","Median inference latency (ms)")):
        fig,ax = plt.subplots()
        for case in ("central","IID","Non-IID"):
            rows = [r for r in results if r["partition"] == case and r["stage"] == "final"]
            ax.plot([r["pruning_pct"] for r in rows],[r[metric] for r in rows],marker="o",label=case)
        ax.set(xlabel="Target channel pruning (%)", ylabel=ylabel, title=name+" / server simulation")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out/(metric+".png"))
        plt.close(fig)
    fig,ax = plt.subplots()
    for pct in (0,20,40):
        for case in ("IID","Non-IID"):
            rows = [r for r in rounds if r["pruning_pct"] == pct and r["partition"] == case]
            ax.plot([r["round"] for r in rows],[r["miou"] for r in rows],label=f"{pct}% {case}")
    ax.set(xlabel="Communication round",ylabel="Validation mIoU",title=name+" / server simulation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out/"round_miou.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode",choices=["smoke","stpls3d"])
    ap.add_argument("--data",type=Path)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--variant",choices=["A","B"],default="A")
    ap.add_argument("--config",type=Path,default=ROOT/"configs/default.json")
    ap.add_argument("--device",default="auto")
    ap.add_argument("--batch-size",type=int)
    ap.add_argument("--cpu-threads",type=int,default=2)
    ap.add_argument("--no-amp",action="store_true")
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    if args.mode == "smoke":
        cfg.update(batch=3,pretrain_epochs=4,finetune_epochs=1,rounds=2,latency_repeats=8)
    if args.batch_size is not None:
        cfg["batch"] = args.batch_size
    if args.no_amp:
        cfg["amp"] = False
    if min(cfg[k] for k in ("batch","pretrain_epochs","finetune_epochs","rounds","local_epochs","latency_repeats")) < 1:
        ap.error("Batch, epochs, rounds and latency repeats must be positive")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if torch.device(device).type == "cuda" and not torch.cuda.is_available():
        ap.error("CUDA unavailable in this Python environment")
    torch.set_num_threads(args.cpu_threads)
    seed_all(cfg["seed"])
    sets,classes,side,name,manifest_hash = load_data(args,cfg)
    central, assignments = (spatial_clients(sets["train"].rows) if isinstance(sets["train"],CompactPatches)
                            else partition(sets["train"].rows,cfg["seed"]))
    central_set = Subset(sets["train"],central)
    meta = dict(dataset=name, variant=args.variant, task="voxel_semantic_segmentation",
                classes=classes, side=side, manifest_sha256=manifest_hash)
    run = args.out
    run.mkdir(parents=True,exist_ok=False)
    ckpt = run/"checkpoints"
    ckpt.mkdir()
    result_dir = run/"results"
    result_dir.mkdir()
    (run/"plots").mkdir()
    meta.update(simulation="two sequential UAV clients", device=device)
    dump(run/"config.json",dict(**meta, settings=cfg, torch=str(torch.__version__), numpy=np.__version__, python=platform.python_version(),
                                machine=platform.machine(), cpu_threads=torch.get_num_threads(),
                                gpu=torch.cuda.get_device_name(device) if torch.device(device).type=="cuda" else None,
                                data=str(args.data.resolve()) if args.data else None))
    dump(run/"splits.json",dict(train=[r["id"] for r in sets["train"].rows],
                                val=[r["id"] for r in sets["val"].rows], test=[r["id"] for r in sets["test"].rows],
                                central_indices=central, client_indices=assignments,
                                client_regions={case:[[sets["train"].rows[i]["region"] for i in ids] for ids in pair]
                                                for case,pair in assignments.items()}))
    model = VoxelUNet(2 if args.variant=="A" else 5,len(classes),cfg["widths"]).to(device)
    logs = {}
    baseline_file = ckpt/"E1_best.pt"
    logs["pretrain"] = train(model,central_set,device,cfg,cfg["pretrain_epochs"],cfg["pretrain_lr"],cfg["seed"],
                             validation=sets["val"],best_path=baseline_file,meta=meta)
    model.cpu()
    base, best = load(baseline_file)
    logs["best_epoch"] = best["best_epoch"]
    dump(run/"training_log.json",logs)
    example = sets["val"][0][0].unsqueeze(0).to(device)
    results, rounds = [], []

    def record(net, path, exp, pct, case, stage, communication):
        net.to(device)
        row = dict(dataset=name,variant=args.variant,id=exp,pruning_pct=pct,partition=case,stage=stage,
                   **evaluate(net,sets["test"],device,cfg["batch"]), **profile(net,example,device,cfg["latency_repeats"]),
                   checkpoint_mb=path.stat().st_size/1e6, communication_mb=communication)
        net.cpu()
        ref = results[0] if results else row
        row.update(parameter_reduction=1-row["parameters"]/ref["parameters"],
                   mac_reduction=1-row["macs"]/ref["macs"], speedup=ref["latency_p50_ms"]/row["latency_p50_ms"])
        if pct and min(row["parameter_reduction"],row["mac_reduction"]) <= 0:
            raise RuntimeError("Expected fewer parameters and Conv3d MACs")
        results.append(row)

    for i,pct in enumerate((0,20,40)):
        exp, fl = f"E{i+1}", f"E{i+4}"
        if pct:
            net, keep = prune(base.to(device),pct/100,example)
            base.cpu()
            dump(ckpt/(exp+"_channels.json"),keep)
            path = ckpt/(exp+"_pruned.pt")
            save(path,net,{**meta,"pruning_pct":pct,"keep":keep})
            record(net,path,exp,pct,"central","before_ft",0.)
            logs[exp] = train(net.to(device),central_set,device,cfg,cfg["finetune_epochs"],cfg["finetune_lr"],cfg["seed"])
            path = ckpt/(exp+"_finetuned.pt")
            save(path,net,{**meta,"pruning_pct":pct,"keep":keep})
            net.cpu()
        else:
            path = baseline_file
        net,_ = load(path)
        record(net,path,exp,pct,"central","final",0.)
        for case,pair in assignments.items():
            global_model,_ = load(path)
            history = simulate(global_model,sets["train"],pair,sets["val"],device,cfg,cfg["seed"])
            for r in history:
                rounds.append(dict(dataset=name,variant=args.variant,id=fl,pruning_pct=pct,partition=case,**r))
            final = ckpt/(fl+"_"+case+".pt")
            save(final,global_model,{**meta,"pruning_pct":pct,"partition":case,"initial":path.name})
            restored,_ = load(final)
            record(restored,final,fl,pct,case,"final",history[-1]["cumulative_mb"])
        dump(run/"training_log.json",logs)
        write_csv(result_dir/"experiment_results.csv",results)
        write_csv(result_dir/"federated_round_metrics.csv",rounds)
    write_csv(result_dir/"pruning_comparison.csv",[r for r in results if r["partition"]=="central"])
    plot(run/"plots",results,rounds,name)
    print(f"{name}: completed {len(results)} evaluations and {len(rounds)} FL rounds; output {run}")


if __name__ == "__main__":
    main()
