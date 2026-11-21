"""Train on official synthetic scenes, freeze candidates, then score WMSC exactly once per checkpoint."""
import argparse
import json
import math
from pathlib import Path
import platform

import torch
from torch.utils.data import DataLoader

from datasets.compact_cache import atomic_json,sha
from datasets.s2r_scenes import OUT, read, table, training_sets, scene_dataset
from federated.server import simulate
from models.voxel_3dcnn import VoxelUNet
from pruning.structured_pruning import prune
from training.evaluate import evaluate,profile
from training.train import seed_all,train,load,save

DEVICE="cuda:0"
CFG=dict(batch=8,amp=True,seed=42,workers=2,pin_memory=True,balanced_patches=False,verify_cuda=True,
         rounds=10,local_epochs=1,local_lr=1e-4)


def initialize():
    if not torch.cuda.is_available():raise RuntimeError("CUDA required")
    torch.set_num_threads(2)
    seed_all(42)


def context(out):
    m=read(out/"dataset_manifest.json")
    central,val,local=training_sets(m)
    return m,central,val,local


def smoke(out):
    m,central,val,_=context(out)
    path=out/"smoke"
    path.mkdir()
    net=VoxelUNet(5,6).to(DEVICE)
    x,y=next(iter(DataLoader(central,batch_size=8,num_workers=2,pin_memory=True)))
    with torch.no_grad():
        shape=list(net.eval()(x.to(DEVICE)).shape)
    if shape!=[8,6,32,32,32] or not x.is_pinned() or not y.is_pinned(): raise ValueError("GPU loader smoke failed")
    logs=train(net,central,DEVICE,CFG,1,1e-3,42,val,path/"smoke.pt",{"source":"synthetic_only_smoke"})
    restored,_=load(path/"smoke.pt",DEVICE)
    score=evaluate(restored,val,DEVICE,8)
    if abs(score["miou"]-logs[0]["val_miou"])>1e-9:raise ValueError("Checkpoint validation mismatch")
    atomic_json(path/"passed.json",dict(manifest_sha256=sha(out/"dataset_manifest.json"),input=list(x.shape),output=shape,
                                       pinned=True,metrics=score,logs=logs,gpu=torch.cuda.get_device_name(0),
                                       torch=str(torch.__version__),python=platform.python_version()))


def meta(m):
    return dict(training_domain="STPLS3D Synthetic_v3",classes=m["classes"],variant="B",side=32,initialization="random_seed_42")


def train_and_select(out):
    proof=read(out/"smoke/passed.json")
    if proof["manifest_sha256"]!=sha(out/"dataset_manifest.json"):raise ValueError("Changed data")
    m,central,val,local=context(out)
    ck=out/"checkpoints"
    ck.mkdir()
    info=meta(m)
    config=dict(settings=CFG,pretrain=dict(epochs=20,lr=.001,scheduler="cosine",selection="highest synthetic val mIoU"),
                ft=dict(lrs=[.0001,.001],epochs=[5,10,20],tie_break="earlier epoch then lower lr",
                        gradual="try matched 2-stage if immediate 40% loss >=0.05 mIoU"),
                fl_selection="fixed final round 10",external_access="after freezing all models")
    atomic_json(out/"protocol.json",config)
    basefile=ck/"E1_best.pt"
    model=VoxelUNet(5,6).to(DEVICE)
    result=train(model,central,DEVICE,CFG,20,.001,42,val,basefile,info)
    table(out/"baseline_training.csv",result)
    model.cpu()
    base,base_meta=load(basefile)
    epoch=base_meta["best_epoch"]
    baseline_score=base_meta["val_miou"]
    print(f"Baseline best epoch {epoch}: {baseline_score:.5f}",flush=True)
    ref=val[0][0].unsqueeze(0).to(DEVICE)
    def specification(i,pct,partition,checkpoint,ft=0,lr=0,method="original",rounds=0,comm=0,peak=0):
        return dict(id=i,pruning_pct=pct,partition=partition,checkpoint=str(checkpoint.relative_to(out)),
                    sha256=sha(checkpoint),pretrain_checkpoint_epoch=epoch,ft_epochs=ft,ft_lr=lr,
                    method=method,fl_rounds=rounds,communication_mb=comm,peak_training_mb=peak,
                    ft_patch_visits=ft*len(central),fl_patch_visits=rounds*len(local))
    frozen=[specification("E1",0,"central",basefile,peak=max(r["peak_training_mb"] for r in result))]
    pruning=[]
    selected={0:frozen[0]}
    for pct in (0,20,40):
        initial=basefile
        if pct:
            small,keep=prune(base.to(DEVICE),pct/100,ref)
            base.cpu()
            initial=ck/f"raw_{pct}.pt"
            save(initial,small.cpu(),{**info,"pruning_pct":pct,"keep":keep})
        fresh,_=load(initial,DEVICE)
        immediate=evaluate(fresh,val,DEVICE,8)
        fresh.cpu()
        pruning.append(dict(pruning_pct=pct,method="one_shot" if pct else "control",lr=0.,epoch=0,
                            val_miou=immediate["miou"],loss=immediate["loss"],peak_training_mb=None,
                            checkpoint=initial.name))
        for rate in (1e-4,1e-3):
            candidate,_=load(initial,DEVICE)
            def snapshot(net,row):
                if row["epoch"] not in (5,10,20):return
                scores=evaluate(net,val,DEVICE,8)
                path=ck/f"ft_{pct}_{rate}_{row['epoch']}.pt"
                save(path,net,{**info,"ft_lr":rate,"ft_epoch":row["epoch"],"pruning_pct":pct})
                pruning.append(dict(pruning_pct=pct,method="one_shot" if pct else "control",lr=rate,epoch=row["epoch"],
                                    val_miou=scores["miou"],loss=scores["loss"],peak_training_mb=row["peak_training_mb"],
                                    checkpoint=path.name))
            train(candidate,central,DEVICE,CFG,20,rate,42,epoch_callback=snapshot)
            candidate.cpu()
        winner=max((r for r in pruning if r["pruning_pct"]==pct and r["epoch"]>0),
                   key=lambda r:(r["val_miou"],-r["epoch"],-r["lr"]))
        if pct==40 and baseline_score-immediate["miou"]>=.05:
            half=winner["epoch"]//2
            rate=winner["lr"]
            early,_=load(ck/"raw_20.pt",DEVICE)
            train(early,central,DEVICE,CFG,half,rate,42)
            final,_=prune(early,.4,ref,target_widths=fresh.spec["widths"])
            early.cpu()
            log=train(final,central,DEVICE,CFG,winner["epoch"]-half,rate,42+half)
            scores=evaluate(final,val,DEVICE,8)
            path=ck/"ft_40_gradual.pt"
            save(path,final.cpu(),{**info,"pruning_pct":40,"ft_epoch":winner["epoch"],"ft_lr":rate})
            row=dict(pruning_pct=40,method="gradual",lr=rate,epoch=winner["epoch"],val_miou=scores["miou"],
                     loss=scores["loss"],peak_training_mb=max(x["peak_training_mb"] for x in log),checkpoint=path.name)
            pruning.append(row)
            if row["val_miou"]>winner["val_miou"]:winner=row
        if pct:
            item=specification("E2" if pct==20 else "E3",pct,"central",ck/winner["checkpoint"],
                               winner["epoch"],winner["lr"],winner["method"],peak=winner["peak_training_mb"])
            selected[pct]=item
            frozen.append(item)
        table(out/"pruning_results.csv",pruning)
        print(f"Grid {pct}% complete",flush=True)
    # Same LR and additional patch visits as each selected pruned model, including a matched unpruned network.
    for pct in (20,40):
        s=selected[pct]
        control=next(r for r in pruning if r["pruning_pct"]==0 and r["epoch"]==s["ft_epochs"] and r["lr"]==s["ft_lr"])
        frozen.append(specification(f"control_for_{pct}",0,"central",ck/control["checkpoint"],control["epoch"],
                                    control["lr"],"matched_control",peak=control["peak_training_mb"]))
    history=[]
    for pct,exp in ((0,"E4"),(20,"E5"),(40,"E6")):
        p=selected[pct]
        for case,pair in m["client_indices"].items():
            model,_=load(out/p["checkpoint"])
            rounds=simulate(model,local,pair,val,DEVICE,CFG,42)
            file=ck/f"{exp}_{case}.pt"
            save(file,model,{**info,"initial_sha256":p["sha256"],"partition":case,"round":10})
            frozen.append(specification(exp,pct,case,file,p["ft_epochs"],p["ft_lr"],p["method"],10,
                                        rounds[-1]["cumulative_mb"],max(r["peak_training_mb"] for r in rounds)))
            for r in rounds:
                history.append(dict(id=exp,pruning_pct=pct,partition=case,split="synthetic_val",**{k:v for k,v in r.items() if k!="per_class_iou"},
                                    per_class_iou=json.dumps(r["per_class_iou"])))
            table(out/"federated_results.csv",history)
            print(f"{exp} {case} finished",flush=True)
    atomic_json(out/"frozen_models.json",dict(dataset_sha256=sha(out/"dataset_manifest.json"),models=frozen))


def external_test(out):
    m=read(out/"dataset_manifest.json")
    frozen=read(out/"frozen_models.json")
    if (out/"synthetic_to_real_comparison.csv").exists():raise FileExistsError("External evaluations already present")
    if frozen["dataset_sha256"]!=sha(out/"dataset_manifest.json"):raise ValueError("Manifest changed after freeze")
    for row in frozen["models"]:
        if sha(out/row["checkpoint"])!=row["sha256"]:raise ValueError("Checkpoint changed after freeze")
    val=scene_dataset(m["scene_assignment"]["val"])
    wmsc=scene_dataset(m["external"])
    ref=val[0][0].unsqueeze(0).to(DEVICE)
    results=[]
    measured={}
    for row in frozen["models"]:
        key=row["sha256"]
        if key not in measured:
            model,_=load(out/row["checkpoint"],DEVICE)
            metrics=profile(model,ref,DEVICE,100)
            for name,d in (("synthetic_val",val),("wmsc_test",wmsc)):
                scores=evaluate(model,d,DEVICE,8)
                for field in ("accuracy","macro_f1","miou","valid_voxels"):
                    metrics[name+"_"+field]=scores[field]
                for c,iou in zip(m["classes"],scores["per_class_iou"]):
                    metrics[name+"_iou_"+c]=iou
            model.cpu()
            measured[key]=metrics
        results.append({**row,**measured[key]})
    for pct in (20,40):
        pr=next(r for r in results if r["pruning_pct"]==pct)
        if any(pr[k]>=results[0][k] for k in ("parameters","macs")):raise RuntimeError("Physical pruning check failed")
    table(out/"synthetic_to_real_comparison.csv",results)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["smoke","fit","test"])
    parser.add_argument("--out",type=Path,default=OUT)
    a=parser.parse_args()
    initialize()
    {"smoke":smoke,"fit":train_and_select,"test":external_test}[a.stage](a.out)


if __name__=="__main__":main()
