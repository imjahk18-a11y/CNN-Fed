"""One central training epoch on a verified real compact cache. Validation only."""
import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace
import torch
from torch.utils.data import DataLoader,Subset
from datasets.compact_cache import spatial_clients,sha
from datasets.storage_budget import budget,require_space
from experiments.run_experiments import load_data
from models.voxel_3dcnn import VoxelUNet
from training.train import train,load,seed_all
from training.evaluate import evaluate


def pilot(root,validation,out,batch,workers,variant):
    dev="cuda:0"
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    stats=json.loads((validation/"stats.json").read_text())
    if stats["manifest_sha256"]!=sha(root/"manifest.json"): raise ValueError("Audit the current manifest")
    for file,digest in json.loads((root/"manifest.json").read_text())["shards"].items():
        if sha(root/file)!=digest: raise ValueError(f"Shard modified: {file}")
    headroom=budget(out)
    require_space(headroom,100_000_000)
    out.mkdir(parents=True,exist_ok=False)
    seed_all(42)
    torch.set_num_threads(2)
    cfg=dict(seed=42,batch=batch,workers=workers,amp=True,pin_memory=True,balanced_patches=True,verify_cuda=True)
    sets,classes,side,dataset,digest=load_data(SimpleNamespace(mode="stpls3d",data=root,variant=variant),cfg)
    central,_=spatial_clients(sets["train"].rows)
    tr=Subset(sets["train"],central)
    loader=DataLoader(tr,batch_size=batch,num_workers=workers,pin_memory=True)
    x,y=next(iter(loader))
    if not x.is_pinned() or not y.is_pinned(): raise RuntimeError("Pinned host tensors expected")
    net=VoxelUNet(x.shape[1],len(classes)).to(dev)
    with torch.no_grad():
        scores=net.eval()(x.to(dev,non_blocking=True))
        if scores.shape!=(len(x),len(classes),side,side,side) or not scores.isfinite().all():
            raise RuntimeError("Invalid GPU batch")
    del scores,loader
    meta=dict(task="voxel_semantic_segmentation",dataset=dataset,classes=classes,side=side,
              variant=variant,manifest_sha256=digest)
    ckpt=out/"best_val.pt"
    begin=time.perf_counter()
    log=train(net,tr,dev,cfg,1,.001,42,sets["val"],ckpt,meta)
    epoch_wall=time.perf_counter()-begin
    del net
    model,_=load(ckpt,dev)
    metrics=evaluate(model,sets["val"],dev,batch)
    if abs(metrics["miou"]-log[0]["val_miou"])>1e-8: raise ValueError("Reloaded validation mismatch")
    data=dict(config=cfg,meta=meta,gpu=torch.cuda.get_device_name(0),pytorch=str(torch.__version__),
              cuda=torch.version.cuda,central_patches=len(tr),val_patches=len(sets["val"]),log=log,
              checkpoint_bytes=ckpt.stat().st_size,wall_train_and_val_s=epoch_wall,validation=metrics,
              input_shape=list(x.shape),labels_shape=list(y.shape),input_pinned=x.is_pinned(),labels_pinned=y.is_pinned(),
              checkpoint_sha256=sha(ckpt),disk_before=headroom,disk_after=budget(out))
    (out/"metrics.json").write_text(json.dumps(data,indent=2))


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cache",type=Path)
    ap.add_argument("--audit",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--batch",type=int,default=8)
    ap.add_argument("--workers",type=int,default=2)
    ap.add_argument("--variant",choices=("A","B"),default="B")
    args=ap.parse_args()
    pilot(args.cache,args.audit,args.out,args.batch,args.workers,args.variant)
