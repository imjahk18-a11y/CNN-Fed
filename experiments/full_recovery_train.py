"""Lazy Full-Z DataLoader, GPU smoke and split-driven synthetic training."""
import argparse
from bisect import bisect_right
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader,Subset
from torch.nn import functional as F

from datasets.compact_cache import atomic_json,sha
from datasets.recovery_cache import Frames
from datasets.recover_synthetic import CACHE,OUT
from experiments.sparse_reproduction import save,load,scores_from_confusion,write_csv
from models.active_voxel import geometry,SparseUNet
from training.train import seed_all

DEVICE=torch.device("cuda:0")


class Patches(Dataset):
    def __init__(self,ids,channels,hashes):
        self.scenes=[]
        self.channels=channels
        sizes=[]
        for ident in ids:
            root=CACHE/ident
            if sha(root/"index.json")!=hashes[ident]:raise IOError(f"Index changed: {ident}")
            ds=Frames(root)
            self.scenes.append(ds)
            sizes.append(len(ds.rows))
        self.bounds=np.r_[0,np.cumsum(sizes)].tolist()

    def __len__(self):
        return self.bounds[-1]

    def __getitem__(self,i):
        sid=bisect_right(self.bounds,i)-1
        scene=self.scenes[sid];xyz,rgb,_,y=scene.read(i-self.bounds[sid])
        x=rgb/np.float32(255)
        if self.channels==6:
            denom=np.array([31,31,max(scene.meta["shape"][2]-1,1)],np.float32)
            x=np.column_stack((x,xyz.astype(np.float32)/denom))
        return xyz,x,y.copy(),sid


def collate(samples):
    coords=[];features=[];labels=[];ids=[]
    for b,(xyz,x,y,s) in enumerate(samples):
        coords.append(np.column_stack((np.full(len(xyz),b,np.int32),xyz)))
        features.append(x);labels.append(y);ids.append(s)
    return (torch.from_numpy(np.concatenate(coords)),torch.from_numpy(np.concatenate(features)),
            torch.from_numpy(np.concatenate(labels)),ids)


def loader(ds,shuffle=False):
    return DataLoader(ds,batch_size=24,shuffle=shuffle,num_workers=2,pin_memory=True,collate_fn=collate,
                      persistent_workers=True,generator=torch.Generator().manual_seed(42))


def on_gpu(batch):
    coords,x,y,_=batch
    return geometry(coords.to(DEVICE,non_blocking=True)),x.to(DEVICE,non_blocking=True),y.to(DEVICE,torch.long,non_blocking=True)


def smoke(split, pilot=False):
    ids=split["train"]["scenes"]+split["validation"]["scenes"]
    results={}
    for channels in (3,6):
        ds=Patches(ids,channels,split["index_hashes"])
        rng=np.random.default_rng(42)
        # Interleave every scene across batches. Nine repeats provide a measured loading interval.
        picks=[int(rng.integers(a,b)) for _ in range(9) for a,b in zip(ds.bounds,ds.bounds[1:])]
        batches=loader(Subset(ds,picks))
        net=SparseUNet(inputs=channels).to(DEVICE)
        seen=set();n=voxels=0;losses=[];waiting=0
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter();it=iter(batches)
        for k in range(len(batches)):
            t=time.perf_counter();batch=next(it);waiting+=time.perf_counter()-t
            c,x,y,sids=batch
            assert (x>=0).all() and (x<=1).all() and torch.isfinite(x).all() and x.shape[1]==channels
            assert y.min()>=0 and y.max()<6 and c[:,1:].min()>=0 and c[:,1:3].max()<32
            for b,sid in enumerate(sids):
                zmax=ds.scenes[sid].meta["shape"][2]
                co=c[c[:,0]==b,1:]
                assert len(co)>0 and co[:,2].max()<zmax
                if channels==6:
                    torch.testing.assert_close(x[c[:,0]==b,3:],co.float()/torch.tensor([31,31,max(1,zmax-1)]))
            seen.update(sids);n+=len(sids);voxels+=len(x)
            if k==0:
                g,x,y=on_gpu(batch)
                with torch.autocast("cuda",dtype=torch.float16):
                    pos,logits=net(g,x)
                    loss=F.cross_entropy(logits.float(),y)
                loss.backward()
                assert pos is g.coords and logits.shape==(len(y),6) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
                losses.append(float(loss))
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        assert len(seen)==len(ids)
        results[str(channels)]=dict(patches=n,voxels=voxels,checked_scenes=len(seen),batch_size=24,
            loss=losses[0],forward_backward_steps=1,optimizer_steps=0,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            total_s=elapsed,end_to_end_patches_per_s=n/elapsed,loader_wait_s=waiting,
            loader_patches_per_wait_s=n/waiting,range_checks_passed=True)
        del batches,it,net
    atomic_json(OUT/("gpu_dataloader_pilot.json" if pilot else "gpu_dataloader_smoke.json"),dict(status="passed",device=str(DEVICE),gpu=torch.cuda.get_device_name(0),
                torch=str(torch.__version__),cuda=torch.version.cuda,seed=42,features="RGB/255, then patch x/31, y/31, scene z/max(1,H-1)",modes=results))


def train(split,args):
    args.out.mkdir(parents=True,exist_ok=False)
    train=loader(Patches(split["train"]["scenes"],args.channels,split["index_hashes"]),True)
    val=loader(Patches(split["validation"]["scenes"],args.channels,split["index_hashes"]))
    net=SparseUNet(inputs=args.channels).to(DEVICE)
    opt=torch.optim.AdamW(net.parameters(),lr=.001)
    scaler=torch.amp.GradScaler("cuda",init_scale=1024)
    best=-1;logs=[]
    atomic_json(args.out/"protocol.json",dict(split_sha256=sha(OUT/"train_val_split.json"),channels=args.channels,
                epochs=args.epochs,seed=42,batch=24,lr=.001,optimizer="AdamW",sites=7))
    for epoch in range(1,args.epochs+1):
        net.train();loss_sum=0;n=0
        for batch in train:
            g,x,y=on_gpu(batch);opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.float16):
                _,logits=net(g,x);loss=F.cross_entropy(logits.float(),y)
            scaler.scale(loss).backward();scaler.step(opt);scaler.update()
            loss_sum+=float(loss.detach())*len(y);n+=len(y)
        net.eval();cm=torch.zeros(6,6,device=DEVICE,dtype=torch.long)
        with torch.inference_mode():
            for batch in val:
                g,x,y=on_gpu(batch);_,scores=net(g,x)
                cm+=torch.bincount(6*y+scores.argmax(1),minlength=36).reshape(6,6)
        score=scores_from_confusion(cm.cpu().numpy())
        logs.append(dict(epoch=epoch,train_loss=loss_sum/n,val_miou=score["miou"],val_accuracy=score["accuracy"],val_f1=score["macro_f1"]))
        if score["miou"]>best:
            best=score["miou"];save(net,args.out/"best.pt")
        write_csv(args.out/"history.csv",logs)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=["smoke","train","pilot"]);p.add_argument("--channels",choices=[3,6],type=int,default=3)
    p.add_argument("--epochs",type=int,default=35);p.add_argument("--out",type=Path,default=Path("results/fullz_63_training"))
    args=p.parse_args()
    split=json.loads((OUT/"train_val_split.json").read_text())
    if args.mode=="pilot":
        from datasets.recover_synthetic import JOURNAL
        records=json.loads(JOURNAL.read_text())["scenes"]
        hashes={k:r["index_sha256"] for k,r in records.items() if r["status"] in ("cache_verified","raw_removed")}
        split=dict(status="ready",train=dict(scenes=list(hashes)),validation=dict(scenes=[]),index_hashes=hashes)
    if split.get("status")!="ready" or not torch.cuda.is_available():raise RuntimeError("Complete dataset split and CUDA required")
    seed_all(42);torch.set_num_threads(2)
    if args.mode in ("pilot","smoke"):smoke(split,args.mode=="pilot")
    else:train(split,args)
