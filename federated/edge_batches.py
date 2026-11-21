"""Load only the requested compressed occupied-voxel patches on each CUDA client."""
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F

from models.active_voxel import geometry


class EdgeBatches:
    def __init__(self, manifest, role, batch_size, device):
        self.root=manifest.parent
        self.meta=json.loads(manifest.read_text())
        if self.meta["role"]!=role or batch_size not in (1,2,4):
            raise ValueError("Role or batch size mismatch")
        self.files=self.meta["patches"]
        if not self.files: raise ValueError("Empty client dataset")
        self.batch_size=batch_size
        self.device=device

    def __len__(self):
        return (len(self.files)+self.batch_size-1)//self.batch_size

    def __getitem__(self,index):
        if not 0<=index<len(self): raise IndexError(index)
        entries=self.files[index*self.batch_size:(index+1)*self.batch_size]
        coords=[]; colors=[]; labels=[]
        for idx,entry in enumerate(entries):
            name=entry["file"]
            if Path(name).name!=name: raise ValueError("Invalid patch name")
            raw=(self.root/name).read_bytes()
            if hashlib.sha256(raw).hexdigest()!=entry["sha256"]: raise ValueError("Corrupt patch")
            with np.load(io.BytesIO(raw),allow_pickle=False) as z:
                c,x,y=z["coords"],z["rgb"],z["labels"]
            if c.shape!=(len(y),3) or x.shape!=c.shape or not len(c) or c.dtype.kind not in "iu":
                raise ValueError("Invalid occupied patch")
            if np.any(c<0) or np.any(c[:,:2]>=32) or not np.isfinite(x).all() or x.min()<0 or x.max()>255 or y.min()<0 or y.max()>5:
                raise ValueError("Invalid input range")
            coords.append(np.column_stack((np.full(len(c),idx,np.int32),c)))
            colors.append(x); labels.append(y)
        pos=torch.from_numpy(np.concatenate(coords)).to(self.device)
        return (geometry(pos),torch.from_numpy(np.concatenate(colors)).to(self.device,torch.float32)/255,
                torch.from_numpy(np.concatenate(labels)).to(self.device,torch.long))


def cap_allocator(gib):
    total=torch.cuda.get_device_properties(0).total_memory
    if gib<=0: raise ValueError("Nonpositive allocator budget")
    torch.cuda.set_per_process_memory_fraction(min(1.,gib*2**30/total),0)


def memory():
    # ru_maxrss is KiB on Linux. NVML includes allocations outside the torch allocator.
    driver=None
    try:
        result=subprocess.run(["nvidia-smi","--query-compute-apps=pid,used_gpu_memory",
                               "--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=4)
        for line in result.stdout.splitlines():
            pid,used=line.split(",")
            if int(pid)==os.getpid(): driver=float(used)
    except (OSError,ValueError,subprocess.TimeoutExpired):
        pass
    return dict(peak_alloc_mib=torch.cuda.max_memory_allocated()/2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                current_reserved_mib=torch.cuda.memory_reserved()/2**20,
                peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                driver_sample_mib=driver)


def measure_inference(net,patch):
    g,x,y=patch
    if not next(net.parameters()).is_cuda or not x.is_cuda or not y.is_cuda:
        raise RuntimeError("Edge execution requires CUDA")
    with torch.inference_mode():
        net.eval()
        coords,logits=net(g,x)
        if coords is not g.coords or logits.shape!=(len(y),6) or not logits.is_cuda or not torch.isfinite(logits).all():
            raise ValueError("CUDA logits/coordinates invalid")
        return logits


def train_epoch(net,data,seed):
    opt=torch.optim.AdamW(net.parameters(),lr=.0001)
    net.train()
    total=points=0
    first=last=None
    before=next(net.parameters()).detach().clone()
    start=time.perf_counter()
    for index in np.random.default_rng(seed).permutation(len(data)):
        g,x,y=data[int(index)]
        if not x.is_cuda or not y.is_cuda or next(net.parameters()).device!=x.device:
            raise RuntimeError("Expected CUDA training")
        opt.zero_grad(set_to_none=True)
        coords,logits=net(g,x)
        if coords is not g.coords or logits.shape!=(len(y),6): raise ValueError("Misaligned voxels")
        loss=F.cross_entropy(logits,y)
        if not torch.isfinite(loss): raise FloatingPointError("Nonfinite loss")
        loss.backward()
        grad_norm=torch.nn.utils.clip_grad_norm_(net.parameters(),float("inf"),error_if_nonfinite=True)
        if not grad_norm.is_cuda: raise RuntimeError("Expected CUDA gradients")
        opt.step()
        last=loss.item()
        if first is None: first=last
        total+=len(y)*last; points+=len(y)
    torch.cuda.current_stream().synchronize()
    if torch.equal(before,next(net.parameters())): raise RuntimeError("Optimizer did not change weights")
    duration=time.perf_counter()-start
    return dict(loss=total/points,first_batch_loss=first,last_batch_loss=last,epoch_s=duration,
                mean_iteration_s=duration/len(data),steps=len(data),patches=len(data.files))
