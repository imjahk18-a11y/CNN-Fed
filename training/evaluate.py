import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets.voxelization import IGNORE
from .train import masked_cross_entropy, sync


@torch.inference_mode()
def evaluate(model, dataset, device, batch):
    model.eval()
    k = model.spec["classes"]
    conf = torch.zeros(k,k, dtype=torch.int64)
    total, loading = 0., 0.
    it = iter(DataLoader(dataset, batch_size=batch))
    for _ in range(len(it)):
        start = time.perf_counter()
        x,y = next(it)
        loading += time.perf_counter()-start
        x,y = x.to(device),y.to(device)
        logits = model(x)
        loss = (masked_cross_entropy(logits, y, "sum") if x.is_cuda and torch.are_deterministic_algorithms_enabled()
                else nn.functional.cross_entropy(logits, y, ignore_index=IGNORE, reduction="sum"))
        total += float(loss)
        valid = y != IGNORE
        pairs = y[valid]*k + logits.argmax(1)[valid]
        conf += torch.bincount(pairs.cpu(), minlength=k*k).reshape(k,k)
    if not conf.sum() or not np.isfinite(total):
        raise ValueError("Invalid evaluation split")
    tp = conf.diag().double()
    support = conf.sum(0)+conf.sum(1)
    union = support-tp
    per = [float(t/u) if u else None for t,u in zip(tp,union)]
    return dict(loss=total/int(conf.sum()), accuracy=float(tp.sum()/conf.sum()),
                miou=float((tp[union>0]/union[union>0]).mean()),
                macro_f1=float((2*tp[support>0]/support[support>0]).mean()),
                per_class_iou=per, valid_voxels=int(conf.sum()), data_loading_s=loading)


@torch.inference_mode()
def profile(model, x, device, repeats):
    model.eval()
    macs = 0
    def count(conv, args, out):
        nonlocal macs
        macs += out.numel()*conv.weight[0].numel()
    hooks = [m.register_forward_hook(count) for m in model.modules() if isinstance(m,nn.Conv3d)]
    try:
        model(x)
    finally:
        for handle in hooks:
            handle.remove()
    for _ in range(3):
        model(x)
    sync(device)
    cuda = torch.device(device).type == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        model(x)
        sync(device)
        times.append((time.perf_counter()-start)*1000)
    return dict(parameters=sum(p.numel() for p in model.parameters()), macs=int(macs),
                model_mb=sum(t.numel()*t.element_size() for t in model.state_dict().values())/1e6,
                latency_p50_ms=float(np.median(times)), latency_p95_ms=float(np.percentile(times,95)),
                peak_gpu_mb=torch.cuda.max_memory_allocated(device)/1e6 if cuda else None)
