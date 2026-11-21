import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torch.nn.functional import cross_entropy

from datasets.voxelization import IGNORE
from models.voxel_3dcnn import VoxelUNet


def seed_all(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def save(path, model, meta):
    torch.save({"spec":model.spec, "state":model.state_dict(), "metadata":meta}, path)


def load(path, device="cpu"):
    blob = torch.load(path, map_location="cpu", weights_only=True)
    model = VoxelUNet(**blob["spec"])
    model.load_state_dict(blob["state"], strict=True)
    return model.to(device).eval(), blob["metadata"]


def masked_cross_entropy(logits, target, reduction="mean"):
    # Deterministic CUDA alternative to the spatial NLL kernel. Accumulate in FP32 under AMP.
    valid = target != IGNORE
    indices = target.masked_fill(~valid, 0).unsqueeze(1)
    values = -logits.float().log_softmax(1).gather(1, indices).squeeze(1)
    total = values.masked_fill(~valid, 0).sum()
    return total if reduction == "sum" else total / valid.sum()


def train(model, dataset, device, cfg, epochs, lr, seed, validation=None, best_path=None, meta=None, epoch_callback=None):
    from .evaluate import evaluate
    gen=torch.Generator().manual_seed(seed)
    sampler=None
    if cfg.get("balanced_patches"):
        rows=([dataset.dataset.rows[i] for i in dataset.indices] if isinstance(dataset,Subset) else dataset.rows)
        present=np.asarray([r["hist"] for r in rows])>0
        weights=(present/np.maximum(1,present.sum(0))).sum(1)
        sampler=WeightedRandomSampler(weights,len(rows),replacement=True,generator=gen)
    pin=cfg.get("pin_memory",False)
    batches = DataLoader(dataset, batch_size=cfg["batch"], shuffle=sampler is None, sampler=sampler,
                         num_workers=cfg.get("workers",0),pin_memory=pin,generator=gen)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs) if validation is not None else None
    amp = torch.device(device).type == "cuda" and cfg["amp"]
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    audit_device = cfg.get("verify_cuda", False)
    if audit_device:
        expected = torch.device("cuda:0")
        if torch.device(device) != expected or any(t.device != expected for t in (*model.parameters(), *model.buffers())):
            raise RuntimeError("Training requires model parameters and buffers on cuda:0")
    log, best = [], -1
    for ep in range(epochs):
        model.train()
        total, count = 0., 0
        sync(device)
        if audit_device:
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        for x,y in batches:
            x,y = x.to(device,non_blocking=pin),y.to(device,non_blocking=pin)
            if audit_device and (x.device != expected or y.device != expected):
                raise RuntimeError("Training inputs and labels must be on cuda:0")
            n = int((y != IGNORE).sum())
            if not n:
                continue
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=torch.device(device).type, enabled=amp):
                logits = model(x)
                loss = (masked_cross_entropy(logits, y) if x.is_cuda and torch.are_deterministic_algorithms_enabled()
                        else cross_entropy(logits, y, ignore_index=IGNORE))
            if not loss.isfinite():
                raise ValueError("Nonfinite training loss")
            scaler.scale(loss).backward()
            if audit_device and (loss.device != expected or logits.device != expected or
                                 any(p.grad is None or p.grad.device != expected for p in model.parameters() if p.requires_grad)):
                raise RuntimeError("Forward and backward tensors must be on cuda:0")
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach())*n
            count += n
        sync(device)
        if not count:
            raise ValueError("No labeled voxels")
        row = dict(epoch=ep+1, loss=total/count, seconds=time.perf_counter()-start,
                   lr=optimizer.param_groups[0]["lr"])
        if audit_device:
            row.update(device=str(expected), allocated_mb=torch.cuda.memory_allocated(device)/1e6,
                       peak_training_mb=torch.cuda.max_memory_allocated(device)/1e6)
        if validation is not None:
            metrics = evaluate(model, validation, device, cfg["batch"])
            row["val_miou"] = metrics["miou"]
            if metrics["miou"] > best:
                best = metrics["miou"]
                save(best_path, model, {**meta, "best_epoch":ep+1, "val_miou":best})
            scheduler.step()
        log.append(row)
        if epoch_callback is not None:
            epoch_callback(model, row)
    return log
