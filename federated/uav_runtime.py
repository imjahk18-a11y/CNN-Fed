"""Shared model, wire codec and patch-bundle operations for the two endpoint roles."""
import math

import numpy as np
import torch
from torch.nn import functional as F

from federated.communication_codec import decode, encode
from federated.process_fedavg import serialize, unpack_state
from models.active_voxel import SparseUNet, geometry
from training.train import seed_all
from experiments.distributed_ai import train_local, LR
from experiments.sparse_reproduction import scores_from_confusion

MODES = ("FP32", "FP16", "INT8", "INT8_SR", "FP32_MATCH")


def device_setup(device):
    seed_all(42)
    torch.set_num_threads(2)
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device unavailable")
    return device


def bundle(path, role, device, *, training_labels=True):
    """Only the patches provisioned on this endpoint are visible through this loader."""
    with np.load(path, allow_pickle=False) as z:
        if str(z["role"]) != role:
            raise ValueError("Wrong role-specific data bundle")
        bounds, coords, rgb = (z[k] for k in ("bounds", "coords", "rgb"))
        labels = z["labels"] if training_labels else None
        if bounds[0] != 0 or bounds[-1] != len(coords) or not (np.diff(bounds) > 0).all():
            raise ValueError("Invalid patch offsets")
        if len(rgb) != len(coords) or (
            labels is not None
            and (len(labels) != len(coords) or not ((labels >= 0) & (labels < 6)).all())
        ):
            raise ValueError("Invalid voxel labels")
        if not np.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 255:
            raise ValueError("Invalid RGB")
        batches = []
        for a in range(0, len(bounds) - 1, 24):
            b = min(a + 24, len(bounds) - 1)
            sl = slice(bounds[a], bounds[b])
            idx = np.repeat(np.arange(b - a, dtype=np.int32), np.diff(bounds[a : b + 1]))
            pos = torch.from_numpy(np.column_stack((idx, coords[sl]))).to(device)
            batches.append(
                (
                    geometry(pos),
                    torch.from_numpy(rgb[sl]).to(device) / 255,
                    None if labels is None else torch.from_numpy(labels[sl]).to(device, torch.long),
                )
            )
        return batches, len(bounds) - 1


def wire_encode(state, mode, rng_seed):
    if mode == "FP32_MATCH":
        mode = "FP32"
    if mode in ("INT8", "FP32"):
        return encode(state, mode)[0]
    if mode == "FP16":
        payload = {k: (t.half() if t.is_floating_point() else t) for k, t in state.items()}
        if any(not torch.isfinite(t).all() for t in payload.values()):
            raise ValueError("FP16 overflow")
        return serialize(payload)
    if mode != "INT8_SR":
        raise ValueError(mode)
    generator = torch.Generator().manual_seed(rng_seed)
    q, scales = {}, []
    for k in sorted(state):
        t = state[k]
        if not t.is_floating_point():
            q[k] = t
            continue
        scale = torch.tensor(float(t.abs().max()) / 127 or 1.0, dtype=torch.float32)
        x = (t / scale).clamp(-127, 127)
        lo = x.floor()
        q[k] = (lo + (torch.rand(x.shape, generator=generator) < (x - lo))).to(torch.int8)
        scales.append(scale)
    return serialize((q, torch.stack(scales)))


def wire_decode(raw, mode, template):
    if mode != "FP16":
        return decode(raw, {"INT8_SR": "INT8", "FP32_MATCH": "FP32"}.get(mode, mode), template)
    state = unpack_state(raw)
    if state.keys() != template.keys():
        raise ValueError("Wrong keys")
    out = {}
    for k, ref in template.items():
        t = state[k]
        expected = torch.float16 if ref.is_floating_point() else ref.dtype
        if t.dtype != expected or t.shape != ref.shape or not torch.isfinite(t).all():
            raise ValueError(k)
        out[k] = t.to(ref.dtype)
    return out


def rounding_seed(seed, r, side):
    return seed * 100000 + r * 10 + side


def group(k):
    return "bn" if k.endswith(("running_mean", "running_var")) else "parameters"


def vector(state, which):
    return torch.cat(
        [
            state[k].flatten().double()
            for k in sorted(state)
            if state[k].is_floating_point() and group(k) == which
        ]
    )


def apply_reference_norms(start, candidate, target):
    out = dict(candidate)
    for g in ("parameters", "bn"):
        n = (vector(candidate, g) - vector(start, g)).norm().item()
        wanted = target[g]
        if not math.isfinite(wanted) or wanted < 0 or (n == 0 and wanted > 0):
            raise ValueError("Invalid reference norm")
        for k, v in candidate.items():
            if v.is_floating_point() and group(k) == g:
                out[k] = (
                    start[k].double() + (v.double() - start[k].double()) * (wanted / n if n else 0.0)
                ).float()
                if k.endswith("running_var") and (out[k] < 0).any():
                    raise ValueError("Negative BN variance")
    return out


def local_epoch(model, data, seed, device):
    if device.type == "cuda":
        return train_local(model, data, 1, seed)[0]
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_sum = count = 0
    for i in np.random.default_rng(seed).permutation(len(data)):
        g, x, y = data[i]
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(g, x)[1], y)
        loss.backward()
        opt.step()
        loss_sum += loss.item() * len(y)
        count += len(y)
    return loss_sum / count


@torch.inference_mode()
def score(model, data):
    model.eval()
    conf = torch.zeros((6, 6), dtype=torch.long, device=next(model.parameters()).device)
    for g, x, y in data:
        coords, logits = model(g, x)
        if coords is not g.coords:
            raise ValueError("Misaligned predictions")
        conf += torch.bincount(6 * y + logits.argmax(1), minlength=36).reshape(6, 6)
    return scores_from_confusion(conf.cpu().numpy())["miou"]


def load_checkpoint(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    net = SparseUNet(tuple(ck.get("widths", (16, 32, 16))))
    net.load_state_dict(ck["state"])
    return net.to(device).eval()


def early_update(best, best_round, score_now, r, patience):
    if score_now > best:
        return score_now, r, False
    return best, best_round, r - best_round >= patience
