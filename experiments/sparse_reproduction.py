"""Synthetic-only active-voxel training and selection; final evaluation on WMSC."""
import argparse
import csv
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch
from torch.nn import functional as F

from datasets.compact_cache import atomic_json, sha
from datasets.fullz import Scene, CLASSES, convert
from training.train import seed_all
from federated.fedavg import fedavg
from models.active_voxel import SparseUNet, geometry, prune_channels

BASE = Path("results/sparse_3dcnn_reproduction")
CACHE = Path("data/sparse_fullz_v1")
DEVICE = torch.device("cuda:0")
BATCH = 24
SEED = 42
ROLES = dict(
    central="13_points_GTv3",
    validation="12_points_GTv3",
    drone_a="3_points_GTv3",
    drone_b="4_points_GTv3",
)
MiB = 1024**2


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def pack(scene, indices, sites=7):
    coords, rgb, labels = [], [], []
    for index, i in enumerate(indices):
        c, x, y = scene[i]
        coords.append(np.column_stack((np.full(len(c), index, np.int32), c)))
        rgb.append(x)
        labels.append(y)
    pos = torch.from_numpy(np.concatenate(coords)).to(DEVICE)
    x = torch.from_numpy(np.concatenate(rgb)).to(DEVICE) / 255
    y = torch.from_numpy(np.concatenate(labels)).to(DEVICE, torch.long)
    return geometry(pos, sites), x, y


def batches(scene, shuffle, sites=7):
    indices = np.arange(len(scene))
    if shuffle:
        np.random.default_rng(SEED).shuffle(indices)
    return [pack(scene, indices[i : i + BATCH], sites) for i in range(0, len(scene), BATCH)]


def cpu_state(net):
    return {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}


def save(net, path):
    torch.save(dict(widths=net.widths, sites=net.sites, inputs=net.inputs, state=cpu_state(net)), path)


def load(path):
    ck = torch.load(path, weights_only=True, map_location="cpu")
    net = SparseUNet(ck["widths"], sites=ck.get("sites", 7), inputs=ck.get("inputs", 3))
    net.load_state_dict(ck["state"])
    return net.to(DEVICE).eval()


def fit_epoch(net, data, optimizer, scaler, seed):
    torch.cuda.synchronize()
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    resident = torch.cuda.memory_allocated()
    net.train()
    total = torch.zeros((), device=DEVICE)
    count = 0
    for i in np.random.default_rng(seed).permutation(len(data)):
        g, x, y = data[i]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            pos, logits = net(g, x)
            loss = F.cross_entropy(logits.float(), y)
        if (
            pos is not g.coords
            or logits.shape != (len(y), 6)
            or logits.device != DEVICE
            or logits.dtype != torch.float16
        ):
            raise RuntimeError("CUDA AMP or coordinate mismatch")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.detach() * len(y)
        count += len(y)
    value = total.item() / count
    if not np.isfinite(value):
        raise ValueError("Nonfinite loss")
    return dict(
        loss=value,
        valid_voxels=count,
        train_s=time.perf_counter() - start,
        peak_mib=torch.cuda.max_memory_allocated() / MiB,
        resident_mib=resident / MiB,
    )


@torch.inference_mode()
def evaluate(net, data):
    net.eval()
    conf = torch.zeros((6, 6), device=DEVICE, dtype=torch.long)
    for g, x, y in data:
        pos, logits = net(g, x)
        if pos is not g.coords:
            raise ValueError("Output coordinate order changed")
        conf += torch.bincount(6 * y + logits.argmax(1), minlength=36).reshape(6, 6)
    return scores_from_confusion(conf.cpu().numpy())


def scores_from_confusion(matrix):
    tp = matrix.diagonal()
    support = matrix.sum(0) + matrix.sum(1)
    union = support - tp
    iou = np.divide(tp, union, out=np.full(6, np.nan), where=union > 0)
    f1 = np.divide(2 * tp, support, out=np.full(6, np.nan), where=support > 0)
    return dict(
        accuracy=float(tp.sum() / matrix.sum()),
        miou=float(np.nanmean(iou)),
        macro_f1=float(np.nanmean(f1)),
        valid_voxels=int(matrix.sum()),
        iou=[float(v) if np.isfinite(v) else None for v in iou],
        confusion=matrix.tolist(),
    )


def fit(net, data, val, epochs, lr, path, name):
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda", init_scale=1024)
    best = -1.0
    log = []
    for epoch in range(1, epochs + 1):
        row = fit_epoch(net, data, opt, scaler, SEED + epoch)
        score = evaluate(net, val)
        row.update(
            epoch=epoch,
            val_miou=score["miou"],
            val_accuracy=score["accuracy"],
            val_macro_f1=score["macro_f1"],
        )
        if score["miou"] > best:
            best = score["miou"]
            save(net, path)
        log.append(row)
        print(
            name,
            epoch,
            round(row["loss"], 4),
            round(score["miou"], 4),
            round(row["train_s"], 2),
            flush=True,
        )
    return log


@torch.inference_mode()
def profile(net, batch):
    g, x, _ = batch
    net.eval()
    times = {}
    for prepare in (False, True):

        def forward():
            return net(geometry(g.coords, net.sites) if prepare else g, x)

        for _ in range(10):
            forward()
        torch.cuda.synchronize()
        start_bytes = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        samples = []
        for _ in range(60):
            t = time.perf_counter()
            forward()
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - t) * 1000)
        label = "with_index" if prepare else "cached"
        for pct in (50, 95):
            times[f"{label}_p{pct}_ms"] = float(np.percentile(samples, pct))
        times[f"{label}_increment_mib"] = (torch.cuda.max_memory_allocated() - start_bytes) / MiB
    return dict(
        parameters=sum(p.numel() for p in net.parameters()),
        macs=net.macs(g),
        voxels=len(x),
        state_bytes=sum(v.numel() * v.element_size() for v in net.state_dict().values()),
        **times,
    )


def run(out):
    seed_all(SEED)
    if not torch.cuda.is_available():
        raise RuntimeError("cuda:0 required")
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    scenes = {k: Scene(CACHE / v) for k, v in ROLES.items()}
    out.mkdir(parents=True, exist_ok=False)
    ck = out / "checkpoints"
    ck.mkdir()
    atomic_json(
        out / "protocol.json",
        dict(
            seed=SEED,
            batch=BATCH,
            central_epochs=35,
            fine_tune_epochs=5,
            rounds=10,
            local_epochs=1,
            lr=0.001,
            ft_lr=0.0001,
            local_lr=0.0001,
            roles=ROLES,
            classes=CLASSES,
            scenes={k: v.meta for k, v in scenes.items()},
            external="WMSC only after freeze",
            central_selection="earliest highest synthetic validation mIoU",
            fl_selection="round 10",
            batch_order="fixed seeded random patch groups; shuffle group order per epoch",
            device=str(DEVICE),
            gpu=torch.cuda.get_device_name(0),
            torch=str(torch.__version__),
            cuda=torch.version.cuda,
            python=platform.python_version(),
            backend="native PyTorch occupied pairs",
            installed={
                k: importlib.util.find_spec(k) is not None
                for k in ("spconv", "MinkowskiEngine", "torchsparse")
            },
        ),
    )
    central = batches(scenes["central"], True)
    val = batches(scenes["validation"], False)
    scene = scenes["validation"]
    probe_idx = int(np.argsort(np.diff(scene.bounds), kind="stable")[len(scene) // 2])
    probe = pack(scene, [probe_idx])
    torch.manual_seed(SEED)
    net = SparseUNet().to(DEVICE)
    first = net.encoder.weight.detach().clone()
    smoke = fit(net, central, val, 1, 0.001, ck / "smoke.pt", "smoke")[0]
    if torch.equal(first, net.encoder.weight) or any(
        p.device != DEVICE or p.grad is None for p in net.parameters()
    ):
        raise RuntimeError("GPU update failed")
    restored = load(ck / "smoke.pt")
    g, x, _ = probe
    with torch.no_grad():
        torch.testing.assert_close(net.eval()(g, x)[1], restored(g, x)[1], rtol=0, atol=0)
    if evaluate(restored, val)["miou"] != smoke["val_miou"]:
        raise RuntimeError("Checkpoint validation mismatch")
    atomic_json(
        out / "smoke.json",
        dict(
            **smoke,
            gpu=str(DEVICE),
            checkpoint_sha256=sha(ck / "smoke.pt"),
            exact_logits=True,
            alignment=True,
            updated_weights=True,
            profile=profile(restored, probe),
        ),
    )
    torch.manual_seed(SEED)
    net = SparseUNet().to(DEVICE)
    logs = fit(net, central, val, 35, 0.001, ck / "baseline.pt", "baseline")
    write_csv(out / "baseline_training.csv", logs)
    frozen = {"baseline": ck / "baseline.pt"}
    profiles, pruning, rounds = [], [], []
    clients = [batches(scenes[k], True) for k in ("drone_a", "drone_b")]
    sizes = [len(scenes[k]) for k in ("drone_a", "drone_b")]
    for pct in (0, 20, 40):
        base = load(ck / "baseline.pt")
        small = prune_channels(base, pct / 100).to(DEVICE)
        g, x, y = probe
        small.train()
        pos, logits = small(g, x)
        F.cross_entropy(logits, y).backward()
        if (
            pos is not g.coords
            or logits.shape != (len(y), 6)
            or any(p.grad is None or not torch.isfinite(p.grad).all() for p in small.parameters())
        ):
            raise RuntimeError("Pruned gradient/alignment check failed")
        small = prune_channels(base, pct / 100).eval()  # discard BN changes from gradient check
        profiles.append(
            dict(pruning=pct, patch_index=probe_idx, widths=str(small.widths), **profile(small, probe))
        )
        before = evaluate(small, val)
        name = f"ft{pct}"
        path = ck / (name + ".pt")
        rows = fit(small, central, val, 5, 0.0001, path, name)
        write_csv(out / (name + "_training.csv"), rows)
        selected = load(path)
        after = evaluate(selected, val)
        pruning.append(
            dict(
                pruning=pct,
                before_val_miou=before["miou"],
                after_val_miou=after["miou"],
                selected_epoch=int(np.argmax([r["val_miou"] for r in rows])) + 1,
            )
        )
        frozen[name] = path
        size = profiles[-1]["state_bytes"]
        for rnd in range(1, 11):
            states, elapsed, peaks = [], 0, []
            for client, data in enumerate(clients):
                local = deepcopy(selected)
                opt = torch.optim.AdamW(local.parameters(), lr=0.0001)
                row = fit_epoch(
                    local,
                    data,
                    opt,
                    torch.amp.GradScaler("cuda", init_scale=1024),
                    SEED + 100 * rnd + client,
                )
                states.append(cpu_state(local))
                elapsed += row["train_s"]
                peaks.append(row["peak_mib"])
            selected.load_state_dict(fedavg(states, sizes))
            score = evaluate(selected, val)
            rounds.append(
                dict(
                    pruning=pct,
                    round=rnd,
                    val_accuracy=score["accuracy"],
                    val_miou=score["miou"],
                    val_macro_f1=score["macro_f1"],
                    train_s=elapsed,
                    peak_mib=max(peaks),
                    cumulative_comm_bytes=rnd * 4 * size,
                )
            )
            print("FedAvg", pct, rnd, round(score["miou"], 4), flush=True)
        path = ck / f"fl{pct}.pt"
        save(selected, path)
        frozen[f"fl{pct}"] = path
    write_csv(out / "profiles.csv", profiles)
    write_csv(out / "pruning.csv", pruning)
    write_csv(out / "federated.csv", rounds)
    candidates = {k: dict(file=str(v.relative_to(out)), sha256=sha(v)) for k, v in frozen.items()}
    atomic_json(out / "frozen.json", candidates)
    wroot = CACHE / "WMSC_points"
    source = Path("data/single_site/WMSC_points.ply")
    if not (wroot / "scene.json").exists():
        convert(source, wroot)
    wmsc = Scene(wroot)
    if wmsc.meta["sha256"] != sha(source):
        raise ValueError("WMSC source mismatch")
    ext = batches(wmsc, False)
    results = {}
    for name, info in candidates.items():
        path = out / info["file"]
        if sha(path) != info["sha256"]:
            raise ValueError("Frozen checkpoint changed")
        net = load(path)
        results[name] = dict(validation=evaluate(net, val), wmsc=evaluate(net, ext))
    atomic_json(out / "evaluation.json", dict(scene=wmsc.meta, models=results))
    write_csv(
        out / "comparison.csv",
        [
            dict(model=k, split=s, **{m: v for m, v in score.items() if m not in ("confusion", "iou")})
            for k, pair in results.items()
            for s, score in pair.items()
        ],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=BASE)
    run(parser.parse_args().out)
