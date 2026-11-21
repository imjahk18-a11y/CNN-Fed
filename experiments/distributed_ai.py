"""Controlled two-client sparse FL on four disjoint synthetic scenes."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch
from torch.nn import functional as F

from datasets.compact_cache import atomic_json, sha
from datasets.fullz import Scene, common, CLASSES
from experiments.sparse_reproduction import (
    CACHE,
    DEVICE,
    pack,
    cpu_state,
    fit_epoch,
    evaluate,
    save,
    load,
    write_csv,
)
from federated.fedavg import fedavg
from models.active_voxel import prune_channels
from training.train import seed_all

OUT = Path("results/distributed_ai")
OLD = Path("results/sparse_3dcnn_reproduction/run02")
BASE = OLD / "checkpoints/baseline.pt"
SEEDS = (42, 43, 44)
SIZES = (0, 20, 40)
LR = 0.0001
THRESHOLD = 0.35
SCENES = {i: f"{i}_points_GTv3" for i in (3, 4, 12, 13)}


def init():
    seed_all(42)
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("cuda:0 required")


def dump(net, path):
    tmp = path.with_suffix(".writing")
    save(net, tmp)
    tmp.replace(path)


def batchify(refs, scenes):
    # Reference pairs permit mixing distinct scenes inside one actual batch.
    class View:
        def __getitem__(self, i):
            scene, patch = refs[i]
            return scenes[scene][patch]

    view = View()
    return [pack(view, list(range(i, min(i + 24, len(refs))))) for i in range(0, len(refs), 24)]


def make_split(scenes):
    rng = np.random.default_rng(2026)
    h = {}
    for i, s in scenes.items():
        labels = common(s.raw)
        h[i] = np.stack(
            [np.add.reduceat((labels == c).astype(np.int64), s.bounds[:-1]) for c in range(6)], axis=1
        )

    def draw(i, n):
        # Fixed random draw within dominant-class strata (proportional allocation).
        group = h[i].argmax(1)
        freq = np.bincount(group, minlength=6)
        counts = np.floor(n * freq / len(group)).astype(int)
        remainder = n - counts.sum()
        counts[np.argsort(-(n * freq / len(group) - counts), kind="stable")[:remainder]] += 1
        ids = [
            (i, int(idx))
            for cl, num in enumerate(counts)
            for idx in rng.choice(np.where(group == cl)[0], num, replace=False)
        ]
        rng.shuffle(ids)
        return ids

    a, b = draw(3, 96), draw(4, 96)
    pool = a + b
    # Each IID client receives half of each scene. Minimize six-class frequency mismatch.
    hist = np.stack([h[i][j] for i, j in pool])
    best = np.inf
    for _ in range(1000):
        idx = np.r_[rng.choice(96, 48, replace=False), 96 + rng.choice(96, 48, replace=False)]
        counts = hist[idx].sum(0)
        other = hist.sum(0) - counts
        distance = np.abs(counts / counts.sum() - other / other.sum()).sum()
        if distance < best:
            best = distance
            selected = set(idx)
    iid = [[p for j, p in enumerate(pool) if (j in selected) == flag] for flag in (True, False)]
    for group in iid:
        rng.shuffle(group)
    rng.shuffle(pool)
    split = dict(
        pretrain_scene=13,
        ft=draw(13, 192),
        val=draw(12, 144),
        central=pool,
        IID=iid,
        NonIID=[a, b],
        split_seed=2026,
    )
    stats = []
    for regime in ("IID", "NonIID"):
        groups = split[regime]
        assert (
            set(groups[0]).isdisjoint(groups[1])
            and set(groups[0] + groups[1]) == set(pool)
            and len(set(pool)) == 192
        )
        for client, refs in enumerate(groups, 1):
            total = sum(h[i][j] for i, j in refs)
            for c, n in enumerate(total):
                stats.append(
                    dict(
                        regime=regime,
                        client=client,
                        class_name=CLASSES[c],
                        patches=len(refs),
                        voxels=int(n),
                        fraction=n / total.sum(),
                    )
                )
    assert all(len({i for i, j in split[k]}) == 1 for k in ("ft", "val"))
    assert len({s.meta["sha256"] for s in scenes.values()}) == 4
    write_csv(OUT / "client_distributions.csv", stats)
    atomic_json(OUT / "split.json", split)
    return split


def data_context():
    scenes = {i: Scene(CACHE / name) for i, name in SCENES.items()}
    path = OUT / "split.json"
    split = json.loads(path.read_text()) if path.exists() else make_split(scenes)
    sets = {k: batchify(split[k], scenes) for k in ("ft", "val", "central")}
    for regime in ("IID", "NonIID"):
        sets[regime] = [batchify(refs, scenes) for refs in split[regime]]
    return sets


def train_local(net, data, epochs, seed, prox=0):
    opt = torch.optim.AdamW(net.parameters(), lr=LR)
    scaler = torch.amp.GradScaler("cuda", init_scale=1024)
    ref = [p.detach().clone() for p in net.parameters()] if prox else None
    logs = []
    for ep in range(epochs):
        if not prox:
            logs.append(fit_epoch(net, data, opt, scaler, seed + ep))
            continue
        net.train()
        torch.cuda.reset_peak_memory_stats()
        total = 0.0
        n = 0
        for i in np.random.default_rng(seed + ep).permutation(len(data)):
            g, x, y = data[i]
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                _, logits = net(g, x)
                ce = F.cross_entropy(logits.float(), y)
                loss = ce + prox / 2 * sum((p - r).square().sum() for p, r in zip(net.parameters(), ref))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += float(ce.detach()) * len(y)
            n += len(y)
        logs.append(dict(loss=total / n, peak_mib=torch.cuda.max_memory_allocated() / 2**20))
    return float(np.mean([r["loss"] for r in logs])), max(r["peak_mib"] for r in logs)


def nbytes(state):
    return sum(x.numel() * x.element_size() for x in state.values())


def startfile(seed, pct):
    return OUT / "starts" / f"s{seed}_p{pct}.pt"


def prepare(sets, smoke=False):
    source = json.loads((OLD / "frozen.json").read_text())
    assert sha(BASE) == source["baseline"]["sha256"]
    for name, meta in source.items():
        assert sha(OLD / meta["file"]) == meta["sha256"]
    test = fedavg([{"w": torch.tensor([1.0])}, {"w": torch.tensor([5.0])}], [1, 3])
    assert test["w"].item() == 4
    profiles = []
    for pct in SIZES:
        net = prune_channels(load(BASE), pct / 100)
        old = load(OLD / source[f"ft{pct}"]["file"])
        assert net.widths == old.widths
        for seed in (42,) if smoke else SEEDS:
            if smoke and pct:
                continue
            path = startfile(seed, pct)
            if path.exists():
                continue
            start = deepcopy(net)
            train_local(start, sets["ft"], 5, seed * 100)
            dump(start, path)
        state = net.state_dict()
        g = sets["val"][0][0]
        profiles.append(
            dict(
                pruning_pct=pct,
                widths=str(net.widths),
                parameters=sum(p.numel() for p in net.parameters()),
                state_bytes=nbytes(state),
                checkpoint_bytes=startfile(42, pct).stat().st_size if startfile(42, pct).exists() else 0,
                fixed_batch_macs=net.macs(g),
                fixed_batch_voxels=len(g.coords),
            )
        )
    if not smoke:
        write_csv(OUT / "model_profiles.csv", profiles)


def config_id(method, pct, regime, le, seed):
    return f"{method}_p{pct}_{regime}_e{le}_s{seed}"


def round_seed(seed, r, client):
    return seed * 10000 + 100 * r + 10 * client


def run_condition(sets, method, pct, regime, le, seed, rounds=30, smoke=False):
    key = config_id(method, pct, regime, le, seed)
    dest = (OUT / "smoke" if smoke else OUT / "runs") / key
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "done.json").exists():
        return
    net = load(startfile(seed, pct))
    val = sets["val"]
    baseline = evaluate(net, val)
    is_fl = method in ("FedAvg", "FedProx", "FedBN")
    groups = (
        sets[regime]
        if is_fl
        else ([sets["central"]] if method == "Central" else [sets[regime][int(method[-1]) - 1]])
    )
    # BN affine and running moments are persistent and private for each FedBN client.
    bn = [{}, {}]
    history = []
    for r in range(1, rounds + 1):
        torch.cuda.synchronize()
        tic = time.perf_counter()
        states = []
        localscores = []
        losses = []
        peaks = []
        client_models = []
        for c, data in enumerate(groups):
            local = deepcopy(net) if is_fl else net
            if method == "FedBN" and bn[c]:
                local.load_state_dict(bn[c], strict=False)
            loss, peak = train_local(
                local, data, le, round_seed(seed, r, c), 0.01 if method == "FedProx" else 0
            )
            score = evaluate(local, val)
            localscores.append(score)
            losses.append(loss)
            peaks.append(peak)
            if is_fl:
                state = cpu_state(local)
                if method == "FedBN":
                    bn[c] = {k: v for k, v in state.items() if ".norm." in k}
                    state = {k: v for k, v in state.items() if k not in bn[c]}
                states.append(state)
                client_models.append(local)
        if is_fl:
            sizes = [sum(int(g.coords[:, 0].max()) + 1 for g, x, y in d) for d in groups]
            averaged = fedavg(states, sizes)
            net.load_state_dict(averaged, strict=(method != "FedBN"))
            if method == "FedBN":

                class Ensemble(torch.nn.Module):
                    def forward(self, g, x):
                        return g.coords, (client_models[0](g, x)[1] + client_models[1](g, x)[1]) / 2

                for local, private in zip(client_models, bn):
                    local.load_state_dict(averaged, strict=False)
                    local.load_state_dict(private, strict=False)
                    local.eval()
                score = evaluate(Ensemble(), val)
            else:
                score = evaluate(net, val)
        else:
            score = localscores[0]
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - tic
        up = 2 * nbytes(states[0]) if is_fl else 0
        row = dict(
            run=key,
            method=method,
            pruning_pct=pct,
            partition=regime,
            local_epochs=le,
            seed=seed,
            round=r,
            initial_miou=baseline["miou"],
            global_miou=score["miou"],
            global_f1=score["macro_f1"],
            global_accuracy=score["accuracy"],
            upload_bytes=up,
            download_bytes=up,
            total_bytes=up * 2,
            cumulative_bytes=r * up * 2,
            round_s=elapsed,
            peak_mib=max(peaks),
            patch_visits_this_round=sum(len(d) * 24 * le for d in groups),
        )
        for idx in range(2):
            pos = idx if is_fl else (0 if method == f"Local{idx+1}" else None)
            row.update(
                {
                    f"client{idx+1}_loss": losses[pos] if pos is not None else "",
                    f"client{idx+1}_val_miou": localscores[pos]["miou"] if pos is not None else "",
                    f"client{idx+1}_val_f1": localscores[pos]["macro_f1"] if pos is not None else "",
                    f"client{idx+1}_peak_mib": peaks[pos] if pos is not None else "",
                }
            )
        history.append(row)
        if r in (10, rounds):
            dump(net, dest / f"r{r}.pt")
            if method == "FedBN":
                torch.save(bn, dest / f"bn{r}.pt")
            write_csv(dest / "rounds.csv", history)
    atomic_json(
        dest / "done.json",
        dict(
            rounds=rounds, split_sha256=sha(OUT / "split.json"), start_sha256=sha(startfile(seed, pct))
        ),
    )
    print(
        "done",
        key,
        rounds,
        round(score["miou"], 4),
        round(sum(v["round_s"] for v in history), 1),
        flush=True,
    )


def run():
    init()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "starts").mkdir(exist_ok=True)
    sets = data_context()
    proto = OUT / "protocol.json"
    if not proto.exists():
        atomic_json(
            proto,
            dict(
                seeds=SEEDS,
                split_seed=2026,
                batch=24,
                local_lr=LR,
                checkpoint=sha(BASE),
                ft_epochs=5,
                ft_patches=192,
                rounds=[10, 30],
                local_epochs=[1, 3],
                target_miou=THRESHOLD,
                conditional_prox="if mean IID minus NonIID round30 mIoU >= 0.02 for a matched pruning and local epoch use mu=.01 for that cell",
                fedbn="NonIID original with local epochs 3, 3 seeds. Average personalized logits with shared aggregated weights.",
                torch=str(torch.__version__),
                cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0),
                python=platform.python_version(),
            ),
        )
    prepare(sets, smoke=True)
    run_condition(sets, "FedAvg", 0, "NonIID", 1, 42, 10, True)
    prepare(sets)
    for seed in SEEDS:
        for le in (1, 3):
            for pct in SIZES:
                for regime in ("IID", "NonIID"):
                    run_condition(sets, "FedAvg", pct, regime, le, seed)
    import csv

    trigger = []
    for le in (1, 3):
        for pct in SIZES:
            means = {}
            for regime in ("IID", "NonIID"):
                scores = []
                for seed in SEEDS:
                    f = OUT / "runs" / config_id("FedAvg", pct, regime, le, seed) / "rounds.csv"
                    with f.open() as h:
                        scores.append(float(list(csv.DictReader(h))[-1]["global_miou"]))
                means[regime] = float(np.mean(scores))
            delta = means["IID"] - means["NonIID"]
            trigger.append(
                dict(
                    local_epochs=le,
                    pruning_pct=pct,
                    iid_minus_noniid=delta,
                    run_prox=bool(delta >= 0.02),
                )
            )
            if delta >= 0.02:
                for seed in SEEDS:
                    run_condition(sets, "FedProx", pct, "NonIID", le, seed)
    atomic_json(OUT / "prox_decision.json", trigger)
    for seed in SEEDS:
        for le in (1, 3):
            run_condition(sets, "Central", 0, "pooled", le, seed)
            for regime in ("IID", "NonIID"):
                for method in ("Local1", "Local2"):
                    run_condition(sets, method, 0, regime, le, seed)
        run_condition(sets, "FedBN", 0, "NonIID", 3, seed)


if __name__ == "__main__":
    run()
