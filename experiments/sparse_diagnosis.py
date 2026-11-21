"""Paired stencil comparison. External data are consumed by a separate frozen evaluation."""
import argparse
from datetime import datetime
import json
from itertools import product
from pathlib import Path
import platform
import socket
import sys
import time

import numpy as np
import torch

from datasets.compact_cache import atomic_json, sha
from datasets.fullz import CLASSES, Scene, common
from datasets.storage_budget import budget
from experiments.sparse_reproduction import (
    BATCH,
    CACHE,
    DEVICE,
    ROLES,
    SEED,
    batches,
    evaluate,
    fit,
    fit_epoch,
    load,
    pack,
    profile,
    save,
    write_csv,
)
from models.active_voxel import SparseUNet, kernel_offsets
from training.train import seed_all

OUT = Path("results/sparse_diagnosis")
MODELS = {"A": 7, "B": 27}


def layer_support(sites):
    """Exact interior inference dependency on a fully occupied lattice, up to pooling parity."""
    offsets = kernel_offsets(sites)
    children = tuple(product((0, 1), repeat=3))

    def expand(coords, steps):
        return {tuple(a + b for a, b in zip(q, v)) for q in coords for v in steps}

    def pool_back(coords):
        return expand({tuple(2 * x for x in q) for q in coords}, children)

    def describe(cells):
        c = np.array(sorted(cells))
        return dict(points=len(c), bbox_cells=(np.ptp(c, axis=0) + 1).tolist())

    coarse = {(0, 0, 0)}
    encoder = expand(coarse, offsets)
    result = dict(
        encoder=describe(encoder),
        pool=describe(expand(pool_back(coarse), offsets)),
        bridge=describe(expand(pool_back(expand(coarse, offsets)), offsets)),
    )
    for q in children:
        fine = expand({q}, offsets)
        parents = {tuple(x // 2 for x in cell) for cell in fine}
        deep = expand(pool_back(expand(parents, offsets)), offsets)
        all_inputs = deep | expand(fine, offsets)
        result[str(q)] = describe(all_inputs)
    return result


def scan(scene):
    hist = np.bincount(common(scene.raw), minlength=6)
    assert hist.tolist() == scene.meta["common_hist"]
    rgb_min, rgb_max = scene.rgb.min(0), scene.rgb.max(0)
    assert np.isfinite(scene.rgb).all() and min(rgb_min) >= 0 and max(rgb_max) <= 255
    assert (
        len(scene.rgb) == len(scene.raw) == len(scene.coords) == scene.bounds[-1] == scene.meta["voxels"]
    )
    assert len(scene) == scene.meta["patches"] and (np.diff(scene.bounds) > 0).all()
    low, high = scene.coords.min(0), scene.coords.max(0)
    assert min(low) >= 0 and max(high[:2]) < 32 and high[2] < scene.meta["shape"][2]
    return dict(
        **scene.meta,
        rgb_min=rgb_min.tolist(),
        rgb_max=rgb_max.tolist(),
        feature_min=(rgb_min / 255).tolist(),
        feature_max=(rgb_max / 255).tolist(),
        coord_min=low.tolist(),
        coord_max=high.tolist(),
    )


def smoke(scene, sites, out):
    ids = [0, len(scene) // 2, len(scene) - 1]
    batch = pack(scene, ids, sites)
    g, x, y = batch
    original = np.concatenate([scene[k][1] for k in ids])
    reference = original / np.float32(255)
    np.testing.assert_array_max_ulp(x.cpu().numpy(), reference, maxulp=1)
    np.testing.assert_array_equal(y.cpu().numpy(), np.concatenate([scene[k][2] for k in ids]))
    np.testing.assert_array_equal(
        g.coords.cpu().numpy(),
        np.concatenate(
            [np.column_stack([np.full(len(scene[k][0]), n), scene[k][0]]) for n, k in enumerate(ids)]
        ),
    )
    if not torch.equal(g.coarse[g.parent, 0], g.coords[:, 0]) or not torch.equal(
        g.coarse[g.parent, 1:], g.coords[:, 1:] // 2
    ):
        raise ValueError("Parent mismatch")
    seed_all(SEED)
    net = SparseUNet(sites=sites).to(DEVICE)
    first = net.encoder.weight.detach().clone()
    captured = []
    hook = net.encoder.register_forward_pre_hook(lambda mod, args: captured.append(args[0] is x))
    opt = torch.optim.AdamW(net.parameters(), lr=0.001)
    row = fit_epoch(net, [batch], opt, torch.amp.GradScaler("cuda", init_scale=1024), SEED + 1)
    hook.remove()
    assert captured == [True] and not torch.equal(first, net.encoder.weight)
    assert all(
        p.grad is not None and p.device == DEVICE and torch.isfinite(p.grad).all()
        for p in net.parameters()
    )
    path = out / f"smoke_{sites}.pt"
    save(net, path)
    with torch.no_grad():
        pos, pred = load(path)(g, x)
        assert pos is g.coords and pred.shape == (len(x), 6)
        torch.testing.assert_close(net.eval()(g, x)[1], pred, atol=0, rtol=0)
    return dict(
        **row,
        patches=ids,
        cpu_gpu_RGB_max_abs=float(np.abs(x.cpu().numpy() - reference).max()),
        exact_reload=True,
        encoder_received_rgb_only=True,
        coordinate_order=["batch", "x_local", "y_local", "z_scene"],
        channels=["R/255", "G/255", "B/255"],
        layer_parameters={
            k: sum(p.numel() for p in module.parameters()) for k, module in net.named_children()
        },
        receptive_field=layer_support(sites),
    )


def archive_inventory(out):
    source = Path("results/synthetic_to_real/acquisition.json")
    listing = json.loads(source.read_text())
    local_ply = {(p.name, p.stat().st_size) for p in Path("data").rglob("*.ply")}
    records = [
        dict(
            version=version,
            **member,
            local=(Path(member["member"]).name, member["bytes"]) in local_ply,
        )
        for version, members in listing["candidates"].items()
        for member in members
    ]
    write_csv(out / "archive_inventory.csv", records)
    atomic_json(
        out / "storage.json",
        dict(
            measured=datetime.now().astimezone().isoformat(),
            quota=budget(out),
            archive_bytes=listing["archive_bytes"],
            source=str(source),
            unpacked_bytes={
                k: sum(m["bytes"] for m in group) for k, group in listing["candidates"].items()
            },
            missing_v3_bytes=sum(r["bytes"] for r in records if r["version"] == "3" and not r["local"]),
            downloaded_bytes=0,
        ),
    )


def main(out):
    seed_all(SEED)
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("cuda:0 required")
    out.mkdir(parents=True, exist_ok=False)
    archive_inventory(out)
    scenes = {k: Scene(CACHE / v) for k, v in ROLES.items()}
    atomic_json(
        out / "protocol.json",
        dict(
            host=socket.gethostname(),
            executable=sys.executable,
            python=platform.python_version(),
            torch=str(torch.__version__),
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0),
            total_vram_mib=torch.cuda.get_device_properties(0).total_memory / 2**20,
            seed=SEED,
            batch=BATCH,
            epochs=35,
            lr=0.001,
            optimizer="AdamW",
            weight_decay=0.01,
            loss="mean unweighted CE over all occupied voxels; unknown raw IDs rejected; no ignored targets",
            selection="earliest maximum synthetic validation mIoU independently for A and B",
            roles=ROLES,
            classes=CLASSES,
            models=MODELS,
            validation_precision="FP32",
            training_precision="FP16 AMP",
            scenes={k: scan(v) for k, v in scenes.items()},
            historical_patches={"train": 145329, "validation": 37837, "wmsc": 1271},
        ),
    )
    atomic_json(out / "smoke.json", {k: smoke(scenes["central"], v, out) for k, v in MODELS.items()})
    print("smoke A/B passed", flush=True)
    frozen = {}
    for name, sites in MODELS.items():
        start = time.perf_counter()
        train = batches(scenes["central"], True, sites)
        val = batches(scenes["validation"], False, sites)
        torch.cuda.synchronize()
        prep_s = time.perf_counter() - start
        seed_all(SEED)
        model = SparseUNet(sites=sites).to(DEVICE)
        path = out / f"{name}.pt"
        logs = fit(model, train, val, 35, 0.001, path, name)
        write_csv(out / f"training_{name}.csv", logs)
        selected = load(path)
        score = evaluate(selected, val)
        best = max(logs, key=lambda r: r["val_miou"])
        assert score["miou"] == best["val_miou"]
        v = scenes["validation"]
        probe = int(np.argsort(np.diff(v.bounds), kind="stable")[len(v) // 2])
        g, x, y = pack(v, [probe], sites)
        metrics = profile(selected, (g, x, y))
        atomic_json(
            out / f"validation_{name}.json",
            dict(
                score=score,
                profile=metrics,
                edges_fine=len(x) + sum(len(dst) for dst, _ in g.fine_edges),
                edges_coarse=len(g.coarse) + sum(len(dst) for dst, _ in g.coarse_edges),
                probe_index=probe,
                prep_s=prep_s,
            ),
        )
        frozen[name] = dict(
            sites=sites,
            file=path.name,
            sha256=sha(path),
            epoch=best["epoch"],
            validation_miou=best["val_miou"],
        )
        del train, val, model, selected, g, x, y
        torch.cuda.empty_cache()
    atomic_json(out / "frozen.json", frozen)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    main(ap.parse_args().out)
