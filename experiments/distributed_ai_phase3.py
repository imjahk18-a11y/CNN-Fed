"""Provision disjoint patches and run the controlled update study with independent TCP endpoints."""
import itertools
import json
import multiprocessing as mp
import os
import socket
from pathlib import Path

import numpy as np
import torch

from datasets.compact_cache import atomic_json, sha
from datasets.fullz import Scene
from experiments.distributed_ai import OLD as REPRO, OUT as P1, SCENES, CACHE, SEEDS, startfile
from experiments.distributed_ai_phase2 import OUT as P2
from federated.fl_server import run_server
from federated.uav_client import serve_client
from federated.uav_runtime import MODES, vector

OUT = Path("results/distributed_ai_phase3")


def provision():
    split = json.loads((P1 / "split.json").read_text())
    assert sha(P1 / "split.json") == json.loads((P2 / "protocol.json").read_text())["split_sha256"]
    roles = dict(client1=split["NonIID"][0], client2=split["NonIID"][1], validation=split["val"])
    for (role, refs), scene_id in zip(roles.items(), (3, 4, 12)):
        assert {r[0] for r in refs} == {scene_id} and len(set(map(tuple, refs))) == (
            144 if role == "validation" else 96
        )
        scene = Scene(CACHE / SCENES[scene_id])
        directory = OUT / "bundles" / role
        directory.mkdir(parents=True, exist_ok=True)
        dest = directory / "patches.npz"
        if not dest.exists():
            xyz, rgb, labels = zip(*(scene[i] for _, i in refs))
            tmp = directory / "partial.npz"
            np.savez_compressed(
                tmp,
                role=role,
                coords=np.concatenate(xyz),
                rgb=np.concatenate(rgb),
                labels=np.concatenate(labels),
                bounds=np.r_[0, np.cumsum(list(map(len, xyz)))],
                refs=refs,
            )
            tmp.replace(dest)
        with np.load(dest, allow_pickle=False) as d:
            assert np.array_equal(d["refs"], refs)
    return {k: OUT / "bundles" / k / "patches.npz" for k in roles}


def run_dir(seed, mode):
    return OUT / "runs" / f"s{seed}_{mode}"


def tp(path):
    return torch.load(path, map_location="cpu", weights_only=True)["state"]


def reference(seed):
    before = tp(startfile(seed, 0))
    rows = []
    for r in range(1, 31):
        now = tp(run_dir(seed, "INT8") / f"global_{r:02}.pt")
        rows.append(
            {g: (vector(now, g) - vector(before, g)).norm().item() for g in ("parameters", "bn")}
        )
        before = now
    return rows


def launch(seed, mode, bundles, early=False):
    root = OUT / "early_smoke" if early else run_dir(seed, mode)
    if (root / "done.json").exists():
        return
    token = os.urandom(16)
    ctx = mp.get_context("spawn")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(4)
        listener.settimeout(100)
        host, port = listener.getsockname()
        clients = [
            ctx.Process(
                target=serve_client,
                args=(
                    host,
                    port,
                    i,
                    Path(bundles[f"client{i+1}"]),
                    root / f"client{i+1}",
                    token,
                    "cuda:0",
                    True,
                    early and i == 1,
                ),
            )
            for i in (0, 1)
        ]
        try:
            for c in clients:
                c.start()
            run_server(
                listener,
                token,
                startfile(seed, 0),
                Path(bundles["validation"]),
                root,
                seed,
                mode,
                patience=3 if early else 0,
                targets=reference(seed) if mode == "FP32_MATCH" else None,
                trace=True,
            )
            for c in clients:
                c.join(20)
                if c.exitcode != 0:
                    raise RuntimeError(f"Client exit {c.exitcode}")
        finally:
            for c in clients:
                if c.is_alive():
                    c.terminate()
                    c.join()
    meta = json.loads((root / "done.json").read_text())
    if not early and mode in ("FP32", "INT8"):
        older = P2 / "runs" / f"s{seed}_p00_{mode}_0"
        ref = json.loads((older / "done.json").read_text())
        assert all(
            abs(a["miou"] - b["miou"]) < 1e-12
            for a, b in zip(meta["rows"][1:], ref["rows"], strict=True)
        )
        old = tp(older / "final.pt")
        assert all(torch.equal(old[k], v) for k, v in tp(root / "server.pt").items())
    print(
        seed,
        mode,
        "early" if early else "full",
        len(meta["rows"]) - 1,
        meta["rows"][-1]["miou"],
        flush=True,
    )


def main():
    OUT.mkdir(exist_ok=True)
    bundles = provision()
    originals = [p for d in (REPRO / "checkpoints", P1, P2) for p in d.rglob("*") if p.is_file()]
    protocol = dict(
        hashes={str(p): sha(p) for p in originals},
        split=sha(P1 / "split.json"),
        bundles={k: sha(v) for k, v in bundles.items()},
        seeds=SEEDS,
        rounds=30,
        modes=MODES,
        patience=3,
        min_delta=0,
        control="per-round per-seed trainable and BN norm matching to deterministic INT8",
        rounding="nearest ties to even; unbiased stochastic alternative",
    )
    p = OUT / "protocol.json"
    if p.exists():
        assert json.loads(p.read_text()) == json.loads(json.dumps(protocol))
    else:
        atomic_json(p, protocol)
    for seed in SEEDS:
        for mode in ("INT8", "FP32", "FP16", "INT8_SR", "FP32_MATCH"):
            launch(seed, mode, bundles)
    launch(42, "INT8", bundles, True)
    assert all(sha(p) == protocol["hashes"][str(p)] for p in originals)
    atomic_json(OUT / "integrity.json", dict(verified_original_files=len(originals)))


if __name__ == "__main__":
    main()
