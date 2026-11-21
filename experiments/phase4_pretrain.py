"""Freeze four-scene splits and train independent original sparse CNNs from scratch."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from datasets.compact_cache import atomic_json, sha
from datasets.fullz import Scene, CLASSES, common
from experiments.distributed_ai import OUT as P1, CACHE, SCENES, batchify, dump, init, train_local
from experiments.sparse_reproduction import fit_epoch, write_csv
from models.active_voxel import SparseUNet
from training.train import seed_all

OUT = Path("results/distributed_ai_phase4")
FOLDS = {"reference": (13, 3, 4, 12), "rotated": (4, 12, 13, 3)}
SEEDS = {"reference": (101, 202, 303), "rotated": (1101, 1202, 1303)}
ROLES = ("pretrain", "finetune", "calibration", "client1", "client2", "validation", "audit")
FL_SEED = 42


def artifact_hashes():
    roots = [
        Path("results") / name
        for name in (
            "distributed_ai",
            "distributed_ai_phase2",
            "distributed_ai_phase3",
            "sparse_3dcnn_reproduction",
        )
    ]
    return {str(p): sha(p) for root in roots for p in sorted(root.rglob("*")) if p.is_file()}


def freeze(path, data):
    canonical = json.loads(json.dumps(data))
    if path.exists():
        if json.loads(path.read_text()) != canonical:
            raise ValueError(f"Inputs differ from frozen {path}")
    else:
        atomic_json(path, canonical)


def distant(xy, chosen, gap=2):
    # Tile origins in units of 32 voxels. Require Chebyshev separation in either axis.
    return np.flatnonzero(
        np.all(np.max(np.abs(xy[:, None] - xy[chosen][None, :]), axis=2) > gap, axis=1)
    )


def make_splits(scenes):
    old = json.loads((P1 / "split.json").read_text())
    stats = []
    folds = {}
    for f, (pre, a, b, v) in FOLDS.items():
        rng = np.random.default_rng(4189)

        def draw(pool, n):
            return rng.choice(pool, n, replace=False).tolist()

        def pairs(s, indices):
            return [[s, int(i)] for i in indices]

        xy = scenes[pre].tile_xy
        # Contiguous corner for calibration. Exclude two surrounding tile rings from training.
        cal = np.lexsort((xy[:, 1], xy[:, 0], xy.max(axis=1)))[:96].tolist()
        train = distant(xy, cal)
        train = rng.permutation(train).tolist()
        val = [p for _, p in old["val"]] if f == "reference" else draw(len(scenes[v]), 144)
        audit = draw(distant(scenes[v].tile_xy, val), 144)
        one, two = (
            old["NonIID"]
            if f == "reference"
            else (pairs(a, draw(len(scenes[a]), 96)), pairs(b, draw(len(scenes[b]), 96)))
        )
        roles = dict(
            pretrain=pairs(pre, train),
            finetune=pairs(pre, draw(train, 192)),
            calibration=pairs(pre, cal),
            client1=one,
            client2=two,
            validation=pairs(v, val),
            audit=pairs(v, audit),
        )
        assert len({scenes[i].meta["sha256"] for i in (pre, a, b, v)}) == 4
        assert set(train).isdisjoint(cal) and set(val).isdisjoint(audit)
        for role, refs in roles.items():
            assert len(set(map(tuple, refs))) == len(refs)
            counts = np.zeros(6, np.int64)
            for sid, i in refs:
                sc = scenes[sid]
                counts += np.bincount(common(sc.raw[sc.bounds[i] : sc.bounds[i + 1]]), minlength=6)
            for name, num in zip(CLASSES, counts):
                stats.append(
                    dict(
                        fold=f,
                        role=role,
                        scene=refs[0][0],
                        patches=len(refs),
                        class_name=name,
                        voxels=int(num),
                        fraction=float(num / counts.sum()),
                    )
                )
        folds[f] = roles
    freeze(OUT / "splits.json", folds)
    write_csv(OUT / "class_distributions.csv", stats)
    return folds


def start_path(f, seed):
    return OUT / "pretraining" / f / f"s{seed}" / "start.pt"


def param_digest(net):
    h = hashlib.sha256()
    for p in net.parameters():
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    init()
    OUT.mkdir(exist_ok=True)
    scenes = {i: Scene(CACHE / name) for i, name in SCENES.items()}
    folds = make_splits(scenes)
    freeze(
        OUT / "protocol.json",
        dict(
            folds=FOLDS,
            seeds=SEEDS,
            fl_seed=FL_SEED,
            rounds=30,
            early_patience=3,
            min_delta=0,
            retention_tolerance_pp=0.5,
            pretrain_epochs=35,
            finetune_epochs=5,
            pretrain_lr=0.001,
            local_lr=0.0001,
            batch=24,
            split_sha=sha(OUT / "splits.json"),
            sources={i: s.meta["sha256"] for i, s in scenes.items()},
            policy="last pretrain epoch; final audit after all FL paths are committed",
            modes=["FP32", "INT8", "FP32_MATCH"],
            bn=["aggregate", "frozen", "private", "recalibrated"],
            gpu=torch.cuda.get_device_name(0),
            torch=str(torch.__version__),
        ),
    )
    freeze(OUT / "prior_hashes.json", artifact_hashes())
    for f, roles in folds.items():
        data = batchify(roles["pretrain"], scenes)
        ft = batchify(roles["finetune"], scenes)
        for seed in SEEDS[f]:
            dest = start_path(f, seed)
            if dest.with_name("done.json").exists():
                assert sha(dest) == json.loads(dest.with_name("done.json").read_text())["start_sha"]
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            seed_all(seed)
            model = SparseUNet().cuda()
            initial = param_digest(model)
            opt = torch.optim.AdamW(model.parameters(), lr=0.001)
            scaler = torch.amp.GradScaler("cuda", init_scale=1024)
            rows = []
            for epoch in range(1, 36):
                row = fit_epoch(model, data, opt, scaler, seed * 100 + epoch)
                rows.append(dict(epoch=epoch, **row))
                if epoch % 10 == 0:
                    print(f, seed, "pretrain epoch", epoch, flush=True)
            dump(model, dest.with_name("pretrained.pt"))
            train_local(model, ft, 5, FL_SEED * 100)
            dump(model, dest)
            write_csv(dest.with_name("epochs.csv"), rows)
            atomic_json(
                dest.with_name("done.json"),
                dict(
                    seed=seed,
                    init_sha=initial,
                    pretrained_sha=sha(dest.with_name("pretrained.pt")),
                    start_sha=sha(dest),
                    patches=len(roles["pretrain"]),
                ),
            )
            print(f, seed, "independent pretraining complete", flush=True)
        del data, ft
    print("All six independent starts ready", flush=True)


if __name__ == "__main__":
    main()
