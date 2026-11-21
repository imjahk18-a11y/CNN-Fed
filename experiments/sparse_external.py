"""Evaluate both frozen models once on all WMSC voxels and retain aligned predictions."""
import argparse
import fcntl
import json
from pathlib import Path

import numpy as np
import torch

from datasets.compact_cache import atomic_json, sha
from datasets.fullz import Scene, common
from experiments.sparse_diagnosis import MODELS
from experiments.sparse_reproduction import BATCH, CACHE, SEED, load, pack, scores_from_confusion
from training.train import seed_all


@torch.inference_mode()
def external(out):
    seed_all(SEED)
    torch.set_num_threads(2)
    frozen = json.loads((out / "frozen.json").read_text())
    assert set(frozen) == set(MODELS)
    for name, info in frozen.items():
        if sha(out / info["file"]) != info["sha256"] or info["sites"] != MODELS[name]:
            raise ValueError("Frozen model mismatch")
    scene = Scene(CACHE / "WMSC_points")
    if sha(Path(scene.meta["source"])) != scene.meta["sha256"]:
        raise ValueError("WMSC source changed")
    score = {}
    for name, info in frozen.items():
        target = out / f"wmsc_predictions_{name}.npz"
        if target.exists():
            with np.load(target, allow_pickle=False) as record:
                if (
                    record["checkpoint"].item() != info["sha256"]
                    or record["source"].item() != scene.meta["sha256"]
                ):
                    raise ValueError("Existing predictions belong to a different checkpoint or dataset")
                pred = record["pred"]
        else:
            net = load(out / info["file"])
            pred = np.empty(len(scene.raw), dtype="uint8")
            for start in range(0, len(scene), BATCH):
                stop = min(start + BATCH, len(scene))
                g, x, y = pack(scene, range(start, stop), info["sites"])
                pos, logits = net(g, x)
                if pos is not g.coords or not torch.isfinite(logits).all():
                    raise ValueError("Invalid external prediction")
                pred[scene.bounds[start] : scene.bounds[stop]] = logits.argmax(1).cpu().numpy()
            staging = target.with_suffix(".writing")
            with staging.open("wb") as f:
                np.savez(f, pred=pred, checkpoint=info["sha256"], source=scene.meta["sha256"])
            staging.replace(target)
        if pred.shape != (len(scene.raw),) or pred.dtype != np.uint8 or np.max(pred) >= 6:
            raise ValueError("Invalid external prediction rows")
        labels = common(scene.raw).astype(np.int64)
        cm = np.bincount(6 * labels + pred, minlength=36).reshape(6, 6)
        assert cm.sum() == scene.meta["voxels"]
        score[name] = scores_from_confusion(cm)
    atomic_json(out / "external.json", dict(scene=scene.meta, models=score))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    out = ap.parse_args().out
    with (out / ".external.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        external(out)
