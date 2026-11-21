"""Freeze the planned candidates and measure full-scene endpoint generalization."""
from copy import deepcopy
import json

import torch

from datasets.compact_cache import atomic_json, sha
from datasets.fullz import Scene
from experiments.distributed_ai import OUT, CACHE, SEEDS, SIZES, config_id, init
from experiments.sparse_reproduction import batches, load, evaluate


class PersonalizedAverage(torch.nn.Module):
    def __init__(self, net, private):
        super().__init__()
        self.models = torch.nn.ModuleList([deepcopy(net), deepcopy(net)])
        for model, bn in zip(self.models, private):
            model.load_state_dict(bn, strict=False)

    def forward(self, g, x):
        return g.coords, (self.models[0](g, x)[1] + self.models[1](g, x)[1]) / 2


def freeze():
    decisions = json.loads((OUT / "prox_decision.json").read_text())
    expected = set()
    for seed in SEEDS:
        for le in (1, 3):
            expected.add(config_id("Central", 0, "pooled", le, seed))
            for regime in ("IID", "NonIID"):
                for pct in SIZES:
                    expected.add(config_id("FedAvg", pct, regime, le, seed))
                for m in ("Local1", "Local2"):
                    expected.add(config_id(m, 0, regime, le, seed))
        expected.add(config_id("FedBN", 0, "NonIID", 3, seed))
        for decision in decisions:
            if decision["run_prox"]:
                expected.add(
                    config_id(
                        "FedProx", decision["pruning_pct"], "NonIID", decision["local_epochs"], seed
                    )
                )
    assert expected == {p.parent.name for p in (OUT / "runs").glob("*/done.json")}
    candidates = {}
    for key in sorted(expected):
        folder = OUT / "runs" / key
        meta = json.loads((folder / "done.json").read_text())
        assert meta["split_sha256"] == sha(OUT / "split.json") and meta["rounds"] == 30
        for r in (10, 30):
            path = folder / f"r{r}.pt"
            bn = folder / f"bn{r}.pt"
            info = dict(run=key, round=r, path=str(path.relative_to(OUT)), sha256=sha(path))
            if key.startswith("FedBN"):
                info.update(private=str(bn.relative_to(OUT)), private_sha256=sha(bn))
            candidates[f"{key}_r{r}"] = info
    f = OUT / "frozen_endpoints.json"
    if f.exists():
        assert json.loads(f.read_text()) == candidates
    else:
        atomic_json(f, candidates)
    return candidates


def main():
    init()
    candidates = freeze()
    datasets = {}
    for role, name in (("synthetic_full", "12_points_GTv3"), ("wmsc", "WMSC_points")):
        scene = Scene(CACHE / name)
        datasets[role] = batches(scene, False)
    f = OUT / "endpoint_scores.json"
    result = json.loads(f.read_text()) if f.exists() else {}
    for ident, info in candidates.items():
        if ident in result:
            continue
        net = load(OUT / info["path"])
        if "private" in info:
            bn = torch.load(OUT / info["private"], weights_only=True, map_location="cpu")
            net = PersonalizedAverage(net, bn)
        result[ident] = {k: evaluate(net, v) for k, v in datasets.items()}
        atomic_json(f, result)
    print("Verified and evaluated", len(result), "frozen endpoints", flush=True)


if __name__ == "__main__":
    main()
