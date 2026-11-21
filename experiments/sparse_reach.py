"""Count actual spatial paths on a label-blind, deterministic Synthetic validation sample."""
import argparse
from pathlib import Path

import numpy as np

from datasets.compact_cache import atomic_json
from datasets.fullz import Scene
from experiments.sparse_diagnosis import MODELS
from experiments.sparse_reproduction import CACHE, ROLES
from models.active_voxel import kernel_offsets


def adjacency(cells, offsets):
    return {
        q: {
            (q[0] + dx, q[1] + dy, q[2] + dz)
            for dx, dy, dz in offsets
            if (q[0] + dx, q[1] + dy, q[2] + dz) in cells
        }
        for q in cells
    }


def count(out):
    scene = Scene(CACHE / ROLES["validation"])
    rng = np.random.default_rng(830)
    patch_ids = rng.choice(len(scene), 48, replace=False)
    reach = {k: [] for k in MODELS}
    degree = {k: [] for k in MODELS}
    for i in patch_ids:
        c = scene.coords[scene.bounds[i] : scene.bounds[i + 1]]
        fine = set(map(tuple, c.tolist()))
        parents = {q: tuple(x // 2 for x in q) for q in fine}
        coarse = set(parents.values())
        child = {p: set() for p in coarse}
        for q, p in parents.items():
            child[p].add(q)
        targets = [tuple(v) for v in c[rng.choice(len(c), min(len(c), 32), replace=False)].tolist()]
        for name, sites in MODELS.items():
            n1 = adjacency(fine, kernel_offsets(sites))
            n2 = adjacency(coarse, kernel_offsets(sites))
            for q in targets:
                degree[name].append(len(n1[q]) - 1)
                coarse_src = set().union(*(n2[parents[x]] for x in n1[q]))
                pooled = set().union(*(child[p] for p in coarse_src))
                reachable = set().union(*(n1[x] for x in pooled | n1[q]))
                reach[name].append(len(reachable))
    for idx in range(len(reach["A"])):
        assert reach["A"][idx] <= reach["B"][idx]
    atomic_json(
        out / "spatial_support.json",
        dict(
            patch_ids=patch_ids.tolist(),
            centers=len(reach["A"]),
            sampling="48 validation patches uniform; up to 32 unique occupied targets each; same targets A/B; seed 830",
            models={
                name: dict(
                    median_reachable=float(np.median(reach[name])),
                    mean_reachable=float(np.mean(reach[name])),
                    p10_p90=np.percentile(reach[name], [10, 90]).tolist(),
                    mean_immediate_neighbors=float(np.mean(degree[name])),
                    center_with_no_neighbors=int(np.sum(np.asarray(degree[name]) == 0)),
                )
                for name in MODELS
            },
            strictly_more_reachable=int(np.sum(np.array(reach["B"]) > reach["A"])),
            axial_isolated_but_cube_connected=int(
                np.sum((np.array(degree["A"]) == 0) & (np.array(degree["B"]) > 0))
            ),
        ),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    count(ap.parse_args().out)
