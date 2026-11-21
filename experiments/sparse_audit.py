"""Verify stored Full-Z scene coordinates and sampled votes against original PLY points."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from datasets.compact_cache import atomic_json, sha
from datasets.fullz import Scene, SIDE
from datasets.ply_stream import header


def verify_scene(root):
    scene = Scene(root)
    meta = scene.meta
    shape = np.array(meta["shape"])
    origin = np.array(meta["origin"])
    n = len(scene.raw)
    checks = np.random.default_rng(918).choice(n, min(n, 96), replace=False)
    coords = scene.coords[checks].copy()
    patch = np.searchsorted(scene.bounds, checks, side="right") - 1
    coords[:, :2] += SIDE * scene.tile_xy[patch]
    keys = np.ravel_multi_index(coords.T, shape)
    target = set(keys.tolist())
    sorted_keys = np.sort(keys)
    votes = defaultdict(list)
    path = Path(meta["source"])
    h = header(path)
    pts = np.memmap(path, mode="r", dtype=h.dtype, offset=h.offset, shape=(h.count,))
    for i in range(0, len(pts), 200_000):
        rows = pts[i : i + 200_000]
        xyz = np.stack([rows[k] for k in "xyz"], axis=1).astype("float64")
        cell = np.floor((xyz - origin) / 0.3).astype("int64")
        key = np.ravel_multi_index(cell.T, shape)
        index = np.searchsorted(sorted_keys, key).clip(max=len(keys) - 1)
        for j in np.flatnonzero(sorted_keys[index] == key):
            rgb = tuple(int(rows[k][j]) for k in ("red", "green", "blue"))
            offset = xyz[j] - (origin + (cell[j] + 0.5) * 0.3)
            votes[int(key[j])].append((float(offset @ offset), i + j, int(rows["class"][j]), rgb))
    assert set(votes) == target
    for key, idx in zip(keys, checks):
        points = votes[int(key)]
        # Plain scalar oracle, independent of vectorized segmented reduction.
        colors = [tuple(c // 16 for c in row[3]) for row in points]

        def vote(values):
            counts = Counter(values)
            best = max(counts.values())
            return min(
                (d, source_i, value)
                for (d, source_i, *_), value in zip(points, values)
                if counts[value] == best
            )[2]

        label = vote([row[2] for row in points])
        color = vote(colors)
        mean = np.mean([row[3] for row, b in zip(points, colors) if b == color], axis=0)
        assert int(scene.raw[idx]) == label
        np.testing.assert_allclose(scene.rgb[idx], mean, rtol=1e-7)
    unique_patches = set(map(tuple, scene.tile_xy))
    assert len(unique_patches) == len(scene)
    assert scene.bounds[0] == 0 and scene.bounds[-1] == n and (np.diff(scene.bounds) > 0).all()
    total = 0
    zlimit = [int(shape[2]), -1]
    for i in range(len(scene)):
        coords, _, _ = scene[i]
        global_coords = coords.copy()
        global_coords[:, :2] += scene.tile_xy[i] * SIDE
        assert (coords >= 0).all() and (coords[:, :2] < SIDE).all()
        global_key = np.ravel_multi_index(global_coords.T, shape)
        assert (np.diff(global_key) > 0).all()
        total += len(coords)
        zlimit = [min(zlimit[0], int(coords[:, 2].min())), max(zlimit[1], int(coords[:, 2].max()))]
    assert total == meta["voxels"] and zlimit == [0, int(shape[2]) - 1]
    assert sha(path) == meta["sha256"]
    return dict(
        checked_votes=len(keys),
        points_in_sampled_voxels=sum(map(len, votes.values())),
        unique_voxels=total,
        unique_xy_patches=len(scene),
        min_max_z=zlimit,
        source_hash_ok=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--cache", type=Path, default=Path("data/sparse_fullz_v1"))
    args = parser.parse_args()
    cfg = json.loads((args.run / "protocol.json").read_text())
    if not (args.run / "frozen.json").exists():
        parser.error("Freeze models before the external audit")
    audits = {name: verify_scene(args.cache / name) for name in [*cfg["roles"].values(), "WMSC_points"]}
    atomic_json(args.run / "cache_audit.json", audits)
