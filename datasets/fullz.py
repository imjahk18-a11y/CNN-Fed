"""Full-scene occupied voxels, partitioned only along X and Y."""
import argparse
import fcntl
import json
from pathlib import Path

import numpy as np

from .compact_cache import atomic_json, sha
from .ply_stream import header

RESOLUTION = 0.3
SIDE = 32
CLASSES = ("Ground", "Building", "Tree", "Car", "Light pole", "Fence")
# STPLS3D README, section 2.5. ID 16 has no published definition.
GROUPS = ((0, 15, 18, 19), (1, 7, 9, 10, 13, 17), (2, 3, 4), (5, 6, 8), (11, 12), (14,))
LOOKUP = np.full(256, -1, dtype=np.int8)
for i, ids in enumerate(GROUPS):
    LOOKUP[list(ids)] = i


def common(raw):
    raw = np.asarray(raw)
    if raw.dtype.kind not in "iu" or (raw < 0).any() or (raw >= len(LOOKUP)).any():
        raise ValueError("Invalid raw semantic ID")
    labels = LOOKUP[raw]
    if (labels < 0).any():
        raise ValueError(f"Undefined raw semantic IDs {np.unique(raw[labels < 0])}")
    return labels


def xyz(rows):
    return np.column_stack([rows[k] for k in "xyz"]).astype(np.float64)


def winner(group, value, distance, radix):
    # Per candidate, find count and closest point. Input row order breaks exact distance ties.
    pair = group * radix + value
    unique, inverse, counts = np.unique(pair, return_inverse=True, return_counts=True)
    nearest = np.full(len(unique), np.inf)
    np.minimum.at(nearest, inverse, distance)
    closest = distance == nearest[inverse]
    first = np.full(len(unique), len(group))
    np.minimum.at(first, inverse[closest], np.flatnonzero(closest))
    voxel = unique // radix
    rank = np.lexsort((first, nearest, -counts, voxel))
    best = rank[np.r_[True, np.diff(voxel[rank]) != 0]]
    return unique[best] % radix


def representatives(group, raw, rgb, distance):
    raw = np.asarray(raw, dtype=np.int64)
    common(raw)
    semantic = winner(group, raw, distance, 256)
    bins = rgb.astype(np.int64) // 16
    colors = (bins[:, 0] * 16 + bins[:, 1]) * 16 + bins[:, 2]
    selected = colors == winner(group, colors, distance, 4096)[group]
    members = group[selected]
    count = np.bincount(members)
    mean = np.stack([np.bincount(members, weights=ch[selected]) / count for ch in rgb.T], axis=1)
    return semantic.astype(np.uint8), mean.astype(np.float32)


def convert(path, dest):
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / ".build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (dest / "scene.json").exists():
            if Scene(dest).meta["sha256"] != sha(path):
                raise ValueError("Source checksum changed")
            return
        _build_scene(path, dest)


def _build_scene(path, dest):
    h = header(path)
    if not h.fmt.startswith("binary") or any(
        h.dtype[k] != np.uint8 for k in ("red", "green", "blue", "class")
    ):
        raise ValueError("Expected binary STPLS3D XYZ + uint8 RGB/class")
    points = np.memmap(path, dtype=h.dtype, mode="r", offset=h.offset, shape=(h.count,))
    block = 300_000
    low, high = np.full(3, np.inf), np.full(3, -np.inf)
    for start in range(0, h.count, block):
        coord = xyz(points[start : start + block])
        if not np.isfinite(coord).all():
            raise ValueError("Nonfinite XYZ")
        low, high = np.minimum(low, coord.min(0)), np.maximum(high, coord.max(0))
    origin = np.floor(low / RESOLUTION) * RESOLUTION
    shape = np.floor((high - origin) / RESOLUTION).astype(np.int64) + 1
    keys = np.empty(h.count, dtype=np.int64)
    for start in range(0, h.count, block):
        cells = np.floor((xyz(points[start : start + block]) - origin) / RESOLUTION).astype(np.int64)
        keys[start : start + len(cells)] = np.ravel_multi_index(cells.T, shape)
    order = np.argsort(keys, kind="stable")
    keys = keys[order]
    starts = np.r_[0, np.flatnonzero(np.diff(keys)) + 1, len(keys)]
    cells = np.array(np.unravel_index(keys[starts[:-1]], shape)).T
    raw = np.empty(len(cells), np.uint8)
    rgb = np.empty((len(cells), 3), np.float32)
    a = 0
    while a < len(cells):
        b = max(a + 1, int(np.searchsorted(starts, starts[a] + block, side="right")) - 1)
        group = np.repeat(np.arange(b - a), np.diff(starts[a : b + 1]))
        rows = points[order[starts[a] : starts[b]]]
        offset = xyz(rows) - (origin + (cells[a:b][group] + 0.5) * RESOLUTION)
        colors = np.column_stack([rows[k] for k in ("red", "green", "blue")])
        raw[a:b], rgb[a:b] = representatives(
            group, rows["class"], colors, np.einsum("ij,ij->i", offset, offset)
        )
        a = b
    tile = cells[:, :2] // SIDE
    key = tile[:, 0] * ((shape[1] + SIDE - 1) // SIDE) + tile[:, 1]
    sort = np.argsort(key, kind="stable")
    bounds = np.r_[0, np.flatnonzero(np.diff(key[sort])) + 1, len(sort)]
    tile_xy = tile[sort[bounds[:-1]]]
    cells[:, :2] %= SIDE
    for name, arr in dict(
        coords=cells[sort].astype(np.int32), rgb=rgb[sort], raw=raw[sort], bounds=bounds, tile_xy=tile_xy
    ).items():
        np.save(dest / name, arr)
    metadata = dict(
        source=str(path),
        sha256=sha(path),
        points=h.count,
        origin=origin.tolist(),
        shape=shape.tolist(),
        voxels=len(cells),
        patches=len(tile_xy),
        resolution=RESOLUTION,
        stride=SIDE,
        raw_hist=np.bincount(raw, minlength=20).tolist(),
        common_hist=np.bincount(common(raw), minlength=6).tolist(),
    )
    atomic_json(dest / "scene.json", metadata)
    print(path.name, metadata, flush=True)


class Scene:
    def __init__(self, root):
        self.meta = json.loads((root / "scene.json").read_text())
        for name in ("coords", "rgb", "raw", "bounds", "tile_xy"):
            setattr(self, name, np.load(root / (name + ".npy"), mmap_mode="r"))

    def __len__(self):
        return len(self.bounds) - 1

    def __getitem__(self, index):
        start, end = self.bounds[index : index + 2]
        return self.coords[start:end], self.rgb[start:end], common(self.raw[start:end])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/sparse_fullz_v1"))
    args = parser.parse_args()
    for source in args.paths:
        convert(source, args.out / source.stem)
