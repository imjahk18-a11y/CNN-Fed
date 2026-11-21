"""Prepare semantic STPLS3D RealWorldData PLYs and a spatial split manifest."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from .voxelization import IGNORE, aggregate, dense

SOURCE = "https://github.com/meidachen/STPLS3D/blob/main/KPConv-PyTorch/datasets/STPLS3D.py"
CLASSES = ["ground", "building", "vegetation", "cars", "lightStreetSigns", "fences"]
# Official changeSemLabels on documented raw IDs (including ID 16 in that code).
RAW_TO_CLASS = [0, 1, 2, 2, 2, 3, 3, 1, 3, 1, 1, 4, 4, 1, 5, 0, 0, 1, 0, 0]
PLY_TYPES = {"char":"i1", "uchar":"u1", "short":"i2", "ushort":"u2", "int":"i4",
             "uint":"u4", "float":"f4", "double":"f8", "int8":"i1", "uint8":"u1",
             "int16":"i2", "uint16":"u2", "int32":"i4", "uint32":"u4",
             "float32":"f4", "float64":"f8"}


def vertices(path):
    with path.open("rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError(f"Missing PLY signature: {path}")
        fmt, count, element = None, None, None
        fields = []
        for raw in f:
            line = raw.decode("ascii").split()
            if not line:
                continue
            if line[0] == "end_header":
                break
            if line[0] == "format":
                fmt = line[1]
            elif line[0] == "element":
                if count is None and line[1] != "vertex":
                    raise ValueError("Expected first element to be vertex")
                element = line[1]
                if element == "vertex":
                    count = int(line[2])
            elif line[0] == "property" and element == "vertex":
                if line[1] not in PLY_TYPES:
                    raise ValueError(f"Unsupported vertex property: {line}")
                fields.append((line[2], PLY_TYPES[line[1]]))
        else:
            raise ValueError("Incomplete PLY header")
        if not count or count < 0 or not fields:
            raise ValueError("Empty PLY vertex element")
        dtype = np.dtype(fields)
        if fmt == "ascii":
            pts = np.loadtxt(f, dtype=dtype, max_rows=count, ndmin=1)
            if len(pts) != count:
                raise ValueError("Truncated ASCII vertices")
            return pts
        if fmt not in {"binary_little_endian","binary_big_endian"}:
            raise ValueError(f"Unsupported PLY format {fmt}")
        dtype = dtype.newbyteorder("<" if fmt == "binary_little_endian" else ">")
        return np.memmap(path, mode="r", dtype=dtype, offset=f.tell(), shape=(count,))


def assign_splits(rows):
    scenes = sorted({r["scene"] for r in rows})
    if len(scenes) >= 3:
        test = next((s for s in scenes if s.lower() == "wmsc_points"), scenes[-1])
        val = next(s for s in reversed(scenes) if s != test)
        for r in rows:
            r["split"] = "test" if r["scene"] == test else "val" if r["scene"] == val else "train"
    else:
        # Sites are independent. Keep vertical stacks together along the longest XY axis.
        for scene in scenes:
            subset = [r for r in rows if r["scene"] == scene]
            coords = np.asarray([r["tile"][:2] for r in subset])
            axis = int(np.argmax(np.ptp(coords, axis=0)))
            columns = np.unique(coords[:, axis])
            if len(columns) < 5:
                raise ValueError(f"{scene}: need 5 occupied horizontal patch columns for buffered splits")
            first = max(1, min(len(columns)-4, int(.65*len(columns))))
            last = max(first+2, min(len(columns)-2, int(.83*len(columns))))
            for r in subset:
                pos = r["tile"][axis]
                if pos in (columns[first], columns[last]):
                    r["split"] = "buffer"
                else:
                    r["split"] = "train" if pos < columns[first] else "val" if pos < columns[last] else "test"
    train = [r for r in rows if r["split"] == "train"]
    sites = {r["scene"] for r in train}
    if len(sites) > 1:
        for r in train:
            r["region"] = r["scene"]
    else:
        xy = np.array([r["tile"][:2] for r in train])
        axis = int(np.argmax(np.ptp(xy, axis=0)))
        cols = np.unique(xy[:, axis])
        if len(cols) < 2:
            raise ValueError("Training footprint needs at least two horizontal columns for regional FL")
        cut = cols[len(cols)//2]
        for r in train:
            r["region"] = r["scene"] + ("-west_or_south" if r["tile"][axis] < cut else "-east_or_north")


def prepare_scene(path, destination, side, resolution, field, schema, ignore_raw):
    points = vertices(path)
    names = set(points.dtype.names)
    if not {"x","y","z",field} <= names:
        raise ValueError(f"{path} requires XYZ and semantic field {field}. Found {sorted(names)}")
    rgb_fields = ["red", "green", "blue"]
    has_rgb = set(rgb_fields) <= names
    xyz = np.column_stack([points[k] for k in "xyz"]).astype("float64", copy=False)
    if not np.isfinite(xyz).all():
        raise ValueError(f"Nonfinite XYZ in {path}")
    origin = xyz.min(axis=0)
    xyz -= origin
    xyz /= resolution
    raw = points[field]
    if not np.issubdtype(raw.dtype, np.integer):
        if not np.isfinite(raw).all() or not np.all(raw == np.floor(raw)):
            raise ValueError("Semantic IDs must be integers")
    observed, counts = np.unique(raw, return_counts=True)
    mapping = RAW_TO_CLASS if schema == "raw" else list(range(len(CLASSES)))
    unknown = set(observed.tolist()) - set(range(len(mapping))) - set(ignore_raw)
    if unknown:
        raise ValueError(f"Unknown semantic IDs {unknown} in {path}; inspect label schema")
    target = np.full(len(raw), IGNORE, dtype=np.int16)
    for i, label in enumerate(mapping):
        if i not in ignore_raw:
            target[raw == i] = label
    rgb = np.column_stack([points[k] for k in rgb_fields]) if has_rgb else None
    if has_rgb and (not np.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 255):
        raise ValueError("Expected RGB in 0..255")
    tiles = np.floor(xyz/side).astype(np.int64)
    order = np.lexsort(tiles.T[::-1])
    sorted_tiles = tiles[order]
    starts = np.r_[0, np.flatnonzero(np.any(np.diff(sorted_tiles, axis=0), axis=1))+1, len(order)]
    scene_dir = destination/path.stem
    scene_dir.mkdir()
    rows = []
    for start, end in zip(starts[:-1], starts[1:]):
        idx = order[start:end]
        tile = tiles[idx[0]]
        slots, features, labels = aggregate(xyz[idx]-tile*side, rgb[idx] if has_rgb else None,
                                           target[idx], side, len(CLASSES))
        valid = labels != IGNORE
        if not valid.any():
            continue
        hist = np.bincount(labels[valid], minlength=len(CLASSES)).tolist()
        name = "_".join(map(str,tile))
        out = scene_dir/(name+".npz")
        np.savez_compressed(out, slots=slots, features=features, target=labels)
        rows.append(dict(id=path.stem+"/"+name, scene=path.stem, tile=tile.tolist(), hist=hist,
                         file=str(out.relative_to(destination))))
    with path.open("rb") as f:
        sha = hashlib.file_digest(f, "sha256").hexdigest()
    return rows, dict(file=str(path.resolve()), sha256=sha, origin_m=origin.tolist(), rgb=has_rgb,
                      raw_counts={str(int(k)):int(v) for k,v in zip(observed,counts)})


class VoxelPatches(Dataset):
    def __init__(self, root, rows, side, variant):
        self.root, self.rows, self.side, self.variant = root, rows, side, variant

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        with np.load(self.root/self.rows[i]["file"], allow_pickle=False) as data:
            return dense(data["slots"], data["features"], data["target"], self.side, self.variant)


def prepare(paths, out, side=32, resolution=.3, field="class", schema="raw", ignore_raw=()):
    if not paths or len({p.stem for p in paths}) != len(paths):
        raise ValueError("Provide uniquely named independent site PLY files")
    if side < 2 or not np.isfinite(resolution) or resolution <= 0 or schema not in {"raw","benchmark"}:
        raise ValueError("Invalid patch parameters or schema")
    out.mkdir(parents=True, exist_ok=False)
    rows, sources = [], []
    for path in sorted(paths):
        patch_rows, meta = prepare_scene(path, out, side, resolution, field, schema, ignore_raw)
        if not patch_rows:
            raise ValueError(f"No supervised patches in {path}")
        rows.extend(patch_rows)
        sources.append(meta)
    if len({s["sha256"] for s in sources}) != len(sources):
        raise ValueError("Duplicate source point clouds")
    assign_splits(rows)
    manifest = dict(dataset="STPLS3D", classes=CLASSES, side=side, resolution=resolution, schema=schema,
                    label_field=field, ignore_raw=list(ignore_raw), mapping_source=SOURCE,
                    raw_to_train_id=RAW_TO_CLASS if schema=="raw" else list(range(len(CLASSES))),
                    sources=sources, rows=rows)
    (out/"manifest.json").write_text(json.dumps(manifest))
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ply", nargs="+", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--resolution", type=float, default=.3)
    ap.add_argument("--side", type=int, default=32)
    ap.add_argument("--label-field", default="class")
    ap.add_argument("--schema", choices=["raw","benchmark"], default="raw")
    ap.add_argument("--ignore-raw", type=int, nargs="*", default=[])
    args = ap.parse_args()
    prepare(args.ply,args.out,args.side,args.resolution,args.label_field,args.schema,args.ignore_raw)
