"""Post-freeze RGB, height and spatial error analysis using saved WMSC predictions."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np

from datasets.compact_cache import atomic_json
from datasets.fullz import CLASSES, Scene, common
from experiments.sparse_reproduction import CACHE, ROLES, write_csv


def values(scene):
    labels = common(scene.raw)
    z = scene.coords[:, 2]
    # Heights represent the voxel center; five scenes have independent coordinate origins.
    relative = (z + 0.5) * 0.3
    tile_floor = np.minimum.reduceat(z, scene.bounds[:-1])
    local = (z - np.repeat(tile_floor, np.diff(scene.bounds))) * 0.3
    return labels, dict(
        R=scene.rgb[:, 0],
        G=scene.rgb[:, 1],
        B=scene.rgb[:, 2],
        z_scene_m=relative,
        z_world_m=relative + scene.meta["origin"][2],
        z_patch_m=local,
    )


def describe(columns, mask, source, group):
    n = int(mask.sum())
    row = dict(source=source, group=group, voxels=n)
    for name, col in columns.items():
        arr = col[mask]
        stats = (
            (
                np.mean(arr, dtype=np.float64),
                np.std(arr, dtype=np.float64),
                *np.percentile(arr, [10, 50, 90]),
            )
            if n
            else [None] * 5
        )
        row.update(
            {f"{name}_{measure}": v for measure, v in zip(("mean", "std", "p10", "p50", "p90"), stats)}
        )
    return row


def rgb_bins(rgb):
    b = np.floor(rgb / 16).astype("int16")
    return b[:, 0] * 256 + b[:, 1] * 16 + b[:, 2]


def spatial_examples(out, scene, labels, preds):
    counts = np.diff(scene.bounds)
    gt = np.add.reduceat((labels == 0).astype("int64"), scene.bounds[:-1])
    wrong = np.add.reduceat(((labels == 0) & (preds["A"] == 1)).astype("int64"), scene.bounds[:-1])
    candidates = np.where((gt >= 100) & (wrong > 0))[0]
    order = candidates[np.argsort(wrong[candidates] / gt[candidates], kind="stable")]
    indices = order[np.rint((len(order) - 1) * np.array([0.1, 0.5, 0.9])).astype(int)]
    rows, patch_rows = [], []
    colors = ["#b19859", "#be554a", "#37865d", "#477ccd", "#8b5da3", "#c785b0"]
    cmap = ListedColormap(colors)
    fig, axes = plt.subplots(len(indices), 4, figsize=(9, 7), layout="constrained")
    for j, (q, i) in enumerate(zip((0.1, 0.5, 0.9), indices)):
        s = slice(scene.bounds[i], scene.bounds[i + 1])
        coords = scene.coords[s]
        # Top surface for RGB/semantic image; example CSV and rates include every height.
        top = np.argsort(coords[:, 2], kind="stable")
        xy = coords[top, :2]
        fields = [scene.rgb[s] / 255, labels[s], preds["A"][s], preds["B"][s]]
        for ax, field in zip(axes[j], fields):
            canvas = np.full((32, 32, 3), 0.97)
            color = field if field.ndim == 2 else cmap(field.astype(int))[:, :3]
            canvas[xy[:, 1], xy[:, 0]] = color[top]
            ax.imshow(canvas, origin="lower", interpolation="nearest")
            ax.set(xticks=[], yticks=[])
        axes[j, 0].set_ylabel(f"Tile {i}\nA G→B {wrong[i]/gt[i]:.0%}")
        for a, title in zip(axes[0], ["RGB-16", "Ground truth", "A · 7 sites", "B · 27 sites"]):
            a.set_title(title)
        patch_rows.append(
            dict(
                quantile=q,
                patch=int(i),
                tile_x=int(scene.tile_xy[i, 0]),
                tile_y=int(scene.tile_xy[i, 1]),
                occupied=int(counts[i]),
                ground=int(gt[i]),
                a_ground_to_building=int(wrong[i]),
                b_ground_to_building=int(np.sum((labels[s] == 0) & (preds["B"][s] == 1))),
            )
        )
        for group, target in (("Ground→Building", 1), ("Ground→Ground", 0)):
            ids = np.where((labels[s] == 0) & (preds["A"][s] == target))[0]
            if not len(ids):
                continue
            idx = ids[len(ids) // 2]
            x, y, z = map(int, coords[idx])
            rows.append(
                dict(
                    patch=int(i),
                    example=group,
                    x=x,
                    y=y,
                    z=z,
                    world_z_m=scene.meta["origin"][2] + (z + 0.5) * 0.3,
                    R=float(scene.rgb[s][idx, 0]),
                    G=float(scene.rgb[s][idx, 1]),
                    B=float(scene.rgb[s][idx, 2]),
                    raw_gt=int(scene.raw[s][idx]),
                    gt="Ground",
                    pred_A=CLASSES[preds["A"][s][idx]],
                    pred_B=CLASSES[preds["B"][s][idx]],
                )
            )
    fig.legend(
        handles=[Patch(facecolor=c, label=l) for c, l in zip(colors, CLASSES)],
        loc="outside lower center",
        ncol=6,
        fontsize=8,
    )
    fig.savefig(out / "wmsc_examples.png", dpi=160)
    plt.close(fig)
    write_csv(out / "example_voxels.csv", rows)
    write_csv(out / "example_patches.csv", patch_rows)


def analyze(out):
    protocol = json.loads((out / "protocol.json").read_text())
    external = json.loads((out / "external.json").read_text())
    scenes = {role: Scene(CACHE / name) for role, name in ROLES.items()}
    scenes["WMSC"] = Scene(CACHE / "WMSC_points")
    inventory = []
    for role, s in scenes.items():
        hist = np.bincount(common(s.raw), minlength=6)
        assert hist.tolist() == s.meta["common_hist"]
        for c, n in zip(CLASSES, hist):
            inventory.append(
                dict(
                    split=role,
                    scene=Path(s.meta["source"]).stem,
                    patches=len(s),
                    occupied=len(s.raw),
                    class_name=c,
                    voxels=int(n),
                    share=float(n / len(s.raw)),
                )
            )
    write_csv(out / "scene_class_distribution.csv", inventory)
    train = scenes["central"]
    train_y = common(train.raw)
    keys = rgb_bins(train.rgb)
    g, b = (np.bincount(keys[train_y == cl], minlength=4096) for cl in (0, 1))
    g_prob, b_prob = g / g.sum(), b / b.sum()
    overlap = float(np.minimum(g_prob, b_prob).sum())
    summaries = []
    for role in ("central", "validation", "WMSC"):
        s = scenes[role]
        labels, cols = values(s)
        for i, c in enumerate(CLASSES):
            summaries.append(describe(cols, labels == i, role, c))
    s = scenes["WMSC"]
    labels, cols = values(s)
    preds = {}
    for k in external["models"]:
        with np.load(out / f"wmsc_predictions_{k}.npz", allow_pickle=False) as record:
            preds[k] = record["pred"]
    rgb = rgb_bins(s.rgb)
    ground_rgb = np.bincount(rgb[labels == 0], minlength=4096)
    ground_prob = ground_rgb / ground_rgb.sum()
    domain_overlap = {
        k: float(np.minimum(ground_prob, p).sum()) for k, p in (("ground", g_prob), ("building", b_prob))
    }
    error_rows = []
    brightness = np.floor(
        (s.rgb[:, 0] * 0.2126 + s.rgb[:, 1] * 0.7152 + s.rgb[:, 2] * 0.0722) / 64
    ).astype(int)
    color_groups = {
        "train_ground_more_likely": g_prob[rgb] > b_prob[rgb],
        "train_building_more_likely": g_prob[rgb] < b_prob[rgb],
        "train_unseen_in_both": (g[rgb] + b[rgb]) == 0,
        "equal_seen": (g_prob[rgb] == b_prob[rgb]) & (g[rgb] > 0),
    }
    height_cells = np.rint(cols["z_patch_m"] / 0.3).astype(np.int32)
    masks = dict(
        raw_ground_material={str(i): s.raw == i for i in (0, 15, 18, 19)},
        train_RGB_prior_equal=color_groups,
        brightness={str(i): brightness == i for i in range(4)},
        z_above_patch_min_m={
            "0–0.9": height_cells < 3,
            "0.9–3": (height_cells >= 3) & (height_cells < 10),
            "3+": height_cells >= 10,
        },
    )
    for name, pred in preds.items():
        np.testing.assert_array_equal(
            np.bincount(6 * labels.astype(np.int64) + pred, minlength=36).reshape(6, 6),
            external["models"][name]["confusion"],
        )
        for cl in (0, 1):
            for guessed in (0, 1):
                summaries.append(
                    describe(
                        cols,
                        (labels == cl) & (pred == guessed),
                        name,
                        CLASSES[cl] + "→" + CLASSES[guessed],
                    )
                )
        for attribute, groups in masks.items():
            for group, mask in groups.items():
                valid = mask & (labels == 0)
                n = int(valid.sum())
                wrong = int((valid & (pred == 1)).sum())
                error_rows.append(
                    dict(
                        model=name,
                        attribute=attribute,
                        group=group,
                        ground=n,
                        ground_to_building=wrong,
                        rate=wrong / n if n else None,
                    )
                )
    write_csv(out / "rgb_height_statistics.csv", summaries)
    write_csv(out / "ground_error_rates.csv", error_rows)
    cm_rows = []
    metric_rows = []
    class_rows = []
    for name in preds:
        val = json.loads((out / f"validation_{name}.json").read_text())
        for split, score in (("validation", val["score"]), ("WMSC", external["models"][name])):
            metric_rows.append(
                dict(
                    model=name,
                    split=split,
                    **{key: score[key] for key in ("accuracy", "macro_f1", "miou", "valid_voxels")},
                )
            )
            cm = np.asarray(score["confusion"])
            for i, c in enumerate(CLASSES):
                cm_rows.append(
                    dict(model=name, split=split, ground_truth=c, **dict(zip(CLASSES, cm[i])))
                )
                class_rows.append(
                    dict(
                        model=name,
                        split=split,
                        class_name=c,
                        support=int(cm[i].sum()),
                        predicted=int(cm[:, i].sum()),
                        iou=score["iou"][i],
                    )
                )
    write_csv(out / "metrics.csv", metric_rows)
    write_csv(out / "class_iou.csv", class_rows)
    write_csv(out / "confusion.csv", cm_rows)
    spatial_examples(out, s, labels, preds)
    ground = labels == 0
    transitions = np.bincount(
        6 * preds["A"][ground].astype(np.int64) + preds["B"][ground], minlength=36
    ).reshape(6, 6)
    write_csv(
        out / "ground_transitions.csv",
        [dict(prediction_A=c, **dict(zip(CLASSES, transitions[i]))) for i, c in enumerate(CLASSES)],
    )
    paired = {}
    for i, c in enumerate(CLASSES):
        part = labels == i
        paired[c] = dict(
            both=int((part & (preds["A"] == i) & (preds["B"] == i)).sum()),
            a_only=int((part & (preds["A"] == i) & (preds["B"] != i)).sum()),
            b_only=int((part & (preds["A"] != i) & (preds["B"] == i)).sum()),
        )
    for name in preds:
        assert int(
            next(r for r in metric_rows if r["split"] == "WMSC" and r["model"] == name)["valid_voxels"]
        ) == len(s.raw)
    atomic_json(
        out / "paired.json",
        dict(
            classes=protocol["classes"],
            both_domains="same WMSC voxels, A vs B",
            paired_correct=paired,
            ground_transitions=transitions.tolist(),
            train_ground_building_rgb_overlap=overlap,
            wmsc_ground_rgb_hist_intersection_with_train=domain_overlap,
        ),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    analyze(ap.parse_args().out)
