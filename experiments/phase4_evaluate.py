"""Evaluate prespecified endpoints only after all FL decisions have been frozen."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from datasets.compact_cache import atomic_json, sha
from datasets.fullz import Scene
from experiments.distributed_ai import CACHE, SCENES, batchify, init
from experiments.sparse_reproduction import evaluate
from experiments.phase4_federated import MODES, POLICIES, folder, moments, result_summary
from experiments.phase4_pretrain import OUT, SEEDS, artifact_hashes, start_path, freeze
from federated.uav_runtime import early_update
from models.active_voxel import SparseUNet


def choices(values):
    best = values[0]
    best_round = 0
    for r, x in enumerate(values[1:], 1):
        best, best_round, stop = early_update(best, best_round, x, r, 3)
        if stop:
            break
    return best_round, r, int(np.argmax(values))


def load_snap(dest, r):
    return torch.load(dest / f"r{r:02}.pt", map_location="cpu", weights_only=True)


def main():
    init()
    manifest = json.loads((OUT / "training_complete.json").read_text())
    assert all(sha(Path(path)) == digest for path, digest in manifest["checkpoint_hashes"].items())
    folds = json.loads((OUT / "splits.json").read_text())
    # No audit score has been used in selecting any of the following rounds.
    selected = {}
    for f in folds:
        for seed in SEEDS[f]:
            for mode in MODES:
                for bn in (*POLICIES, "recalibrated"):
                    source = "aggregate" if bn == "recalibrated" else bn
                    run = json.loads((folder(f, seed, source, mode) / "done.json").read_text())
                    metric = "recal_miou" if bn == "recalibrated" else "miou"
                    vals = [row[metric] for row in run["rows"]]
                    selected[f"{f}/{seed}/{bn}/{mode}"] = dict(
                        zip(("deployed", "stop", "best"), choices(vals))
                    )
    freeze(OUT / "selected_rounds.json", selected)
    ind = []
    ablation = []
    early = []
    trajectories = []
    support_checks = []
    swaps = []
    net = SparseUNet().cuda()
    for f, roles in folds.items():
        audit = batchify(
            roles["audit"],
            {i: Scene(CACHE / s) for i, s in SCENES.items() if i == roles["audit"][0][0]},
        )
        for seed in SEEDS[f]:
            initial = torch.load(start_path(f, seed), map_location="cpu", weights_only=True)["state"]
            startmeta = json.loads(start_path(f, seed).with_name("done.json").read_text())
            for mode in MODES:
                for bn in (*POLICIES, "recalibrated"):
                    info = dict(fold=f, seed=seed, mode=mode, bn=bn)
                    src = "aggregate" if bn == "recalibrated" else bn
                    dest = folder(f, seed, src, mode)
                    run = json.loads((dest / "done.json").read_text())
                    rows = run["rows"]
                    assert run["start_sha"] == startmeta["start_sha"] and len(rows) == 31
                    metric = "recal_miou" if bn == "recalibrated" else "miou"
                    d = selected[f"{f}/{seed}/{bn}/{mode}"]
                    stop, deploy, best = d["stop"], d["deployed"], d["best"]
                    estimates = {}
                    for r in {0, 10, 30, stop, deploy, best}:
                        ck = load_snap(dest, r)
                        state = {**ck["state"], **(ck["recal_moments"] if bn == "recalibrated" else {})}
                        if bn == "aggregate":
                            net.load_state_dict(state)
                            scored = evaluate(net, audit)
                            estimates[r] = scored["miou"]
                            ious = np.array(scored["iou"], dtype=float)
                            conf = np.array(scored["confusion"])
                            assert (conf.sum(1)[:5] > 0).all()
                            support_checks.append(
                                dict(
                                    **info,
                                    round=r,
                                    miou=scored["miou"],
                                    five_class_miou=ious[:5].mean(),
                                    classes_in_union=int(np.isfinite(ious).sum()),
                                    true_fence=int(conf.sum(1)[5]),
                                    predicted_fence=int(conf.sum(0)[5]),
                                )
                            )
                        else:
                            estimates[r] = result_summary(net, state, audit, ck["private"])[0]
                    ck = load_snap(dest, 30)
                    reset = {k: v for k, v in initial.items() if moments(k)}
                    st = ck["state"]
                    priv = ck["private"]
                    if priv:
                        priv = [{**p, **reset} for p in priv]
                    same_initial = result_summary(net, {**st, **reset}, audit, priv)[0]
                    full = rows[-1]["cumulative_bytes"] + run["closing_bytes"]
                    spent = rows[stop]["cumulative_bytes"] + run["closing_bytes"]
                    summary = dict(
                        **info,
                        start_sha=startmeta["start_sha"],
                        init_sha=startmeta["init_sha"],
                        pretrained_sha=startmeta["pretrained_sha"],
                        pretrain_patches=startmeta["patches"],
                        val_r0=rows[0][metric],
                        val_r30=rows[30][metric],
                        audit_r0=estimates[0],
                        audit_r30=estimates[30],
                        audit_r30_same_parameters_initial_stats=same_initial,
                        audit_r30_moment_effect_pp=100 * (estimates[30] - same_initial),
                        full_session_bytes=full,
                        model_frame_bytes=sum(r["model_bytes"] for r in rows),
                        validation_message_bytes=sum(r["evaluation_bytes"] for r in rows),
                        valid_patch_visits=30 * 192,
                        optimizer_steps=30 * 8,
                        calibration_forward_batches=31 * 3 * 4 if bn == "recalibrated" else 0,
                    )
                    ablation.append(summary)
                    if bn == "aggregate":
                        ind.append(summary)
                    early.append(
                        dict(
                            **info,
                            best_round=best,
                            best_val=rows[best][metric],
                            best_audit=estimates[best],
                            stop_round=stop,
                            restored_round=deploy,
                            stop_val=rows[stop][metric],
                            restored_val=rows[deploy][metric],
                            restored_audit=estimates[deploy],
                            audit_at_stop_before_restore=estimates[stop],
                            fixed10_audit=estimates[10],
                            r30_val=rows[30][metric],
                            r30_audit=estimates[30],
                            r0_audit=estimates[0],
                            delta_vs_r30_pp=100 * (estimates[deploy] - estimates[30]),
                            delta_vs_r0_pp=100 * (estimates[deploy] - estimates[0]),
                            retained_vs_r30=int(estimates[deploy] >= estimates[30] - 0.005),
                            retained_vs_r0=int(estimates[deploy] >= estimates[0] - 0.005),
                            early_session_bytes=spent,
                            full_session_bytes=full,
                            saved_pct=100 * (1 - spent / full),
                            bytes_to_best_model=rows[best]["cumulative_bytes"],
                            bytes_to_confirm_hindsight_best=full,
                        )
                    )
                    if bn != "recalibrated":
                        for row in rows:
                            trajectories.append(dict(**info, **row))
            # Cross only the running moments at fixed trainable parameters. No additional fitting.
            states = {m: load_snap(folder(f, seed, "aggregate", m), 30)["state"] for m in MODES}
            for weights, a in states.items():
                for stats, b in states.items():
                    st = {k: b[k] if moments(k) else v for k, v in a.items()}
                    swaps.append(
                        dict(
                            fold=f,
                            seed=seed,
                            parameters=weights,
                            moments=stats,
                            audit_miou=result_summary(net, st, audit)[0],
                        )
                    )
    for name, records in (
        ("independent_runs", ind),
        ("bn_ablation", ablation),
        ("early_stopping_generalization", early),
        ("update_scaling", trajectories),
        ("bn_crossed", swaps),
        ("class_support_sensitivity", support_checks),
    ):
        pd.DataFrame(records).to_csv(OUT / f"{name}.csv", index=False)
    assert artifact_hashes() == json.loads((OUT / "prior_hashes.json").read_text())
    atomic_json(
        OUT / "verification.json",
        dict(
            independent_starts=len({r["start_sha"] for r in ind}),
            unique_initializations=len({r["init_sha"] for r in ind}),
            analysis_cells=len(early),
            checkpoint_hashes_verified=len(manifest["checkpoint_hashes"]),
            historical_files_verified=len(artifact_hashes()),
            archive_replayed=True,
        ),
    )


if __name__ == "__main__":
    main()
