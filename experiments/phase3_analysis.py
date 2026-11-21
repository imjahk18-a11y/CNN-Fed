"""Reconstruct locally recorded update vectors and compare with committed server states."""
import json
import itertools
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from datasets.compact_cache import sha
from experiments.distributed_ai import SEEDS, startfile
from experiments.distributed_ai_phase2 import OUT as P2
from experiments.distributed_ai_phase3 import OUT, tp, run_dir
from federated.uav_runtime import (
    MODES,
    bundle,
    device_setup,
    early_update,
    group,
    rounding_seed,
    score,
    vector,
    wire_decode,
    wire_encode,
)
from federated.fedavg import fedavg
from models.active_voxel import SparseUNet

GROUPS = ("parameters", "bn", "all_float")
COLORS = {
    "FP32": "#214c73",
    "FP16": "#d48028",
    "INT8": "#158172",
    "FP32_MATCH": "#9c4b64",
    "INT8_SR": "#7761ad",
}


def cos(a, b):
    den = a.norm() * b.norm()
    return float(a.dot(b) / den) if den else np.nan


def unchanged_bin_fraction(down, local, section):
    same = changed = 0
    for k,t in local.items():
        if not t.is_floating_point() or (section != 'all_float' and group(k) != section):
            continue
        mask = down[k] != t
        scale = float(t.abs().max()) / 127 or 1.0
        q0 = (down[k] / scale).round().clamp(-127,127)
        q1 = (t / scale).round().clamp(-127,127)
        same += int(((q0 == q1) & mask).sum())
        changed += int(mask.sum())
    return same/changed if changed else np.nan


def flatten(state, section):
    if section == "all_float":
        return torch.cat([vector(state, "parameters"), vector(state, "bn")])
    return vector(state, section)


def state_after_wire(state, mode, seed, r, side):
    return wire_decode(wire_encode(state, mode, rounding_seed(seed, r, side)), mode, state)


def analyze_run(seed, mode):
    root = run_dir(seed, mode)
    meta = json.loads((root / "done.json").read_text())
    rows = meta["rows"]
    assert len(rows) == 31
    state = tp(startfile(seed, 0))
    initial = state
    if mode in ("FP32", "INT8"):
        previous = P2 / "runs" / f"s{seed}_p00_{mode}_0"
        old = json.loads((previous / "done.json").read_text())
        assert [r["miou"] for r in rows[1:]] == [r["miou"] for r in old["rows"]]
        prior = tp(previous / "final.pt")
        assert all(torch.equal(prior[k], v) for k, v in tp(root / "server.pt").items())
    info = dict(seed=seed, mode=mode)
    stats = [dict(**info, section=g, **rows[0], drift_from_start_l2=0) for g in GROUPS]
    for row in rows[1:]:
        r = row["round"]
        down = state_after_wire(state, mode, seed, r, 0)
        trained = [tp(root / f"client{i}" / f"local_{r:02}.pt") for i in (1, 2)]
        up = [state_after_wire(t, mode, seed, r, i) for i, t in enumerate(trained, 1)]
        actual = tp(root / f"global_{r:02}.pt")
        agg = fedavg(up, [row["client1_patches"], row["client2_patches"]])
        if mode != "FP32_MATCH":
            assert all(torch.equal(v, actual[k]) for k, v in agg.items())
        for g in GROUPS:
            base, received, next_global, uncorrected = (
                flatten(s, g) for s in (state, down, actual, agg)
            )
            a = [flatten(t, g) for t in trained]
            b = [flatten(t, g) for t in up]
            opt = [t - received for t in a]
            restored = [t - received for t in b]
            effective = [t - base for t in b]
            update = next_global - base
            # The same global was broadcast to both clients; weights are 96:96.
            assert row["client1_patches"] == row["client2_patches"] == 96
            local_avg = (opt[0] + opt[1]) / 2
            out = dict(
                **info,
                section=g,
                **row,
                global_l2=float(update.norm()),
                unscaled_global_l2=float((uncorrected - base).norm()),
                global_vs_local_mean_cos=cos(update, local_avg),
                down_error_l2=float((received - base).norm()),
                mean_local_l2=float(local_avg.norm()),
                pre_client_cos=cos(*opt),
                post_client_cos=cos(*restored),
                effective_client_cos=cos(*effective),
                drift_from_start_l2=float((next_global - flatten(initial, g)).norm()),
                global_minus_localmean_l2=float((update - local_avg).norm()),
                uplink_mean_error_l2=float(((b[0] - a[0] + b[1] - a[1]) / 2).norm()),
            )
            for idx, (u, v, w, t, z) in enumerate(zip(opt, restored, effective, a, b), 1):
                if mode == 'INT8':
                    out[f'client{idx}_same_uplink_bin_fraction'] = unchanged_bin_fraction(down,trained[idx-1],g)
                changed = u != 0
                out.update(
                    {
                        f"client{idx}_local_l2": float(u.norm()),
                        f"client{idx}_decoded_local_l2": float(v.norm()),
                        f"client{idx}_effective_l2": float(w.norm()),
                        f"client{idx}_uplink_error_l2": float((z - t).norm()),
                        f"client{idx}_uplink_error_over_local": float((z - t).norm() / u.norm())
                        if u.norm()
                        else np.nan,
                        f"client{idx}_pre_post_cos": cos(u, v),
                        f"client{idx}_erased_fraction": float((v[changed] == 0).double().mean())
                        if changed.any()
                        else np.nan,
                    }
                )
            stats.append(out)
        state = actual
    return stats


def policies(data):
    out = []
    for (seed, mode), d in data[data.section == "parameters"].groupby(["seed", "mode"]):
        d = d.sort_values("round").set_index("round")
        best = d.miou.iloc[0]
        br = 0
        stop = 30
        for r in range(1, 31):
            best, br, hit = early_update(best, br, d.loc[r, "miou"], r, 3)
            if hit:
                stop = r
                break
        choices = {
            "0R": (0, 0),
            "best_hindsight": (int(d.miou.idxmax()), int(d.miou.idxmax())),
            "10R": (10, 10),
            "20R": (20, 20),
            "30R": (30, 30),
            "online_patience3": (stop, br),
        }
        for name, (r, b) in choices.items():
            out.append(
                dict(
                    seed=seed,
                    mode=mode,
                    policy=name,
                    stop_round=r,
                    model_round=b,
                    deployed_miou=d.loc[b, "miou"],
                    observed_stop_miou=d.loc[r, "miou"],
                    bytes_to_model=d.loc[b, "cumulative_bytes"],
                    bytes_spent=d.loc[r, "cumulative_bytes"],
                    bytes_to_identify=(
                        d.loc[30, "cumulative_bytes"]
                        if name == "best_hindsight"
                        else d.loc[r, "cumulative_bytes"]
                    ),
                )
            )
    return pd.DataFrame(out)


def summarize(data):
    out = []
    for (seed, mode), d in data.groupby(["seed", "mode"]):
        p = d[d.section == "parameters"].sort_values("round").set_index("round")
        bn = d[d.section == "bn"].sort_values("round").set_index("round")
        late = p.loc[2:30]
        ref = data[
            (data.seed == seed) & (data["mode"] == "INT8") & (data.section == "parameters")
        ].sort_values("round")
        out.append(
            dict(
                seed=seed,
                mode=mode,
                r0=p.miou[0],
                r30=p.miou[30],
                delta_pp=100 * (p.miou[30] - p.miou[0]),
                total_MB=p.cumulative_bytes[30] / 1e6,
                r1_global_l2=p.global_l2[1],
                r2_30_global_l2=late.global_l2.mean(),
                r2_30_bn_l2=bn.loc[2:30, "global_l2"].mean(),
                r2_30_global_vs_local_cos=late.global_vs_local_mean_cos.mean(),
                r2_30_local_l2=late[["client1_local_l2", "client2_local_l2"]].to_numpy().mean(),
                r2_30_erased=late[["client1_erased_fraction", "client2_erased_fraction"]]
                .to_numpy()
                .mean(),
                r2_30_pre_client_cos=late.pre_client_cos.mean(),
                r2_30_post_client_cos=late.post_client_cos.mean(),
                r2_30_uplink_error_ratio=late[
                    ["client1_uplink_error_over_local", "client2_uplink_error_over_local"]
                ]
                .to_numpy()
                .mean(),
                final_parameter_drift=p.drift_from_start_l2[30],
                final_bn_drift=bn.drift_from_start_l2[30],
                max_norm_gap_vs_int8=np.max(
                    np.abs(p.loc[1:30, "global_l2"].to_numpy() - ref.global_l2.to_numpy()[1:])
                ),
                mean_client1_loss=p.client1_loss.mean(),
                mean_client2_loss=p.client2_loss.mean(),
            )
        )
    return pd.DataFrame(out)


def swap_bn():
    device = device_setup("cuda:0")
    val, _ = bundle(OUT / "bundles/validation/patches.npz", "validation", device)
    net = SparseUNet().to(device)
    out = []
    for seed, r in itertools.product(SEEDS, (1, 10, 20, 30)):
        states = {m: tp(run_dir(seed, m) / f"global_{r:02}.pt") for m in ("FP32", "INT8")}
        for p, b in itertools.product(states, repeat=2):
            net.load_state_dict({k: states[b if group(k) == "bn" else p][k] for k in states[p]})
            value = score(net, val)
            if p == b:
                original = json.loads((run_dir(seed, p) / "done.json").read_text())["rows"][r]["miou"]
                assert value == original
            out.append(dict(seed=seed, round=r, parameters_from=p, bn_moments_from=b, miou=value))
    return pd.DataFrame(out)


def figures(d):
    d = d[d.section == "parameters"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    fig2, ax2 = plt.subplots(figsize=(7, 4), layout="constrained")
    for mode, g in d.groupby("mode"):
        s = g.groupby("round").agg(
            miou=("miou", "mean"),
            sd=("miou", "std"),
            l2=("global_l2", "mean"),
            bytes=("cumulative_bytes", "mean"),
        )
        for ax, x in ((axes[0], s.index), (ax2, s.bytes / 1e6)):
            ax.plot(
                x,
                100 * s.miou,
                label=mode,
                color=COLORS[mode],
                ls="--" if mode == "FP32_MATCH" else "-",
            )
            ax.fill_between(
                x, 100 * (s.miou - s.sd), 100 * (s.miou + s.sd), color=COLORS[mode], alpha=0.10
            )
        axes[1].plot(
            s.index[1:],
            s.l2.iloc[1:],
            label=mode,
            color=COLORS[mode],
            ls="--" if mode == "FP32_MATCH" else "-",
        )
    axes[0].set(xlabel="Completed rounds", ylabel="Validation mIoU (%)")
    axes[1].set(xlabel="Completed rounds", ylabel="Global trainable-parameter update L2", yscale="log")
    ax2.set(xlabel="Measured cumulative TCP application bytes (MB)", ylabel="Validation mIoU (%)")
    axes[0].legend(fontsize=8)
    ax2.legend(fontsize=8)
    fig.savefig(OUT / "model_comparison.png", dpi=170)
    fig2.savefig(OUT / "communication_accuracy.png", dpi=170)
    plt.close(fig)
    plt.close(fig2)


def main():
    torch.set_num_threads(2)
    d = pd.DataFrame(
        [row for seed, mode in itertools.product(SEEDS, MODES) for row in analyze_run(seed, mode)]
    )
    assert len(d) == 15 * 31 * 3
    early = policies(d)
    stats = summarize(d)
    live = json.loads((OUT / "early_smoke/done.json").read_text())
    target = early[
        (early.seed == 42) & (early["mode"] == "INT8") & (early.policy == "online_patience3")
    ].iloc[0]
    assert len(live["rows"]) - 1 == target.stop_round and live["best_round"] == target.model_round
    assert sum(r.get("retries", 0) for r in live["rows"]) == 1 and live["stopped"]
    assert max(r["miou"] for r in live["rows"]) == target.deployed_miou
    saved = tp(OUT / "early_smoke/best.pt")
    best = (
        tp(run_dir(42, "INT8") / f"global_{live['best_round']:02}.pt")
        if live["best_round"]
        else tp(startfile(42, 0))
    )
    assert all(torch.equal(v, saved[k]) for k, v in best.items())
    swapped = swap_bn()
    for name, df in (
        ("update_norms", d),
        ("early_stopping", early),
        ("quantization_analysis", stats),
        ("bn_swap", swapped),
    ):
        df.to_csv(OUT / f"{name}.csv", index=False)
    figures(d)


if __name__ == "__main__":
    main()
