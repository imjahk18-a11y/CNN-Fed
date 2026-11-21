"""Summaries of real transport, retrospective validation and fixed-budget comparisons."""
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from datasets.compact_cache import sha
from experiments.distributed_ai_phase2 import OLD, OUT, PROFILES, MODES, SEEDS, SIZES, config_name
from experiments.distributed_ai import startfile
from federated.communication_codec import encode, decode

GROUP = ["pruning", "codec"]
BUDGETS = [1_000_000, 1_800_000, 3_000_000]
COLORS = {0: "#234f78", 20: "#008369", 40: "#b35d1d"}


def save(name, rows):
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    df.to_csv(OUT / name, index=False)
    return df


def intervals(rounds):
    groups = []
    for r in rounds:
        if not groups or r > groups[-1][-1] + 1:
            groups.append([])
        groups[-1].append(int(r))
    return ", ".join(str(g[0]) if len(g) == 1 else f"{g[0]}–{g[-1]}" for g in groups) or "없음"


def trace_metrics(scores):
    arr = np.asarray(scores)
    delta = np.diff(arr)
    passed = arr[1:] >= arr[0] + 0.01
    sustained = np.flatnonzero(passed[2:] & passed[1:-1] & passed[:-2]) + 3
    return dict(
        best_round=int(arr.argmax()),
        best_miou=arr.max(),
        best_to_end_pp=100 * (arr[-1] - arr.max()),
        below_initial=int((arr[1:] < arr[0]).sum()),
        falling_rounds=intervals(np.flatnonzero(delta < 0) + 1),
        down_steps=int((delta < 0).sum()),
        step_rms_pp=100 * np.sqrt(np.mean(delta**2)),
        max_drawdown_pp=100 * (np.maximum.accumulate(arr) - arr).max(),
        target_confirm_round=int(sustained[0]) if len(sustained) else np.nan,
    )


def phase1():
    trajectories = pd.concat(
        [pd.read_csv(OLD / f) for f in ["federated_rounds.csv", "control_rounds.csv"]]
    )
    seeds, assets = [], []
    for run, g in trajectories.groupby("run"):
        g = g.sort_values("round")
        assert list(g["round"]) == list(range(1, 31))
        r = g.iloc[0]
        scores = [r.initial_miou, *g.global_miou]
        info = dict(
            run=run,
            method=r.method,
            pruning=r.pruning_pct,
            partition=r.partition,
            local_epochs=r.local_epochs,
            seed=r.seed,
        )
        seeds.append(
            dict(**info, **{f"r{i}": scores[i] for i in (0, 10, 20, 30)}, **trace_metrics(scores))
        )
        for i in (0, 10, 20, 30):
            assets.append(
                dict(**info, round=i, miou=scores[i], change_pp=100 * (scores[i] - scores[0]))
            )
    save("prior_runs.csv", seeds)
    d = save("prior_endpoints.csv", assets)
    return save(
        "prior_summary.csv",
        d.groupby(["method", "pruning", "partition", "local_epochs", "round"])
        .miou.agg(["mean", "std"])
        .reset_index(),
    )


def collect():
    all_rows, results, net, budgets = [], [], [], []
    for seed, pct, mode, profile in itertools.product(SEEDS, SIZES, MODES, PROFILES):
        ident = config_name(seed, pct, mode, profile)
        part = OUT / "runs" / ident
        run = json.loads((part / "done.json").read_text())
        n = 30 if profile == "loopback" else 5
        assert run["config"] == [seed, pct, mode, profile]
        rs = run["rows"]
        meta = dict(seed=seed, pruning=pct, codec=mode, profile=profile)
        initial = run["initial"]["miou"]
        zero = dict(
            **meta,
            round=0,
            miou=initial,
            macro_f1=run["initial"]["macro_f1"],
            delta=0.0,
            cumulative_bytes=0,
            total_bytes=0,
            elapsed_s=0.0,
        )
        all_rows.append(zero)
        assert [r["round"] for r in rs] == list(range(1, n + 1))
        cumulative = 0
        for r in rs:
            assert r["up_bytes"] + r["down_bytes"] + r["handshake_bytes"] == r["total_bytes"]
            cumulative += r["total_bytes"]
            assert r["cumulative_bytes"] == cumulative
            assert max(r["down_max_step_fraction"], r["up_max_step_fraction"]) <= 0.50002
            all_rows.append(dict(**meta, **r))
        if profile == "loopback":
            trace = trace_metrics([initial] + [r["miou"] for r in rs])
            target = trace["target_confirm_round"]
            results.append(
                dict(
                    **meta,
                    **trace,
                    **{f"r{i}": rs[i - 1]["miou"] for i in (10, 20, 30)},
                    r0=initial,
                    delta30_pp=rs[-1]["delta"] * 100,
                    bytes=cumulative,
                    elapsed_s=run["elapsed_s"],
                    target_bytes=np.nan if pd.isna(target) else rs[int(target) - 1]["cumulative_bytes"],
                    peak_mib=max(max(r["client1_peak"], r["client2_peak"]) for r in rs),
                )
            )
            for budget in BUDGETS:
                r = ([zero] + [r for r in rs if r["cumulative_bytes"] <= budget])[-1]
                budgets.append(
                    dict(
                        **meta,
                        budget_bytes=budget,
                        round=r["round"],
                        used_bytes=r["cumulative_bytes"],
                        miou=r["miou"],
                        delta_pp=r["delta"] * 100,
                    )
                )
        first = rs[:5]
        net.append(
            dict(
                **meta,
                rounds=5,
                miou=first[-1]["miou"],
                time_s=first[-1]["elapsed_s"],
                bytes=first[-1]["cumulative_bytes"],
                up_bytes=sum(r["up_bytes"] for r in first),
                down_bytes=sum(r["down_bytes"] for r in first),
                retries=sum(r["retransmissions"] for r in first),
                retry_bytes=sum(r["retry_frame_bytes"] for r in first),
                failures=sum(r["failures"] for r in first),
                recovered=sum(r["recovered"] for r in first),
                client1_proc_s=sum(r["client1_proc"] for r in first),
                client2_proc_s=sum(r["client2_proc"] for r in first),
                client1_train_s=sum(r["client1_train"] for r in first),
                client2_train_s=sum(r["client2_train"] for r in first),
                peak_mib=max(max(r["client1_peak"], r["client2_peak"]) for r in first),
            )
        )
    df = save("rounds.csv", all_rows)
    clean = save("seed_results.csv", results)
    network = pd.DataFrame(net)
    keys = ["seed", "pruning", "codec"]
    ref = network[network.profile == "loopback"].set_index(keys)
    network = network.join(ref[["time_s", "miou", "bytes"]].add_prefix("clean_"), on=keys)
    network["time_delta_s"] = network.time_s - network.clean_time_s
    assert np.array_equal(network.miou, network.clean_miou)
    for _, g in df[df["round"] <= 5].groupby(keys + ["round"]):
        assert len(g) == 5 and g.miou.nunique() == 1
    # All four network profiles must end with identical actual model states for the same cell/seed.
    for values in itertools.product(SEEDS, SIZES, MODES):
        ref_state = None
        for prof in PROFILES[1:]:
            path = OUT / "runs" / config_name(*values, prof) / "final.pt"
            state = torch.load(path, map_location="cpu", weights_only=True)["state"]
            if ref_state is not None:
                assert all(torch.equal(v, ref_state[k]) for k, v in state.items())
            ref_state = state
    assert network.failures.sum() == network.recovered.sum() == network.retries.sum() == 18
    save("network_five_rounds.csv", network)
    return df, clean, network, save("equal_budget.csv", budgets)


def aggregate(df, by, metrics):
    summary = df.groupby(by)[metrics].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    return summary.reset_index()


def summaries(rows, clean, network, budgets):
    metrics = [
        "r0",
        "r10",
        "r20",
        "r30",
        "delta30_pp",
        "bytes",
        "elapsed_s",
        "peak_mib",
        "max_drawdown_pp",
        "step_rms_pp",
        "target_confirm_round",
        "target_bytes",
    ]
    sums = aggregate(clean, GROUP, metrics)
    sums["targets_of_3"] = clean.groupby(GROUP).target_confirm_round.count().values
    base = sums[(sums.pruning == 0) & (sums.codec == "FP32")].bytes_mean.item()
    sums["reduction_vs_original_pct"] = 100 * (1 - sums.bytes_mean / base)
    paired = []
    for pct, g in clean.groupby("pruning"):
        a = g[g.codec == "FP32"].set_index("seed").sort_index()
        b = g[g.codec == "INT8"].set_index("seed").sort_index()
        for seed in SEEDS:
            paired.append(
                dict(
                    pruning=pct,
                    seed=seed,
                    r30_delta_pp=100 * (b.loc[seed, "r30"] - a.loc[seed, "r30"]),
                    reduction_pct=100 * (1 - b.loc[seed, "bytes"] / a.loc[seed, "bytes"]),
                    time_delta_s=b.loc[seed, "elapsed_s"] - a.loc[seed, "elapsed_s"],
                )
            )
    save("int8_paired.csv", paired)
    save(
        "network_summary.csv",
        aggregate(
            network,
            GROUP + ["profile"],
            [
                "time_s",
                "time_delta_s",
                "bytes",
                "client1_proc_s",
                "client2_proc_s",
                "peak_mib",
                "retries",
                "recovered",
            ],
        ),
    )
    save(
        "equal_budget_summary.csv",
        aggregate(budgets, GROUP + ["budget_bytes"], ["round", "used_bytes", "miou", "delta_pp"]),
    )
    return save("summary.csv", sums)


def profiles():
    old = pd.read_csv(OLD / "model_profiles.csv").set_index("pruning_pct")
    records = []
    for pct, mode, seed in itertools.product(SIZES, MODES, SEEDS):
        f = startfile(seed, pct)
        state = torch.load(f, weights_only=True, map_location="cpu")["state"]
        raw, (absolute, units) = encode(state, mode)
        copy = decode(raw, mode, state)
        error = torch.cat([(v - copy[k]).flatten() for k, v in state.items() if v.is_floating_point()])
        ref = torch.cat([v.flatten() for v in state.values() if v.is_floating_point()])
        records.append(
            dict(
                pruning=pct,
                codec=mode,
                seed=seed,
                params=old.loc[pct, "parameters"],
                checkpoint_bytes=f.stat().st_size,
                state_bytes=old.loc[pct, "state_bytes"],
                serialized_bytes=len(raw),
                max_absolute_error=absolute,
                max_quant_step_fraction=units,
                rmse=error.square().mean().sqrt().item(),
                relative_l2=(error.norm() / ref.norm()).item(),
                start_sha256=sha(f),
            )
        )
    return save("model_codec_profiles.csv", records)


def curves(df):
    clean = df[df.profile == "loopback"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained", sharey=True)
    for (pct, mode), d in clean.groupby(GROUP):
        stats = d.groupby("round").agg(
            y=("miou", "mean"), sd=("miou", "std"), b=("cumulative_bytes", "mean")
        )
        for ax, x in zip(axes, [stats.index, stats.b / 1e6]):
            ax.plot(
                x,
                100 * stats.y,
                color=COLORS[pct],
                ls="-" if mode == "FP32" else "--",
                label=f"{pct}% / {mode}",
            )
            ax.fill_between(
                x, 100 * (stats.y - stats.sd), 100 * (stats.y + stats.sd), color=COLORS[pct], alpha=0.07
            )
    axes[0].set(xlabel="Committed rounds", ylabel="Synthetic validation mIoU (%)")
    axes[1].set_xlabel("Measured cumulative application traffic (MB)")
    axes[1].legend(fontsize=8, ncol=2)
    fig.savefig(OUT / "communication_convergence.png", dpi=165)
    plt.close(fig)


def main():
    prior = phase1()
    df, clean, network, budgets = collect()
    summary = summaries(df, clean, network, budgets)
    prof = profiles()
    curves(df)
    from experiments.communication_report import write_report

    write_report(prior, df, clean, network, budgets, summary, prof)


if __name__ == "__main__":
    main()
