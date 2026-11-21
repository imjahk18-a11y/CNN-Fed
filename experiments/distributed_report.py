"""Generate paired seed summaries, communication accounting, figures and the research report."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.distributed_ai import OUT, THRESHOLD, SEEDS

GROUP = ["method", "partition", "pruning_pct", "local_epochs", "rounds"]


def summarize():
    scores = json.loads((OUT / "endpoint_scores.json").read_text())
    frames = []
    results = []
    for path in sorted((OUT / "runs").glob("*/rounds.csv")):
        df = pd.read_csv(path)
        assert list(df["round"]) == list(range(1, 31))
        frames.append(df)
        for rounds in (10, 30):
            prefix = df.iloc[:rounds]
            last = prefix.iloc[-1]
            ext = scores[f"{last.run}_r{rounds}"]
            miou = np.r_[last.initial_miou, prefix.global_miou]
            hit = np.flatnonzero(miou >= THRESHOLD)
            result = dict(
                run=last.run,
                method=last.method,
                partition=last.partition,
                pruning_pct=last.pruning_pct,
                local_epochs=last.local_epochs,
                seed=last.seed,
                rounds=rounds,
                initial_miou=last.initial_miou,
                miou=last.global_miou,
                macro_f1=last.global_f1,
                accuracy=last.global_accuracy,
                synthetic_full_miou=ext["synthetic_full"]["miou"],
                synthetic_full_f1=ext["synthetic_full"]["macro_f1"],
                wmsc_miou=ext["wmsc"]["miou"],
                wmsc_f1=ext["wmsc"]["macro_f1"],
                delta_miou=last.global_miou - last.initial_miou,
                tail5_miou=prefix.tail(5).global_miou.mean(),
                round_change_rms=np.mean(np.diff(miou) ** 2) ** 0.5,
                client_val_gap=abs(prefix.client1_val_miou - prefix.client2_val_miou).mean(),
                total_bytes=prefix.total_bytes.sum(),
                upload_bytes=prefix.upload_bytes.sum(),
                download_bytes=prefix.download_bytes.sum(),
                train_round_seconds=prefix.round_s.sum(),
                peak_mib=prefix.peak_mib.max(),
                patch_visits=prefix.patch_visits_this_round.sum(),
                reached_target=int(bool(len(hit))),
                round_to_target=int(hit[0]) if len(hit) else np.nan,
                bytes_to_target=int(hit[0]) * last.total_bytes if len(hit) else np.nan,
            )
            results.append(result)
    rounds = pd.concat(frames, ignore_index=True)
    rounds[rounds.method.isin(["FedAvg", "FedBN", "FedProx"])].to_csv(
        OUT / "federated_rounds.csv", index=False
    )
    rounds[~rounds.method.isin(["FedAvg", "FedBN", "FedProx"])].to_csv(
        OUT / "control_rounds.csv", index=False
    )
    res = pd.DataFrame(results)
    res.to_csv(OUT / "experiment_results.csv", index=False)
    metrics = [
        "miou",
        "macro_f1",
        "synthetic_full_miou",
        "synthetic_full_f1",
        "wmsc_miou",
        "wmsc_f1",
        "delta_miou",
        "tail5_miou",
        "round_change_rms",
        "client_val_gap",
        "total_bytes",
        "round_to_target",
        "train_round_seconds",
        "peak_mib",
        "patch_visits",
    ]
    summary = res.groupby(GROUP)[metrics].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary["reached_n"] = res.groupby(GROUP).reached_target.sum()
    summary["seeds_n"] = res.groupby(GROUP).seed.nunique()
    assert (summary.seeds_n == 3).all()
    summary.reset_index().to_csv(OUT / "experiment_summary.csv", index=False)
    fed = res[res.method == "FedAvg"]
    non = []
    for key, df in fed.groupby(["pruning_pct", "local_epochs", "rounds"]):
        a = df[df.partition == "IID"].set_index("seed").sort_index()
        b = df[df.partition == "NonIID"].set_index("seed").sort_index()
        row = dict(zip(["pruning_pct", "local_epochs", "rounds"], key))
        for metric in [
            "miou",
            "macro_f1",
            "synthetic_full_miou",
            "wmsc_miou",
            "round_change_rms",
            "client_val_gap",
        ]:
            diff = b[metric] - a[metric]
            row.update(
                {
                    f"iid_{metric}": a[metric].mean(),
                    f"noniid_{metric}": b[metric].mean(),
                    f"paired_delta_{metric}": diff.mean(),
                    f"paired_sd_{metric}": diff.std(ddof=1),
                }
            )
        non.append(row)
    paired = pd.DataFrame(non)
    paired.to_csv(OUT / "noniid_comparison.csv", index=False)
    transport = pd.read_csv(OUT / "transport_rounds.csv")
    comm = []
    for (method, pct, n), df in res[res.method.isin(["FedAvg", "FedBN", "FedProx"])].groupby(
        ["method", "pruning_pct", "rounds"]
    ):
        assert df.total_bytes.nunique() == 1
        comm.append(
            dict(
                mode="simulated_state_tensor_bytes",
                method=method,
                pruning_pct=pct,
                rounds=n,
                upload_bytes=df.upload_bytes.iloc[0],
                download_bytes=df.download_bytes.iloc[0],
                duplicate_bytes=0,
                abort_bytes=0,
                handshake_bytes=0,
                committed_bytes=df.total_bytes.iloc[0],
                session_bytes=df.total_bytes.iloc[0],
            )
        )
    for pct, d in transport.groupby("pruning_pct"):
        abort = 0
        handshake = int(d.handshake_bytes.sum())
        if pct == 0:
            roll = json.loads((OUT / "transport/p0/rollback.json").read_text())
            abort = roll["upload_bytes"] + roll["download_bytes"]
        comm.append(
            dict(
                mode="tcp_application_actual",
                method="FedAvg",
                pruning_pct=pct,
                rounds=10,
                upload_bytes=d.upload_bytes.sum(),
                download_bytes=d.download_bytes.sum(),
                duplicate_bytes=d.duplicate_bytes.sum(),
                abort_bytes=abort,
                handshake_bytes=handshake,
                committed_bytes=d.total_bytes.sum(),
                session_bytes=d.total_bytes.sum() + abort + handshake,
            )
        )
    comm = pd.DataFrame(comm)
    comm["clean_model_frame_bytes"] = comm.committed_bytes - comm.duplicate_bytes
    comm["clean_reduction_pct"] = np.nan
    for (mode, n), d in comm.groupby(["mode", "rounds"]):
        base = d[(d.pruning_pct == 0) & (d.method == "FedAvg")].clean_model_frame_bytes.iloc[0]
        comm.loc[d.index, "clean_reduction_pct"] = 100 * (1 - d.clean_model_frame_bytes / base)
    comm.to_csv(OUT / "communication_cost.csv", index=False)
    return res, rounds, paired, comm, summary.reset_index()


def plots(res, rounds, comm):
    colors = {0: "#355d84", 20: "#2a826a", 40: "#c0743b"}
    fed = rounds[rounds.method == "FedAvg"]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True, sharey=True, layout="constrained")
    for ax, (reg, ep) in zip(axes.flat, [(r, e) for r in ("IID", "NonIID") for e in (1, 3)]):
        for pct, col in colors.items():
            d = fed[(fed.partition == reg) & (fed.local_epochs == ep) & (fed.pruning_pct == pct)]
            z = d.groupby("round").global_miou.agg(["mean", "std"])
            initial = d[d["round"] == 1].initial_miou
            x = np.arange(31)
            y = np.r_[initial.mean(), z["mean"]] * 100
            sd = np.r_[initial.std(), z["std"]] * 100
            ax.plot(x, y, label=f"{pct}% pruning", color=col)
            ax.fill_between(x, y - sd, y + sd, color=col, alpha=0.16)
        ax.axhline(100 * THRESHOLD, color=".55", ls=":", lw=1)
        ax.axvline(10, color=".76", ls="--", lw=0.8)
        ax.set_title(f"{reg} · {ep} local epoch{'s' if ep>1 else ''}")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.supxlabel("Communication round")
    fig.supylabel("Synthetic validation mIoU (%)")
    fig.savefig(OUT / "convergence.png", dpi=175)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.8, 3.6), layout="constrained")
    q = comm[(comm.method == "FedAvg") & (comm.rounds == 10)].sort_values("pruning_pct")
    x = np.arange(3)
    for off, k, l, col in [
        (-0.18, "simulated_state_tensor_bytes", "Tensor payload", "#355d84"),
        (0.18, "tcp_application_actual", "Measured TCP frames", "#bd7542"),
    ]:
        y = q[q["mode"] == k].clean_model_frame_bytes.to_numpy() / 1e6
        ax.bar(x + off, y, 0.36, color=col, label=l)
    ax.set(
        xticks=x,
        xticklabels=["Original", "20% pruning", "40% pruning"],
        ylabel="Upload + download over 10 rounds (MB)",
    )
    ax.legend(frameon=False)
    fig.savefig(OUT / "communication_comparison.png", dpi=175)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.9), sharey=True, layout="constrained")
    for ax, reg in zip(axes, ("IID", "NonIID")):
        for ep, mark in ((1, "o"), (3, "s")):
            d = res[
                (res.method == "FedAvg")
                & (res.partition == reg)
                & (res.local_epochs == ep)
                & (res.rounds == 30)
            ]
            for pct, col in colors.items():
                sub = d[d.pruning_pct == pct]
                x = sub.total_bytes.mean() / 1e6
                ax.errorbar(
                    x,
                    100 * sub.miou.mean(),
                    yerr=100 * sub.miou.std(),
                    color=col,
                    marker=mark,
                    ls="none",
                    capsize=3,
                    label=f"{pct}%, E{ep}",
                )
        ax.set(title=reg, xlabel="30-round tensor payload (MB)")
    axes[0].set_ylabel("Synthetic validation mIoU (%)")
    axes[1].legend(frameon=False, fontsize=7, ncol=2)
    fig.savefig(OUT / "pruning_tradeoff.png", dpi=175)
    plt.close(fig)


def report(res, rounds, paired, comm, summary):
    protocol = json.loads((OUT / "protocol.json").read_text())
    profiles = pd.read_csv(OUT / "model_profiles.csv")
    dist = pd.read_csv(OUT / "client_distributions.csv")
    transport = json.loads((OUT / "transport_checks.json").read_text())

    def table(header, rows):
        return "\n".join(
            [
                "|" + "|".join(header) + "|",
                "|" + "|".join(["---"] * len(header)) + "|",
                *["|" + "|".join(map(str, r)) + "|" for r in rows],
            ]
        )

    def pm(r, key, scale=100):
        return f"{scale*r[key+'_mean']:.2f} ± {scale*r[key+'_std']:.2f}"

    report = [
        "# 구조적 프루닝과 2-client 연합학습",
        "## 연구 질문과 조건",
        "두 클라이언트가 같은 폭의 경량 sparse 3D CNN을 교환할 때 채널 축소가 전송 바이트, 수렴과 Non-IID 안정성에 주는 영향을 측정했다. 주평가지표는 고정 Synthetic validation의 점유 복셀 mIoU다. 모든 ± 값은 같은 분할에서 실행한 세 시드(42·43·44)의 평균과 표본 표준편차다.",
        f"{protocol['gpu']} 한 장과 PyTorch {protocol['torch']}를 사용했다. RGB-16, 0.3 m Full-Z XY 32×32, 7점 submanifold convolution, encoder/pool/bridge/skip decoder, BN, 6개 클래스를 기존 구현에서 재사용했다. 이전 Synthetic 장면 13 전체 35 epoch 중 장면 12 검증으로 선택한 32 epoch 기준 체크포인트를 공통 출발점으로 삼았다. 이전 0·20·40% 체크포인트의 해시와 구조, 불균등 샘플 수 1:3에서 FedAvg의 수치 결과를 점검했다. 원본 모델의 seed 42, Non-IID, E1 smoke를 10라운드 완료한 뒤 본실험을 시작했다.",
        "프루닝은 각 블록의 출력 필터 L1 순위로 실제 채널을 제거하고 BN, 다음 입력, coarse/skip 인덱스와 head를 함께 줄였다. 같은 출발점에서 구조를 만든 뒤 세 모델 모두 장면 13의 동일한 192패치로 5 epoch 미세조정했다. 마지막 epoch를 사용했다. 이전 실험에서 각자 다른 epoch로 선택한 미세조정 모델을 새 비교에 그대로 쓰지 않았다. 시드는 미세조정·로컬 배치 순서에 적용된다. 사전학습 자체를 독립적으로 세 번 반복한 실험은 아니다.",
        table(
            ["요청 채널 축소", "실제 폭", "파라미터", "상태 텐서 bytes", "체크포인트 bytes", "검증 24패치 MACs"],
            [
                (
                    f"{r.pruning_pct}%",
                    r.widths,
                    f"{r.parameters:,}",
                    f"{r.state_bytes:,}",
                    f"{r.checkpoint_bytes:,}",
                    f"{r.fixed_batch_macs:,}",
                )
                for r in profiles.itertuples()
            ],
        ),
        "## 고정 데이터 분할",
        "하나의 GPU에서 전체 요인 조합을 비교할 수 있도록 FL은 장면 3·4에서 각각 96패치, 라운드 검증은 장면 12에서 144패치를 고정했다. 각 장면 안에서 지배 클래스 비율에 맞춰 시드 2026으로 추출했다. 원본 각 장면에는 2,704패치가 있다. 장면 13은 사전학습·미세조정 전용이고, 장면 12는 공통 검증 전용이다. 두 조건의 클라이언트 합집합과 패치 방문 수가 같고 한 조건 내 중복 패치는 0이다. 장면 12 전체 4,226,451복셀에 대한 별도 종점 점수도 집계했다. 장면 12는 종전 사전학습 모델 선택에도 사용되었다.",
        "Non-IID는 Drone 1에 장면 3, Drone 2에 장면 4를 배정했다. IID 근사는 두 클라이언트가 각 장면에서 48패치씩 받게 하고 고정 후보 배정 중 여섯 클래스의 점유 빈도 차이가 가장 작은 것을 선택했다. 이 조건은 같은 장면의 인접 타일이 서로 다른 클라이언트에 들어갈 수 있다. 별도 장면 12와 WMSC 패치는 어떤 클라이언트 학습에도 들어가지 않는다. [실제 패치 ID](split.json) · [클래스별 복셀 수](client_distributions.csv)",
        table(
            ["분할", "클라이언트", "Ground %", "Building %", "Tree %", "Car %", "Light pole %", "Fence %"],
            [
                (
                    reg,
                    i,
                    *[f"{v*100:.2f}" for v in dist[(dist.regime == reg) & (dist.client == i)].fraction],
                )
                for reg in ("IID", "NonIID")
                for i in (1, 2)
            ],
        ),
        "## 라운드와 대조군",
        "E는 local epochs다. AdamW LR 0.0001, weight decay 0.01, batch 24, CUDA AMP, 결정적 연산을 사용했다. 매 라운드 로컬 optimizer·scaler를 초기화했다. 동일 구조의 클라이언트 상태를 패치 수 96:96으로 가중 평균하고 BN batch counter는 최댓값을 취했다. 각 라운드 집계 직전 두 client 모델과 집계 직후 global 모델을 같은 144패치에 FP32로 평가했다. mIoU와 Macro F1은 분할의 혼동행렬을 누적해 계산하며, 정답과 예측 양쪽에서 빠진 클래스만 각각의 평균에서 제외한다.",
        "중앙 대조군은 같은 192패치를 풀링하고 하나의 모델로 라운드당 E epoch를 학습했다. 중앙과 FL은 라운드당 같은 192×E 패치, 같은 총 optimizer step 시도 수(8×E)다. 각 Local-only 모델은 한 클라이언트의 96×E 패치만 방문하며 통신이 없다. 중앙 및 Local-only는 0% 미세조정 출발점을 공유한다. 10라운드 결과는 동일 30라운드 궤적의 10라운드 체크포인트다. 두 예산을 독립 반복으로 세지 않았다.",
        "## 수렴과 통신 효율",
        "각 그래프의 음영과 오차막대는 세 시드의 표본 표준편차다. 통신량은 십진 MB, 메모리는 MiB다. 도달 기준 mIoU 0.35는 smoke 전 고정했다. 미세조정 직후에 이미 도달하면 0라운드, 미도달은 결측으로 기록했다. 마지막 시점의 성능과 최초 도달은 별도 지표다.",
        "![라운드별 mIoU](convergence.png)",
    ]
    main = summary[summary.method == "FedAvg"].sort_values(
        ["rounds", "partition", "local_epochs", "pruning_pct"]
    )
    body = []
    for _, r in main.iterrows():
        hit = f"{r.round_to_target_mean:.1f} ({int(r.reached_n)}/3)" if r.reached_n else "미도달 (0/3)"
        body.append(
            [
                r["partition"],
                int(r.local_epochs),
                int(r.rounds),
                int(r.pruning_pct),
                pm(r, "miou"),
                pm(r, "macro_f1"),
                pm(r, "synthetic_full_miou"),
                f"{r.total_bytes_mean/1e6:.3f}",
                hit,
                f"{r.train_round_seconds_mean:.1f}",
                f"{r.peak_mib_mean:.1f}",
            ]
        )
    report.append(
        table(
            [
                "분할",
                "E",
                "R",
                "축소 %",
                "val mIoU %",
                "val F1 %",
                "전체 장면 12 mIoU %",
                "통신 MB",
                "목표 R (도달 수)",
                "벽시계 초",
                "peak MiB",
            ],
            body,
        )
    )
    ref = main[
        (main["partition"] == "NonIID") & (main.local_epochs == 1) & (main.rounds == 30)
    ].set_index("pruning_pct")
    report.append(
        "30라운드 Non-IID·E1에서 원본 대비 관측된 변화는 "
        + " / ".join(
            f"{pct}% 채널 축소 시 통신 {100*(1-ref.loc[pct,'total_bytes_mean']/ref.loc[0,'total_bytes_mean']):.1f}% 절감, mIoU {100*(ref.loc[pct,'miou_mean']-ref.loc[0,'miou_mean']):+.2f}%p"
            for pct in (20, 40)
        )
        + "였다. [시드별 원자료](experiment_results.csv) · [모든 조건 평균·표준편차](experiment_summary.csv)"
    )
    # Same bidirectional tensor-byte cap, no interpolation between observed rounds.
    budget = int(
        comm[
            (comm["mode"] == "simulated_state_tensor_bytes")
            & (comm.method == "FedAvg")
            & (comm.rounds == 10)
            & (comm.pruning_pct == 0)
        ].clean_model_frame_bytes.iloc[0]
    )
    within = rounds[(rounds.method == "FedAvg") & (rounds.cumulative_bytes <= budget)]
    equal = within.sort_values("round").groupby("run").tail(1).copy()
    equal.to_csv(OUT / "equal_byte_budget.csv", index=False)
    report += [
        "### 같은 통신 예산에서의 비교",
        f"원본 10라운드에 해당하는 {budget/1e6:.4f} MB를 한도로 두고 각 실행의 마지막 관측 라운드를 비교했다. 작은 모델은 같은 통신 예산으로 로컬 학습을 더 많이 수행할 수 있다. 표는 Non-IID·E1 조건이며 다른 조건도 CSV에 저장했다.",
        table(
            ["채널 축소", "가능한 라운드", "실제 사용 MB", "패치 방문", "val mIoU %"],
            [
                [
                    f"{pct}%",
                    int(d["round"].iloc[0]),
                    f"{d.cumulative_bytes.iloc[0]/1e6:.4f}",
                    int(d["round"].iloc[0]) * 192,
                    f"{100*d.global_miou.mean():.2f} ± {100*d.global_miou.std():.2f}",
                ]
                for pct, d in equal[(equal.partition == "NonIID") & (equal.local_epochs == 1)].groupby(
                    "pruning_pct"
                )
            ],
        ),
        "[동일 통신량 한도에서의 시드별 값](equal_byte_budget.csv)",
    ]
    report += [
        "![통신량과 성능](pruning_tradeoff.png)",
        "### 중앙집중식과 Local-only 대조",
        table(
            ["방법", "분할", "E", "30R val mIoU %", "Macro F1 %", "패치 방문", "시간 초"],
            [
                [
                    r.method,
                    r["partition"],
                    int(r.local_epochs),
                    pm(r, "miou"),
                    pm(r, "macro_f1"),
                    int(r.patch_visits_mean),
                    pm(r, "train_round_seconds", 1),
                ]
                for _, r in summary[
                    (summary.method.isin(["Central", "Local1", "Local2"])) & (summary.rounds == 30)
                ].iterrows()
            ],
        ),
        "### Non-IID 안정성",
        "아래 차이는 같은 시드의 Non-IID − IID를 짝지어 계산했다. 변동성은 초기값을 포함해 연속 라운드 간 mIoU 변화량의 RMS다. 라운드별 client 검증 격차는 같은 검증 패치에서 두 client mIoU 차이의 절댓값이다. 작은 값은 움직임이 작다는 뜻이며 최종 정확도와 함께 해석해야 한다.",
        table(
            [
                "축소 %",
                "E",
                "30R mIoU 차이 %p",
                "라운드 변화 RMS, IID / Non-IID %p",
                "client 격차, IID / Non-IID %p",
            ],
            [
                [
                    int(r.pruning_pct),
                    int(r.local_epochs),
                    f"{100*r.paired_delta_miou:+.2f} ± {100*r.paired_sd_miou:.2f}",
                    f"{100*r.iid_round_change_rms:.2f} / {100*r.noniid_round_change_rms:.2f}",
                    f"{100*r.iid_client_val_gap:.2f} / {100*r.noniid_client_val_gap:.2f}",
                ]
                for r in paired[paired.rounds == 30].itertuples()
            ],
        ),
    ]
    decisions = json.loads((OUT / "prox_decision.json").read_text())
    trig = [str(d["pruning_pct"]) + "%, E" + str(d["local_epochs"]) for d in decisions if d["run_prox"]]
    report.append(
        "FedProx 실행 조건은 30라운드에서 동일 E·폭의 세 시드 평균 Non-IID mIoU가 IID보다 2.00%p 이상 낮은 경우로 미리 고정했다. "
        + (
            "해당 조건 " + ", ".join(trig) + "에서 μ=0.01을 적용했다."
            if trig
            else "임계값을 넘은 조합은 0개여서 FedProx 추가 학습은 실행하지 않았다."
        )
        + " [조건별 판정](prox_decision.json)"
    )
    report.append(
        "FedBN은 원본·Non-IID·E3에서 세 시드로 별도 실행했다. 각 client는 BN affine 파라미터, running mean/variance와 batch counter를 라운드 사이에 로컬로 보존했다. 서버는 나머지 텐서만 평균했다. 글로벌 검증과 WMSC 평가는 공통 집계 가중치에 두 private BN 상태를 각각 결합한 뒤 두 모델의 logits를 같은 가중치로 평균했다. 이 평가에는 두 personalized 모델이 필요하다. 통신 계산에서 private BN은 제외했다."
    )
    report.append(
        table(
            ["방법", "축소 %", "E", "30R val mIoU %", "val F1 %", "통신 MB"],
            [
                [
                    r.method,
                    int(r.pruning_pct),
                    int(r.local_epochs),
                    pm(r, "miou"),
                    pm(r, "macro_f1"),
                    f"{r.total_bytes_mean/1e6:.3f}",
                ]
                for _, r in summary[
                    (summary["partition"] == "NonIID")
                    & (summary.rounds == 30)
                    & (
                        (summary.method.isin(["FedBN", "FedProx"]))
                        | (
                            (summary.method == "FedAvg")
                            & (summary.pruning_pct == 0)
                            & (summary.local_epochs == 3)
                        )
                    )
                ].iterrows()
            ],
        )
    )
    report += [
        "## 실제 직렬화와 프로세스 통신",
        "같은 linuxserver2에서 독립 CPU 서버 프로세스와 CUDA client 프로세스 두 개가 loopback TCP를 사용했다. seed 42, Non-IID, E1, 각 구조별 10라운드다. 각 연결에서 실제 직렬화한 가중치와 BN 상태, 길이 필드, JSON 메타데이터 및 해시를 송수신했다. 서버가 센 송신 길이와 클라이언트 수신 길이를 대조했다. 아래 clean 열은 정상 모델 프레임만, 전체 세션 열은 핸드셰이크·재전송·폐기된 부분 라운드까지 포함한다. TCP/IP 헤더와 링크 계층 트래픽은 측정 범위 밖이다.",
        table(
            ["축소 %", "텐서 추정 MB", "TCP 정상 프레임 MB", "직렬화·프레임 부가 %", "TCP 세션 전체 MB", "정상 프레임 절감 %"],
            [
                [
                    pct,
                    f"{sim/1e6:.4f}",
                    f"{actual.clean_model_frame_bytes/1e6:.4f}",
                    f"{100*(actual.clean_model_frame_bytes/sim-1):.1f}",
                    f"{actual.session_bytes/1e6:.4f}",
                    f"{actual.clean_reduction_pct:.1f}",
                ]
                for pct in (0, 20, 40)
                for sim in [
                    comm[
                        (comm["mode"] == "simulated_state_tensor_bytes")
                        & (comm.method == "FedAvg")
                        & (comm.rounds == 10)
                        & (comm.pruning_pct == pct)
                    ].clean_model_frame_bytes.iloc[0]
                ]
                for _, actual in comm[
                    (comm["mode"] == "tcp_application_actual") & (comm.pruning_pct == pct)
                ].iterrows()
            ],
        ),
        "원본 모델에서는 3라운드 도중 첫 client 응답만 받은 뒤 서버 세션을 중단했다. 두 업데이트가 모두 검증돼야 원자적으로 다음 버전을 commit하도록 해, 2라운드 체크포인트에서 복구했다. 재시작 후 이전 라운드 업데이트를 한 번 더 전송해 서버가 중복을 제외하는지 확인했다. run ID·client ID·모델 버전·부모 해시를 검사한다. 세 구조의 최종 상태와 시뮬레이션 간 최대 텐서 절대 차이는 "
        + ", ".join(f"{pct}% {transport['models'][str(pct)]['max_delta']:.3g}" for pct in (0, 20, 40))
        + "였다. [TCP 실측](transport_rounds.csv) · [검증 기록](transport_checks.json) · [바이트별 회계](communication_cost.csv)",
        "![실제 프레임과 상태 텐서 바이트](communication_comparison.png)",
        "## 보조 외부 평가",
        "모든 10·30라운드 후보의 가중치와 FedBN private 상태 해시를 고정한 뒤 WMSC 전체 2,139,806개 점유 복셀에서 평가했다. 이 장면은 이전 연구에서 반복 사용됐으므로 신규 외부 홀드아웃 수준의 독립성은 없다. 하이퍼파라미터·목표 성능·FedProx 판정에는 WMSC를 사용하지 않았다.",
        table(
            ["분할", "E", "축소 %", "30R WMSC mIoU %", "Macro F1 %"],
            [
                [
                    r["partition"],
                    int(r.local_epochs),
                    int(r.pruning_pct),
                    pm(r, "wmsc_miou"),
                    pm(r, "wmsc_f1"),
                ]
                for _, r in main[main.rounds == 30].iterrows()
            ],
        ),
        "## 해석 범위와 재현",
        "시드 표준편차는 같은 사전학습 가중치와 고정 패치 풀에서의 최적화 변동성이다. 두 개 학습 장면과 한 Synthetic 검증 장면에 대한 결과이며, 독립 지역의 다양성이나 실제 비행의 변동성을 대표하지 않는다. 단일 GPU 순차 시뮬레이션의 통신량은 무압축 상태 텐서 크기 합산이고, 로컬 학습 GPU 메모리는 PyTorch allocator 기준이며 상주 배치 캐시를 포함한다. round 시간은 client/global 검증을 포함한다. TCP 실험의 실측은 LAN·무선 링크, 드론 전력·비행시간을 포함하지 않는다.",
        "프로젝트 루트에서 다음 순서로 같은 조건을 재현한다. 완료 표시가 있는 조건은 재사용하며, 새 기준으로 실험할 때는 결과 루트를 분리해야 한다.",
        "```bash\n.venv-gpu/bin/python -m experiments.distributed_ai\n.venv-gpu/bin/python -m federated.process_fedavg\n.venv-gpu/bin/python -m experiments.distributed_evaluation\n.venv-gpu/bin/python -m experiments.distributed_report\n```",
        "[라운드별 global/client 지표·시간·GPU 메모리](federated_rounds.csv) · [Non-IID 짝비교](noniid_comparison.csv) · [모델 구조·저장량](model_profiles.csv) · [동결된 외부 평가 후보](frozen_endpoints.json)",
    ]
    starts = res[
        (res.method == "FedAvg")
        & (res["partition"] == "NonIID")
        & (res.local_epochs == 1)
        & (res.rounds == 30)
    ]
    init_table = table(
        ["채널 축소", "미세조정 직후 val mIoU %", "0.35 충족 시드", "30R 변화 %p"],
        [
            [
                f"{p}%",
                f"{100*d.initial_miou.mean():.2f} ± {100*d.initial_miou.std():.2f}",
                f"{(d.initial_miou>=THRESHOLD).sum()}/3",
                f"{100*d.delta_miou.mean():+.2f}",
            ]
            for p, d in starts.groupby("pruning_pct")
        ],
    )
    report.insert(
        report.index("![라운드별 mIoU](convergence.png)"),
        "Non-IID·E1의 FL 시작 수준과 30라운드 변화다. 초기값이 이미 목표를 넘은 경우 도달 라운드로 학습 속도 차이를 추론할 수 없다.\n\n"
        + init_table,
    )
    # Attach computed interpretations near the corresponding result tables.
    ni = paired[paired.rounds == 30]
    gap = 100 * ni.paired_delta_miou
    central = summary[(summary.method == "Central") & (summary.rounds == 30)].set_index("local_epochs")
    original = main[
        (main["partition"] == "NonIID") & (main.rounds == 30) & (main.pruning_pct == 0)
    ].set_index("local_epochs")
    bn = summary[(summary.method == "FedBN") & (summary.rounds == 30)].iloc[0]
    budget_rows = equal[(equal.partition == "NonIID") & (equal.local_epochs == 1)]
    budget_means = budget_rows.groupby("pruning_pct").global_miou.mean()
    report.insert(
        report.index("[동일 통신량 한도에서의 시드별 값](equal_byte_budget.csv)"),
        f"같은 통신 예산에서 원본 대비 mIoU 차이는 20% 모델 {100*(budget_means[20]-budget_means[0]):+.2f}%p, "
        f"40% 모델 {100*(budget_means[40]-budget_means[0]):+.2f}%p였다. 이 비교에는 미세조정 직후의 성능 차이와 서로 다른 로컬 학습량이 함께 반영된다.",
    )
    where = report.index("## 실제 직렬화와 프로세스 통신")
    report[where:where] = [
        f"30라운드 여섯 조합의 짝지은 Non-IID − IID mIoU 차이는 {gap.min():+.2f}~{gap.max():+.2f}%p였다. E1에서 E3로 바꾸면 학습 방문 수는 세 배가 되고 통신량은 같다. 이때 중앙 대조군의 mIoU 변화는 {100*(central.loc[3,'miou_mean']-central.loc[1,'miou_mean']):+.2f}%p, 원본 Non-IID FedAvg는 {100*(original.loc[3,'miou_mean']-original.loc[1,'miou_mean']):+.2f}%p였다.",
        f"Non-IID·E3에서 FedBN의 원본 FedAvg 대비 30라운드 mIoU 차이는 {100*(bn.miou_mean-original.loc[3,'miou_mean']):+.2f}%p였다. 이 수치에는 BN 로컬화와 두 personalized 모델의 logits 평균을 사용하는 평가 방식이 함께 반영된다.",
    ]
    if (OUT / "split_audit.json").is_file():
        audit = json.loads((OUT / "split_audit.json").read_text())
        report.insert(
            report.index("## 라운드와 대조군"),
            f"서로 다른 client에 배정된 동일 장면의 8방향 인접 XY 타일 쌍은 IID {audit['IID']['adjacent_cross_client_xy_pairs']}개, Non-IID {audit['NonIID']['adjacent_cross_client_xy_pairs']}개였다. [분할 검증](split_audit.json)",
        )
    (OUT / "DISTRIBUTED_AI_REPORT.md").write_text("\n\n".join(report) + "\n")


if __name__ == "__main__":
    results, rounds, paired, comm, summary = summarize()
    plots(results, rounds, comm)
    report(results, rounds, paired, comm, summary)
