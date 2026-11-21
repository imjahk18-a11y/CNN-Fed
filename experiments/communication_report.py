"""Write the Korean phase-two experimental report using the measured artifacts."""
import json

import numpy as np
import pandas as pd

from experiments.distributed_ai_phase2 import OUT, PROFILES
from experiments.communication_analysis import GROUP, intervals

LABEL = {
    "loopback": "기본 loopback",
    "bandwidth": "방향별 256 kB/s",
    "latency": "추가 RTT 50 ms",
    "disconnect": "3R Client 2 단절",
    "straggler": "Client 2 처리 +150 ms",
}


def table(header, rows):
    return "\n".join(
        [
            "|" + "|".join(header) + "|",
            "|" + "|".join(["---"] * len(header)) + "|",
            *["|" + "|".join(map(str, r)) + "|" for r in rows],
        ]
    )


def pm(series, factor=1.0, digits=2):
    return f"{factor*series.mean():.{digits}f} ± {factor*series.std():.{digits}f}"


def write_report(prior, df, clean, network, budgets, summary, prof):
    old = json.loads((OUT / "integrity.json").read_text())
    protocol = json.loads((OUT / "protocol.json").read_text())
    paired = pd.read_csv(OUT / "int8_paired.csv")
    groups = list(clean.groupby(GROUP))
    original = clean[(clean.pruning == 0) & (clean.codec == "FP32")]
    base = original.bytes.mean()
    clean_rounds = df[df.profile == "loopback"]
    sec = [
        "# 두 클라이언트 연합학습의 프루닝과 양방향 INT8 통신",
        "## 주요 결과",
        "세 시드의 고정 Synthetic 검증 패치에서 0·20·40% 구조적 채널 축소와 FP32·INT8 전송을 교차 비교했다. 아래 바이트는 서버에서 실제로 송수신한 TCP 애플리케이션 바이트다. MB는 1,000,000바이트다. 표의 ±는 시드 42·43·44의 표본 표준편차다.",
        table(
            [
                "프루닝",
                "전송",
                "0R mIoU %",
                "10R",
                "20R",
                "30R",
                "초기 대비 30R %p",
                "30R MB",
                "원본 FP32 대비 절감 %",
                "30R 시간 초",
                "클라이언트 peak MiB",
            ],
            [
                [
                    pct,
                    mode,
                    *[pm(g[f"r{i}"], 100) for i in (0, 10, 20, 30)],
                    pm(g.delta30_pp),
                    f"{g.bytes.mean()/1e6:.3f}",
                    f"{100*(1-g.bytes.mean()/base):.1f}",
                    pm(g.elapsed_s),
                    pm(g.peak_mib),
                ]
                for (pct, mode), g in groups
            ],
        ),
        "같은 폭과 시드의 INT8 − FP32 차이를 짝지어 집계했다.",
        table(
            ["프루닝", "30R mIoU 차이 %p", "같은 폭 대비 통신 절감 %", "30R 완료시간 차이 초"],
            [
                [p, pm(g.r30_delta_pp), pm(g.reduction_pct), pm(g.time_delta_s)]
                for p, g in paired.groupby("pruning")
            ],
        ),
        "## 초기값을 고려한 수렴과 성능 하락",
        "각 시드·폭의 FL 시작점에서 +1.00%p 이상을 3라운드 연속 유지하면 세 번째 라운드를 도달 시점으로 기록했다. 0R은 원본 서버 체크포인트를 전송 전에 평가한 값이다. 평균 궤적의 최고점은 0R을 포함하며, 동률에서는 첫 라운드를 택했다. 최고 라운드는 사후 기술 통계다.",
        "![라운드와 누적 통신량별 검증 mIoU](communication_convergence.png)",
    ]
    body = []
    for (pct, mode), g in groups:
        tr = (
            clean_rounds[(clean_rounds.pruning == pct) & (clean_rounds.codec == mode)]
            .groupby("round")
            .miou.mean()
        )
        arr = tr.to_numpy()
        body.append(
            [
                pct,
                mode,
                int(tr.idxmax()),
                f"{tr.max()*100:.2f}",
                f"{100*(arr[-1]-tr.max()):+.2f}",
                intervals(np.flatnonzero(np.diff(arr) < 0) + 1),
                pm(g.step_rms_pp),
                "/".join(
                    "미도달" if pd.isna(x) else str(int(x))
                    for x in g.sort_values("seed").target_confirm_round
                ),
            ]
        )
    sec += [
        table(
            [
                "축소 %",
                "전송",
                "평균 궤적 최고 R",
                "최고 mIoU %",
                "최고→30R %p",
                "추가 전송 후 mIoU가 직전보다 낮아진 R",
                "라운드 변화 RMS %p",
                "도달 R (42/43/44)",
            ],
            body,
        ),
        "라운드 하락은 인접 시점의 관측 차이다. 통신량 자체가 하락을 유발했다는 인과 추론은 어렵다. 시드별 최고점·하락 라운드·최대 낙폭·도달 바이트는 [시드별 결과](seed_results.csv)에 있다.",
        "## 같은 실제 통신 예산에서의 비교",
        "각 예산 이하에서 완료 가능한 마지막 라운드를 비교했다. 30라운드 상한을 먼저 채운 경우 그 시점의 성능과 사용량을 표시한다. 각 방법이 수행한 로컬 학습량은 라운드 수에 비례한다.",
        table(
            ["예산 MB", "축소 %", "전송", "완료 R 범위", "사용 MB", "mIoU %", "초기 대비 %p"],
            [
                [
                    f"{budget/1e6:.1f}",
                    p,
                    m,
                    f"{g['round'].min()}–{g['round'].max()}",
                    f"{g.used_bytes.mean()/1e6:.3f}",
                    pm(g.miou, 100),
                    pm(g.delta_pp),
                ]
                for (budget, p, m), g in budgets.groupby(["budget_bytes", *GROUP])
            ],
        ),
        "## 중앙집중식·Local-only 대조와 기존 기록 검증",
        "동일한 사전학습, Non-IID, 로컬 1 epoch의 기존 대조군이다. 중앙 모델은 양쪽 패치 192개를, Local-only 모델은 해당 클라이언트 패치 96개를 각 라운드마다 학습했다. 이 대조군들의 모델 교환량은 0이다. 원본 FP32 FedAvg를 세 폭·세 시드 모두 실제 TCP로 재실행하여 매 라운드 검증과 10·30R 체크포인트의 모든 텐서를 기존 실행과 대조했다.",
        table(
            ["방법", "0R %", "10R %", "20R %", "30R %"],
            [
                [
                    method,
                    *[
                        f"{100*r['mean']:.2f} ± {100*r['std']:.2f}"
                        for _, r in g.sort_values("round").iterrows()
                    ],
                ]
                for method, g in prior[
                    (prior.local_epochs == 1)
                    & prior.method.isin(["Central", "Local1", "Local2"])
                    & prior.partition.isin(["pooled", "NonIID"])
                ].groupby("method")
            ],
        ),
        "과거 IID·Non-IID와 E1·E3, FedBN을 포함한 69개 실행의 0/10/20/30R 및 하락 구간도 별도로 추출했다. [이전 실행 재분석](prior_runs.csv) · [이전 조건별 평균·표준편차](prior_summary.csv)",
        "## 네트워크 조건과 복구",
        "GPU 한 장을 공유하는 독립 클라이언트 2개가 순차적으로 CUDA 학습을 수행했고, 별도 서버 프로세스가 CPU에서 집계하고 GPU에서 검증했다. 각 조합에서 기본 loopback의 첫 5라운드와 같은 시작점·학습 순서를 사용했다. 모델 프레임에만 애플리케이션 대역폭 또는 지연을 부과했다. 대역폭 조건은 방향별 256,000 B/s로 4 KiB 단위 송신을 조절했다. 지연 조건은 요청과 응답 각각에 25 ms를 더했다. 느린 클라이언트 조건에서는 Client 2의 처리에 150 ms를 더했다.",
        "단절 조건에서 Client 2는 3라운드 모델을 받은 직후 응답 전에 실제 소켓을 닫고 150 ms 뒤 다시 접속했다. 서버는 이미 받은 Client 1 업데이트를 보관하고 같은 버전을 Client 2에 한 번 재전송했다. 두 업데이트의 라운드·ID·부모 해시·샘플 수가 일치해야 집계했다. 모든 네트워크 조건의 첫 다섯 검증 점수가 기본 연결과 같았고, 네 가지 장애·제약 조건의 5라운드 최종 가중치도 서로 일치했다.",
        table(
            ["조건", "18회 평균 5R 초", "페어링한 기본 대비 초", "5R 평균 MB", "Client 1/2 평균 처리 초", "재전송 횟수", "복구/실패"],
            [
                [
                    LABEL[q],
                    f"{g.time_s.mean():.2f}",
                    f"{g.time_delta_s.mean():+.2f}",
                    f"{g.bytes.mean()/1e6:.3f}",
                    f"{g.client1_proc_s.mean():.2f} / {g.client2_proc_s.mean():.2f}",
                    int(g.retries.sum()),
                    f"{int(g.recovered.sum())}/{int(g.failures.sum())}"
                    if g.failures.sum()
                    else "실패 없음",
                ]
                for q in PROFILES
                for g in [network[network.profile == q]]
            ],
        ),
        "이 표의 평균은 6개 모델·전송 조합 × 3시드에 걸친 값이다. 클라이언트 처리시간은 수신 완료부터 응답 직렬화·검증까지며 대기 주입을 포함한다. 순수 로컬 학습시간도 별도 측정했다. 라운드·총시간은 준비된 연결에서 첫 학습 라운드 시작부터 최종 검증까지의 경과 시간으로, 로딩·프로세스 기동·최초 연결·0R 평가·체크포인트 저장은 제외했다. 재연결은 경과 시간에 포함했다. 첫 연결의 양방향 합계 34바이트와 재연결 17바이트는 전송량에 포함했다. [프로필별 시간·바이트·복구 원자료](network_five_rounds.csv) · [조합별 평균·표준편차](network_summary.csv)",
        "## 모델 크기와 양자화 검증",
        "전체 모델 상태를 양방향으로 전송했다. 각 부동소수 텐서에서 s=max(|x|)/127을 구하고 round(x/s)를 −127…127 범위의 signed INT8로 저장했다. 영 텐서는 s=1이다. 스케일은 FP32로 전송하고 정렬된 키 순서에 대응시켰다. 수신자는 각 텐서의 스케일로 FP32 값을 복원했다. BN 배치 카운터는 정수 원형으로 보존했다. 서버는 복원값을 샘플 수 96:96으로 평균하고 배치 카운터는 기존 FedAvg 규칙인 최댓값을 사용했다. 단말 내부 학습은 기존 AMP와 FP32 마스터 가중치를 유지했다.",
        table(
            [
                "축소 %",
                "전송",
                "파라미터",
                "초기 체크포인트 B (42)",
                "초기 payload B (42)",
                "초기 상대 L2 오차",
                "전체 전송 최대 절대 오차",
                "최대 오차/step",
            ],
            [
                [
                    p,
                    m,
                    int(g.params.iloc[0]),
                    int(g[g.seed == 42].checkpoint_bytes.iloc[0]),
                    int(g[g.seed == 42].serialized_bytes.iloc[0]),
                    f"{g.relative_l2.mean():.5f}",
                    f"{q[['down_max_error','up_max_error']].max().max():.5f}",
                    f"{q[['down_max_step_fraction','up_max_step_fraction']].max().max():.6f}",
                ]
                for (p, m), g in prof.groupby(GROUP)
                for q in [df[(df.pruning == p) & (df.codec == m)]]
            ],
        ),
        "송신자는 모든 직렬화된 상태를 재로딩해 FP32·정수의 일치와 INT8 원소별 오차 상한(반 스텝, 부동소수 오차 허용)을 확인했다. 수신자는 SHA-256, 키, shape, dtype, 스케일과 유한값을 검사했다. 위 payload는 직렬화한 가중치와 스케일이며 통신량 표에는 추가로 헤더·해시·텔레메트리·핸드셰이크가 포함된다. 패킷 헤더와 OS 수준 재전송은 측정하지 않았다. 기존 TCP 실험과 메타데이터가 달라 절감률의 기준은 이번 구현 안에서 맞췄다. [모든 초기 모델의 양자화 오차](model_codec_profiles.csv)",
    ]
    speed = network[network.profile == "bandwidth"].groupby(GROUP).time_s.mean()
    absolute = summary.loc[summary.r30_mean.idxmax()]
    bybudget = budgets.groupby(["budget_bytes", *GROUP]).miou.mean().reset_index()
    best = bybudget.loc[bybudget.groupby("budget_bytes").miou.idxmax()]
    sec += [
        "## 다중 드론 적용 시 해석",
        f"이 분할에서 30라운드 절대 mIoU가 가장 높은 조합은 {int(absolute.pruning)}% 프루닝·{absolute.codec}({absolute.r30_mean*100:.2f}%)였다. 같은 바이트 한도의 관측 순위는 "
        + ", ".join(
            f"{r.budget_bytes/1e6:.1f} MB에서 {r.pruning}%·{r.codec} {100*r.miou:.2f}%"
            for r in best.itertuples()
        )
        + "였다. 예산별 순위는 관측된 라운드와 이 검증 장면에 한정된다.",
        f"대역폭 제한에서 원본 FP32의 5라운드 평균은 {speed.loc[0,'FP32']:.2f}초, 원본 INT8은 {speed.loc[0,'INT8']:.2f}초, 40% 프루닝·INT8은 {speed.loc[40,'INT8']:.2f}초였다. 원본 INT8은 채널 수와 단말 계산 구조를 유지하며 전송량을 줄인다. 구조적 프루닝은 단말에 상주하는 파라미터와 로컬 학습 메모리까지 줄일 수 있지만 FL 시작 성능도 바뀐다. 둘을 결합하면 통신량은 가장 작아지고 작은 모델에 반복 양자화 오차가 누적될 수 있다. 이 비용은 각 폭의 FP32 짝대조로 평가했다.",
        "실제 드론에는 링크 용량뿐 아니라 연산 여유, 손실과 재접속 빈도에 따라 적용 후보가 달라진다. 이 실험은 단일 호스트의 순차 학습과 동기식 FedAvg를 사용했다. 느린 클라이언트에 의한 시간 증가에는 순차 실행 방식이 반영된다. 무선 링크·다른 장치·실제 패킷 손실·에너지·비동기 집계는 측정하지 않았다. 단절 복구율은 한 가지 지정된 단절에서만 관측했다.",
        "## 데이터와 재현",
        f"{protocol['gpu']}, PyTorch {protocol['torch']}, batch 24, AdamW LR 0.0001, 라운드마다 optimizer 초기화. 장면 13 사전학습·미세조정, FL 장면 3과 4 각 96패치, 라운드 검증 장면 12의 동일 144패치를 사용했다. 세 시드는 동일한 사전학습 체크포인트와 split을 공유한다. 시드 간 편차는 같은 자료에서의 학습 순서 변동을 나타내며 다른 장면·다른 사전학습·드론 집단으로의 일반화는 검증하지 않았다. 이전 사전학습 모델 선택에도 장면 12를 사용했다. 후속 실험에는 WMSC를 사용하지 않았다.",
        f"기존 실험 디렉터리와 공통 사전학습 파일 {old['old_files_verified']}개의 SHA-256을 시작·종료 시점에 대조했다. [환경·원본 해시·평가 기준](protocol.json) · [무결성 확인](integrity.json) · [모든 라운드의 시간·메모리·방향별 바이트](rounds.csv).",
        "프로젝트 루트에서 실행한다. 완료된 조합은 재사용한다.",
        "```bash\n.venv-gpu/bin/python -m experiments.distributed_ai_phase2\n.venv-gpu/bin/python -m experiments.communication_analysis\n```",
    ]
    orig_int8 = clean[(clean.pruning == 0) & (clean.codec == "INT8")]
    sec.insert(sec.index("## 모델 크기와 양자화 검증"),
               "측정 시간에는 송신자의 직렬화 왕복 검사와 서버의 라운드별 검증도 들어간다. FP32는 과거 검증 점수를 대조하고 10·30라운드에서 과거 가중치도 읽어 비교했다. 시간 비교에는 이 검증 비용과 호스트의 실행 편차가 포함된다.")
    sec.insert(sec.index("## 데이터와 재현"),
               f"원본 INT8의 30라운드 mIoU는 같은 가중치를 전송 전에 평가한 0라운드보다 "
               f"{orig_int8.delta30_pp.mean():+.2f}%p였다. 원본 FP32의 같은 변화는 "
               f"{original.delta30_pp.mean():+.2f}%p였다. FL을 적용할 때는 추가 라운드가 "
               "사전학습 모델보다 실제로 나아졌는지도 함께 확인할 필요가 있다.")
    (OUT / "COMMUNICATION_REPORT.md").write_text("\n\n".join(sec) + "\n")
