"""Derive paired summaries and Korean Phase 4 report from committed measurements."""
import json

import pandas as pd
from experiments.phase4_pretrain import FOLDS, OUT, SEEDS


def table(header, rows):
    return "\n".join(
        [
            "|" + "|".join(header) + "|",
            "|" + "|".join(["---"] * len(header)) + "|",
            *["|" + "|".join(map(str, r)) + "|" for r in rows],
        ]
    )


def pm(x, scale=100, digits=2):
    return f"{x.mean()*scale:.{digits}f} ± {x.std()*scale:.{digits}f}"


def main():
    ind = pd.read_csv(OUT / "independent_runs.csv")
    bn = pd.read_csv(OUT / "bn_ablation.csv")
    trajectories = pd.read_csv(OUT / "update_scaling.csv")
    e = pd.read_csv(OUT / "early_stopping_generalization.csv")
    swaps = pd.read_csv(OUT / "bn_crossed.csv")
    classes = pd.read_csv(OUT / "class_distributions.csv")
    verify = json.loads((OUT / "verification.json").read_text())
    sp = json.loads((OUT / "split_audit.json").read_text())
    assert verify["independent_starts"] == verify["unique_initializations"] == 6
    assert (
        len(ind) == 18 and len(bn) == len(e) == 72 and len(trajectories) == 54 * 31 and len(swaps) == 54
    )
    pair = []
    for f, g in ind.groupby("fold"):
        for m, n in (("INT8", "FP32"), ("FP32_MATCH", "FP32"), ("INT8", "FP32_MATCH")):
            a = g[g["mode"] == m].set_index("seed")
            b = g[g["mode"] == n].set_index("seed")
            for seed in a.index:
                pair.append(
                    dict(
                        fold=f,
                        seed=seed,
                        contrast=f"{m} − {n}",
                        val_delta_pp=100 * (a.at[seed, "val_r30"] - b.at[seed, "val_r30"]),
                        audit_delta_pp=100 * (a.at[seed, "audit_r30"] - b.at[seed, "audit_r30"]),
                    )
                )
    paired = pd.DataFrame(pair)
    paired.to_csv(OUT / "paired_differences.csv", index=False)
    ref = paired[paired.contrast == "INT8 − FP32"]
    scaled = paired[paired.contrast == "FP32_MATCH − FP32"]
    main_early = e[e.bn == "aggregate"]
    int8_early = main_early[main_early["mode"] == "INT8"]
    deltas = []
    for f in FOLDS:
        dd = ref[ref.fold == f]
        deltas.append(f"{f} 분할 {pm(dd.audit_delta_pp,1)}%p ({int((dd.audit_delta_pp>0).sum())}/3 양수)")
    body = [
        "# 독립 사전학습과 장면 변경에 따른 2-client FedAvg 재현성",
        "## 결과 요약",
        "독립 사전학습 모델의 30라운드 최종 평가에서 INT8 − FP32 차이는 "
        + " · ".join(deltas)
        + "였다. 아래의 ‘최종 평가’는 조기 중단에 쓰지 않은 별도 패치(audit)에서의 측정이다. ±는 각 분할 안의 서로 다른 초기화 3개의 표본 표준편차다.",
        f"FP32 업데이트 크기 조절의 최종 평가 이득은 독립 시작점 6개 중 {int((scaled.audit_delta_pp>0).sum())}개에서 양수였다. INT8 온라인 조기 중단은 30R 대비 0.5%p 이내 유지 기준을 {int(int8_early.retained_vs_r30.sum())}/6회 충족했고, 0R 대비로는 {int(int8_early.retained_vs_r0.sum())}/6회 충족했다. 기준은 실행 전에 고정했다. 작은 표본에서 동등성을 통계적으로 검정한 결과는 아니다.",
        "## 기존 결과 재현과 새 실험 구성",
        "기존 Phase 3의 시드 42 체크포인트와 클라이언트·검증 패치를 불러온 검산에서 FP32, INT8, FP32_MATCH 세 조건 모두 0~30R mIoU와 30R 모든 가중치가 기존 기록과 정확히 일치했다. 과거 세 시드 평균 35.33%, 36.74%, 36.48% 및 INT8 조기 중단 37.53%는 한 사전학습 모델에서 파생된 수치다. 검산 실행은 독립 표본 수에서 제외했다.",
        table(
            ["분할", "사전학습 / 미세조정 / 보정 장면", "Client 1 / 2", "검증 및 최종 평가 장면", "사전학습 시드"],
            [
                [f, p, f"{a} / {b}", v, ", ".join(map(str, SEEDS[f]))]
                for f, (p, a, b, v) in FOLDS.items()
            ],
        ),
        "각 모델은 무작위 초기화 후 0.3 m Full-Z RGB-16 원본 sparse CNN을 35 epoch(AdamW LR 0.001), 같은 장면의 192패치를 5 epoch(LR 0.0001) 학습했다. 35번째 epoch를 사용해 사전학습 중 검증 장면을 모델 선택에 쓰지 않았다. Phase 3는 장면 13 전체로 사전학습 후 검증 기준으로 epoch를 선택했다. reference는 원래 client/검증의 패치와 순서를 유지하며 사전학습·미세조정 구성은 달라졌다. rotated는 사전학습·두 클라이언트·검증 장면 역할을 모두 바꿨다. 분할 내 시드에 따라 초기 가중치와 사전학습 batch 순서가 달라지고 FL 순서 시드는 모든 실험에서 42다.",
        "### 분할과 클래스 분포",
        "장면당 XY 52×52 타일 중 훈련 장면 한쪽의 96타일을 BN 보정용으로 예약했다. 이 영역과 인접 두 겹을 사전학습·미세조정에서 제외했다. 검증 장면에서 조기 중단용 144패치와 두 겹 이상 떨어진 별도 144패치를 최종 평가용으로 뒀다. 각 client에는 다른 장면의 96패치를 배정했다. 보정 패치 레이블은 분포 감사에만 사용했다. 배치 구성 후 BN 함수에는 좌표와 RGB만 전달했다.",
        table(
            ["분할", "역할", "패치", "Ground %", "Building %", "Tree %", "Car %", "Light pole %", "Fence %"],
            [
                [
                    f,
                    role,
                    g.patches.iloc[0],
                    *[
                        f"{100*g.set_index('class_name').loc[c,'fraction']:.2f}"
                        for c in ("Ground", "Building", "Tree", "Car", "Light pole", "Fence")
                    ],
                ]
                for (f, role), g in classes.groupby(["fold", "role"])
            ],
        ),
        f"두 분할 모두 학습·보정·검증·최종 평가 간 점유 좌표와 RGB 전체가 같은 패치는 0개였다. 검증용과 최종 평가용 간 최소 Chebyshev 타일 거리는 reference {sp['reference']['min_audit_validation_tile_distance']}, rotated {sp['rotated']['min_audit_validation_tile_distance']}였다. 네 합성 장면은 장면 로컬 XY 좌표계를 사용한다. 동일 장면 내부의 거리와 패치 중복은 검사했으나 독립 지리적 촬영 구역은 확인하지 않았다. 교차 분할은 같은 네 장면을 재사용하므로 독립 지역 6곳으로 집계하지 않았다. [패치 ID](splits.json) · [클래스별 복셀 수](class_distributions.csv) · [누수 점검](split_audit.json)",
        "## 기본 BN 집계에서의 30라운드 결과",
        table(
            ["분할", "조건", "검증 0R %", "검증 30R %", "최종 평가 0R %", "최종 평가 30R %", "TCP MB"],
            [
                [
                    f,
                    m,
                    pm(g.val_r0),
                    pm(g.val_r30),
                    pm(g.audit_r0),
                    pm(g.audit_r30),
                    f"{g.full_session_bytes.mean()/1e6:.3f}",
                ]
                for (f, m), g in ind.groupby(["fold", "mode"])
            ],
        ),
        table(
            ["분할", "짝비교", "검증 차이 %p", "최종 평가 차이 %p", "최종 평가 양수/3"],
            [
                [
                    f,
                    k,
                    pm(g.val_delta_pp, 1),
                    pm(g.audit_delta_pp, 1),
                    int((g.audit_delta_pp > 0).sum()),
                ]
                for (f, k), g in paired.groupby(["fold", "contrast"])
            ],
        ),
        "모든 FL 조건은 같은 시작점에서 라운드당 클라이언트 각 1 epoch, batch 24, LR 0.0001, 30라운드로 학습했다. 경로당 총 5,760패치 방문, optimizer step 시도 240회다. 두 클라이언트의 전체 상태 또는 공유 부분을 샘플 수 96:96으로 가중 평균했다. 서버와 클라이언트는 같은 CUDA 프로세스에서 순차 실행하고, 모델은 두 localhost TCP 연결에서 실제 직렬화해 송수신했다. 실제 드론 링크와 지연·전력은 측정하지 않았다.",
        "## 업데이트 크기와 방향",
        "FP32_MATCH는 같은 시드·BN 정책의 INT8에서 얻은 각 라운드의 글로벌 학습 파라미터 L2와 BN running mean/variance L2를 각각 FP32 업데이트에 적용한다. 그룹별 FP32 방향을 유지한다. private 또는 frozen 정책에서 서버가 갱신하지 않는 상태의 글로벌 이동은 0이다. 이 크기 대조군은 INT8 참고 궤적을 필요로 하며, 목표 크기를 정하는 데 검증·최종 평가 점수는 사용하지 않았다. 아래는 처음 전체 가중치를 양자화하는 1R과 그 후의 평균을 분리한 값이다.",
        table(
            [
                "분할",
                "조건",
                "1R 파라미터 L2",
                "2~30R 파라미터 L2",
                "2~30R BN L2",
                "학습 직후 client 간 cos",
                "업로드 복원 후 client 간 cos",
            ],
            [
                [
                    f,
                    m,
                    f"{g[g['round']==1].global_parameters_l2.mean():.5f}",
                    f"{g[g['round']>=2].global_parameters_l2.mean():.5f}",
                    f"{g[g['round']>=2].global_bn_l2.mean():.5f}",
                    f"{g[g['round']>=2].client_update_cos.mean():.3f}",
                    f"{g[g['round']>=2].client_decoded_cos.mean():.3f}",
                ]
                for (f, m), g in trajectories[trajectories.bn == "aggregate"].groupby(["fold", "mode"])
            ],
        ),
        "코사인은 두 클라이언트가 수신한 공유 파라미터에서 각각 얼마나 어느 방향으로 움직였는지 비교한다. BN affine은 공유할 때만 포함했다. 각 client와 글로벌 업데이트 사이의 코사인, loss, 업·다운로드 바이트도 라운드별로 저장했다. 0R에는 업데이트가 없어 L2와 코사인을 결측 처리했다. [라운드별 측정](update_scaling.csv)",
        "## BN 처리 방식별 결과",
        "- **aggregate**는 BN affine·running moments를 공유하고 배치 카운터는 기존 FedAvg 규칙인 최댓값으로 집계했다.\n- **frozen**은 로컬 forward에서도 시작 모델의 running moments를 사용하고 고정했다. affine에는 gradient를 적용했다.\n- **private**는 BN affine·moments·counter를 클라이언트에 유지하고 나머지만 교환했다. 공통 입력에 대한 두 personalized 모델의 혼동행렬을 합산해 mIoU를 구했다. 실제로는 두 개인화 모델의 성능을 나타내며 한 서버 모델의 예측과 다르다.\n- **recalibrated**는 aggregate와 학습 경로·파라미터·통신량이 같다. 각 라운드 별도 학습 영역의 입력 96패치에서 encoder→bridge→decoder 순서로 활성값의 점유 사이트별 모집단 평균·분산을 구해 평가 복사본에만 적용했다. 보정은 라운드당 순전파 12배치를 추가하며 0R에도 적용했다. 보정 결과가 다음 라운드 학습에 들어가지 않는다.",
        table(
            [
                "분할",
                "BN",
                "FP32 최종 평가 %",
                "FP32_MATCH 최종 평가 %",
                "INT8 최종 평가 %",
                "FP32 세션 MB",
                "추가 검증 전송 MB¹",
            ],
            [
                [
                    f,
                    b,
                    *[pm(g[g["mode"] == m].audit_r30) for m in ("FP32", "FP32_MATCH", "INT8")],
                    f"{g[g['mode']=='FP32'].full_session_bytes.mean()/1e6:.3f}",
                    f"{g.validation_message_bytes.mean()/1e6:.4f}",
                ]
                for (f, b), g in bn.groupby(["fold", "bn"])
            ],
        ),
        "¹ private에서만 매 라운드 클라이언트별 6×6 confusion count와 프레임 메타데이터를 서버로 전송했다. 데이터·시작 모델의 최초 설치 비용은 별도다. TCP 바이트는 애플리케이션 양방향 합계로, 연결 식별 메시지와 종료 왕복을 포함한다.",
        "같은 파라미터에서 BN 통계만 바꾸는 비교는 학습 경로가 달라지는 frozen/private 비교보다 좁은 질문에 답한다. 아래 표는 30R에서 FP32와 INT8 체크포인트의 학습 파라미터와 running moments를 교차해 같은 최종 평가 패치에서 얻은 mIoU다.",
        table(
            ["분할", "학습 파라미터 출처", "FP32 BN %", "INT8 BN %", "BN 교체 효과 %p"],
            [
                [
                    f,
                    weights,
                    pm(g[g.moments == "FP32"].audit_miou),
                    pm(g[g.moments == "INT8"].audit_miou),
                    pm(
                        g[g.moments == "INT8"].set_index("seed").audit_miou
                        - g[g.moments == "FP32"].set_index("seed").audit_miou
                    ),
                ]
                for (f, weights), g in swaps[swaps.parameters.isin(["FP32", "INT8"])].groupby(
                    ["fold", "parameters"]
                )
            ],
        ),
        "개인화에는 BN affine의 로컬 학습과 클라이언트별 추론이 함께 반영된다. 교차 장착은 최종 추론 시의 통계 민감도를 측정한다. BN이 학습 내내 파라미터에 미친 효과까지 독립적으로 식별할 수는 없다. [시드별 BN 실험](bn_ablation.csv) · [교차 평가](bn_crossed.csv)",
        "## 조기 중단의 일반화",
        "Phase 3와 같은 온라인 규칙(min_delta=0, patience=3)을 고정했다. 전체 궤적에서 현재까지의 점수만 순차로 읽고 중단 시점까지 최고였던 모델을 복원했다. 30R까지 관측한 사후 최고 라운드는 따로 기록했다. 모든 방법의 선택 라운드와 체크포인트 해시를 고정한 뒤 최종 평가용 패치에서 처음으로 모델 점수를 계산했다. 그 전에는 패치 중복과 클래스 분포만 점검했다. 아래 세션 비용은 실측한 중단 prefix에 같은 길이의 종료 프레임을 더한 값이며, 실제 각 실행은 비교를 위해 30라운드까지 수행했다.",
        table(
            [
                "분할",
                "조건",
                "사후 최고 R",
                "중단 R",
                "복원 R",
                "복원 검증 %",
                "복원 최종 평가 %",
                "30R 대비 최종 평가 %p",
                "조기 MB",
                "30R 대비 유지/3",
            ],
            [
                [
                    f,
                    m,
                    ",".join(map(str, g.best_round)),
                    ",".join(map(str, g.stop_round)),
                    ",".join(map(str, g.restored_round)),
                    pm(g.restored_val),
                    pm(g.restored_audit),
                    pm(g.delta_vs_r30_pp, 1),
                    f"{g.early_session_bytes.mean()/1e6:.3f}",
                    int(g.retained_vs_r30.sum()),
                ]
                for (f, m), g in main_early.sort_values("seed").groupby(["fold", "mode"])
            ],
        ),
        "유지는 중단 모델의 최종 평가 mIoU가 비교 시점보다 0.5%p를 초과해 낮아지지 않은 경우로 정의했다. 시작점과 30R에 대해 별도 판정했으며, 10R 고정 종료 성능과 네 BN 정책의 결과도 저장했다. 같은 검증 장면의 공간적으로 분리된 패치이므로 새로운 비행 지역에서의 독립 시험과는 평가 범위가 다르다. [시드별 중단·통신량·유지 판정](early_stopping_generalization.csv)",
        "**실제 서버에서 필요한 자료.** aggregate·frozen·recalibrated의 온라인 중단에는 서버가 보유한 대표 입력과 정답이 필요하다. private는 양쪽 클라이언트에도 같은 읽기 전용 검증 세트를 배포해 각각 지표를 계산했다. 이 연구 서버에서는 가능했지만 실기체 서버에 이런 레이블이 제공된다는 증거는 없다. 별도 보정용 영역의 입력도 설치 전에 확보해야 한다. 레이블 없는 서버는 이 mIoU 중단 규칙을 직접 실행할 수 없으며, 사전에 합의한 고정 라운드의 측정값은 그 경우의 대조 기준이다.",
        "## 해석 범위와 재현",
        f"무작위 초기 가중치·독립 사전학습·최종 시작 체크포인트 해시를 기록했다. 동결된 경로 체크포인트 {verify['checkpoint_hashes_verified']:,}개와 과거 실험 파일 {verify['historical_files_verified']:,}개의 해시를 확인했다. 새로운 데이터 다운로드는 수행하지 않았고 WMSC 원본·레이블은 사용하지 않았다. [프로토콜](protocol.json) · [독립 실행](independent_runs.csv) · [시드 짝비교](paired_differences.csv)",
        "프로젝트 루트에서 다음 순서로 실행한다.",
        """```bash
.venv-gpu/bin/python -m experiments.phase4_pretrain
.venv-gpu/bin/python -m experiments.phase4_split_audit
.venv-gpu/bin/python -m experiments.phase4_federated
.venv-gpu/bin/python -m experiments.phase4_evaluate
.venv-gpu/bin/python -m experiments.phase4_report
```""",
    ]
    calibration = classes[classes.role == "calibration"].set_index(["fold", "class_name"])
    notes = [
        "보정 영역의 구성은 "
        + ", ".join(
            f"{f} Ground {100*calibration.loc[(f,'Ground'),'fraction']:.2f}% · Fence {int(calibration.loc[(f,'Fence'),'voxels'])}복셀"
            for f in FOLDS
        )
        + "이었다. 재보정 결과는 이 96패치의 입력 분포에 조건부이며 다른 보정 영역을 뽑았을 때의 변동성은 측정하지 않았다."
    ]
    for fold in FOLDS:
        g = bn[(bn.fold == fold) & (bn["mode"] == "INT8")].pivot(
            index="seed", columns="bn", values="audit_r30"
        )
        es = int8_early[int8_early.fold == fold]
        notes.append(
            f"{fold}에서 동일한 INT8 학습 파라미터에 보정용 영역의 BN 통계를 적용했을 때 최종 평가 변화는 "
            f"{pm(g.recalibrated-g['aggregate'])}%p였다. 같은 분할에서 기본 INT8의 조기 중단은 평균 "
            f"{es.early_session_bytes.mean()/1e6:.3f} MB를 사용했고 30R 대비 최종 평가 차이는 "
            f"{pm(es.delta_vs_r30_pp,1)}%p였다."
        )
    notes.append(
        "기존 연구는 장면 12 전체를 평가한 이력이 있다. reference 최종 평가 패치는 이번 독립 모델의 학습과 라운드 선택에서 분리했지만 연구 전체에서 처음 공개되는 자료는 아니다. rotated의 장면 3 역시 과거 모델의 클라이언트 학습 장면이었다. 새 분할의 모델은 이 과거 가중치를 불러오지 않았다."
    )
    support = pd.read_csv(OUT / "class_support_sensitivity.csv")
    comparison = []
    for f, g in support[support["round"] == 30].groupby("fold"):
        p = g.pivot(index="seed", columns="mode", values="five_class_miou")
        comparison.append(f"{f} {pm(p.INT8-p.FP32)}%p")
    notes.append(
        "기본 mIoU는 정답·예측 모두에 없는 클래스를 평균에서 제외한다. reference의 최종 평가 패치에는 Fence 정답이 0개여서 분할 간 절대 점수를 직접 비교하기 어렵다. 양쪽 모두 정답이 존재하는 동일 다섯 클래스(Ground~Light pole)로 30R mIoU를 다시 집계한 INT8 − FP32 차이는 "
        + ", ".join(comparison)
        + "였다. 이 민감도 분석은 중단 결정에 사용하지 않았다. [클래스 지원 수와 민감도](class_support_sensitivity.csv)"
    )
    body[body.index("## 해석 범위와 재현") + 1 : body.index("## 해석 범위와 재현") + 1] = notes
    (OUT / "PHASE4_REPORT.md").write_text("\n\n".join(body) + "\n")


if __name__ == "__main__":
    main()
