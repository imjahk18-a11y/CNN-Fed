"""Summarize the measured synthetic CUDA runs."""
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def report(out,central,rounds,milestones,fl_epoch):
    env=json.loads((out/"gpu_environment.json").read_text())
    curves = [r for r in central if r["ft_epochs"] in {0,fl_epoch}]
    chosen = [r for r in curves if r["ft_epochs"]==fl_epoch or r["pruning_pct"]==0]
    last = [r for r in rounds if r["final_test_miou"] is not None]
    dest = out/"plots"
    dest.mkdir()
    fig,axes = plt.subplots(2,3,figsize=(12,7))
    for ax,(key,label) in zip(axes.flat,[("parameters","Parameters"),("macs","Conv3d MACs/patch"),
                                         ("model_mb","State size (MB)"),("latency_p50_ms","Latency p50 (ms)"),
                                         ("latency_p95_ms","Latency p95 (ms)"),("peak_gpu_mb","Inference peak (MB)")]):
        ax.bar([str(r["pruning_pct"])+"%" for r in chosen],[r[key] for r in chosen])
        ax.set_title(label)
    fig.suptitle("Synthetic / RTX GPU / FP32 / batch 1")
    fig.tight_layout()
    fig.savefig(dest/"gpu_model_comparison.png",dpi=150)
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    for ax,(key,label) in zip(axes.flat,[("miou","Test mIoU"),("macro_f1","Test macro F1"),
                                         ("ft_time_s","Cumulative training (s)"),("peak_training_mb","Training peak (MB)")]):
        for pct in (0,20,40):
            rows=[r for r in central if r["pruning_pct"]==pct and r[key] is not None]
            if rows:
                ax.plot([r["ft_epochs"] for r in rows],[r[key] for r in rows],marker="o",label=f"{pct}%")
        ax.set(xlabel="Fine-tuning epochs",ylabel=label)
        ax.legend()
    fig.tight_layout()
    fig.savefig(dest/"gpu_recovery.png",dpi=150)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for pct in (0,20,40):
        for case in ("IID","Non-IID"):
            rows=[r for r in rounds if r["pruning_pct"]==pct and r["partition"]==case]
            for ax,key,label in zip(axes,["miou","cumulative_mb"],["Validation mIoU","Cumulative payload (MB)"]):
                ax.plot([r["round"] for r in rows],[r[key] for r in rows],label=f"{pct}% {case}")
                ax.set(xlabel="Round",ylabel=label)
    for ax in axes:
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(dest/"gpu_fl.png",dpi=150)
    plt.close(fig)
    with (out/"comparison_telemetry.csv").open() as f:
        util=[int(r["utilization_pct"]) for r in csv.DictReader(f)]
    lines=["# RTX A6000 합성 분할 실험", "",
           f"{env['gpu_name']} · PyTorch {env['torch']} · CUDA {env['torch_cuda']} · cuda:0. "
           "FP32, seed 42, batch 3. 합성 train 25개 중 중앙 13개와 로컬 12개를 분리했고 val 5개, test 7개를 고정했다.",
           "", "2 epoch GPU 사전학습 smoke 성공 후, CPU 비교에 사용했던 최적 E1 체크포인트를 다시 읽어 "
           "원본 가중치와 공간 분할을 맞췄다. 미세조정은 동일 optimizer로 연속 학습했다. "
           f"FL 시작점은 사전 지정한 {fl_epoch} epoch이다.",
           "", "|채널 제거|FT epoch|test mIoU|Macro F1|Parameters|MACs|p50/p95 ms|FT 누적 s|",
           "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in central:
        lines.append(f"|{r['pruning_pct']}%|{r['ft_epochs']}|{r['miou']:.4f}|{r['macro_f1']:.4f}|"
                     f"{r['parameters']}|{r['macs']}|{r['latency_p50_ms']:.3f}/{r['latency_p95_ms']:.3f}|{r['ft_time_s']:.3f}|")
    lines += ["","지연시간은 각 구조별 단일 장치 내 패치를 3회 예열 후 100회 동기화하여 측정했다. 같은 구조의 epoch 행은 "
              "이 구조별 측정값을 공유한다. 최종 FL 모델의 추론은 별도로 실측했다. "
              "GPU peak는 PyTorch가 프로세스에 할당한 바이트를 십진 MB로 표현한다. "
              f"nvidia-smi의 장치 전체 사용률 표본 범위는 {min(util)}~{max(util)}%이다.",
              "", "|FL 모델|분할|최종 test mIoU|Macro F1|10라운드 통신 MB|", "|---|---|---:|---:|---:|"]
    for r in last:
        lines.append(f"|{r['pruning_pct']}%|{r['partition']}|{r['final_test_miou']:.4f}|{r['final_test_macro_f1']:.4f}|{r['cumulative_mb']:.6f}|")
    lines += ["","라운드 CSV의 miou와 macro_f1은 validation 값이며 마지막 행의 final_test 열에 test 측정값을 기록했다. "
              "통신량은 서버와 두 순차 CUDA 클라이언트 사이에서 왕복한 모델 텐서 크기다. 서버 집계는 CPU로 수행했다.",
              "", "## 40% 프루닝 진단", "",
              "원본 너비 8/16/8을 5/10/5로 줄였다. 과거 CPU 기록에서 프루닝 직후와 FT 1 epoch의 두 번째 클래스 IoU가 모두 0이었다. "
              "아래 값은 동일 가중치·동일 validation 데이터에서 한 계층씩 제거하거나 회복 방법을 바꾼 결과다.",
              "", "|조건|너비|Epoch|LR|validation mIoU|", "|---|---|---:|---:|---:|"]
    with (out/"gpu_pruning_diagnostics.csv").open() as f:
        for r in csv.DictReader(f):
            lines.append(f"|{r['condition']}|{r['widths']}|{r['epochs']}|{r['lr']}|{float(r['miou']):.4f}|")
    lines+=["","점진적 조건은 20% 모델에서 절반의 epoch를 학습한 후 40% 목표 너비로 다시 줄이고 optimizer를 초기화했다. "
            "원샷과 총 학습 epoch는 같다. 이 비교는 1개 seed와 작은 합성 데이터로 측정했다. "
            "실제 STPLS3D의 회복률과 탑재 지연은 별도 검증이 필요하다.", ""]
    (out/"gpu_experiment_report.md").write_text("\n".join(lines))
