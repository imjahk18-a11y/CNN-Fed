"""Render measured selection and external-test results. No training or model selection."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from datasets.s2r_scenes import OUT,read,table


def records(path):
    with path.open() as f: return list(csv.DictReader(f))


def label(row):
    return row["id"] if row["partition"]=="central" else row["id"]+" "+row["partition"]


def figures(out,main,rounds,classes):
    plots=out/"plots"
    plots.mkdir(exist_ok=True)
    names=[label(r) for r in main]
    x=np.arange(len(main))
    colors=("#23648f","#ba6139","#38795f")
    def finish(fig,file):
        fig.tight_layout()
        fig.savefig(plots/file,dpi=170,bbox_inches="tight")
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4.2))
    for j,key in enumerate(("synthetic_val_miou","wmsc_test_miou")):
        ax.bar(x+(j-.5)*.38,[float(r[key]) for r in main],width=.37,color=colors[j],
               label=("Synthetic validation","WMSC external test")[j])
    ax.set(xticks=x,xticklabels=names,ylabel="Voxel mIoU",ylim=(0,1))
    ax.tick_params(axis="x",labelrotation=25)
    ax.legend()
    finish(fig,"domain_miou.png")
    for file,kx,ky,xlabel,ylabel in (
        ("pruning_wmsc.png","pruning_pct","wmsc_test_miou","Target channel pruning (%)","WMSC mIoU"),
        ("size_wmsc.png","model_mb","wmsc_test_miou","Model state size (decimal MB)","WMSC mIoU"),
        ("gpu_latency.png","pruning_pct","latency_p50_ms","Target channel pruning (%)","RTX A6000 latency (ms)")):
        fig,ax=plt.subplots(figsize=(7,4))
        for i,case in enumerate(("central","IID","Non-IID")):
            group=sorted((r for r in main if r["partition"]==case),key=lambda r:float(r[kx]))
            xx=[float(r[kx]) for r in group]
            yy=[float(r[ky]) for r in group]
            ax.plot(xx,yy,marker=["o","s","^"][i],label=case,color=colors[i])
            if ky=="latency_p50_ms":
                ax.plot(xx,[float(r["latency_p95_ms"]) for r in group],ls="--",color=colors[i],alpha=.7)
        if ky=="latency_p50_ms":ax.set_title("Batch 1 FP32: solid p50, dashed p95")
        ax.set(xlabel=xlabel,ylabel=ylabel)
        ax.legend()
        finish(fig,file)
    fig,ax=plt.subplots(figsize=(8,4.4))
    for i,exp in enumerate(("E4","E5","E6")):
        for case in ("IID","Non-IID"):
            group=[r for r in rounds if r["id"]==exp and r["partition"]==case]
            ax.plot([int(r["round"]) for r in group],[float(r["miou"]) for r in group],
                    color=colors[i],linestyle="-" if case=="IID" else "--",marker=".",label=f"{exp} {case}")
    ax.set(xlabel="FedAvg round",ylabel="Synthetic validation mIoU",xticks=range(1,11))
    ax.legend(ncol=3)
    finish(fig,"fl_rounds.png")
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,split,title in zip(axes,("synthetic_val","wmsc_test"),("Synthetic validation","WMSC external test")):
        a=np.array([[float(r[split+"_iou_"+c] or np.nan) for c in classes] for r in main])
        ax.imshow(a,aspect="auto",vmin=0,vmax=1,cmap="viridis")
        ax.set(yticks=x,yticklabels=names,xticks=range(len(classes)),xticklabels=classes,title=title)
        ax.tick_params(axis="x",labelrotation=30,labelsize=8)
        for (i,j),v in np.ndenumerate(a):
            ax.text(j,i,"—" if np.isnan(v) else f"{v:.2f}",ha="center",va="center",fontsize=8,
                    color="white" if v<.55 else "black")
    finish(fig,"class_iou.png")


def report(out):
    m=read(out/"dataset_manifest.json")
    r=records(out/"synthetic_to_real_comparison.csv")
    rounds=records(out/"federated_results.csv")
    classes=m["classes"]
    main=[a for a in r if a["id"].startswith("E")]
    central=[a for a in main if a["partition"]=="central"]
    by_pct={int(a["pruning_pct"]):a for a in central}
    figures(out,main,rounds,classes)
    support=records(out/"wmsc_class_distribution.csv")
    val_support=[x for x in records(out/"synthetic_class_distribution.csv") if x["split"]=="cached_val"]
    expanded=[]
    for model in r:
        for split,support_rows in (("synthetic_val",val_support),("wmsc_test",support)):
            for c,s in zip(classes,support_rows):
                expanded.append(dict(model=label(model),split=split,semantic_class=c,true_voxels=s["voxels"],
                                     iou=model[split+"_iou_"+c]))
    table(out/"per_class_results.csv",expanded)
    lines=["# STPLS3D Synthetic-to-Real 분할 실험", "",
           "공식 Synthetic_v3의 독립 장면 네 곳에서만 사전학습·미세조정·연합학습·모델 선택을 수행했다. "
           "WMSC는 모든 체크포인트를 확정한 뒤 동일한 외부 평가에 사용했다.", "",
           "## 데이터와 검증", ""]
    acq=read(out/"acquisition.json")
    lines.append(f"공유 ZIP에서 가장 작은 GTv3 네 장면 {sum(s['bytes'] for s in acq['selected'])/1e9:.3f} GB만 선택적으로 받았다. "
                 "v1·v2·v3 배포 목록을 비교했고, 한 버전 안에서 스키마를 고정했다. 다운로드 전 부분 표본의 ID, "
                 "전체 다운로드 후 원시 ID 빈도·CRC32·SHA-256을 각각 기록했다. "
                 "[아카이브·헤더 기록](acquisition.json) · [공식 데이터 정의](https://github.com/meidachen/STPLS3D#2-dataset)")
    lines+=["","|용도|합성 장면|사용 패치|","|---|---|---:|"]
    for role,group in m["scene_assignment"].items():
        lines.append(f"|{role}|{group['scene']}|{len(group['patch_ids'])}|")
    lines+=["",f"0.3 m, 32³, occupancy·패치 바닥 기준 정규화 높이·RGB의 5채널이다. 중앙과 검증은 독립된 장면이다. "
            "각 로컬 장면을 수평으로 나누고 경계 한 열을 제외했다. 같은 로컬 풀을 Non-IID에서는 장면별로, IID 근사에서는 "
            "양 장면의 띠를 섞어 크기·클래스 분포 차이가 작은 조합으로 배정했다. 중앙과 로컬은 클래스별 해시 표집으로, "
            "검증은 균일 해시 표집으로 패치 수를 제한했다. 독립 검증 장면은 하나다.",
            "", "GTv3 PLY의 class 필드와 WMSC 캐시의 원시 ID 변환을 공식 KPConv의 공통 6-class 매핑과 대조했다. "
            "instance 필드는 사용하지 않았다. 직접 읽은 원시 클래스 히스토그램, 양 도메인의 유효 복셀 빈도, 기존 WMSC 캐시 "
            "외부 인덱스는 별도 파일로 보관했다. [공식 변환 코드](https://github.com/meidachen/STPLS3D/blob/main/KPConv-PyTorch/datasets/STPLS3D.py#L573-L613) · [데이터 manifest](dataset_manifest.json)",
            "", "## 학습 및 선택", "",
            "RTX A6000 한 장, CUDA 12.6, PyTorch 2.8.0+cu126, seed 42, batch 8, AMP. 별도 1 epoch smoke 성공 후 무작위 초기화한 "
            "기준 모델을 AdamW 1e-3와 cosine annealing으로 20 epochs 학습했다. 모든 실험은 같은 최적 합성 validation 체크포인트에서 유래한다. "
            f"채택 epoch는 {central[0]['pretrain_checkpoint_epoch']}이다. 기존 WMSC 학습 가중치는 로드하지 않았다.",
            "", "각 비율에서 AdamW LR 1e-4/1e-3 × 5/10/20 epochs를 비교했다. 한 LR 안에서 optimizer를 유지했으며 "
            "선택은 합성 validation mIoU 최대, 동률이면 짧은 학습과 낮은 LR 순이다. 40% 직후 validation 하락이 0.05 이상이면 "
            "단계적 20→40% 후보도 같은 총 epoch로 비교했다. 이 후보는 두 번째 축소 후 optimizer를 초기화한다. "
            "원본에도 같은 추가 epoch·LR을 적용하여 보조 대조군을 만들었다.",
            "", "FedAvg는 모든 비교에서 10 rounds, 1 local epoch, LR 1e-4로 고정해 두 클라이언트를 순차 학습했다. "
            "샘플 수로 파라미터와 BN 평균·분산을 가중평균하고 BN batch counter는 최댓값을 쓴다. "
            "WMSC 평가 시작 전에 최종 모델 경로·해시·학습량을 고정했다. [선택 기록](pruning_results.csv) · [최종 모델 목록](frozen_models.json)",
            "", "## 최종 평가", "",
            "|모델|FT epoch / LR|합성 val mIoU|WMSC mIoU|WMSC − val (%p)|WMSC F1|WMSC Accuracy|GPU p50/p95 ms|통신 MB|",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for a in r:
        v,w=map(float,(a["synthetic_val_miou"],a["wmsc_test_miou"]))
        lines.append(f"|{label(a)}|{a['ft_epochs']} / {a['ft_lr']}|{v:.4f}|{w:.4f}|{100*(w-v):+.2f}|"
                     f"{float(a['wmsc_test_macro_f1']):.4f}|{float(a['wmsc_test_accuracy']):.4f}|"
                     f"{float(a['latency_p50_ms']):.3f}/{float(a['latency_p95_ms']):.3f}|{float(a['communication_mb']):.4f}|")
    lines+=["","## 프루닝 효과와 학습량", "",
            "|채널 축소|선택 방식|Parameters|MACs/patch|모델 MB|WMSC E1 대비 변화 (%p)|동일 추가 학습 원본 대비 (%p)|",
            "|---:|---|---:|---:|---:|---:|---:|"]
    base=by_pct[0]
    for pct,a in by_pct.items():
        w=float(a["wmsc_test_miou"])
        control=next((x for x in r if x["id"]==f"control_for_{pct}"),base)
        lines.append(f"|{pct}%|{a['method']}|{a['parameters']}|{a['macs']}|{float(a['model_mb']):.6f}|"
                     f"{100*(w-float(base['wmsc_test_miou'])):+.2f}|{100*(w-float(control['wmsc_test_miou'])):+.2f}|")
    for pct in (20,40):
        a=by_pct[pct]
        size=1-float(a['model_mb'])/float(base['model_mb'])
        lat=100*(float(a['latency_p50_ms'])/float(base['latency_p50_ms'])-1)
        lines+=["",f"{pct}% 모델에서 state_dict 크기와 동일 라운드 통신 비용은 원본 대비 {100*size:.1f}% 감소했다. "
                   f"중앙 모델 p50 지연 변화는 {lat:+.1f}%였다. 지연시간은 동일한 장치 내 batch 1 FP32 패치를 "
                   "예열 후 CUDA 동기화해 100회 측정한 값이다."]
    lines+=["", "## FL이 WMSC 성능에 미친 변화", "",
            "|초기 모델|분할|10라운드 후 시작점 대비 WMSC mIoU 변화 (%p)|","|---|---|---:|"]
    for a in main:
        if a["partition"]=="central":continue
        initial=by_pct[int(a["pruning_pct"])]
        diff=float(a["wmsc_test_miou"])-float(initial["wmsc_test_miou"])
        lines.append(f"|{initial['id']}|{a['partition']}|{100*diff:+.2f}|")
    lines+=["","통신량은 전체 모델을 두 클라이언트에 배포하고 수신한 텐서 크기의 합이다. 실제 무선 전송을 측정하지 않았다. "
            "FL과 시작점 차이에는 추가 로컬 학습 효과도 포함된다. 훈련 peak는 해당 단계의 CUDA 학습 로그, 추론 peak는 "
            "최종 체크포인트 단일 패치 프로파일이다. 상세 지표에는 도메인별 Macro F1·Accuracy·클래스별 IoU도 있다.",
            "", "## 클래스별 WMSC IoU", "", "|클래스|WMSC 정답 복셀|"+"|".join(label(a) for a in main)+"|",
            "|---|---:|"+"---:|"*len(main)]
    for c,s in zip(classes,support):
        lines.append(f"|{c}|{s['voxels']}|"+"|".join(f"{float(a['wmsc_test_iou_'+c]):.4f}" if a['wmsc_test_iou_'+c] else "—" for a in main)+"|")
    lines+=["","양쪽 도메인의 고정 split에서 유효 복셀 전체의 혼동행렬을 누적했다. 빈 복셀은 제외한다. "
            "정답·예측 양쪽에서 비어 있는 클래스 IoU는 미정의이며 mIoU에서 제외한다. 정답이 없어도 예측이 있으면 IoU는 0이다. "
            "WMSC 외부 인덱스에는 기존 캐시의 유효 패치 1,683개를 모두 포함했다. 원래 캐시 생성 때 제외된 공간 완충열은 포함되지 않는다. "
            "단일 WMSC와 단일 seed의 결과이므로 다른 지역이나 실기체의 일반화 성능과 변동성은 측정 전이다.",
            "", "[전체 지표](synthetic_to_real_comparison.csv) · [양 도메인의 클래스별 정답 복셀과 IoU](per_class_results.csv) · "
            "[FL 라운드 기록](federated_results.csv) · [그래프](plots/domain_miou.png)"]
    base_score=float(base["wmsc_test_miou"])
    match20=next(x for x in r if x["id"]=="control_for_20")
    diff20=100*(float(by_pct[20]["wmsc_test_miou"])-float(match20["wmsc_test_miou"]))
    gaps=[100*(float(x["wmsc_test_miou"])-float(by_pct[int(x["pruning_pct"])]["wmsc_test_miou"]))
          for x in main if x["partition"]!="central"]
    lines[2:2]=[f"기준 모델의 합성 검증 mIoU는 {float(base['synthetic_val_miou']):.4f}, WMSC 외부 mIoU는 {base_score:.4f}였다. "
                f"20% 모델은 같은 LR·epoch로 더 학습한 원본과 WMSC에서 {diff20:+.2f}%p 차이를 보였다. "
                f"FL 후 여섯 조건의 WMSC 변화 범위는 시작점 대비 {min(gaps):+.2f}~{max(gaps):+.2f}%p였다.", ""]
    lines.extend(["", "## 그래프", "",
                  "[도메인별 mIoU](plots/domain_miou.png) · [프루닝 비율과 WMSC 성능](plots/pruning_wmsc.png) · "
                  "[라운드별 합성 검증 성능](plots/fl_rounds.png) · [모델 크기와 WMSC 성능](plots/size_wmsc.png) · "
                  "[A6000 추론 지연](plots/gpu_latency.png) · [클래스별 IoU](plots/class_iou.png)"])
    (out/"final_report.md").write_text("\n".join(lines)+"\n")


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out",type=Path,default=OUT)
    report(ap.parse_args().out)
