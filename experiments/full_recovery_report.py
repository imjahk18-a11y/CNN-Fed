"""Publish the verified corpus inventory. Generate a split only for all 63 scenes."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import fcntl

import numpy as np

from datasets.compact_cache import atomic_json
from datasets.fullz import CLASSES, common
from datasets.recover_synthetic import CACHE, JOURNAL, OUT, space
from datasets.recovery_cache import Frames, verify_cache


def csv_table(path,rows):
    if not rows:return
    temp=path.with_suffix(".writing")
    with temp.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    temp.replace(path)


def aggregate(ids,metas):
    mm=[metas[k] for k in ids]
    patches=sum(x["patches"] for x in mm)
    hist=np.sum([m["voxel_hist"] for m in mm],axis=0,dtype=np.int64) if mm else np.zeros(6,dtype=np.int64)
    p_hist=np.sum([m["class_patch_hist"] for m in mm],axis=0,dtype=np.int64) if mm else np.zeros(6,dtype=np.int64)
    return dict(scenes=sorted(ids),patches=patches,voxels=int(hist.sum()),voxel_hist=hist.tolist(),class_patch_hist=p_hist.tolist())


def make_split(metas,records):
    # Group differently rendered versions of a numbered/named site and exact source duplicates.
    parent={k:k for k in metas}
    def root(k):
        while parent[k]!=k:k=parent[k]
        return k
    def merge(a,b):
        parent[root(b)]=root(a)
    first={}
    # Match same-content, same-origin occupied patches using the actual lossless frame payload.
    spatial={}
    common_patches=0
    for k,m in metas.items():
        site=Path(m["member"]).stem.lower().split("_points_gt")[0]
        for tag in (("site",site),("source",m["source_sha256"])):
            if tag in first:merge(k,first[tag])
            else:first[tag]=k
        for s in m["shards"]:
            with (CACHE/k/s["file"]).open("rb") as f:
                for p in s["patches"]:
                    f.seek(p["offset"]);frame=f.read(p["size"])
                    tag=(tuple(m["origin"]),m["shape"][2],p["x"],p["y"],hashlib.sha256(frame).digest())
                    if tag in spatial:
                        merge(k,spatial[tag]);common_patches+=1
                    else:spatial[tag]=k
    groups=defaultdict(list)
    for k in metas:groups[root(k)].append(k)
    order=list(sorted(groups))
    rng=np.random.default_rng(42);rng.shuffle(order)
    # Subset-sum over whole site groups; fixed tie breaking, independent of labels.
    total=sum(m["patches"] for m in metas.values())
    target=round(total*37837/(145329+37837))
    paths={0:()}
    for g in order:
        n=sum(metas[k]["patches"] for k in groups[g])
        for v,seq in list(paths.items()):
            paths.setdefault(v+n,seq+(g,))
    chosen=min((n for n in paths if 0<n<total),key=lambda n:(abs(n-target),n))
    val={k for g in paths[chosen] for k in groups[g]}
    train=set(metas)-val
    assert train and val and train.isdisjoint(val) and len(train|val)==63
    assert {root(k) for k in train}.isdisjoint(root(k) for k in val)
    return dict(status="ready",seed=42,classes=CLASSES,train=aggregate(train,metas),validation=aggregate(val,metas),
                target_train=145329,target_validation=37837,train_patch_difference=total-chosen-145329,
                validation_patch_difference=chosen-37837,site_groups=dict(groups),identical_spatial_patches_grouped=common_patches,
                scene_overlap=0,site_group_overlap=0,wmsc_included=False,
                spatial_scope="Within each scene all XY tiles stay together. Group same site identifier, source digest, or identical geolocated lossless patch; external CRS/site correspondence unavailable.",
                method="new fixed whole-site grouped split; closest historical validation fraction, no label-based tuning",
                index_hashes={k:records[k]["index_sha256"] for k in metas})


def publish():
    state=json.loads(JOURNAL.read_text());records=state["scenes"]
    metas={};rows=[];classes=[]
    for k,r in records.items():
        if r["status"] not in ("cache_verified","raw_removed"):continue
        verify_cache(CACHE/k,r,deep=False)
        m=Frames(CACHE/k).meta;metas[k]=m
        point_hist=np.array(m["raw_point_hist"]);ids=np.flatnonzero(point_hist);c_hist=np.zeros(6,np.int64)
        np.add.at(c_hist,common(ids),point_hist[ids])
        rows.append(dict(scene=k,version=m["version"],points=m["points"],voxels=m["voxels"],patches=m["patches"],
                         xmin=m["xyz_min"][0],xmax=m["xyz_max"][0],ymin=m["xyz_min"][1],ymax=m["xyz_max"][1],
                         zmin=m["xyz_min"][2],zmax=m["xyz_max"][2],z_cells=m["shape"][2],source_bytes=r["bytes"],
                         zstd_bytes=m["cache_bytes"],rgb_mean=json.dumps((np.array(m["rgb_sum"])/m["voxels"]).tolist()),
                         rgb_min=json.dumps(m["rgb_min"]),rgb_max=json.dumps(m["rgb_max"]),
                         raw_point_hist=json.dumps({int(i):int(point_hist[i]) for i in ids})))
        for i,c in enumerate(CLASSES):
            classes.append(dict(scene=k,version=m["version"],class_name=c,points=int(c_hist[i]),
                                voxels=m["voxel_hist"][i],patches_with_class=m["class_patch_hist"][i]))
    csv_table(OUT/"scene_statistics.csv",rows);csv_table(OUT/"class_distribution.csv",classes)
    split=make_split(metas,records) if len(metas)==63 else dict(status="not_ready",verified_scenes=len(metas),required_scenes=63,train=None,validation=None,
               historical_train_patches=145329,historical_validation_patches=37837)
    atomic_json(OUT/"train_val_split.json",split)
    done=[r for r in records.values() if r["id"] in metas]
    fresh=[r for r in done if not r["existing"]]
    old=[r for r in done if r["existing"]]
    failed={k:r["error"] for k,r in records.items() if r["status"]=="failed"}
    pending=[k for k,r in records.items() if r["status"]=="pending"]
    after=space()
    atomic_json(OUT/"download_integrity.json",dict(targets=63,new_verified=[r["member"] for r in fresh],failed=failed,pending=pending,
        reused=[r["member"] for r in old],raw_removed=[r["member"] for r in done if r["raw_removed"]],
        verified={r["id"]:{key:r[key] for key in ("bytes","crc32","sha256","index_sha256","audit","raw_removed")} for r in done},
        quota_before_bytes=state["quota_before"],quota_after_bytes=after))
    payload=sum(m["cache_bytes"] for m in metas.values())
    exact=payload+sum((CACHE/k/"index.json").stat().st_size for k in metas)
    versions={str(v):aggregate([k for k,m in metas.items() if m["version"]==v],metas) for v in (1,2,3)}
    tot=aggregate(metas,metas)
    atomic_json(OUT/"cache_statistics.json",dict(verified=len(metas),required=63,total=tot,versions=versions,cache_bytes_including_indexes=exact,
        zstd_frame_bytes=payload,points=sum(m["points"] for m in metas.values()),
        rgb_mean=(np.sum([m["rgb_sum"] for m in metas.values()],axis=0)/tot["voxels"]).tolist() if tot["voxels"] else None,
        format="fullz-rgb16-frames-v1",reserve_bytes=8*2**30,quota_before_bytes=state["quota_before"],quota_after_bytes=after))
    gpu_path=OUT/"gpu_dataloader_smoke.json"
    if split["status"]!="ready":atomic_json(gpu_path,dict(status="not_run_full_corpus",verified=len(metas),required=63))
    gpu=json.loads(gpu_path.read_text()) if gpu_path.exists() else {}
    lines=["# STPLS3D 장면별 다운로드와 Full-Z 캐시 복원",
           f"**63개 장면 중 {len(metas)}개 압축 캐시 검증 완료.** 신규 다운로드 검증 {len(fresh)}개, 기존 PLY 재사용 {len(old)}개, 시도 후 실패 {len(failed)}개, 대기 {len(pending)}개.",
           "## 확보 범위", "|버전|검증 장면 / 대상|패치|점유 복셀|\n|---|---:|---:|---:|\n"+"\n".join(f"|v{v}|{len(group['scenes'])} / {limit}|{group['patches']:,}|{group['voxels']:,}|" for (v,group),limit in zip(versions.items(),(14,24,25))),
           f"전체 검증분은 {tot['patches']:,}패치, {tot['voxels']:,}개 점유 복셀이다. 압축 프레임과 인덱스 합계는 {exact:,}바이트({exact/2**20:.2f} MiB)다. 사용자 quota 잔여는 시작 {state['quota_before']/2**30:.2f} GiB, 보고 시점 {after/2**30:.2f} GiB다.",
           "좌표와 색상은 무손실 Zstandard 프레임으로 저장했다. 각 XY 패치는 시작 좌표·장면 원점·Full-Z 높이를 갖는다. 원본의 raw 다수결, RGB-16 최빈 결합 bin 평균, Common 6을 보존한다. 매 장면에서 모든 프레임을 디코드해 좌표·범위·빈도·체크섬을 검증하고, 최대 96개 복셀을 원본 점의 독립 투표·평균과 대조했다.",
           "[성공·실패·대기 파일별 목록](download_manifest.csv) · [CRC·SHA·삭제 파일 목록](download_integrity.json) · [단계별 상태](stream_state.json) · [장면별 XYZ·RGB 통계](scene_statistics.csv) · [클래스별 점·복셀·패치 수](class_distribution.csv)"]
    if failed:lines+=["## 중단 원인",*["- "+k+" — "+v for k,v in failed.items()]]
    lines+=["## 학습 데이터 분할과 GPU 검증"]
    if split["status"]=="ready":
        lines.append(f"새로운 고정 장면 그룹 split은 train {split['train']['patches']:,}, validation {split['validation']['patches']:,}패치다. 과거 145,329 / 37,837 기준 차이는 각각 {split['train_patch_difference']:+,} / {split['validation_patch_difference']:+,}패치다. 시드 42, 장면·그룹 공통 원소 0개. 같은 장면 또는 확인된 동일 공간 패치는 한쪽에만 배정했다. 이름이 다른 익명 합성 장면 간 실제 공간 동일성은 별도의 제작 지리 메타데이터 없이는 확정할 수 없다.")
    else:lines.append("전체 63개가 완성될 때 장면 그룹 단위 split을 확정한다. 현재 train/validation 패치 수 및 과거 145,329 / 37,837과의 차이는 미측정이다.")
    lines.append("전체 GPU 결과: "+str(gpu.get("status","대기")))
    if gpu.get("status")=="passed":
        for ch,t in gpu["modes"].items():lines.append(f"- {ch}채널 batch 24, {t['checked_scenes']}개 장면, forward/backward {t['forward_backward_steps']}회, 최대 {t['peak_allocated_mib']:.1f} MiB, {t['end_to_end_patches_per_s']:.1f} patches/s.")
    pilot=OUT/"gpu_dataloader_pilot.json"
    if pilot.exists():
        pilot_info=json.loads(pilot.read_text())
        lines.append(f"별도 파일럿은 {pilot_info['modes']['3']['checked_scenes']}개 장면에 대해 RGB와 RGB+Local XYZ를 검사했다. [파일럿 실측](gpu_dataloader_pilot.json)")
    lines+=["[Split](train_val_split.json) · [전체 GPU 결과](gpu_dataloader_smoke.json)",
            "## 재현 명령","프로젝트 루트에서 다운로드를 재개한다. 동일한 공식 ZIP이 로컬 디스크에 있으면 첫 명령에 `--archive /path/to/STPLS3D.zip`을 추가할 수 있다.",
            "```bash\n.venv-gpu/bin/python -m datasets.recover_synthetic\n.venv-gpu/bin/python -m experiments.full_recovery_report\n.venv-gpu/bin/python -m experiments.full_recovery_train smoke\n.venv-gpu/bin/python -m experiments.full_recovery_report\n```",
            "전체 캐시·split과 GPU smoke 검증이 끝난 후 35 epoch RGB 학습을 실행하는 명령이다. 이번 데이터 복원에서는 실행하지 않았다. 6채널은 `--channels 6`과 별도 출력 경로를 사용한다.",
            "```bash\n.venv-gpu/bin/python -m experiments.full_recovery_train train --channels 3 --epochs 35 --out results/fullz_63_rgb35\n```"]
    dest=OUT/"DATA_RECOVERY_REPORT.md";tmp=dest.with_suffix(".writing");tmp.write_text("\n\n".join(lines)+"\n");tmp.replace(dest)


if __name__=="__main__":
    with (OUT/".stream.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);publish()
