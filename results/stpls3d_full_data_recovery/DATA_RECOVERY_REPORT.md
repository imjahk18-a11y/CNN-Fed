# STPLS3D 장면별 다운로드와 Full-Z 캐시 복원

**63개 장면 중 5개 압축 캐시 검증 완료.** 신규 다운로드 검증 1개, 기존 PLY 재사용 4개, 시도 후 실패 1개, 대기 57개.

## 확보 범위

|버전|검증 장면 / 대상|패치|점유 복셀|
|---|---:|---:|---:|
|v1|1 / 14|3,600|4,441,128|
|v2|0 / 24|0|0|
|v3|4 / 25|10,816|16,958,667|

전체 검증분은 14,416패치, 21,399,795개 점유 복셀이다. 압축 프레임과 인덱스 합계는 175,503,659바이트(167.37 MiB)다. 사용자 quota 잔여는 시작 38.23 GiB, 보고 시점 38.06 GiB다.

좌표와 색상은 무손실 Zstandard 프레임으로 저장했다. 각 XY 패치는 시작 좌표·장면 원점·Full-Z 높이를 갖는다. 원본의 raw 다수결, RGB-16 최빈 결합 bin 평균, Common 6을 보존한다. 매 장면에서 모든 프레임을 디코드해 좌표·범위·빈도·체크섬을 검증하고, 최대 96개 복셀을 원본 점의 독립 투표·평균과 대조했다.

[성공·실패·대기 파일별 목록](download_manifest.csv) · [CRC·SHA·삭제 파일 목록](download_integrity.json) · [단계별 상태](stream_state.json) · [장면별 XYZ·RGB 통계](scene_statistics.csv) · [클래스별 점·복셀·패치 수](class_distribution.csv)

## 중단 원인

- v2_5_points_GTv2 — RuntimeError: Drive download quota exceeded at member byte 0; source file retained if partial

## 학습 데이터 분할과 GPU 검증

전체 63개가 완성될 때 장면 그룹 단위 split을 확정한다. 현재 train/validation 패치 수 및 과거 145,329 / 37,837과의 차이는 미측정이다.

전체 GPU 결과: not_run_full_corpus

별도 파일럿은 5개 장면에 대해 RGB와 RGB+Local XYZ를 검사했다. [파일럿 실측](gpu_dataloader_pilot.json)

[Split](train_val_split.json) · [전체 GPU 결과](gpu_dataloader_smoke.json)

## 재현 명령

프로젝트 루트에서 다운로드를 재개한다. 동일한 공식 ZIP이 로컬 디스크에 있으면 첫 명령에 `--archive /path/to/STPLS3D.zip`을 추가할 수 있다.

```bash
.venv-gpu/bin/python -m datasets.recover_synthetic
.venv-gpu/bin/python -m experiments.full_recovery_report
.venv-gpu/bin/python -m experiments.full_recovery_train smoke
.venv-gpu/bin/python -m experiments.full_recovery_report
```

전체 캐시·split과 GPU smoke 검증이 끝난 후 35 epoch RGB 학습을 실행하는 명령이다. 이번 데이터 복원에서는 실행하지 않았다. 6채널은 `--channels 6`과 별도 출력 경로를 사용한다.

```bash
.venv-gpu/bin/python -m experiments.full_recovery_train train --channels 3 --epochs 35 --out results/fullz_63_rgb35
```
