# CNN-Fed

Lightweight voxel segmentation for two simulated UAV clients

STPLS3D의 점군에서 0.3 m 복셀과 32³ 패치를 만들고 작은 3D U-Net을 학습한다. 예측 단위는 레이블이 있는 복셀이다. 실험은 E1 원본, E2/E3 20%·40% 중앙 프루닝 및 미세조정 모델, E4/E5/E6 각 모델로 시작한 두 클라이언트 FedAvg이다. 세 FL 조건마다 IID와 지역 기반 Non-IID를 각각 실행한다.

## Native Sparse 3D CNN

Full-Z 실험은 원본 PLY에서 장면 단위로 0.3 m 복셀, RGB-16 대표색, raw 다수결 레이블을 만들고 XY만 32×32로 나눈다. PyTorch에서 점유 좌표 쌍에 sparse convolution을 적용한다. 기존 Dense 실험과 별도 모듈·체크포인트를 사용한다. [실험 보고서와 재실행 명령](results/sparse_3dcnn_reproduction/run02/report.md)

7점과 27점 sparse convolution을 같은 35 epoch로 비교한 결과와 Ground 오분류 사례는 [Full-Z 성능 진단](results/sparse_diagnosis/run01/report.md)에 있다.

## 구조적 프루닝과 두 클라이언트 연합학습

고정된 v3 장면 분할에서 중앙집중식, Local-only, FedAvg, FedBN을 비교한다. 시드별 학습량, 실제 TCP 직렬화 비용과 재실행 명령은 [분산학습 실험 보고서](results/distributed_ai/DISTRIBUTED_AI_REPORT.md)에 있다.

같은 분할에서 FP32·INT8 양방향 전송과 프루닝을 조합한 실측 바이트, 통신 예산별 mIoU, 연결 제약별 완료시간은 [통신 압축 후속 실험](results/distributed_ai_phase2/COMMUNICATION_REPORT.md)에 정리했다.

원본 모델의 FP16·INT8 업데이트 분석, FP32 크기 대조, 온라인 조기 중단 결과와 독립 서버·클라이언트 실행 명령은 [Phase 3 보고서](results/distributed_ai_phase3/PHASE3_REPORT.md)에 있다.

장면 역할을 바꾼 두 분할에서 독립 사전학습, BN 처리, 조기 중단을 검증한 결과와 재실행 명령은 [Phase 4 보고서](results/distributed_ai_phase4/PHASE4_REPORT.md)에 있다.

## 검증된 실행

CPU에서 합성 분할 패치로 E1~E6을 실행했다. 합성 2개 클래스의 손실, 프루닝 전후 연산량, 2라운드 통신량 및 실행 범위를 [합성 검증 기록](results/synthetic_smoke_seed42/SMOKE_REPORT.md)에 정리했다. 이 CPU 실행 이후 RTX A6000에서 별도의 합성 CUDA 실험을 수행했다. 이후 실제 STPLS3D WMSC 한 지역의 저용량 캐시와 A6000 1-epoch pilot을 검증했다. 실기체 측정은 남아 있다.

## 준비와 최소 실행

Python 3.11 이상 환경에서 프로젝트 루트를 작업 디렉터리로 사용한다. CPU 검증 환경은 Python 3.11.3, PyTorch 2.0.1, NumPy 1.24.3, Matplotlib 3.7.1이다. GPU 서버에서는 드라이버가 지원하는 CUDA용 PyTorch를 먼저 설치한다. [PyTorch 설치 안내](https://pytorch.org/get-started/locally/)

```bash
cd /home/gpu_01/voxel_uav_project
python3 -m pip install -r requirements.txt
python3 -m pytest -q
python3 -m experiments.run_experiments smoke --device cpu --out results/smoke_repeat
```

각 실행과 전처리는 새 출력 디렉터리를 사용한다. CPU 스레드는 기본 2개이며 `--cpu-threads`로 조절한다. smoke 모드는 고정된 32³ 합성 패치, 사전학습 4 epochs, 미세조정 1 epoch, FL 2 rounds를 사용한다.

## 실제 데이터 및 라벨

공식 Semantic Segmentation 배포의 독립된 실측 지역별 PLY를 준비한다. 예를 들어 `data/STPLS3D/RealWorldData/WMSC_points.ply`를 사용할 수 있다. 같은 지리 영역을 나눈 파일들은 하나의 지역 PLY로 합쳐 입력한다. 소스별 XYZ는 미터 단위의 직교좌표를 사용한다. [공식 데이터 다운로드와 디렉터리](https://github.com/meidachen/STPLS3D#2-dataset)

기본 `--schema raw`는 공식 KPConv의 원시 `class` 변환 규칙을 사용한다. 학습 ID는 0 ground, 1 building, 2 vegetation, 3 cars, 4 lightStreetSigns, 5 fences이다. 이미 이 순서의 평가용 ID로 변환한 PLY는 `--schema benchmark`로 읽는다. 일부 ID만 등장한 파일은 값의 범위만으로 두 체계를 판별하기 어려우므로 배포본의 전처리 이력을 확인한다. **원시 ID의 범위만으로 스키마를 자동 선택하지 않는다.** 변환표 밖의 ID는 오류로 보고한다. 파일의 무효 레이블이 문서화되어 있으면 예를 들어 `--ignore-raw -100`으로 지정한다. manifest에 변환표, 출처, 원시 ID 빈도, SHA-256, 파일별 원점과 패치 위치가 저장된다. [공식 클래스 순서와 필드 변환 코드](https://github.com/meidachen/STPLS3D/blob/main/KPConv-PyTorch/datasets/STPLS3D.py#L573-L613)

```bash
python3 -m datasets.stpls3d_loader data/STPLS3D/RealWorldData/*.ply \
  --schema raw --out data/realworld_030m_32
```

리더는 첫 번째 PLY element인 vertex의 스칼라 property를 읽는다. ASCII와 양쪽 endian의 이진 PLY를 지원한다. XYZ, 의미 레이블 속성(기본 `class`, 변경 시 `--label-field`)이 필요하고, Variant B에는 `red/green/blue` 0–255가 필요하다. 장면을 하나씩 읽고 점을 패치 ID로 정렬하므로 전처리에는 가장 큰 장면의 좌표·정렬 배열을 담을 RAM이 필요하다. 학습은 디스크의 패치별 희소 NPZ를 읽어 배치 단위로 dense 텐서를 구성한다.

## 복셀과 공간 분할

- A는 occupancy와 패치 바닥 기준 상대 높이를 사용한다. B는 평균 RGB 세 채널을 추가한다. 높이는 패치의 물리 높이(기본 9.6 m)로 나눈 복셀 내 점 높이의 평균이며, RGB는 255로 나눈다. 저장 텐서 축은 C,Z,Y,X이다.
- 점이 같은 복셀에 모이면 유효 레이블의 다수결을 적용한다. 동률은 작은 학습 ID를 선택한다. 빈 복셀 및 유효 표가 없는 복셀의 target은 −100이며 loss와 혼동행렬에서 제외된다. 유효 target이 전혀 없는 패치는 건너뛴다.
- 3곳 이상의 독립된 지역이 있으면 전체 지역 단위로 분리한다. WMSC가 있으면 test, 나머지 정렬 순서의 마지막 지역은 val, 그 외는 train이다. WMSC가 없으면 마지막 지역이 test이다. 1~2곳이면 각 지역의 가장 긴 수평 축에서 연속 공간 구간을 나누고 경계마다 점유 패치 열 하나를 제거한다. 높이가 다른 동일 XY 열은 함께 배정한다. 데이터가 너무 좁아 필요한 split이나 두 클라이언트를 확보할 수 없으면 사유를 표시한다.
- train 내부에서 지역별 대략 절반을 중앙 사전학습과 미세조정에 예약하고, 나머지 패치를 두 클라이언트가 사용한다. 중앙 풀과 로컬 풀의 패치 ID는 겹치지 않으며 같은 지역을 공유할 수 있다. IID에서는 지역·최빈 복셀 클래스별로 패치를 섞어 번갈아 배정한다. Non-IID에서는 각 훈련 지역 전체를 한 클라이언트에 배정한다. 훈련 지역이 하나면 수평 축을 기준으로 두 구역을 만든다. 분할과 배정은 전 실험에서 고정되며 splits.json에서 재구성할 수 있다.

## 전체 학습 및 GPU 서버 실행

```bash
python3 -m experiments.run_experiments stpls3d \
  --data data/realworld_030m_32 --variant A --out results/stpls3d_A

CUDA_VISIBLE_DEVICES=0 python3 -m experiments.run_experiments stpls3d \
  --device cuda --data data/realworld_030m_32 --variant B --out results/stpls3d_B
```

기본은 CUDA 자동 선택이며 CPU로도 실행된다. GPU 한 장에서 두 클라이언트를 순차 학습한다. 메모리가 부족하면 `--batch-size 2`처럼 줄인다. CUDA 학습의 AMP는 `--no-amp`로 끌 수 있다. 최종 평가와 지연 측정은 FP32로 수행한다. 기본 설정을 복사해 `--config 파일.json`으로 지정할 수 있다. 입력 변형 A/B를 비교할 때는 같은 데이터 캐시와 seed를 사용한다.

| 단계 | 설정 및 측정 |
|---|---|
|사전학습|AdamW, 1e-3, cosine annealing, 20 epochs, batch 8, seed 42. 가장 높은 validation mIoU 체크포인트를 모든 비율의 시작점으로 읽는다|
|모델|Conv3d–BN–ReLU encoder → max-pool → bridge → nearest upsampling → encoder skip과 채널 결합 → decoder → 1³ segmentation head. 기본 너비 8/16/8. 입력과 출력의 D,H,W가 같다|
|프루닝|원본 Conv3d 출력 필터의 L1 합을 이용해 encoder·bridge·decoder에서 각각 채널을 실제 제거한다. 대응하는 BN·후속 입력과 skip 결합 인덱스를 함께 선택한다. 분류 헤드의 출력 클래스는 보존한다. 정수 반올림으로 실제 채널 감소율은 목표와 다를 수 있다|
|미세조정|각 모델을 중앙 train에서 AdamW 1e-4로 10 epochs 학습한다. 직후 및 마지막 epoch 성능을 같은 test에서 보고한다|
|FedAvg|각 비율의 두 분포 실험은 같은 저장 체크포인트에서 시작한다. 양쪽 클라이언트의 구조와 채널 순서는 동일하다. 10 rounds, 1 local epoch, AdamW 1e-4. optimizer는 지역 학습 때마다 초기화한다|
|BN 정책|모델 가중치, BN affine와 running mean/variance는 **로컬 패치 수**로 가중 평균한다. num_batches_tracked는 최댓값을 사용한다. running variance 집계는 클라이언트별 추정량을 평균하는 정책이다|
|정확도|고정 test의 유효 복셀 전체에서 혼동행렬 하나를 누적한다. Overall Accuracy, 클래스별 IoU, mIoU 및 Macro F1을 계산한다. 정답·예측 양쪽에 없는 클래스는 클래스 평균에서 제외하고 해당 IoU는 null로 저장한다. 라운드 추이는 val에서 측정한다|
|효율|Conv3d MAC은 단일 패치의 합성곱 누적곱 수이다. BN·보간·활성화는 제외한다. 파라미터 수, state_dict MB, 체크포인트 파일 MB, 로딩 시간, warm-up 후 장치 내 입력의 추론 p50/p95, CUDA peak allocated memory를 기록한다. CUDA 구간은 동기화한다|
|통신|round당 두 클라이언트에 내려보내고 다시 받는 전체 state_dict 텐서 바이트 합이다. 누적값은 round 수를 곱하며 단위는 십진 MB이다. 무선 지연, 재전송, 프로토콜 헤더는 별도 실측 대상이다|

## 파일 목록

| 경로 | 내용 |
|---|---|
|`configs/default.json`, `requirements.txt`|학습 기본값, 패키지 의존성|
|`datasets/stpls3d_loader.py`, `datasets/voxelization.py`, `datasets/synthetic_smoke.py`|PLY·공간 분할, 희소 복셀→dense 패치, 검증 전용 생성 데이터|
|`models/voxel_3dcnn.py`, `pruning/structured_pruning.py`|경량 U-Net 및 skip 연결을 보존하는 채널 프루닝|
|`training/train.py`, `training/evaluate.py`|사전학습·미세조정·로컬 학습 루프, 체크포인트 복원, 평가와 프로파일링|
|`federated/server.py`, `federated/fedavg.py`|중앙/로컬 데이터 분할, 두 클라이언트 시뮬레이션 및 가중 집계|
|`experiments/run_experiments.py`, `tests/test_pipeline.py`|CLI 및 통합·단위 테스트|
|`*/__init__.py`|Python 패키지 초기화|
|`실행폴더/config.json`, `splits.json`, `training_log.json`|실행 환경·seed·데이터 출처, 실제 샘플 배정, epoch별 손실과 validation mIoU|
|`실행폴더/checkpoints/`|E1 최적, E2/E3 직후 및 미세조정, E4~E6의 IID/Non-IID 최종 가중치와 E2/E3 선택 채널|
|`실행폴더/results/`|experiment_results.csv, pruning_comparison.csv, federated_round_metrics.csv|
|`실행폴더/plots/`|miou.png, model_mb.png, latency_p50_ms.png, round_miou.png|

체크포인트는 구조 명세, state_dict, 클래스·데이터 메타데이터를 담는다. `from training.train import load` 후 `net, metadata = load(path, device="cpu")`로 원본 또는 프루닝 모델을 복원한다.

## 실제 UAV 연구로의 확장

1. 두 보드에서 입력 센서의 좌표계·해상도·RGB 범위를 일치시키고 서버와 같은 패치에 대한 전처리 텐서와 logits를 대조한다.
2. 대상 보드에서 점군 수집부터 복셀화·추론까지 지연시간, peak RAM/VRAM, 전력, 온도, 지속 추론 속도를 측정한다. 로컬 학습의 메모리와 에너지 예산을 별도로 확인한다.
3. 모델·클래스·채널 스키마와 round 버전을 검증하는 배포 형식, 인증된 가중치 전송, 끊긴 round의 재시도 및 중복 업데이트 방지를 구현한다.
4. 실제 두 지역의 로컬 점군과 독립된 평가 비행 데이터를 확보해 분포 차이, BN 통계 정책, 링크별 실제 통신량과 round 시간을 다시 평가한다.

## RTX A6000 GPU 실험 재현

linuxserver2의 전용 가상환경에서 PyTorch 2.8.0+cu126, Python 3.11.3, CUDA 12.6, 드라이버 580.178.04로 검증했다. 기존 Conda 기본 환경은 CPU용이다. GPU를 노출한 셸에서 다음 명령을 실행한다. 첫 단계는 새 결과 디렉터리를 생성한다. 두 번째 단계는 그 smoke 성공 기록과 체크포인트를 확인한 후 진행한다.

```bash
.venv-gpu/bin/python -m experiments.gpu_validation smoke --out results/gpu_run_repeat
.venv-gpu/bin/python -m experiments.gpu_validation compare --out results/gpu_run_repeat \
  --ft-epochs 5 10 20 --fl-ft-epoch 10 --ft-lr 0.0001
```

가상환경을 새로 구성할 때는 Python 3.11의 `venv`를 만든 뒤 해당 환경의 `python -m pip install --upgrade pip`를 먼저 실행한다. 이어 `python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126`와 `python -m pip install -r requirements.txt`를 실행한다. 이 실험은 동일 조건 비교를 위해 AMP를 끄고 모델과 데이터를 cuda:0에 둔다. 비교 단계는 보관된 CPU E1 체크포인트와 분할을 재사용한다. [공식 PyTorch 배포표](https://pytorch.org/get-started/previous-versions/) · [NVIDIA 드라이버 호환성](https://docs.nvidia.com/deploy/cuda-compatibility/latest/minor-version-compatibility.html)

[GPU 환경 기록](results/gpu_experiments/gpu_environment.json) · [GPU smoke 결과](results/gpu_experiments/gpu_smoke_report.md) · [프루닝·연합학습 보고서](results/gpu_experiments/gpu_experiment_report.md)

## 단일 PLY 저용량 경로

실제 WMSC 결과와 공간·시간 예산은 [저용량 파이프라인 보고서](results/low_storage/report.md)에 있다. 다음은 기존 결과를 보존하는 재현 예이다. 작업의 기준 디렉터리는 프로젝트 루트이다.

```bash
python3 -m datasets.compact_cache data/single_site/WMSC_points.ply --out data/wmsc_repeat
python3 -m experiments.cache_audit data/wmsc_repeat --out results/wmsc_repeat_audit
.venv-gpu/bin/python -m experiments.compact_pilot data/wmsc_repeat \
  --audit results/wmsc_repeat_audit --out results/wmsc_repeat_pilot --batch 8 --workers 2 --variant B
```

- 전처리는 현재 마운트의 사용자 quota와 실제 가용 공간을 확인한다. quota 조회가 제한된 셸에서는 호스트에서 방금 확인한 **잔여 바이트**를 `--quota-free-bytes`로 전달할 수 있다. 5 GB와 가용량의 20% 중 큰 값을 남긴다. 다운로드의 최대 추가 사용량은 개별 PLY 크기 + 캐시 예산 + 임시 작업 예산 + 결과 여유분이다. 원본이 이미 있으면 다운로드 단계는 생략한다.
- 승인된 Drive 파일에서는 `python3 -m datasets.fetch_member FILE_ID --out data/new_source --report results/new_acquisition.json`으로 서버의 HTTP 206과 Content-Range를 검증하고 ZIP 중앙 디렉터리와 개별 PLY만 읽는다. Drive 바이러스 검사 안내에 동의한 경우에만 `--consent-scan-warning`을 사용한다. 전체 ZIP은 로컬에 저장하지 않는다. 원본 후보가 예산에 맞지 않으면 더 작은 실측 지역, 그다음 공식 synthetic 후보를 확인한다. 공식 synthetic를 사용할 경우 출처는 별도로 표기한다.
- `--limits 5000 500 500`, `--chunk 100000`, `--cache-bytes 4000000000`, `--work-bytes 1000000000`가 기본값이다. 데이터의 원점·범위와 레이블을 청크로 확인하고 가장 긴 XY축을 7개 구간으로 나누며 사이마다 1개 패치 열을 비운다. 중앙 구간과 4개 로컬 구간이 train 한도를 공유한다. IID는 양 지역의 독립된 구간을 섞고, Non-IID는 각 지역별로 묶는다. 각 구간에 레이블이 있는 패치가 적어도 하나 필요하다. 단일 지역 내부의 공간 홀드아웃이다.
- 학습 후보는 구간별 클래스 순환 선택을 적용하고, val/test는 레이블과 독립적인 고정 해시 순서로 제한한다(유효 타깃은 필요하다). pilot은 중앙 train에서 클래스 출현 패치 수의 역수에 비례하는 가중치를 사용해 복원 추출한다. 한 epoch 추출 횟수는 중앙 패치 수와 같다. 완전 학습 실행기의 동일한 동작은 config에 `balanced_patches: true`, `pin_memory: true`, `workers: 2`로 설정할 수 있다.
- 캐시는 32개 패치씩 묶은 ZIP shard이며 각 멤버는 점유 복셀의 uint16 슬롯, uint8 높이/RGB/레이블을 담는다. 슬롯 존재로 occupancy를 복원한다. 레이블 255와 빈 복셀은 학습에서 −100이다. 청크 경계를 넘어 point 수·높이·색 합과 클래스별 표를 누적하고 최종 단계에서만 반올림·다수결을 적용한다. 청크 또는 점유 공간이 커도 전 장면 dense grid를 만들지 않는다. 기본 임시 예산으로 모든 bucket을 만들 수 없으면 shard별로 원본을 다시 읽어 공간 사용량을 줄인다.
- 파일 해시·파라미터가 같은 경로로 재실행하면 이미 커밋된 shard를 재사용한다. 캐시 생성 로그와 progress.json에 실제 사용 바이트를 기록하고, 완성한 manifest에 shard SHA-256을 저장한다. `experiments.cache_audit`는 해시, 읽기 CRC, 모든 패치의 실제 텐서·히스토그램·공간 경계를 검증한다. 5종 시각화와 클래스 통계를 생성한다.
- 캐시는 기존 `experiments.run_experiments stpls3d`에서도 읽을 수 있다. 위의 pilot 명령만 실행하면 중앙 학습 1 epoch에서 종료한다. 데이터 다운로드 도구에는 requests와 beautifulsoup4가 필요하다. 저장 포맷 비교 재현에는 선택 의존성 Zarr 2.18.7과 numcodecs 0.15.1이 필요하다. `python3 -m experiments.cache_benchmark data/wmsc_compact --out results/benchmark_repeat.csv`로 측정한다.

## Synthetic-to-Real 외부 평가

Synthetic_v3 네 장면으로만 학습·검증한 뒤 WMSC 기존 캐시 1,683개 패치에서 외부 평가를 수행했다. 결과와 클래스별 정답 복셀 수는 [최종 보고서](results/synthetic_to_real/final_report.md)에 있다.

동일 데이터로 다시 실행할 때는 새 결과 디렉터리로 기존 데이터 manifest와 분포 기록을 복사한다. manifest는 현재 서버의 검증된 캐시와 샘플 ID를 가리킨다.

```bash
mkdir results/s2r_repeat
cp results/synthetic_to_real/{acquisition.json,dataset_manifest.json,synthetic_class_distribution.csv,wmsc_class_distribution.csv} results/s2r_repeat/
.venv-gpu/bin/python -m experiments.synthetic_to_real smoke --out results/s2r_repeat
.venv-gpu/bin/python -m experiments.synthetic_to_real fit --out results/s2r_repeat
.venv-gpu/bin/python -m experiments.synthetic_to_real test --out results/s2r_repeat
python3 -m experiments.s2r_report --out results/s2r_repeat
```

단계별 실행에서 `smoke/passed.json`은 GPU 입력과 1 epoch 검증 완료, `frozen_models.json`은 합성 데이터로 학습·모델 선택 완료, `synthetic_to_real_comparison.csv`는 WMSC 평가 완료를 나타낸다. 그림과 보고서는 저장된 CSV만으로 재생성할 수 있다.

## Phase 5A 가상 Jetson 실험

CPU Ground Server와 두 CUDA 클라이언트를 별도 프로세스로 실행해 같은 RTX A6000에서 원본·20% 프루닝 모델과 FP32·INT8 전송을 비교했다. 실기기 검증 상태는 PENDING_HARDWARE다. 재현에는 서버에 보관된 기존 체크포인트·패치 캐시·검증 번들이 필요하다. [Phase 5A 실험 보고서](results/jetson_phase5_virtual/PHASE5_VIRTUAL_REPORT.md) · [Jetson 실행 안내](deployment/JETSON.md)
