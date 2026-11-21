# 실제 STPLS3D WMSC 저용량 파일 처리 및 1-epoch GPU 검증

2026년 11월 18일 linuxserver2에서 실제 WMSC 점군 9,977,764점을 처리했다. 최종 희소 캐시는 12.26 MB이며 RTX A6000에서 재로딩한 체크포인트의 validation mIoU는 0.3242였다. 대상은 단일 실측 지역 내부의 공간 홀드아웃이다. 이 실행에서 장기 학습과 FL 라운드를 수행하지 않았다.

## 저장 공간과 다운로드

|측정 대상|값|
|---|---:|
|홈 파일시스템 물리적 가용 공간|다운로드 직전 454.21 GiB|
|계정 quota 및 기존 사용량|50.00 GiB / 약 9.11 GiB|
|quota 기준 가용 공간|40.89 GiB|
|유지할 여유 공간|8.18 GiB (가용 공간의 20%)|
|추가 기록 예산|32.71 GiB, 약 35.12 GB|
|기존 프로젝트와 전용 GPU 환경|프로젝트 약 5.6 GiB, 대부분 .venv-gpu|
|기존 results 및 모델 하위 폴더|results 약 1.5 MiB, 체크포인트 폴더 합계 약 852 KiB. 최상위 checkpoints 디렉터리 4 KiB|
|공용 Python 및 사용자 캐시|/opt/anaconda3 약 6.6 GiB, 공유 conda 패키지 캐시 약 6.2 GiB. 사용자 pip 캐시 약 2.0 MiB, 사용자 .cache 총 약 3.6 MiB|

Google Drive의 대용량 파일 안내에 사용자가 동의한 후 HTTP 206과 Content-Range를 검증했다. 전체 아카이브는 37,220,897,716 bytes로 초기 기록 예산보다 크다. 중앙 디렉터리와 WMSC 멤버만 읽었다. ZIP 메타데이터 및 멤버 reader가 읽은 합계는 159,652,863 bytes이다. 확보한 원본은 **한 파일**이다. 멤버는 ZIP 안에서 Stored 방식이며 원본과 압축 멤버 크기가 모두 159,644,443 bytes다. SHA-256은 `b654ee6f360c67c4c8292e93260ede8892d611aa159f0938c2df23dfac18aa32`, CRC32는 `39b6d9f2`이다. [원본 PLY](../../data/single_site/WMSC_points.ply) · [아카이브 목록과 사전 예산](acquisition.json) · [사용자가 지정한 배포 파일](https://drive.google.com/file/d/1U8dduxCdhAXbCpfGLDMCSRHpdb5R7cXb/view)

다운로드 전에 PLY 0.160 GB + 캐시 4 GB + 임시 작업 1.1 GB + 결과 0.1 GB, 합계 **약 5.36 GB**를 상한으로 잡았다. 헤더 확인 후 캐시 상한 추정치는 96,420,348 bytes, bucket 임시 공간 상한은 269,399,628 bytes였다. 이 파일에서 원본·임시 공간·캐시와 결과 여유를 동시에 합쳐도 **약 0.63 GB**이다. 완료 후 보관할 원본과 캐시의 합계는 **171,903,120 bytes**다. 1-epoch 실행 직후 quota 잔여는 약 40.73 GiB였다. [실행 전후 디스크 측정](gpu_pilot/metrics.json)

## 전처리와 공간 검증

실제 파일은 binary little-endian이며 float32 XYZ, uint8 RGB 및 uint8 class를 제공한다. 원시 ID 1, 2, 5, 11, 13, 14, 15, 18, 19가 관측되었다. 공식 KPConv 변환과 원본 정의에 따라 유효한 6개 학습 ID로 매핑했다. 파일명이나 나타난 클래스 수로 스키마를 추정하지 않는다. [공식 변환 코드](https://github.com/meidachen/STPLS3D/blob/main/KPConv-PyTorch/datasets/STPLS3D.py#L573-L613)

100,000점씩 읽어 장면 원점을 정하고, 가장 긴 수평 축의 7개 공간 띠와 그 사이 9.6 m 경계 구간을 먼저 확정했다. 이후 선택된 타일만 0.3 m 단위로 복셀화했다. 32³ 패치에서 빈 복셀과 무효 레이블은 ignore −100으로 복원한다. 점유는 슬롯 존재로, 높이와 평균 RGB는 각 1바이트로 저장했다. 정규화 채널의 실측 최대 양자화 오차는 높이 0.0019607834, RGB 각 0.0019607844 이하였다. 디코딩한 occupancy·정규화 높이·RGB 범위는 각각 [0, 1]이었다.

|구역|패치|의미|
|---|---:|---|
|Central|632|사전학습용, 이번 pilot의 유일한 훈련 풀|
|Drone A / B, Non-IID|434 / 402|서로 다른 연속 지역|
|Drone A / B, IID 근사|429 / 407|같은 로컬 풀을 분포가 비슷한 공간 띠 조합으로 재배정|
|Validation / Test|132 / 83|두 클라이언트 조건이 공유하는 고정 홀드아웃|
|합계|1,683|train 1,468, val 132, test 83|

모든 shard 해시와 모든 패치의 디코딩·히스토그램을 확인했다. 서로 다른 공간 띠 간 중복 패치와 26방향 인접 위반은 모두 0건이다. 유효 복셀 수는 1,882,154개이며 모든 보관 패치 셀 중 3.4129%다. 빈 복셀 비율은 96.5871%였다. [분할별 통계](validation/stats.json) · [실제 패치 배정](validation/partitions.json)

|클래스|유효 복셀|등장 패치|Central 등장 패치|Val 등장 패치|Test 등장 패치|
|---|---:|---:|---:|---:|---:|
|ground|1,258,154|1,470|590|125|66|
|building|212,788|344|90|29|0|
|vegetation|387,965|898|287|73|63|
|cars|7,043|34|3|0|0|
|lightStreetSigns|1,387|23|3|0|0|
|fences|14,817|123|64|15|0|

중앙 train에서 표지·신호 클래스는 35개 복셀뿐이다. 가중 패치 샘플링은 빈도가 낮은 클래스가 포함된 패치를 더 자주 고르지만 새로운 독립 표본을 만들지는 않는다. 차량·표지 클래스가 있는 독립 검증 구역, 그리고 건물·차량·표지·울타리가 있는 독립 테스트 구역이 추가로 필요하다.

[원본 RGB 점군](validation/raw_points.png) · [점유 복셀](validation/occupied_voxels.png) · [시맨틱 색상](validation/semantic_voxels.png) · [공간 분할](validation/spatial_splits.png) · [클라이언트 분포](validation/client_distributions.png)

## 캐시 형식 실측

실제 WMSC에서 seed 42로 추출한 동일한 128개 패치에 동일한 희소 정수 레코드를 사용했다. 무작위 전체 읽기 3회의 중앙값이며 OS 캐시가 활성화된 단일 프로세스 환경이다. 표에는 레코드·인덱스·컨테이너 파일 길이가 포함된다. 디스크 블록 할당량과 GPU 처리량은 별도 지표다.

|형식|파일 수|bytes|무작위 읽기 ms/패치|
|---|---:|---:|---:|
|NumPy mmap + 오프셋|2|993,237|0.0034|
|Zarr 2 / Zstd / 128 KiB chunk|10|897,839|0.2925|
|32개 패치 단위 ZIP / 패치별 Deflate|4|855,639|0.2115|

mmap이 가장 빨랐다. 이 프로젝트의 용량 우선 조건에서는 압축 shard가 가장 작았고 각 worker가 필요한 패치 하나만 CRC를 검증하며 읽을 수 있어 이를 채택했다. 완성 캐시는 53개 ZIP과 계획·진행·manifest 등의 메타데이터를 합해 12,258,677 bytes이다. 기본 캐시 상한은 4 GB다. [벤치마크 CSV](cache_benchmark.csv) · [캐시 manifest](../../data/wmsc_compact/manifest.json)

## 실제 GPU 1 epoch

PyTorch 2.8.0+cu126, CUDA 12.6, RTX A6000에서 입력 B(occupancy, 상대 높이, RGB), batch 8, worker 2개, pinned input/label, CUDA AMP를 사용했다. 입력 형태 [8,5,32,32,32], 레이블 [8,32,32,32]를 확인했다. 중앙 패치 수에 맞춰 가중 복원 추출로 632개를 학습했고 validation의 모든 패치를 평가했다.

|측정값|결과|
|---|---:|
|Train loss / validation loss|0.8340 / 0.6197|
|Validation overall accuracy / Macro F1|0.8306 / 0.4020|
|Validation mIoU (체크포인트 재로딩 검증)|0.3242|
|PyTorch peak allocated GPU memory|103.45 MB|
|훈련 루프 / 훈련·검증·저장 경과시간|1.62 s / 2.75 s|
|저장된 체크포인트|47,041 bytes|

검증 IoU는 ground 0.8482, building 0.2622, vegetation 0.5102, cars 0, lightStreetSigns 미정의, fences 0.00047이다. 차량은 정답이 없지만 예측이 있어 IoU가 0이고, 표지·신호는 정답과 예측 양쪽에 없어 기존 평가 정책상 평균에서 제외했다. 시험용 test에 대한 모델 성능 평가는 실행하지 않았다. [GPU 로그와 클래스별 IoU](gpu_pilot/metrics.json) · [체크포인트](gpu_pilot/best_val.pt)

## 전체 실행 추정과 검증 범위

동일 WMSC 캐시와 batch·worker·GPU 조건에서 중앙 20 epochs는 1회 경과시간의 단순 외삽으로 약 **55초**다. E1~E6 전체는 중앙 40 epochs와 로컬 풀 836개를 3개 모델 × 2개 분포 × 10 rounds 학습한다고 가정하면, 훈련 루프만 약 **193초**다. 이는 중앙 1 epoch의 1.62초를 패치 수에 비례해 환산한 값이다. 라운드별 평가·초기화·집계·저장·프로파일링은 추가되며, 다른 지역 및 5,000개 train 한도까지 확장한 데이터에는 이 시간을 그대로 적용할 수 없다. 전처리한 WMSC를 그대로 재사용할 때는 원본과 캐시 171.90 MB에 결과·체크포인트 계획 여유분 100 MB를 더해 약 **272 MB**를 잡았다. 기존 Python 및 CUDA 환경은 이 추가 예산에서 제외했다.

호스트 전체 회귀 테스트는 17개 통과했다. ASCII와 양쪽 endian 바이너리, 청크 경계 표 집계, 무효 레이블, 낮은 임시 공간에서 원본 재스캔, 커밋 후 중단·재개, 캐시 변조 탐지, 다중 worker 입력을 포함한다. 다중 worker에 필요한 로컬 소켓이 샌드박스에서 제한되어 호스트에서 검증했다. [구현 및 재현 방법](../../README.md)
