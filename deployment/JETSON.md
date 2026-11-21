# Jetson Orin Nano 배포 준비

실기기 검증 상태는 두 대 모두 **PENDING_HARDWARE**다. 각 보드에서 프로젝트 코드와 해당 클라이언트의 패치 디렉터리만 준비한다. Ground Server에는 시작 체크포인트, 선택한 공통 검증 번들, 별도 작업 디렉터리가 필요하다.

## 환경 확인

프로젝트 루트에서 다음 명령으로 모델명, 아키텍처, L4T, JetPack 패키지, RAM, 디스크, 전력 모드, 온도 센서, PyTorch 및 CUDA 텐서 연산 결과를 기록한다. 읽을 수 없는 항목은 JSON에서 null이다. 메타패키지가 없을 때는 L4T 버전으로 설치 조합을 확인한다.

```bash
python3 -m deployment.jetson preflight --output board-preflight.json
```

보드에서 확인한 JetPack/L4T와 Python ABI에 맞춰 NVIDIA의 ARM64 PyTorch wheel 또는 컨테이너를 선택하고 그 환경에 NumPy를 준비한다. 서버의 x86 가상환경은 서버에서 사용한다. Orin Nano 제품명, RAM 용량 및 Super 모드 여부는 보드와 전력 설정에서 각각 확인한다. 보드에서 전력·온도를 기록할 때는 같은 전력 모드와 냉각 조건에서 tegrastats 로그와 실험 시각을 함께 보관한다. [NVIDIA PyTorch 설치 안내](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html) · [NVIDIA tegrastats 안내](https://docs.nvidia.com/jetson/archives/r36.4.4/DeveloperGuide/AT/JetsonLinuxDevelopmentTools/TegrastatsUtility.html)

## Ground Server

서버에서 아래 명령을 프로젝트 루트에서 실행한다. 준비 단계에서 생성된 `results/jetson_phase5_virtual/data/client1`과 `client2`는 각각 해당 보드에만 배치한다. 16바이트 토큰은 같은 세션의 세 프로세스에 비공개로 프로비저닝한다.

```bash
umask 077
python3 -c 'import os; open("token.bin", "xb").write(os.urandom(16))'
.venv-gpu/bin/python -m deployment.jetson server \
  --start results/distributed_ai/starts/s42_p0.pt \
  --validation results/distributed_ai_phase3/bundles/validation/patches.npz \
  --work results/jetson_live_int8/server --token-file token.bin \
  --port 6543 --batch 1 --mode INT8 --rounds 10
```

실제 운영에 대표 레이블이 없다면 `--validation`을 생략한다. 고정 라운드로 실행하고 mIoU는 비워 둔다. 20% 모델은 `s42_p20.pt`, FP32 전송은 `--mode FP32`를 사용한다. 설정이 다른 세션마다 별도 작업 디렉터리를 사용한다.

## 두 클라이언트

기존 앱 프로토콜은 공유 토큰으로 접속을 확인하고 SHA-256으로 메시지 손상을 검사한다. 무선망에서는 서버의 loopback 포트까지 인증·암호화된 터널을 구성하고, 각 보드에서 로컬 터널 포트를 사용한다. 토큰 자체는 전송을 암호화하지 않는다.

```bash
# 보드 1에서 실행
python3 -m deployment.jetson client --id 1 --data client1/manifest.json \
  --host 127.0.0.1 --port 6543 --work work/session1/client1 \
  --token-file token.bin --batch 1 --allocator-gib 2

# 보드 2에서 실행
python3 -m deployment.jetson client --id 2 --data client2/manifest.json \
  --host 127.0.0.1 --port 6543 --work work/session1/client2 \
  --token-file token.bin --batch 1 --allocator-gib 2
```

클라이언트는 Orin Nano와 사용 가능한 CUDA를 확인하고 입력·모델·출력·그래디언트의 장치를 검증한다. 각 라운드에서 수신한 구조와 가중치를 로드해 그 보드의 첫 패치를 추론한 후 전체 로컬 epoch를 학습한다. 재시작은 세션의 동일 인자와 작업 경로로 실행한다. 서버는 정상 커밋한 마지막 라운드에서 이어가고 클라이언트는 일치하는 부모 모델의 캐시된 응답을 재전송한다. 보드에서는 배치 1에서 먼저 확인하고 실측 통합 메모리 여유에 따라 배치 2·4를 점검한다.

실기기에서 아직 필요한 측정은 동일 패치의 A6000 대비 logits 오차·클래스 일치율, 두 독립 GPU의 지속 학습·동시 추론, 무선 RTT·처리량·재접속, 통합 메모리 최고치, 온도와 실제 센서 전력이다.
