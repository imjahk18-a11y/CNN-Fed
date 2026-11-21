# 합성 CUDA smoke test

NVIDIA RTX A6000 · cuda:0 · PyTorch 2.8.0+cu126.

사전학습 2 epochs, 20% 프루닝, 미세조정 2 epochs, 두 클라이언트 IID 2 rounds 완료. 모든 배치에서 모델·입력·레이블·출력·손실·gradient의 CUDA 위치를 검사했다. FedAvg 집계는 CPU에서 수행했다. 최종 체크포인트 재로딩 후 test mIoU 0.987939. 학습 peak 49.39 MB.
