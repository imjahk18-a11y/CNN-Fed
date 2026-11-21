# 첫 시도

CUDA 텐서 검증 성공 후 첫 학습 배치에서 결정적 CUDA 공간 NLL 커널이 없어 중단되었다. MaxPool3d 역전파도 별도 점검에서 동일한 제약을 보였다. 이후 ignore mask를 적용한 log-softmax/gather와 겹치지 않는 2³ 셀의 최댓값 구현을 사용해 smoke를 통과했다.
