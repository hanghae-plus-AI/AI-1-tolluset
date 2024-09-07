## hh-ai-1

학습 기록 저장용

## 주차

### 1주차

1-1
gradient descent 로 구현한 linear regression

- epoch 만큼 lr 값을 기반으로 w(가중치), b(편향) 을 업데이트하며 오류와의 거리를 줄여나감
- MSE의 미분값을 이용하여 w, b를 업데이트함
- torch.T 는 전치(transpose)를 의미함
- mulmat 은 행렬곱을 의미함
- y[None] 은 y의 차원을 늘려주는 역할을 함
- mean 은 평균을 구하는 함수
- 행렬 곱을 하기위해 dimension을 맞춰줘야함

1-2
MLP로 XOR 문제 해결

1-3
MNIST 데이터셋으로 MLP 구현
