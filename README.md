## hh-ai-1

## How to

- activate source `source venv/bin/activate` -> `a`
- check pip list `pip list` -> `pl`
- install jupyter `jupyter notebook` -> `jn`

-> `s`

학습 기록 저장용

refs: https://github.com/hanghae-plus-AI/AI-1-tolluset

## 1주차

### 1-1

gradient descent 로 구현한 linear regression

- epoch 만큼 lr 값을 기반으로 w(가중치), b(편향) 을 업데이트하며 오류와의 거리를 줄여나감
- MSE의 미분값을 이용하여 w, b를 업데이트함
- torch.T 는 전치(transpose)를 의미함
- mulmat 은 행렬곱을 의미함
- y[None] 은 y의 차원을 늘려주는 역할을 함
- mean 은 평균을 구하는 함수
- 행렬 곱을 하기위해 dimension을 맞춰줘야함

### 1-2

Schotastic Gradient Descent 로 구현한 linear regression

- SGD를 이용하여 빠르고 다양한 방향으로 값을 찾아감
- ReLU 함수로 비선형성을 추가함
- 학습 전에는 기존 계산된 기울기 초기화를 해주어야함
- backward로 역전파

### 1-3

MNIST regression 구현

- 배치 사이즈를 나눠 학습을 진행함
- 레이어 마다 활성화 함수를 적용함
- GPU 사용 (mac은 cuda 대신 mps)
- mean 함수는 tensor 객체를 반환하기에 loss에서 backwards 역전파 가능
- torchviz로 모델의 구조를 시각화함

### 1-basic

MNIST classfication 구현

- 학습 데이터와 테스트 데이터 분리
- CrossEntropyLoss 사용
  - softmax 활성화 함수와 nllloss 손실 함수를 적용한 함수
  - 마지막 레이어에 적용되기에 기존 모델의 마지막 활성화 함수제거하고 사용
- classfication 예측 값 확인
  - 마지막 레이어 아웃풋 개수는 예측하려는 클래스 개수와 통일 시켜야함
- 학습 데이터와 테스트 데이터 간에 정확도 시각화

### 1-advance

- sgd와 adam 차이
- ReLU와 Leaky ReLU 차이
- Dropout에 대한 이해
- 검증 시 추가 학습 방지

---

- 모델 학습에 필요한 값들 dataclass로 관리
- 학습 함수에 필요한 값들 파라미터로 관리

## 2주차

### 2-rnn

- 자연어 처리를 위해 토크나이저 사용
- 특정 동작을 위해 특별 토큰들이 존재
- 여러 문장을 처리하기 위해 패딩을 사용
- 학습시에는 패딩을 제거하고 학습을 진행
