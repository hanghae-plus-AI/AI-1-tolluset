## Rank에 따른 loss, 학습 속도, 그리고 메모리 점유율 공유

- train/loss

![스크린샷 2024-11-06 오전 9 04 32](https://github.com/user-attachments/assets/f9754ee2-1e0b-42fd-a68b-3be41f98ee01)

- Runtime

![스크린샷 2024-11-06 오전 9 10 56](https://github.com/user-attachments/assets/953560cb-72cd-4156-805c-3173b893f4c3)

- 메모리 점유율

```
- rank 8
Max Alloc: 0.0 GB
Max Alloc After: 0.6 GB

- rank 128
Max Alloc: 5.4 GB
Max Alloc After: 5.4 GB

- rank 256
Max Alloc: 6.1 GB
Max Alloc After: 6.1 GB
```

## LoRA의 장단점 분석

### 장점

- 학습해야하는 파라미터의 수가 적어 더 적은 메모리를 사용하게 되고 학습 속도 또한 빠르다.
- 기존 코드에 쉽게 적용 가능하다.

### 단점

- 기존 모델에 추가로 구현해야 하기 때문에 약간의 적응이 필요하다.
- 모든 상황에서 성능이 향상되지는 않는다.
