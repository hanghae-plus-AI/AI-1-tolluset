## LLM 서비스 경량화 진행

기존 챗봇 PoC 하기 위한 코드에서 성능을 위해 facebook/opt-350m 보다 큰 facebook/opt-1.3b 모델을 사용하기 위해 경량화를 진행하였습니다.

기존 코드에서 facebook/opt-1.3b 모델을 사용하였을 때는 아래와 같이 메모리 부족으로 해당 모델을 사용할 수 없었습니다.

```bash
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 16.00 MiB. GPU 0 has a total capacity of 14.58 GiB of which 15.56 MiB is free. Including non-PyTorch memory, this process has 14.56 GiB memory in use. Of the allocated memory 14.04 GiB is allocated by PyTorch, and 404.14 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
```

```bash
- attempt-2

torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 64.00 MiB. GPU 0 has a total capacity of 14.58 GiB of which 35.56 MiB is free. Including non-PyTorch memory, this process has 14.54 GiB memory in use. Of the allocated memory 13.80 GiB is allocated by PyTorch, and 619.73 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation. See documentation for Memory Management (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
```

AMP와 lora를 적용하여 경량화를 진행 후 더 큰 모델을 사용하여 PoC를 할 수 있게 되었습니다.

## 경량화를 했을 때 얻을 수 있었던 이점 공유

기존에 사용해보고 싶었던 모델의 크기가 너무 큰 경우 위에 사용한 경량화 방식을 통해 모델을 학습하고 사용해볼 수 있는게 큰 이점인 것 같습니다.

하지만, 작은 모델에서 학습을 하는 것보다 큰 모델에서는 메모리 부족으로 다양하게 학습을 하지 못함으로 결과값에 대해서는 크게 개선되었다고 볼 수 없었습니다.

상황에 따라 큰 모델을 사용하는게 유리한 경우와 작은 모델로 더 큰 데이터로 학습하는 방식을 고려하면 될 것 같습니다.
