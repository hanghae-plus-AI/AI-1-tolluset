## 챗봇 컨셉

산업군: 금융
사용자: 요즘 오를 것 같은 주식을 찾고 싶은 사람
문제 상황: 정성적인 분석으로는 주식을 찾기 어려움
목적: 투자 타입에 따라 요즘 유망한 주식을 찾아주는 서비스

## wandb

train/loss: 값이 적어 로그에 출력되지 않음

valid/loss: https://api.wandb.ai/links/tolluset-ai/mwd6vlyk

```log
{'train_runtime': 0.0106, 'train_samples_per_second': 7335.823, 'train_steps_per_second': 846.441, 'train_loss': 0.0, 'epoch': 3.0}
  0%|                                                                                                            | 0/9 [00:00<?, ?it/s]
***** train metrics *****
  epoch                    =        3.0
  total_flos               =    18980GF
  train_loss               =        0.0
  train_runtime            = 0:00:00.01
  train_samples_per_second =   7335.823
  train_steps_per_second   =    846.441
[INFO|trainer.py:4021] 2024-10-22 16:37:52,404 >>
***** Running Evaluation *****
[INFO|trainer.py:4023] 2024-10-22 16:37:52,404 >>   Num examples = 6
[INFO|trainer.py:4026] 2024-10-22 16:37:52,404 >>   Batch size = 8
100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 1187.18it/s]
***** valid metrics *****
  epoch                   =        3.0
  eval_loss               =     0.2856
  eval_runtime            = 0:00:00.71
  eval_samples_per_second =      8.434
  eval_steps_per_second   =      1.406
```

```

```
