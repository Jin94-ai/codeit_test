# src/models/

모델 정의 및 학습 코드

## 용도

- 모델 아키텍처 정의
- 학습 로직
- 평가 로직
- 모델 유틸리티

## 주요 파일

| 파일 | 설명 | 담당 |
|:-----|:-----|:-----|
| `baseline.py` | 베이스라인 모델 | Model Architect |
| `yolo.py` | YOLO 모델 (선택한 모델) | Model Architect |
| `train.py` | 학습 스크립트 | Model Architect |
| `evaluate.py` | 평가 스크립트 | Experimentation Lead |

## 예시 코드

### baseline.py
```python
import torch.nn as nn

class BaselineModel(nn.Module):
    """베이스라인 모델"""
    def __init__(self):
        super().__init__()
        # 모델 정의

    def forward(self, x):
        # Forward pass
        return x
```

### train.py
```python
def train_one_epoch(model, dataloader, optimizer, device):
    """1 epoch 학습"""
    model.train()
    for batch in dataloader:
        # 학습 로직
        pass

def train(model, train_loader, val_loader, config):
    """전체 학습 루프"""
    for epoch in range(config['epochs']):
        train_one_epoch(model, train_loader, optimizer, device)
        # 검증
```

### evaluate.py
```python
def evaluate(model, dataloader, device):
    """모델 평가"""
    model.eval()
    # 평가 로직
    return metrics
```

## 모델 저장

```python
# 체크포인트 저장
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, 'checkpoints/model.pth')
```

## 담당

- Model Architect: 김보윤
- Experimentation Lead: 황유민 (평가)
