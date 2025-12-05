# src/data/

데이터 처리 코드 모듈

## 용도

- 데이터 전처리
- 데이터 증강
- 데이터 로더
- 데이터셋 클래스

## 주요 파일

| 파일 | 설명 | 담당 |
|:-----|:-----|:-----|
| `preprocessing.py` | 전처리 함수 | Data Engineer |
| `augmentation.py` | 데이터 증강 | Data Engineer |
| `dataset.py` | PyTorch Dataset 클래스 | Data Engineer |
| `transforms.py` | 변환 함수 | Data Engineer |

## 예시 코드

### preprocessing.py
```python
def preprocess_image(image_path):
    """이미지 전처리"""
    pass

def normalize(image):
    """정규화"""
    pass
```

### augmentation.py
```python
import albumentations as A

def get_train_transforms():
    """학습용 데이터 증강"""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, p=0.5),
    ])
```

### dataset.py
```python
from torch.utils.data import Dataset

class PillDataset(Dataset):
    """알약 데이터셋"""
    def __init__(self, data_dir, transform=None):
        pass

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass
```

## 담당

- Data Engineer: 김민우, 김나연
