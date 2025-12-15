# src/data/

데이터 처리 코드 모듈

## 용도

- COCO → YOLO 포맷 변환
- 데이터 전처리
- 데이터 증강
- 데이터 로더
- 데이터셋 클래스

## 주요 모듈

### yolo_dataset/ (COCO → YOLO 변환)

| 파일 | 설명 | 상태 |
|:-----|:-----|:----:|
| `config.py` | 경로 및 설정 | ✅ 완료 |
| `coco_parser.py` | COCO JSON 파싱 + 232개 필터링 | ✅ 완료 |
| `yolo_export.py` | YOLO 변환 + Stratified split | ✅ 완료 |
| `README.md` | 사용 방법 문서 | ✅ 완료 |

**특징**:
- file_name 기반 필터링 (232개)
- Stratified Train/Val split (8:2)
- 단일 이미지 클래스 처리
- COCO bbox → YOLO 정규화

**실행**:
```bash
python -m src.data.yolo_dataset.yolo_export
```

### ADD 데이터셋 병합 모듈 (중간 처리 단계)

**`add_json_category_mapper.py`**

외부 ADD 데이터셋을 기존 YOLO 학습 파이프라인에 사용하기 위해  
annotation(json) 구조를 수정하고 데이터셋을 병합하는 모듈

**역할**:
- ADD annotation(json) 내부
  - `categories.id` ← `images.dl_idx`
  - `categories.name` ← `images.dl_name`
- 수정 전/후 json diff 검증
- 수정 완료 후
  - annotation → `train_annotations/`
  - image → `train_images/`
- 디렉토리 없을 경우 자동 생성

**특징**:
- category_id 중복 허용
- YOLO 변환 파이프라인과 독립
- COCO → YOLO 변환 **이전 단계**에서 1회 실행

**실행**:
```bash
python -m src.data.yolo_dataset.add_json_category_mapper

### 향후 추가 예정

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
