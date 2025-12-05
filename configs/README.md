# configs/

설정 파일 모음 (YAML 형식)

## 용도

- 모델 하이퍼파라미터 설정
- 학습 설정
- 데이터 경로 설정
- 실험별 설정 관리

## 파일명 규칙

```
모델명_설명.yaml

예:
yolov8_baseline.yaml
yolov8_augmented.yaml
faster_rcnn_exp001.yaml
```

## 주요 파일

| 파일 | 설명 | 담당 |
|:-----|:-----|:-----|
| `base.yaml` | 기본 설정 | Model Architect |
| `yolov8_baseline.yaml` | YOLOv8 베이스라인 | Model Architect |
| `experiment_template.yaml` | 실험 템플릿 | Experimentation Lead |

## 예시: base.yaml

```yaml
# 데이터
data:
  train_dir: data/raw/train
  val_dir: data/raw/val
  test_dir: data/raw/test
  image_size: 640

# 모델
model:
  name: yolov8n
  pretrained: true
  num_classes: 4

# 학습
training:
  epochs: 50
  batch_size: 16
  learning_rate: 0.001
  optimizer: adam
  device: cuda

# 데이터 증강
augmentation:
  horizontal_flip: true
  vertical_flip: false
  rotate: 10
  brightness: 0.2

# 기타
seed: 42
save_dir: checkpoints/
```

## 예시: yolov8_baseline.yaml

```yaml
# base.yaml 상속 후 변경사항만 기록

model:
  name: yolov8n
  pretrained: true

training:
  epochs: 50
  batch_size: 16
  learning_rate: 0.001

augmentation:
  horizontal_flip: true
  vertical_flip: false
```

## 사용법

```python
import yaml

# 설정 로드
with open('configs/base.yaml') as f:
    config = yaml.safe_load(f)

# 설정 사용
batch_size = config['training']['batch_size']
learning_rate = config['training']['learning_rate']
```

## 담당

- Model Architect: 모델 관련 설정
- Experimentation Lead: 실험 설정 관리
