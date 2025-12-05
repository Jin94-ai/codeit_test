# scripts/

실행 스크립트 모음

## 용도

- 학습 실행
- 추론 실행
- Kaggle 제출 파일 생성
- 유틸리티 스크립트

## 주요 파일

| 파일 | 설명 | 사용법 | 담당 |
|:-----|:-----|:-------|:-----|
| `train.py` | 모델 학습 | `python scripts/train.py --config configs/base.yaml` | Model Architect |
| `inference.py` | 추론 실행 | `python scripts/inference.py --model checkpoints/best.pth` | Integration Specialist |
| `make_submission.py` | 제출 파일 생성 | `python scripts/make_submission.py` | Integration Specialist |

## 예시 코드

### train.py
```python
import argparse
import yaml

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    # 설정 로드
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # 학습 실행
    train(config)

if __name__ == '__main__':
    main()
```

### inference.py
```python
def inference(model, image_path):
    """단일 이미지 추론"""
    # 추론 로직
    return predictions

if __name__ == '__main__':
    # 추론 실행
    pass
```

### make_submission.py
```python
import pandas as pd

def make_submission(model, test_loader, output_path='submission.csv'):
    """Kaggle 제출 파일 생성"""
    predictions = []
    for batch in test_loader:
        pred = model(batch)
        predictions.append(pred)

    # CSV 생성
    df = pd.DataFrame(predictions)
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
```

## 실행 예시

```bash
# 학습
python scripts/train.py --config configs/yolov8.yaml

# 추론
python scripts/inference.py --model checkpoints/best.pth --input data/test/

# 제출 파일 생성
python scripts/make_submission.py --model checkpoints/best.pth --output submission.csv
```

## 담당

- Model Architect: train.py
- Integration Specialist: inference.py, make_submission.py
