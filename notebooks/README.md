# notebooks/

Jupyter Notebook 저장 폴더

## 용도

- EDA (탐색적 데이터 분석)
- 실험 및 프로토타이핑
- 결과 시각화

## 파일명 규칙

```
XX_설명.ipynb

예:
01_eda.ipynb
02_data_augmentation_test.ipynb
03_model_comparison.ipynb
```

## 주요 노트북

| 파일 | 설명 | 담당 |
|:-----|:-----|:-----|
| `01_eda.ipynb` | 데이터 탐색 및 분석 | Data Engineer |
| `02_baseline_test.ipynb` | 베이스라인 모델 테스트 | Model Architect |

## 작성 가이드

1. **마크다운 셀 활용**: 코드 설명 추가
2. **결과 저장**: 중요한 그래프는 `images/` 폴더에 저장
3. **재현 가능성**: 랜덤 시드 설정

```python
import random
import numpy as np
import torch

# 재현성을 위한 시드 고정
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
```

## 주의사항

- 노트북 실행 후 **Output 포함해서 커밋** (결과 공유 목적)
- 너무 큰 Output은 Clear 후 커밋
- 데이터 파일 경로는 상대 경로 사용

## 담당

- Data Engineer: EDA 노트북
- Model Architect: 모델 실험 노트북
- Experimentation Lead: 성능 분석 노트북
