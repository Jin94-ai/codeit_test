# images/

이미지 및 시각화 결과 저장 폴더

## 용도

- EDA 결과 그래프
- 학습 곡선 (loss, accuracy)
- 실험 결과 비교 차트
- 예측 결과 시각화
- 발표 자료용 이미지

## 파일명 규칙

```
카테고리_설명.png

예:
eda_class_distribution.png
eda_image_size_distribution.png
train_loss_curve_exp001.png
result_prediction_samples.png
comparison_model_performance.png
```

## 폴더 구조 (권장)

```
images/
├── eda/              # EDA 결과
├── training/         # 학습 곡선
├── results/          # 예측 결과
├── comparisons/      # 모델 비교
└── presentation/     # 발표 자료용
```

## 이미지 저장 예시

### Matplotlib

```python
import matplotlib.pyplot as plt

# 그래프 생성
plt.figure(figsize=(10, 6))
plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# 저장
plt.savefig('images/training/loss_curve_exp001.png', dpi=300, bbox_inches='tight')
plt.close()
```

### Seaborn

```python
import seaborn as sns

# 그래프 생성
sns.barplot(x='class', y='count', data=df)

# 저장
plt.savefig('images/eda/class_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
```

## 이미지 사용

### README.md에 삽입
```markdown
![Loss Curve](images/training/loss_curve_exp001.png)
```

### 실험 로그에 삽입
```markdown
## 결과
- Train Loss Curve: [images/training/loss_curve_exp001.png](../../images/training/loss_curve_exp001.png)
```

## 주의사항

- **고해상도 저장**: `dpi=300` 권장
- **파일명 명확히**: 나중에 찾기 쉽게
- **너무 큰 파일 주의**: GitHub 용량 제한 (100MB)
- **Git에 커밋**: 이미지는 공유 목적으로 커밋

## 담당

- Data Engineer: EDA 이미지
- Model Architect: 학습 곡선
- Experimentation Lead: 비교 차트
- Leader: 발표 자료용 이미지
