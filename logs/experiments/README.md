# 실험 로그 작성 가이드

실험 시작 전 또는 완료 후 바로 작성하세요.

---

## 파일명 규칙

```
exp_XXX.md
```

**XXX는 3자리 숫자** (001, 002, 003...)

**예시**:
- `exp_001.md` - 첫 번째 실험 (베이스라인)
- `exp_002.md` - 데이터 증강 실험
- `exp_015.md` - 15번째 실험

---

## 템플릿

아래 내용을 복사해서 사용하세요:

```markdown
# Experiment XXX

## 실험 정보
- **날짜**: YYYY-MM-DD
- **담당**: 이름
- **목적**: 이 실험의 목표

## 모델 설정
- **모델**: (예: YOLOv8n, Faster R-CNN)
- **Pretrained**: Yes / No
- **Backbone**: (예: ResNet50, CSPDarknet)

## 데이터
- **Train/Val Split**: (예: 80/20)
- **데이터 증강**: 사용한 증강 기법 나열
- **이미지 크기**: (예: 640x640)

## 하이퍼파라미터
- **Epochs**:
- **Batch Size**:
- **Learning Rate**:
- **Optimizer**: (예: Adam, SGD)
- **기타**:

## 결과
- **Train mAP@50**:
- **Val mAP@50**:
- **Kaggle Score**:
- **학습 시간**: (예: 2시간 30분)

## 분석
### 잘된 점
-

### 문제점
-

### 개선 방향
-

## 다음 실험 아이디어
-

## 참고 자료
-
```

---

## 작성 팁

### 실험 목적
- 명확하고 구체적으로 (예: "데이터 증강으로 과적합 해결")
- 이전 실험과의 차이점 명시

### 모델 설정
- 재현 가능하도록 상세히 기록
- config 파일 경로 링크

### 결과
- 숫자는 소수점 3자리까지
- 그래프/차트 이미지 링크 추가

### 분석
- 객관적 데이터 기반
- 추측은 명확히 표시 (예: "아마도 ~인 것 같음")

---

## 예시

```markdown
# Experiment 001

## 실험 정보
- **날짜**: 2025-12-10
- **담당**: 박수진
- **목적**: YOLOv8n 베이스라인 모델 구축 및 성능 확인

## 모델 설정
- **모델**: YOLOv8n (nano)
- **Pretrained**: Yes (COCO weights)
- **Backbone**: CSPDarknet

## 데이터
- **Train/Val Split**: 80/20 (랜덤)
- **데이터 증강**: 없음 (기본 설정만)
- **이미지 크기**: 640x640

## 하이퍼파라미터
- **Epochs**: 50
- **Batch Size**: 16
- **Learning Rate**: 0.01
- **Optimizer**: SGD (momentum=0.9)
- **기타**: weight_decay=0.0005

## 결과
- **Train mAP@50**: 0.723
- **Val mAP@50**: 0.654
- **Kaggle Score**: 0.612
- **학습 시간**: 2시간 15분 (RTX 3060)

## 분석
### 잘된 점
- 빠른 학습 속도로 베이스라인 확보
- COCO pretrained 덕분에 초기 성능 준수
- 과적합 징후 없음 (train/val 차이 작음)

### 문제점
- Validation mAP가 예상보다 낮음 (목표: 0.7+)
- 작은 알약 검출 실패율 높음 (50x50px 이하)
- 겹친 알약 검출 어려움

### 개선 방향
- 데이터 증강 추가 (특히 작은 객체 대응)
- Mosaic augmentation 시도
- 앵커 박스 사이즈 조정 검토

## 다음 실험 아이디어
1. exp_002: Albumentations로 데이터 증강 (RandomRotate90, HorizontalFlip, 밝기/대비)
2. exp_003: 이미지 크기 증가 (640 → 1024)
3. exp_004: YOLOv8s (small) 모델로 용량 증가

## 참고 자료
- W&B 실험 결과: https://wandb.ai/team8/exp001
- 학습 곡선 그래프: [images/exp001_curves.png](../../images/exp001_curves.png)
- Config 파일: [configs/exp001.yaml](../../configs/exp001.yaml)
```

---

**실험 완료 후 README.md의 "실험 결과" 테이블에도 추가하세요!**
