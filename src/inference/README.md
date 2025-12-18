# 2-Stage Pill Detection & Classification Pipeline

알약 검출 및 74개 클래스 분류를 위한 2-Stage 파이프라인

---

## 전체 파이프라인 흐름

```
[1. 데이터 준비]
    ├── AIHub 원본 이미지 추출 (Detector용)
    └── AIHub 크롭 이미지 추출 (Classifier용)
           ↓
[2. YOLO 데이터셋 생성]
    ├── Detector용: 단일 클래스 "Pill"
    └── Classifier용: 74개 클래스 폴더 구조
           ↓
[3. 모델 학습]
    ├── Stage 1: Pill Detector
    └── Stage 2: Pill Classifier
           ↓
[4. 추론 & 제출]
    └── 2-Stage Pipeline → submission.csv
```

---

## 1. 데이터 준비

### 1.1 AIHub 원본 이미지 추출 (Detector용)

```bash
python -m src.data.aihub.extract_for_detector
```

| 항목 | 내용 |
|------|------|
| 입력 | AIHub ZIP (TL_*_조합.zip, TS_*_조합.zip) |
| 출력 | `data/aihub_detector/images/`, `data/aihub_detector/annotations/` |
| 용도 | Detector 학습용 원본 이미지 + bbox |
| 설정 | `MAX_IMAGES = 5000` (스크립트 내 수정) |

### 1.2 AIHub 크롭 이미지 추출 (Classifier용)

```bash
python -m src.data.aihub.extract_and_crop
```

| 항목 | 내용 |
|------|------|
| 입력 | AIHub ZIP + Kaggle 데이터 |
| 출력 | `data/cropped/images/`, `data/cropped/annotations/` |
| 용도 | Classifier 학습용 크롭 이미지 (74개 클래스) |
| 설정 | `TARGET_PER_CLASS = 210` (클래스당 최대 개수) |

### 1.3 수동 데이터 정제 (중요!)

크롭 후 잘못된 이미지를 **수동으로 삭제**해야 합니다:

```
data/cropped/images/
├── K-001900_0001.png  ← 정상
├── K-001900_0002.png  ← 잘못 크롭됨 → 삭제!
└── ...
```

**삭제 대상:**
- 알약이 잘린 이미지
- 빈 이미지 또는 배경만 있는 이미지
- 여러 알약이 섞인 이미지
- 흐릿하거나 품질이 낮은 이미지

### 1.4 고아 어노테이션 정리

이미지를 삭제한 후, 대응하는 어노테이션도 삭제:

```bash
python -m src.data.aihub.cleanup_orphan_annotations
```

| 항목 | 내용 |
|------|------|
| 역할 | 이미지 없는 JSON 파일 삭제 |
| 대상 | `data/cropped/annotations/` |

**정리 흐름:**
```
1. extract_and_crop 실행 → 크롭 이미지 생성
2. 수동으로 잘못된 이미지 삭제 (탐색기에서)
3. cleanup_orphan_annotations 실행 → 고아 JSON 삭제
4. yolo_export_classifier 실행
```

---

## 2. YOLO 데이터셋 생성

### 2.1 Detector용 데이터셋

```bash
python -m src.data.yolo_dataset.yolo_export_detector
```

| 항목 | 내용 |
|------|------|
| 입력 | Kaggle + AIHub detector 데이터 |
| 출력 | `data/yolo/images/`, `data/yolo/labels/`, `data/yolo/pills.yaml` |
| 클래스 | 1개 (Pill) |
| 분할 | Train 80% / Val 20% |

### 2.2 Classifier용 데이터셋

```bash
python -m src.data.yolo_dataset.yolo_export_classifier
```

| 항목 | 내용 |
|------|------|
| 입력 | `data/cropped/` |
| 출력 | `data/yolo_cls/train/`, `data/yolo_cls/val/`, `data/yolo_cls/class_mapping.json` |
| 클래스 | 74개 (K-code 폴더 구조) |
| 분할 | Train 80% / Val 20% (stratified) |

---

## 3. 모델 학습

### 3.1 Stage 1: Pill Detector

```bash
python -m src.models.pill_detector
```

| 항목 | 내용 |
|------|------|
| 모델 | YOLO11s |
| 데이터 | `data/yolo/pills.yaml` |
| 출력 | `runs/detect/pill_detector/weights/best.pt` |
| 설정 | epochs=30, imgsz=640, patience=10 |

### 3.2 Stage 2: Pill Classifier

```bash
python -m src.models.pill_classifier
```

| 항목 | 내용 |
|------|------|
| 모델 | YOLO11s-cls |
| 데이터 | `data/yolo_cls/` |
| 출력 | `runs/classify/pill_classifier/weights/best.pt` |
| 설정 | epochs=50, imgsz=224, patience=15 |

---

## 4. 추론 & 제출

### 4.1 Kaggle 제출 파일 생성

```bash
# 기본 실행
python -m src.inference.submit

# 옵션 지정
python -m src.inference.submit --det_conf 0.05 --output submission.csv
```

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--test_dir` | `data/test_images` | 테스트 이미지 폴더 |
| `--output` | `submission.csv` | 출력 CSV 파일 |
| `--det_conf` | `0.05` | Detector confidence |
| `--cls_conf` | `0.3` | Classifier confidence |

### 4.2 단일 이미지 테스트

```bash
python -m src.inference.pill_pipeline data/test_images/1.png
```

### 4.3 Python 코드에서 사용

```python
from src.inference import PillPipeline

# 파이프라인 초기화
pipeline = PillPipeline(detector_conf=0.05)

# 예측
results = pipeline.predict("image.png")
# [{"bbox": [x1,y1,x2,y2], "k_code": "K-001900", "classifier_conf": 0.95}, ...]

# 시각화 포함
results, output_path = pipeline.predict_with_visualization("image.png")
```

---

## 파일 구조

```
src/
├── data/
│   ├── aihub/
│   │   ├── extract_for_detector.py       # AIHub 원본 추출
│   │   ├── extract_and_crop.py           # AIHub 크롭 추출
│   │   └── cleanup_orphan_annotations.py # 고아 어노테이션 삭제
│   └── yolo_dataset/
│       ├── config.py                     # 경로 설정
│       ├── yolo_export_detector.py       # Detector 데이터셋
│       └── yolo_export_classifier.py     # Classifier 데이터셋
├── models/
│   ├── pill_detector.py                  # Stage 1 학습
│   └── pill_classifier.py                # Stage 2 학습
└── inference/
    ├── __init__.py
    ├── pill_pipeline.py                  # 2-Stage 파이프라인
    ├── submit.py                         # Kaggle 제출
    └── README.md                         # 이 문서
```

---

## 출력 형식

### submission.csv
```csv
annotation_id,image_id,category_id,bbox_x,bbox_y,bbox_w,bbox_h,score
1,1,27925,558,73,394,403,0.95
2,1,16550,173,743,181,288,0.92
```

### class_mapping.json
```json
{
  "k_code_to_idx": {"K-001900": 0, "K-002483": 1, ...},
  "idx_to_k_code": {"0": "K-001900", "1": "K-002483", ...}
}
```

---

## 학습된 모델 경로

| 모델 | 경로 |
|------|------|
| Detector | `runs/detect/pill_detector/weights/best.pt` |
| Classifier | `runs/classify/pill_classifier/weights/best.pt` |
| 클래스 매핑 | `data/yolo_cls/class_mapping.json` |

---

## 전체 실행 순서 (처음부터)

```bash
# 1. 데이터 준비
python -m src.data.aihub.extract_for_detector
python -m src.data.aihub.extract_and_crop

# 1-1. 수동 정제 (중요!)
# → data/cropped/images/ 폴더에서 잘못된 이미지 수동 삭제
# → 알약이 잘린 이미지, 빈 이미지, 여러 알약 섞인 이미지 등

# 1-2. 고아 어노테이션 삭제
python -m src.data.aihub.cleanup_orphan_annotations

# 2. YOLO 데이터셋 생성
python -m src.data.yolo_dataset.yolo_export_detector
python -m src.data.yolo_dataset.yolo_export_classifier

# 3. 모델 학습
python -m src.models.pill_detector
python -m src.models.pill_classifier

# 4. 추론 & 제출
python -m src.inference.submit
```

---

## 성능

| 항목 | 결과 |
|------|------|
| Kaggle Score | **0.96** |
| 평가 지표 | mAP@[0.75:0.95] |
| 이전 방식 | 0.815 (End-to-end) |
| 개선폭 | +0.145 |
