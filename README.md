# Health Eat - AI 알약 인식 프로젝트

<div align="center">

![Status](https://img.shields.io/badge/Status-Completed-success)
![Score](https://img.shields.io/badge/Kaggle%20Score-0.96703-blue)
![Python](https://img.shields.io/badge/Python-3.12-yellow)

**2-Stage Pipeline 기반 알약 검출 및 분류 시스템**

**기간**: 2024.12.05 ~ 12.23 | **평가**: Kaggle Private Competition (mAP@[0.75:0.95])

</div>

---

## 최종 결과

| Metric | Score |
|--------|-------|
| **Kaggle Score** | **0.96703** |
| Baseline | 0.815 |
| 개선율 | +18.7% |

---

## 팀원

| 역할 | 이름 | GitHub |
|:----:|:-----|:-------|
| **Leader / Integration** | 이진석 | [@Jin94-ai](https://github.com/Jin94-ai) |
| **Data Engineer** | 김민우 | @mw-kim |
| **Data Engineer** | 김나연 | @ny-kim |
| **Model Architect** | 김보윤 | @by-kim |
| **Experimentation Lead** | 황유민 | @ym-hwang |

---

## 모델 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                    2-Stage Pipeline                         │
├─────────────────────────────────────────────────────────────┤
│  Stage 1: YOLO11m Detector                                  │
│  ├── Input: 이미지 (1280x960)                               │
│  ├── Output: Bounding Boxes + Confidence                    │
│  └── Task: 알약 위치 검출 (Single class: "Pill")            │
├─────────────────────────────────────────────────────────────┤
│  Stage 2: ConvNeXt Classifier                               │
│  ├── Input: 크롭된 알약 이미지 (224x224)                    │
│  ├── Output: K-code + Confidence                            │
│  └── Task: 알약 종류 분류 (74 classes)                      │
└─────────────────────────────────────────────────────────────┘
```

### 왜 2-Stage인가?

1. **데이터 한계 극복**: 학습 데이터(74종) vs 테스트 데이터(196종) 클래스 불일치
2. **일반화 성능**: Detector는 "알약"만 검출 → 새로운 클래스에도 대응 가능
3. **모듈화**: 각 Stage 독립적 개선 가능

---

## 빠른 시작

### 1. 환경 설정

```bash
# 저장소 클론
git clone https://github.com/Jin94-ai/codeit_team8_project1.git
cd codeit_team8_project1

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 2. 데이터 준비

```bash
# Kaggle 데이터 다운로드 후
data/
├── train_images/        # 학습 이미지
├── train_annotations/   # COCO JSON 어노테이션
└── test_images/         # 테스트 이미지 (843개)
```

### 3. 모델 학습

```bash
# Stage 1: Detector 학습
python src/models/yolo11m_detector.py

# Stage 2: Classifier 학습
python src/models/convnext_classifier.py
```

### 4. 추론 및 제출

```bash
# Kaggle 제출 파일 생성
python -m src.inference.submit_v2

# 출력: submission_N.csv
```

---

## 프로젝트 구조

```
codeit_team8_project1/
├── src/
│   ├── data/                    # 데이터 처리
│   │   ├── aihub/              # AIHub 데이터 추출
│   │   └── yolo_dataset/       # YOLO 포맷 변환
│   ├── models/                  # 모델 학습
│   │   ├── yolo11m_detector.py # Stage 1: Detector
│   │   └── convnext_classifier.py # Stage 2: Classifier
│   └── inference/               # 추론
│       ├── pill_pipeline_v2.py # 2-Stage 파이프라인
│       └── submit_v2.py        # Kaggle 제출
├── data/                        # 데이터 (gitignore)
├── docs/                        # 문서
│   ├── PROJECT_AGENDA.md       # 비즈니스 아젠다
│   └── APP_ARCHITECTURE.md     # Cloud 아키텍처
├── logs/                        # 로그
│   ├── collaboration/          # 협업 일지
│   └── meetings/               # 회의록
└── notebooks/                   # 실험 노트북
```

---

## 실험 기록

| Submission | Score | 설명 |
|------------|-------|------|
| #1 | 0.815 | Baseline (YOLO 단일 모델) |
| #2 | 0.690 | End-to-End 196 클래스 |
| #3 | 0.920 | 2-Stage (YOLO + YOLO-cls) |
| #4 | 0.963 | 2-Stage (YOLO + ConvNeXt) |
| #5 | 0.965 | AIHub 데이터 추가 |
| **#6** | **0.967** | **데이터 정제 + 최적화** |

---

## 기술 스택

| 분류 | 기술 |
|------|------|
| **Deep Learning** | PyTorch, Ultralytics YOLO11, timm (ConvNeXt) |
| **Data** | AIHub 의약품 데이터셋, Albumentations |
| **Experiment Tracking** | Weights & Biases |
| **협업** | Git, GitHub, Discord |

---

## 문서

- [보고서 PDF](docs/report.pdf) *(발표 자료)*
- [프로젝트 아젠다](docs/PROJECT_AGENDA.md)
- [앱 아키텍처](docs/APP_ARCHITECTURE.md)

## 협업 일지

| 팀원 | 링크 |
|------|------|
| 이진석 | [logs/collaboration/이진석/](logs/collaboration/이진석/) |
| 김민우 | [logs/collaboration/김민우/](logs/collaboration/김민우/) |
| 김나연 | [logs/collaboration/김나연/](logs/collaboration/김나연/) |
| 김보윤 | [logs/collaboration/김보윤/](logs/collaboration/김보윤/) |
| 황유민 | [logs/collaboration/황유민/](logs/collaboration/황유민/) |

---

## 라이선스

이 프로젝트는 코드잇 스프린트 교육 과정의 일환으로 진행되었습니다.

---

<div align="center">

**코드잇 8팀 - Health Eat**

</div>
