# 코드잇 8팀 - 알약 검출 프로젝트

<div align="center">

![Week](https://img.shields.io/badge/Week-0%2F3-blue)
![Status](https://img.shields.io/badge/Status-Team%20Building-yellow)

**목표**: 알약 이미지에서 최대 4개 검출 (Object Detection)

**기간**: 3주 | **평가**: Kaggle Private Competition

</div>

---

## 팀원

| 역할 | 이름 | GitHub | 상태 |
|:----:|:-----|:-------|:----:|
| **Leader** | 이진석 | [@Jin94-ai](https://github.com/Jin94-ai) | ![](https://img.shields.io/badge/-active-green) |
| **Data Engineer** | [김민우, 김나연] | @username | ![](https://img.shields.io/badge/-active-green) |
| **Model Architect** | [김보윤] | @username | ![](https://img.shields.io/badge/-active-green) |
| **Experimentation Lead** | [황유민] | @username | ![](https://img.shields.io/badge/-active-green) |
| **Integration Specialist** | [이진석석] | @username | ![](https://img.shields.io/badge/-active-green) |

> 역할 상세: [TEAM_ROLES.md](TEAM_ROLES.md)

---

## 진행 현황

```mermaid
gantt
    title 프로젝트 타임라인 (12/5 ~ 12/23)
    dateFormat YYYY-MM-DD
    section Week 0
    팀 빌딩           :done, 2025-12-05, 1d
    EDA              :active, 2025-12-05, 3d
    section Week 1
    베이스라인 구축   :2025-12-06, 7d
    첫 제출          :milestone, 2025-12-11, 0d
    section Week 2
    실험 및 개선     :2025-12-11, 11d
    section Week 3
    발표 준비        :2025-12-22, 2d
    최종 발표        :milestone, 2025-12-23, 0d
```

**체크리스트**:
- [x] 팀 구성 완료
- [x] 첫 미팅 완료
- [ ] EDA 완료 (12/5)
- [ ] 베이스라인 모델 구축 (12/12까지)
- [ ] 첫 Kaggle 제출 (12/11)
- [ ] 실험 및 개선 (12/11-21)
- [ ] 최종 발표 (12/23)

---

## 빠른 시작

### 1. 저장소 클론
```bash
git clone https://github.com/Jin94-ai/codeit_team8_project1.git
cd codeit_team8_project1
```

### 2. 환경 설정
```bash
# 가상환경 생성
python -m venv venv

# 활성화 (Windows)
venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 3. 데이터 다운로드
- Kaggle Competition 페이지에서 데이터 다운로드
- `data/` 폴더에 압축 해제

### 4. 시작하기
- 역할 확인: [TEAM_ROLES.md](TEAM_ROLES.md)
- 협업 일지 작성: [logs/collaboration/](logs/collaboration/)
- 회의록 확인: [logs/meetings/](logs/meetings/)

---

## 기술 스택

### Object Detection
- **모델**: YOLOv8 / Faster R-CNN / EfficientDet (첫 미팅에서 결정)
- **프레임워크**: PyTorch / TensorFlow (첫 미팅에서 결정)

### 데이터 처리
- **증강**: Albumentations
- **전처리**: OpenCV, Pillow

### 실험 추적
- **도구**: Weights & Biases / MLflow (첫 미팅에서 결정)
- **로그**: [logs/experiments/](logs/experiments/)

### 협업
- **버전 관리**: Git, GitHub
- **커뮤니케이션**: Discord / Slack (첫 미팅에서 결정)
- **일지**: [logs/collaboration/](logs/collaboration/)

---

## 실험 결과

| ID | 모델 | mAP@50 | Kaggle Score | 날짜 | 담당 | 상태 |
|:--:|:-----|:------:|:------------:|:----:|:-----|:----:|
| - | - | - | - | - | - | - |

> 실험 상세: [logs/experiments/](logs/experiments/)

---

## 프로젝트 구조

```
codeit_team8_project1/
├── README.md                    # 프로젝트 메인
├── TEAM_ROLES.md               # 팀 역할 정의
├── requirements.txt            # Python 패키지
├── .gitignore
│
├── logs/                       # 📁 작업 로그
│   ├── collaboration/          # 협업 일지 (날짜별)
│   ├── meetings/              # 회의록 (날짜별)
│   └── experiments/           # 실험 로그 (ID별)
│
├── data/                      # 📁 데이터 (gitignore)
│   ├── raw/
│   └── processed/
│
├── notebooks/                 # 📁 Jupyter 노트북
│   └── 01_eda.ipynb
│
├── src/                       # 📁 소스 코드
│   ├── data/                  # 데이터 처리
│   └── models/                # 모델 구현
│
└── scripts/                   # 📁 실행 스크립트
    └── inference.py
```

---

## 주요 링크

- **Kaggle Competition**: [링크 추가 예정]
- **협업 일지**: [logs/collaboration/](logs/collaboration/)
- **회의록**: [logs/meetings/](logs/meetings/)
- **실험 로그**: [logs/experiments/](logs/experiments/)

---

## 협업 규칙

### 일일 스탠드업
- **시간**: 매일 오전 10시 (15분) - 첫 미팅에서 조정
- **형식**: 어제 한 일, 오늘 할 일, 막힌 점

### 협업 일지
- **작성 주기**: 매일
- **위치**: `logs/collaboration/YYYY-MM-DD_이름.md`
- **가이드**: [logs/collaboration/README.md](logs/collaboration/README.md)

### 주간 회고
- **시간**: 매주 금요일 저녁 (1시간)
- **형식**: KPT (Keep, Problem, Try)

---

## 커밋 메시지 규칙

```
[Week X] 작업 내용

예:
[Week 0] Add first meeting notes
[Week 1] Implement baseline YOLOv8 model
[Week 2] Experiment with data augmentation
```

---

<div align="center">

**8팀 프로젝트 화이팅!**

</div>
