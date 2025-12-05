# 팀 역할

<div align="center">

![Team](https://img.shields.io/badge/Team-5%20members-blue)

**5가지 역할로 프로젝트 완성하기**

</div>

---

## 역할 정의

### 1. Leader ([이진석])

**핵심 책임**:
- 프로젝트 조율 및 의사결정 촉진
- 일일 스탠드업 주도
- 코드 통합 관리
- 발표 자료 조율

**주요 산출물**:
- README.md 관리 및 업데이트
- 회의록 관리
- 최종 발표 자료

**작업 비중**: 조율 40% | 코딩 40% | 문서 20%

---

### 2. Data Engineer ([김민우], [김나연])

**핵심 책임**:
- EDA (탐색적 데이터 분석) 주도
- 데이터 전처리 파이프라인 구축
- 데이터 증강 전략 수립 및 실험
- 데이터셋 버전 관리

**주요 산출물**:
- `notebooks/01_eda.ipynb`
- `src/data/preprocessing.py`
- `src/data/augmentation.py`

**작업 비중**: EDA 30% | 파이프라인 50% | 증강 20%

---

### 3. Model Architect ([김보윤])

**핵심 책임**:
- Object Detection 모델 리서치 및 선정
- 베이스라인 모델 구현
- 모델 아키텍처 설계 및 최적화
- Transfer Learning 전략 수립

**주요 산출물**:
- `src/models/baseline.py`
- `src/models/yolo.py` (또는 선택한 모델)
- 학습된 모델 체크포인트

**작업 비중**: 리서치 20% | 구현 60% | 최적화 20%

---

### 4. Experimentation Lead ([황유민])

**핵심 책임**:
- 실험 추적 시스템 구축 (W&B/MLflow)
- 하이퍼석])

**핵심 책임**:
- Pull Request 리뷰
- 코드 통합 및 충돌 해결
- 추론 파이프라인 구축
- Kaggle 제출 파일 생성 및 제출

**주요 산출물**:
- `scripts/inference.py`
- `scripts/make_submission.py`
- 코드 리뷰 피드백

**작업 비중**: 리뷰 40% | 통합 40% | 제출 20%

---

## 협업 매트릭스

| 작업 | 주 담당 | 협업 필요 |
|:-----|:--------|:----------|
| **EDA** | Data Engineer | 전체 참여 |
| **전처리 파이프라인** | Data Engineer | Model Architect |
| **데이터 증강** | Data Engineer | Experimentation Lead |
| **모델 선정** | Model Architect | Experimentation Lead |
| **베이스라인 구현** | Model Architect | Integration Specialist |
| **실험 설계** | Experimentation Lead | Model Architect |
| **하이퍼파라미터 튜닝** | Experimentation Lead | Model Architect |
| **Kaggle 제출** | Integration Specialist | Leader |
| **발표 자료** | Leader | 전체 참여 |

---

## 협업 규칙

### 일일 스탠드업
- **시간**: 매일 오전 10시 (15분) - 첫 미팅에서 조정
- **형식**: 어제 한 일 / 오늘 할 일 / 막힌 점 (각 1-2분)

### 주간 회고
- **시간**: 매주 금요일 저녁 (1시간)
- **형식**: KPT (Keep, Problem, Try)

### 의사결정
1. 문제 정의
2. 옵션 수집 (24시간)
3. 팀 논의 (스탠드업 또는 별도 미팅)
4. 합의 또는 리더 최종 결정
5. 문서화 및 실행

---

## 팀 약속

- [ ] 일일 스탠드업에 빠짐없이 참여
- [ ] 협업 일지를 매일 작성
- [ ] 질문은 주저하지 않고 즉시 공유
- [ ] 코드는 반드시 리뷰 후 병합
- [ ] 실패를 비난하지 않고 학습 기회로 전환
- [ ] 서로의 시간을 존중
- [ ] 즐겁게 프로젝트 진행

---

## 팀원 정보

> 첫 미팅 후 업데이트하세요

| 역할 | 이름 | 강점 | 관심사 | GitHub |
|:----:|:-----|:-----|:-------|:-------|
| Leader | 이진석 | [첫 미팅 후 작성] | [첫 미팅 후 작성] | @Jin94-ai |
| Data Engineer | [이름] | [첫 미팅 후 작성] | [첫 미팅 후 작성] | @username |
| Model Architect | [이름] | [첫 미팅 후 작성] | [첫 미팅 후 작성] | @username |
| Experimentation Lead | [이름] | [첫 미팅 후 작성] | [첫 미팅 후 작성] | @username |
| Integration Specialist | [이름] | [첫 미팅 후 작성] | [첫 미팅 후 작성] | @username |

---

<div align="center">

**역할은 가이드라인입니다. 서로 돕고 배우세요!**

</div>
