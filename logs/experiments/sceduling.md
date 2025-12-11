# 실험 전략 기반 단계

## 1. 데이터 관련 최적화
**Augmentation:**  
- 회전, 색상 변형, 노이즈, MixUp, CutMix  

**Input size:**  
- 작은 이미지 → 학습 속도 ↑  
- 큰 이미지 → 정확도 ↑  

**목적:**  
- 모델 과적합 방지  
- 다양한 환경 대응  

---



## 2. Learning rate & Scheduler 조정
**lr sweep:**  
- Grid, Random, YOLO Evolve  

**Scheduler 비교:**  
- Cosine, Step, OneCycleLR  

**목적:**  
- 빠른 수렴  
- 안정적 학습  

---

## 3. Optimizer & Batch size 조정
**Optimizer:**  
- Adam, AdamW, SGD  

**Batch size:**  
- GPU 메모리 고려  
- 학습 안정성  

**Regularization:**  
- Weight decay  
- Dropout  

---

## 4. Advanced Hyperparameter (YOLO Evolve)
- Promising 후보 기반 전체 하이퍼파라미터 자동 탐색  
- 시간 소요가 크므로 **v8n → v8s -> 11 fine-tune** 순으로 진행  

- Promising 후보: 유망한 하이퍼파라미터/모델 설정 후보
---

## 5. Fine-tune + 앙상블
- Promising 후보 학습  
- Augmentation, LR, Optimizer 미세 조정  
- 여러 모델 앙상블 → 최종 성능 극대화  

# 실험 스케줄링

- 단순화한 초안, 실험&피드백을 통해 스케줄링 변동 있을 예정(ex.다양한 기법 적용)
- 5명이 하루 2번 실험으로 가정.

✅ 12/11 실험 플랜 (데이터 최적화: Augmentation + Input size)

목표: 탐색 단계에서 다양한 Augmentation + Input size 조합을 빠르게 돌려 Top-5 promising 조합을 선별. 그 후 Narrow-down(한 변수씩 테스트)으로 원인 규명.

| 팀원      | 실험1 (기본 실험 — 공통 시작점)                                                 | 실험2 (집중 실험 / 담당 영역 Fine-tune)                                                                                                              |
| ------- | -------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| **이진석** | YOLOv8n으로 **입력 크기 512 / 640 / 768** 각각 테스트 → 빠르게 3개 기본 런             | **입력 크기 민감도 분석**: Augment는 고정한 상태에서 **512 ↔ 640 ↔ 768** 성능 비교 + 작은 객체/큰 객체가 어떻게 달라지는지 사례 수집                                                |
| **김민우** | YOLOv8n으로 **Aug 조합 A/B/C**(기본형 세 가지) 실험 → imgsz 640로 빠른 3회 탐색        | **강한 Aug 영향 분석**: 예) MixUp을 고정 → Mosaic 강도(0/0.1/0.2)만 변화시키는 식으로, **한 번에 한 요소만 조절하는 통제 실험**                                                |
| **김나연** | YOLOv8n으로 **컬러 기반 Aug**(HSV 약/중/강) 비교 — 입력 크기 640 고정                 | **컬러 민감도 분석**: imgsz는 고정하고, **hsv_s, hsv_v, gamma**를 각각 하나씩만 변화시키며 성능 차이 분석                                                                |
| **김보윤** | YOLOv8n으로 **기하학 Aug**(회전 degree 0/5/10, scale 범위) 비교 — imgsz 640     | **기하학 Ablation**: 전체 Aug는 고정하고 **degree만 0→3→5→10으로 변화시키는 식**의 단일 변수 실험                                                                    |
| **황유민** | YOLOv8n으로 **기본 Baseline 검증(640)** + 팀원 결과 중 유망한 Aug 1~2개 Cross-check | **유망 Aug 미세 조정 + 고해상도 실험**: (1) 팀이 뽑은 top-5 Aug 중 상위 2개를 **fine-tune(epoch 증가 + multi-seed)** (2) 768이 성능 상승을 보이면 **조건부 imgsz 960 테스트 진행** |


### 세부 실행 규칙

- Baseline 먼저: 모든 팀원은 실험 시작 전에 baseline(640, default aug) 기본 성능 동일하게 나오는 지 확인.

- Exploration 규칙: 실험1 단계에서는 aug와 imgsz를 혼합해도 괜찮다 (빠른 후보 발굴 목적). 각 팀원은 최대 3 run(빠른)만 수행.

- Promising 기준(승격 조건): promising = baseline mAP@[.5:.95] 대비 +1.5% 이상 OR small-object mAP@0.5 개선 + inference latency 허용 범위 내(예: <1.5×)

- Early stop 기준: val mAP 개선 없으면 patience=5로 조기중단 권장(자원 절약).




















12/12 전략: Learning rate & Scheduler 조정

| 팀원 | 이진석 | 김민우 | 김나연 | 김보윤 | 황유민 |
|------|---|---|---|---|---|
| 실험1 | v8n LR sweep #1 | v8n LR sweep #2 | v8s Fine-tune #1 | v8s Fine-tune #2 | promising 후보 lr/scheduler 미세 조정 #1 |
| 실험2 | v8n Scheduler #1 | v8n Scheduler #2 | v8s Fine-tune #3 | v8s Fine-tune #4 | promising 후보 lr/scheduler 미세 조정 #2 |

12/13 전략: YOLO Evolve 후보 확보

| 팀원 | 이진석 | 김민우 | 김나연 | 김보윤 | 황유민 |
|------|---|---|---|---|---|
| 실험1 | v8n Evolve #1 | v8n Evolve #2 | v8s Fine-tune #5 | v8s Fine-tune #6 | v8n Evolve promising 후보 Fine-tune #1 |
| 실험2 | v8n Evolve #3 | v8n Evolve #4 | v8s Fine-tune #7 | v8s Fine-tune #8 | v8n Evolve promising 후보 Fine-tune #2 |

12/14 전략: promising 후보 기반 Fine-tune

| 팀원 | 이진석 | 김민우 | 김나연 | 김보윤 | 황유민 |
|------|---|---|---|---|---|
| 실험1 | v8n Promising Fine-tune #1 | v8n Promising Fine-tune #2 | v8s Fine-tune #9 | v8s Fine-tune #10 | promising 후보 Fine-tune + lr/optimizer 조정 #1 |
| 실험2 | v8n Promising Fine-tune #3 | v8n Promising Fine-tune #4 | v8s Fine-tune #11 | v8s Fine-tune #12 | promising 후보 Fine-tune + lr/optimizer 조정 #2 |

12/15 전략: 최종 Fine-tune 준비

| 팀원 | 이진석 | 김민우 | 김나연 | 김보윤 | 황유민 |
|------|---|---|---|---|---|
| 실험1 | v8n 최종 Fine-tune #1 | v8n 최종 Fine-tune #2 | v8s Fine-tune #13 | v8s Fine-tune #14 | promising 후보 Fine-tune + augmentation 미세 조정 #1 |
| 실험2 | v8n 최종 Fine-tune #3 | v8n 최종 Fine-tune #4 | v8s Fine-tune #15 | v8s Fine-tune #16 | promising 후보 Fine-tune + augmentation 미세 조정 #2 |

12/18 전략: 앙상블 후보 검증

| 팀원 | 이진석 | 김민우 | 김나연 | 김보윤 | 황유민 |
|------|---|---|---|---|---|
| 실험1 | promising 후보 검증 #1 | promising 후보 검증 #2 | 앙상블 실험 #1 | 앙상블 실험 #2 | 앙상블 후보 조합/미세 튜닝 #1 |
| 실험2 | promising 후보 검증 #3 | promising 후보 검증 #4 | 앙상블 실험 #3 | 앙상블 실험 #4 | 앙상블 후보 조합/미세 튜닝 #2 |

12/19 전략: 앙상블 성능 검증

| 팀원 | 이진석 | 김민우 | 김나연 | 김보윤 | 황유민 |
|------|---|---|---|---|---|
| 실험1 | promising 후보 반복 학습 #1 | promising 후보 반복 학습 #2 | 앙상블 실험 #5 | 앙상블 실험 #6 | 앙상블 후보 최종 Fine-tune #1 |
| 실험2 | promising 후보 반복 학습 #3 | promising 후보 반복 학습 #4 | 앙상블 실험 #7 | 앙상블 실험 #8 | 앙상블 후보 최종 Fine-tune #2 |

12/20 전략: 앙상블 후보 최종 점검

| 팀원 | 이진석 | 김민우 | 김나연 | 김보윤 | 황유민 |
|------|---|---|---|---|---|
| 실험1 | promising 후보 검증 #5 | promising 후보 검증 #6 | 앙상블 테스트 #9 | 앙상블 테스트 #10 | 앙상블 최종 후보 조합 실험 #1 |
| 실험2 | promising 후보 검증 #7 | promising 후보 검증 #8 | 앙상블 테스트 #11 | 앙상블 테스트 #12 | 앙상블 최종 후보 조합 실험 #2 |

12/21 전략: 최종 Fine-tune & 앙상블 결정

| 팀원 | 이진석 | 김민우 | 김나연 | 김보윤 | 황유민 |
|------|---|---|---|---|---|
| 실험1 | promising 후보 반복 Fine-tune | promising 후보 반복 Fine-tune | promising 후보 반복 Fine-tune | promising 후보 반복 Fine-tune | 최종 앙상블 결정 #1 |
| 실험2 | promising 후보 반복 Fine-tune | promising 후보 반복 Fine-tune | promising 후보 반복 Fine-tune | promising 후보 반복 Fine-tune | 최종 앙상블 결정 #2 |
