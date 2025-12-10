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
- 시간 소요가 크므로 **v8n → v8s fine-tune** 순으로 진행  

- Promising 후보: 유망한 하이퍼파라미터/모델 설정 후보
---

## 5. Fine-tune + 앙상블
- Promising 후보 학습  
- Augmentation, LR, Optimizer 미세 조정  
- 여러 모델 앙상블 → 최종 성능 극대화  

# 실험 스케줄링

- 단순화한 초안, 실험&피드백을 통해 스케줄링 변동 있을 예정(ex.다양한 기법 적용)
- 5명이 하루 2번 실험으로 가정.

12/11 전략: 데이터 관련 최적화 (Augmentation + Input size)

| 팀원 | A | B | C | D | E |
|------|---|---|---|---|---|
| 실험1 | v8n Augmentation #1 | v8n Augmentation #2 | v8s Baseline #1 | v8s Baseline #2 | v8n/Aug promising 후보 Fine-tune #1 |
| 실험2 | v8n Input size #1 | v8n Input size #2 | v8s Baseline #3 | v8s Baseline #4 | v8n/Aug promising 후보 Fine-tune #2 |

12/12 전략: Learning rate & Scheduler 조정

| 팀원 | A | B | C | D | E |
|------|---|---|---|---|---|
| 실험1 | v8n LR sweep #1 | v8n LR sweep #2 | v8s Fine-tune #1 | v8s Fine-tune #2 | promising 후보 lr/scheduler 미세 조정 #1 |
| 실험2 | v8n Scheduler #1 | v8n Scheduler #2 | v8s Fine-tune #3 | v8s Fine-tune #4 | promising 후보 lr/scheduler 미세 조정 #2 |

12/13 전략: YOLO Evolve 후보 확보

| 팀원 | A | B | C | D | E |
|------|---|---|---|---|---|
| 실험1 | v8n Evolve #1 | v8n Evolve #2 | v8s Fine-tune #5 | v8s Fine-tune #6 | v8n Evolve promising 후보 Fine-tune #1 |
| 실험2 | v8n Evolve #3 | v8n Evolve #4 | v8s Fine-tune #7 | v8s Fine-tune #8 | v8n Evolve promising 후보 Fine-tune #2 |

12/14 전략: promising 후보 기반 Fine-tune

| 팀원 | A | B | C | D | E |
|------|---|---|---|---|---|
| 실험1 | v8n Promising Fine-tune #1 | v8n Promising Fine-tune #2 | v8s Fine-tune #9 | v8s Fine-tune #10 | promising 후보 Fine-tune + lr/optimizer 조정 #1 |
| 실험2 | v8n Promising Fine-tune #3 | v8n Promising Fine-tune #4 | v8s Fine-tune #11 | v8s Fine-tune #12 | promising 후보 Fine-tune + lr/optimizer 조정 #2 |

12/15 전략: 최종 Fine-tune 준비

| 팀원 | A | B | C | D | E |
|------|---|---|---|---|---|
| 실험1 | v8n 최종 Fine-tune #1 | v8n 최종 Fine-tune #2 | v8s Fine-tune #13 | v8s Fine-tune #14 | promising 후보 Fine-tune + augmentation 미세 조정 #1 |
| 실험2 | v8n 최종 Fine-tune #3 | v8n 최종 Fine-tune #4 | v8s Fine-tune #15 | v8s Fine-tune #16 | promising 후보 Fine-tune + augmentation 미세 조정 #2 |

12/18 전략: 앙상블 후보 검증

| 팀원 | A | B | C | D | E |
|------|---|---|---|---|---|
| 실험1 | promising 후보 검증 #1 | promising 후보 검증 #2 | 앙상블 실험 #1 | 앙상블 실험 #2 | 앙상블 후보 조합/미세 튜닝 #1 |
| 실험2 | promising 후보 검증 #3 | promising 후보 검증 #4 | 앙상블 실험 #3 | 앙상블 실험 #4 | 앙상블 후보 조합/미세 튜닝 #2 |

12/19 전략: 앙상블 성능 검증

| 팀원 | A | B | C | D | E |
|------|---|---|---|---|---|
| 실험1 | promising 후보 반복 학습 #1 | promising 후보 반복 학습 #2 | 앙상블 실험 #5 | 앙상블 실험 #6 | 앙상블 후보 최종 Fine-tune #1 |
| 실험2 | promising 후보 반복 학습 #3 | promising 후보 반복 학습 #4 | 앙상블 실험 #7 | 앙상블 실험 #8 | 앙상블 후보 최종 Fine-tune #2 |

12/20 전략: 앙상블 후보 최종 점검

| 팀원 | A | B | C | D | E |
|------|---|---|---|---|---|
| 실험1 | promising 후보 검증 #5 | promising 후보 검증 #6 | 앙상블 테스트 #9 | 앙상블 테스트 #10 | 앙상블 최종 후보 조합 실험 #1 |
| 실험2 | promising 후보 검증 #7 | promising 후보 검증 #8 | 앙상블 테스트 #11 | 앙상블 테스트 #12 | 앙상블 최종 후보 조합 실험 #2 |

12/21 전략: 최종 Fine-tune & 앙상블 결정

| 팀원 | A | B | C | D | E |
|------|---|---|---|---|---|
| 실험1 | promising 후보 반복 Fine-tune | promising 후보 반복 Fine-tune | promising 후보 반복 Fine-tune | promising 후보 반복 Fine-tune | 최종 앙상블 결정 #1 |
| 실험2 | promising 후보 반복 Fine-tune | promising 후보 반복 Fine-tune | promising 후보 반복 Fine-tune | promising 후보 반복 Fine-tune | 최종 앙상블 결정 #2 |
