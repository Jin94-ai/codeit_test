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

# ✅ DAY1(12/12) (test용) 실험 플랜 (모델 특성 분석 및 wandb 최적화)

## 목표:

- input size 분석
- 모델 특성 분석: yolov8n/ yolo8s/ yolo11n/ yolo11s/ yolo12n
- wandb: image predict, tag, 실험 이름 형식 표준화

#### 실험 이름(expID) 규칙

실험 이름은 exp{Day}{Order}_{model}_{imgsz} 형식으로 만든다.
앞의 {Day}{Order}는 해당 날짜의 실험 순서를 뜻한다. 예를 들어 exp103은 Day 1의 3번째 실험이다. 뒤의 {model}_{imgsz}는 사용한 모델 종류와 입력 크기를 나타낸다. 예: exp103_y8n_768.


| 실험 번호      | 데이터셋     | 증강 기법  | 모델                  | 입력 크기 | 실험 이름(expID 규칙) |
| ---------- | -------- | ------ | ------------------- | ----- | --------------- |
| **exp101** | pills v1 | 기본 Aug | yolov8n      | 512   | exp101_y8n_512  |
| **exp102** | pills v1 | 기본 Aug | yolov8n      | 640   | exp102_y8n_640  |
| **exp103** | pills v1 | 기본 Aug | yolov8n      | 768   | exp103_y8n_768  |
| **exp104** | pills v1 | 기본 Aug | yolov8s     | 640   | exp104_y8s_640  |
| **exp105** | pills v1 | 기본 Aug | yolo11     | 640   | exp105_y11_640  |
| **exp106** | pills v1 | 기본 Aug | yolo11s | 640   | exp106_y11s_640 |
| **exp107** | pills v1 | 기본 Aug | yolo12n   | 640   | exp107_y12_640  |



# ✅ DAY2(12/15) 실험 플랜 (데이터 최적화: Augmentation + Input size)

목표: 탐색 단계에서 다양한 Augmentation + Input size 조합을 빠르게 돌려 Top-5 promising 조합을 선별. 그 후 Narrow-down(한 변수씩 테스트)으로 원인 규명.


## 1차 Best Aug Exploration:

- 목표: 각각 Aug 변수를 약/중/강 조합으로 나누어 빠르게 1차 Best aug 조합을 잡는다.

| 담당자     | 데이터셋          | 증강 기법                    | 하이퍼파라미터                        | 입력 크기              |
| ------- | ------------- | ------------------------ | ------------------------------ | ------------------ |
| **이진석** | pills 데이터(v2) | 기본 Aug (YOLO 기본값)        | epochs 50 / lr 0.01 / batch 16 | 512 / 640 / 768    |
| **김민우** | pills 데이터(v2) | Aug 조합 A·B·C 세트          | epochs 50 / lr 0.01 / batch 16          | 640                |
| **김나연** | pills 데이터(v2) | HSV 약·중·강                | epochs 50 / lr 0.01 / batch 16         | 640                |
| **김보윤** | pills 데이터(v2) | rotate(0/5/10), scale 범위 | epochs 50 / lr 0.01 / batch 16        | 640                |
| **황유민** | pills 데이터(v2) | baseline + 유망한 Aug 1~2개  | epochs 증가(50→80), multi-seed   | 640 → 필요 시 768/960 |

### 세부 실행 규칙

- Baseline 먼저: 모든 팀원은 실험 시작 전에 baseline(640, default aug) 기본 성능 동일하게 나오는 지 확인.


- Exploration 규칙: 실험1 단계에서는 aug와 imgsz를 혼합해도 괜찮다 (빠른 후보 발굴 목적). 각 팀원은 최대 3 run(빠른)만 수행.

- Promising 기준(승격 조건): promising = baseline mAP@[.5:.95] 대비 +1.5% 이상 OR small-object mAP@0.5 개선 + inference latency 허용 범위 내(예: <1.5×)

- Early stop 기준: val mAP 개선 없으면 patience=5로 조기중단 권장(자원 절약).

## 2차: 정교 분석:

- 목표: 1차 Best Aug를 베이스로, 각각 조합에서 단일 변수를 조정해가며 fine tune으로 best 5 조합을 찾는다.

| 담당자     | 데이터셋     | 증강 기법(실험 1 best Aug 반영)                     | 하이퍼 파라미터                                   | 입력 크기           |
| ------- | -------- | ------------------------------------------- | ------------------------------------------ | --------------- |
| **이진석** | pills v2 | best Aug  + imgsz 변화 + 작은 객체 탐지 세부 분석            | epoch 50, lr 0.01, batch 16, multi-seed(3) | 512 / 640 / 768 |
| **김민우** | pills v1 | best Aug + Mosaic/ 강도 변경(0 / 0.1 / 0.2)  | epoch 50, batch 16                         | 640             |
| **김나연** | pills v1 | best HSV + hsv_s / hsv_v / gamma 단일 변수  | epoch 50, batch 16                         | 640             |
| **김보윤** | pills v1 | best rotate + degree 변경(0 → 3 → 5 → 10) | epoch 50, batch 16                         | 640             |
| **황유민** | pills v1 | best top Aug + fine-tune + 고해상도 시험      | epoch 60, multi-seed(3)                    | 640 → 768 → 960 |


