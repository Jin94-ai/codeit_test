# 모델별 기본 특징

1) 모델 크기별 특징

nano (n): 최소 파라미터

small (s): n보다 표현력 증가

medium (m): m보다 복잡한 패턴·밀집 

2) YOLO 버전별 구조 특징 (v8 → v11 → v12)

v8: 균형

v11: 탐지 민감도 강화, recall 증가

v12: attention 기반, 중요한 feature 집중

3) 지표에서 나타나는 모델별 차이

v8n: 안정적임, recall/precision 편향 없음

v11n: recall 우수, precision 저하 가능, 종합 점수는 v8n보다 낮음.

v12s/m: 모델 크기 증가에 따라 성능 안정적 상승



| expID                 | Dataset | Input Size | Model   | mAP@50  | mAP@50-95 | Kaggle Score |
| -----------------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------- | ---------- | ------------ |
| exp101_yolo8n_512     | V1      | 512        | YOLO8n  | N/A | N/A        | 0.68113 |
| exp102_yolo11n        | V1      | 640        | YOLO11n | 0.8749 | 0.85564 | 0.68536 |
| exp103_yolo8n         | V1      | 640        | YOLO8n  | 0.91543 | 0.90185 | 0.70231 |
| exp104_yolo8s         | V1      | 640        | YOLO8s  | 0.98857 | 0.98263 | 0.79748 |
| exp105_yolo11s        | V1      | 640        | YOLO11s | 0.99025 | 0.97986 | 0.79382 |
| exp106_yolo12s        | V1      | 640        | YOLO12s | 0.97656 | 0.96395 | 0.79837 |
| exp107_yolo12m        | V1      | 640        | YOLO12m | 0.9737 | 0.96746 | 0.81182 |
| exp108_yolo8n_V2      | V2      | 640        | YOLO8n  | N/A  | N/A        | 0.57000 |



# Best: **exp107_yolo12m** 0.81182  

insight:
데이터 보충 버전의 성능 개선 필요, 최신 버전일수록, 모델 사이즈가 클수록 좋은 성능, 편가 기준이 map@0.75:0.95여서 자세한 검출이 효과 있음.

데이터set 종류 wandb에서 구분되지 않음. 개선 필요한가