"""
Stage 2: Pill Classifier 학습
- 74개 클래스 분류
- YOLO Classification 모드 사용
"""

import warnings
warnings.filterwarnings('ignore')

import wandb
from ultralytics import YOLO


################### Stage 2: Pill Classifier ###################

# W&B 초기화
try:
    wandb.init(
        project="codeit_team8",
        entity="codeit_team8",
        name="pill_classifier",
        config={
            "model": "yolo11s-cls.pt",
            "data": "data/yolo_cls",
            "epochs": 50,
            "imgsz": 224,
            "batch": 32,
            "task": "Stage2_Classifier",
        },
        settings=wandb.Settings(init_timeout=120)
    )
except Exception as e:
    print(f"W&B 연결 실패, offline 모드로 전환: {e}")
    wandb.init(mode="offline")

# YOLO11s Classification 모델
model = YOLO("yolo11s-cls.pt")

# 학습 - Classification
model.train(
    data="data/yolo_cls",
    epochs=50,
    imgsz=224,
    batch=32,
    patience=15,

    # Augmentation (Classification에 효과적)
    hsv_h=0.015,
    hsv_s=0.4,
    hsv_v=0.4,
    degrees=15,
    translate=0.1,
    scale=0.2,
    fliplr=0.5,
    flipud=0.0,

    # 최적화
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,

    # 저장
    project="runs/classify",
    name="pill_classifier",
    exist_ok=True,

    # 메모리 최적화
    workers=2,
)

print("\n" + "=" * 60)
print("Stage 2: Pill Classifier 학습 완료!")
print("best.pt 위치: runs/classify/pill_classifier/weights/best.pt")
print("=" * 60)

wandb.finish()
