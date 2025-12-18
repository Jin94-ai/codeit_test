"""
Stage 1: Pill Detector 학습
- 단일 클래스 (Pill) 탐지
- 목표: 이미지에서 알약 위치(bbox)만 정확히 찾기
"""

import os
import warnings
warnings.filterwarnings('ignore')

import wandb
from ultralytics import YOLO


################### Stage 1: Pill Detector ###################

# W&B 초기화
try:
    wandb.init(
        project="codeit_team8",
        entity="codeit_team8",
        name="pill_detector",
        config={
            "model": "yolo11s.pt",
            "data": "data/yolo/pills.yaml",
            "epochs": 30,
            "imgsz": 640,
            "batch": 16,
            "task": "Stage1_Detector",
        },
        settings=wandb.Settings(init_timeout=120)
    )
except Exception as e:
    print(f"W&B 연결 실패, offline 모드로 전환: {e}")
    wandb.init(mode="offline")

# YOLO11s 모델
model = YOLO("yolo11s.pt")

# 학습 - 단일 클래스 탐지에 최적화
model.train(
    data="data/yolo/pills.yaml",
    epochs=30,
    imgsz=640,
    batch=16,
    patience=10,
    half=True,

    # Augmentation (객체 탐지에 효과적인 설정)
    mosaic=1.0,
    mixup=0.1,
    fliplr=0.5,
    flipud=0.0,
    scale=0.5,
    translate=0.1,

    # 최적화
    optimizer="AdamW",
    lr0=0.01,
    lrf=0.01,

    # 저장
    project="runs/detect",
    name="pill_detector",
    exist_ok=True,

    # 메모리 최적화
    workers=2,
)

print("\n" + "=" * 60)
print("Stage 1: Pill Detector 학습 완료!")
print("best.pt 위치: runs/detect/pill_detector/weights/best.pt")
print("=" * 60)

wandb.finish()
