import os
from pathlib import Path

# 프로젝트 루트 기준으로 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent  # src/data/yolo_dataset에서 3단계 위
BASE_DIR = PROJECT_ROOT / "data"

TRAIN_IMG_DIR = str(BASE_DIR / "train_images")
TRAIN_ANN_DIR = str(BASE_DIR / "train_annotations")

YOLO_ROOT = str(BASE_DIR / "yolo")

VAL_RATIO = 0.2
SPLIT_SEED = 42
