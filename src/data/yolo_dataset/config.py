import os
from pathlib import Path

# 프로젝트 루트 기준으로 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent  # src/data/yolo_dataset에서 3단계 위
BASE_DIR = PROJECT_ROOT / "data"

# 원본 Kaggle 데이터 경로 (Stage 1: Detector용)
TRAIN_IMG_DIR = str(BASE_DIR / "train_images")
TRAIN_ANN_DIR = str(BASE_DIR / "train_annotations")

# AIHub Detector 데이터 경로 (Stage 1: Detector용 - 원본 조합 이미지)
AIHUB_DETECTOR_IMG_DIR = str(BASE_DIR / "aihub_detector" / "images")
AIHUB_DETECTOR_ANN_DIR = str(BASE_DIR / "aihub_detector" / "annotations")

# Cropped 데이터 경로 (Stage 2: Classifier용)
CROPPED_IMG_DIR = str(BASE_DIR / "cropped" / "images")
CROPPED_ANN_DIR = str(BASE_DIR / "cropped" / "annotations")

YOLO_ROOT = str(BASE_DIR / "yolo")

VAL_RATIO = 0.2
SPLIT_SEED = 42
