import os

# 기본 경로 설정
BASE_DIR = "data"

TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train_images")
TEST_IMG_DIR = os.path.join(BASE_DIR, "test_images")
TRAIN_ANN_DIR = os.path.join(BASE_DIR, "train_annotations")

# YOLOv8용 출력 루트
YOLO_ROOT = "datasets/pills"

# Train/Val split 설정
VAL_RATIO = 0.2
SPLIT_SEED = 42
