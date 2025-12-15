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

# ================= ADD DATASET =================

ADD_BASE_DIR = BASE_DIR / "add"
ADD_ANN_DIR = ADD_BASE_DIR / "annotations"
ADD_IMG_DIR = ADD_BASE_DIR / "images"
ADDED_TRAIN_ANN_DIR = str(BASE_DIR / "train_annotations" / "added_train_annotations")

AIHUB_SINGLE_DIR = "data/aihub_single"  # 다운로드된 single 데이터
AIHUB_BASE_DIR = "D:/166.약품식별 인공지능 개발을 위한 경구약제 이미지 데이터/01.데이터/1.Training"
AIHUB_ANN_DIR = f"{AIHUB_BASE_DIR}/라벨링데이터/단일경구약제 5000종"
AIHUB_IMG_DIR = f"{AIHUB_BASE_DIR}/원천데이터/단일경구약제 5000종"
