import os

BASE_DIR = "data"

TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train_images")
TRAIN_ANN_DIR = os.path.join(BASE_DIR, "train_annotations")

YOLO_ROOT = "datasets/pills"

VAL_RATIO = 0.2
SPLIT_SEED = 42
