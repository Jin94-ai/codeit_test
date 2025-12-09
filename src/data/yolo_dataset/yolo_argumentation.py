# src/data/yolo_dataset/yolo_augmentation.py
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import random
import torch

SEED = 42


def set_global_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_train_transform():
    """
    YOLO 형식 bbox (x_center, y_center, w, h, 0~1)를 사용하는
    on-the-fly 증강 파이프라인을 반환한다.
    - 플립 없음
    - 밝기/대비 ±0.3
    - 색조/채도/밝기 소폭 변화
    - 회전 ±15도
    """
    transform = A.Compose(
        [
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.7,
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=20,
                p=0.5,
            ),
            A.Rotate(
                limit=15,
                border_mode=0,  # 패딩은 검정
                p=0.7,
            ),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="yolo",          # x_c, y_c, w, h (0~1)
            label_fields=["labels"],
            min_visibility=0.1,
            min_area=0.0,
        ),
    )
    return transform