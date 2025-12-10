import os
import sys
import cv2
import numpy as np
import torch

_current_script_file = os.path.abspath(__file__)
_current_script_dir = os.path.dirname(_current_script_file)
PROJECT_ROOT = os.path.abspath(os.path.join(_current_script_dir, '../../..'))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    print(f"DEBUG: Added '{PROJECT_ROOT}' to sys.path in ny_preprocessing.py.")

from .config import TRAIN_IMG_DIR, TRAIN_ANN_DIR

def preprocess_image_for_yolo_tensor(image: np.ndarray) -> torch.Tensor:
    """
    Albumentations transforms가 없을 경우에 적용되는 기본 이미지 전처리 함수.
    이미지 픽셀 값을 0-1 범위로 정규화하고 PyTorch 텐서 (CxHxW)로 변환.
    """
    image_float = image.astype(np.float32) / 255.0  # 0-1 정규화
    image_tensor = torch.from_numpy(image_float).permute(2, 0, 1) # HWC -> CHW
    return image_tensor

def denormalize_image_for_display(image_tensor: torch.Tensor) -> np.ndarray:
    """
    시각화를 위해 정규화된 이미지 텐서를 다시 NumPy 배열로 변환하고 픽셀 범위 클리핑.
    """
    image_np = image_tensor.permute(1, 2, 0).cpu().numpy() # CHW -> HWC
    image_np = np.clip(image_np * 255.0, 0, 255).astype(np.uint8) # 0-1 -> 0-255 및 클리핑
    return image_np