import os
import sys 
import albumentations as A
import cv2 
from albumentations.pytorch import ToTensorV2

_current_script_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_current_script_dir, '../../..'))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    print(f"DEBUG: Added '{PROJECT_ROOT}' to sys.path in ny_augmentation.py.")

TARGET_IMAGE_SIZE = (640, 640) 

def get_train_transforms(target_size: tuple = TARGET_IMAGE_SIZE) -> A.Compose:
    """
    훈련 데이터셋에 적용할 Albumentations 데이터 증강 파이프라인을 정의
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5), 
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
        A.Blur(blur_limit=3, p=0.1),
        A.GaussNoise(p=0.1),
        A.MotionBlur(blur_limit=3, p=0.1),
        A.RandomGamma(p=0.2),

        A.LongestMaxSize(max_size=max(target_size), p=1.0),
        A.PadIfNeeded(
            min_height=target_size[1],
            min_width=target_size[0],
            border_mode=cv2.BORDER_CONSTANT,
            value=(0,0,0), 
            p=1.0
        ),
        
        A.Normalize(
            mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0, p=1.0
        ),
        ToTensorV2(p=1.0) 
    ], 
    bbox_params=A.BboxParams(
        format='yolo', # 중요! YOLO를 위한 것이라면 'yolo' 포맷
        label_fields=['class_labels'],
        min_area=1.0,         
        min_visibility=0.0,   
        clip=True,            
        min_width=1,          
        min_height=1          
    ))

def get_val_transforms(target_size: tuple = TARGET_IMAGE_SIZE) -> A.Compose:
    """
    검증 데이터셋에 적용할 Albumentations 변환 파이프라인을 정의
    """
    return A.Compose([
        A.LongestMaxSize(max_size=max(target_size), p=1.0),
        A.PadIfNeeded(
            min_height=target_size[1],
            min_width=target_size[0],
            border_mode=cv2.BORDER_CONSTANT,
            value=(0,0,0), 
            p=1.0
        ),
        A.Normalize(
            mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0, p=1.0
        ),
        ToTensorV2(p=1.0)
    ], 
    bbox_params=A.BboxParams(
        format='yolo', # 중요! YOLO를 위한 것이라면 'yolo' 포맷
        label_fields=['class_labels'],
        min_area=1.0,
        min_visibility=0.0,
        clip=True,
        min_width=1,
        min_height=1
    ))