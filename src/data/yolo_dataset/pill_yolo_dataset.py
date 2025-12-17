import os
from ultralytics.data.dataset import YOLODataset
import cv2 
import yaml 

from src.data.yolo_dataset.ny_augmentation import (
    get_train_transforms_conservative,
    get_train_transforms_balanced,
    get_train_transforms_aggressive,
    get_val_transforms
)

class PillYoloDataset(YOLODataset):
    def __init__(self, *args, **kwargs):
        print(f"\n--- DEBUG: PillYoloDataset __init__ called. ---")
        print(f"--- DEBUG: Received kwargs: {kwargs} ---") # kwargs 전체를 출력
        
        # kwargs에서 'data' 딕셔너리를 안전하게 추출
        data_from_kwargs = kwargs.get('data', {})
        print(f"--- DEBUG: Extracted 'data' from kwargs: {data_from_kwargs} ---")
        
        # 'data' 딕셔너리에서 'augmentation_pipeline_choice'를 추출. 없으면 'balanced'가 기본값
        augmentation_choice = data_from_kwargs.get('augmentation_pipeline_choice', 'balanced')
        print(f"--- DEBUG: Determined augmentation_choice: '{augmentation_choice}' ---")
        # --- END DEBUG ---

        imgsz = kwargs.get('imgsz', 640)
        
        super().__init__(*args, **kwargs)
        
        if self.split == 'train':
            print(f"[{self.__class__.__name__}] Applying '{augmentation_choice}' Albumentations transforms to training data (imgsz={imgsz})...")
            if augmentation_choice == 'conservative':
                self.transforms = get_train_transforms_conservative(target_size=(imgsz, imgsz))
            elif augmentation_choice == 'balanced':
                self.transforms = get_train_transforms_balanced(target_size=(imgsz, imgsz))
            elif augmentation_choice == 'aggressive':
                self.transforms = get_train_transforms_aggressive(target_size=(imgsz, imgsz))
            else: # 알 수 없는 선택일 경우 balanced를 기본값으로
                print(f"[{self.__class__.__name__}] Warning: Unknown augmentation choice '{augmentation_choice}'. Applying 'balanced' transforms as default.")
                self.transforms = get_train_transforms_balanced(target_size=(imgsz, imgsz))
        elif self.split == 'val': # 검증 데이터셋에는 기본 전처리 (리사이즈/정규화)만 적용
            print(f"[{self.__class__.__name__}] Applying validation transforms to validation data (imgsz={imgsz})...")
            self.transforms = get_val_transforms(target_size=(imgsz, imgsz))
        else: 
            self.transforms = get_val_transforms(target_size=(imgsz, imgsz)) # 예측에도 동일한 전처리 적용 (정규화/리사이즈)
        
        print(f"[{self.__class__.__name__}] Dataset initialized for split: {self.split}")