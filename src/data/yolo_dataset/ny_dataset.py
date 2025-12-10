import os
import sys
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, List, Tuple

# 이미지 처리를 위한 라이브러리
import cv2
import numpy as np
from PIL import Image

# Albumentations 및 PyTorch 텐서 변환 임포트
import albumentations as A
from albumentations.pytorch import ToTensorV2

_current_script_file = os.path.abspath(__file__)
_current_script_dir = os.path.dirname(_current_script_file)
PROJECT_ROOT = os.path.abspath(os.path.join(_current_script_dir, '../../..'))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    print(f"DEBUG: Added '{PROJECT_ROOT}' to sys.path in ny_dataset.py.")

from src.data.yolo_dataset.config import TRAIN_IMG_DIR, YOLO_ROOT 
from src.data.yolo_dataset.coco_parser import load_coco_tables_with_consistency 
from src.data.yolo_dataset.ny_preprocessing import preprocess_image_for_yolo_tensor 
from src.data.yolo_dataset.ny_augmentation import get_train_transforms, get_val_transforms, TARGET_IMAGE_SIZE 

# 전역 변수로 Dataset에서 필요한 데이터들을 캐시
_global_train_df = None
_global_val_df = None
_global_category_id_to_name = None
_global_class_name_to_id = None
_global_images_df = None


def _load_and_prepare_data_for_dataset():
    """
    `src/data/yolo_dataset/yolo_export.py`가 이미 실행되어 YOLO 디렉토리 구조와 라벨 파일들이 생성되었다고 가정하고,
    Dataset에서 필요한 DataFrame과 클래스 매핑을 이 파일 시스템으로부터 재구성
    """
    global _global_train_df, _global_val_df, _global_category_id_to_name, _global_class_name_to_id, _global_images_df
    if _global_train_df is not None:
        print("데이터가 이미 로드되어 있습니다. 재구성 건너뜀.")
        return 

    print("데이터셋 생성을 위해 필요한 데이터를 로드/재구성 중...")

    # 1. 원본 COCO 데이터 로드 (load_coco_tables_with_consistency 함수 사용)
    images_df_orig, annotations_df_orig, categories_df_orig = load_coco_tables_with_consistency()

    # 2. 클래스 매핑 재구성 (yolo_export.py에서 사용한 로직과 동일해야 함)
    unique_cat_ids_original = sorted(categories_df_orig["id"].unique().tolist())
    catid_to_yoloid = {cid: idx for idx, cid in enumerate(unique_cat_ids_original)} # 원본 COCO ID -> 0~N-1 ID
    _global_category_id_to_name = {idx: categories_df_orig[categories_df_orig["id"] == cid]["name"].iloc[0] 
                                   for idx, cid in enumerate(unique_cat_ids_original)} # 0~N-1 ID -> Name
    _global_class_name_to_id = {name: cid for cid, name in _global_category_id_to_name.items()} # Name -> 0~N-1 ID
    
    # 3. yolo_export.py가 생성한 라벨 파일로부터 이미지 ID 추출하여 훈련/검증 이미지 ID 분할 재구성
    train_labels_dir = os.path.join(YOLO_ROOT, "labels", "train")
    val_labels_dir = os.path.join(YOLO_ROOT, "labels", "val")

    if not os.path.exists(train_labels_dir) or not os.path.exists(val_labels_dir):
        raise FileNotFoundError(
            f"오류: YOLO 라벨 디렉토리 ({YOLO_ROOT}/labels/train 또는 val)를 찾을 수 없습니다."
            "`data/yolo_dataset/yolo_export.py`를 먼저 실행하여 데이터를 준비해주세요."
        )

    # 훈련 이미지 ID 추출
    train_image_ids_from_labels = []
    for label_file in os.listdir(train_labels_dir):
        if label_file.endswith(".txt"):
            image_stem = label_file.replace(".txt", "") # 확장자 없는 파일 이름 (예: K-00123)
            image_id_row = images_df_orig[images_df_orig['file_name'].apply(lambda x: x.split('.')[0] == image_stem)]
            if not image_id_row.empty:
                train_image_ids_from_labels.append(image_id_row['id'].iloc[0])

    # 검증 이미지 ID 추출
    val_image_ids_from_labels = []
    for label_file in os.listdir(val_labels_dir):
        if label_file.endswith(".txt"):
            image_stem = label_file.replace(".txt", "")
            image_id_row = images_df_orig[images_df_orig['file_name'].apply(lambda x: x.split('.')[0] == image_stem)]
            if not image_id_row.empty:
                val_image_ids_from_labels.append(image_id_row['id'].iloc[0])

    # 4. train_df 및 val_df 재구성
    full_ann_df_orig = pd.merge(annotations_df_orig, images_df_orig[['id', 'file_name', 'width', 'height']], 
                           left_on='image_id', right_on='id', suffixes=('', '_img'))
    full_ann_df_orig['class_id'] = full_ann_df_orig['category_id'].map(catid_to_yoloid)

    _global_train_df = full_ann_df_orig[
        full_ann_df_orig['image_id'].isin(train_image_ids_from_labels)
    ].reset_index(drop=True)

    _global_val_df = full_ann_df_orig[
        full_ann_df_orig['image_id'].isin(val_image_ids_from_labels)
    ].reset_index(drop=True)

    _global_images_df = images_df_orig 

    print("데이터 재구성 완료.")


class PillYoloDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: str = TRAIN_IMG_DIR, transforms=None):
        self.img_dir = img_dir
        self.transforms = transforms
        self.image_ids = df['image_id'].unique()
        self.grouped_df = df.groupby('image_id')
        self.error_count = 0
        self.max_retries = 10 

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, List[List[float]], Tuple[int, int]]:
        for _ in range(self.max_retries): 
            try:
                current_image_idx = idx 
                image_id = self.image_ids[current_image_idx]
                image_annotations = self.grouped_df.get_group(image_id) 

                file_name = image_annotations['file_name'].iloc[0]
                image_path = os.path.join(self.img_dir, file_name)
                
                # 이미지 로드 (Albumentations는 NumPy 배열을 선호)
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # OpenCV는 BGR, Albumentations는 RGB

                original_h, original_w = image.shape[:2]

                bboxes = []
                class_labels = []
                for _, row in image_annotations.iterrows():
                    # bbox: [xmin, ymin, width, height] (COCO format)
                    xmin, ymin, width, height = row['bbox']
                    
                    # YOLO format for Albumentations: [x_center_norm, y_center_norm, width_norm, height_norm]
                    x_center = (xmin + width / 2.0) / original_w
                    y_center = (ymin + height / 2.0) / original_h
                    norm_width = width / original_w
                    norm_height = height / original_h

                    bboxes.append([x_center, y_center, norm_width, norm_height])
                    class_labels.append(row['class_id'])

                # 데이터 증강 적용
                if self.transforms:
                    transformed = self.transforms(image=image, bboxes=bboxes, class_labels=class_labels)
                    image = transformed['image'] # PyTorch Tensor (Albumentations의 ToTensorV2가 이미 처리)
                    bboxes = transformed['bboxes'] # 변환된 YOLO 형식 바운딩 박스
                    class_labels = transformed['class_labels'] # 변환된 클래스 라벨 (정수)
                else:
                    # transforms가 없을 경우 preprocess_image_for_yolo_tensor 사용
                    image = preprocess_image_for_yolo_tensor(image) # ny_preprocessing.py에서 임포트된 함수 사용
                
                # 라벨 텐서 구성
                final_labels = []
                for i in range(len(bboxes)):
                    final_labels.append([float(class_labels[i])] + list(bboxes[i]))
                
                if not final_labels: # 라벨이 없으면 빈 텐서
                    labels_tensor = torch.empty((0, 5), dtype=torch.float32)
                else:
                    labels_tensor = torch.tensor(final_labels, dtype=torch.float32)


                # 타겟 딕셔너리 구성 (PillYoloDataset에서 리턴되는 형식)
                target = {
                    'boxes': labels_tensor[:, 1:],  # YOLO 학습 시 필요한 정규화된 바운딩 박스 좌표 (N, 4)
                    'labels': labels_tensor[:, 0].long(), # 클래스 ID (N,)
                    'image_id': image_id,
                    'orig_size': torch.tensor([original_h, original_w]), # 원본 이미지 크기
                    'idx': torch.tensor(idx) # 원본 데이터셋 인덱스
                }

                return image, target # 여기서 (image, target) 반환
                
            except Exception as e:
                self.error_count += 1
                if self.error_count % 100 == 0: # 100개마다 에러 카운트 출력 (너무 많으면 생략)
                    print(f"경고: 데이터 로드/처리 중 오류 발생 (현재 오류 {self.error_count}개): {e} 이미지 ID: {image_id} - 다음 재시도 또는 다음 이미지로 이동")
                idx = (idx + 1) % len(self) # 다음 인덱스로 이동하여 재시도
                if self.error_count > self.max_retries * 2: # 너무 많이 재시도하면 오류 발생시키고 종료
                     raise RuntimeError(f"최대 재시도 횟수 초과. 데이터셋 로드에 심각한 오류가 있습니다. 마지막 오류: {e}")
        
        # for 루프가 끝났는데도 데이터를 찾지 못했다면
        raise RuntimeError(f"데이터셋에서 {self.max_retries}회 재시도 후에도 유효한 데이터를 찾을 수 없습니다.")

# 테스트 코드 (이 파일 직접 실행 시)
if __name__ == '__main__':
    print("--- PillYoloDataset Module Test ---")
    _load_and_prepare_data_for_dataset() # 전역 데이터 로드

    if _global_train_df is not None and not _global_train_df.empty:
        print(f"\nInitialized PillYoloDataset with {len(_global_train_df['image_id'].unique())} unique training images.")
        
        # Albumentations 변환 파이프라인 가져오기
        train_transforms = get_train_transforms(TARGET_IMAGE_SIZE) 
        
        dataset = PillYoloDataset(_global_train_df, img_dir=TRAIN_IMG_DIR, transforms=train_transforms)

        from torch.utils.data import DataLoader
        
        def collate_fn_yolo(batch):
            """
            YOLO 모델용 커스텀 collate_fn: 
            이미지 리스트와 타겟(라벨, 박스 등) 리스트를 받아 배치로 묶음.
            """
            images = [item[0] for item in batch]
            targets = [item[1] for item in batch]
            
            # 이미지는 스택 (PyTorch의 default_collate는 이미지를 스택)
            images = torch.stack(images, 0) # [batch_size, C, H, W]

            # 타겟은 객체 수가 다르므로 리스트 오브 딕셔너리로 유지
            return images, targets

        data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn_yolo, num_workers=2) # num_workers는 디버깅 시 0으로 설정

        print("\nTesting DataLoader...")
        for i, (images, targets) in enumerate(data_loader):
            if i >= 2: # 2개의 배치만 테스트
                break
            print(f"\n--- Batch {i+1} ---")
            print(f"Images (batch_size={len(images)}): {images.shape} (batch tensor shape)")
            
            first_target = targets[0]
            print(f"First target in batch:")
            print(f"  Boxes: {first_target['boxes'].shape} (e.g., [num_objects, 4])")
            print(f"  Labels: {first_target['labels'].shape} (e.g., [num_objects])")
            print(f"  Image ID: {first_target['image_id']}")
            print(f"  Original Size: {first_target['orig_size']}")
        
        print("\nDataLoader test complete.")

    else:
        print("Global train DataFrame is empty. Cannot initialize dataset for test.")