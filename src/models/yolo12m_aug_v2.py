import os
import sys
import json

current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file_path, '..', '..', '..'))

if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.notebook import tqdm
import random
import yaml
import torch
from src.models.callbacks import wandb_train_logging, wandb_val_logging

import wandb
from ultralytics import YOLO

# 경고 무시 (선택 사항)
import warnings
warnings.filterwarnings('ignore')

# Matplotlib 폰트 매니저
from matplotlib import font_manager

# ---- 코랩 기본 한글 폰트 자동 설정 ----
# 코랩에 기본적으로 설치된 폰트 후보들
korean_fonts = ["NanumGothic", "AppleGothic", "Malgun Gothic", "DejaVu Sans"]

# 사용 가능한 폰트를 자동으로 탐색
available_fonts = set(f.name for f in font_manager.fontManager.ttflist)
selected_font = None

for font in korean_fonts:
    if font in available_fonts:
        selected_font = font
        break

if selected_font:
    plt.rcParams['font.family'] = selected_font
    plt.rcParams['axes.unicode_minus'] = False
    print(f"코랩에서 사용 가능한 한글 폰트 설정 완료: {selected_font}")
else:
    print("경고: 사용 가능한 기본 한글 폰트를 찾지 못했습니다. 수동 설정 필요")



# =================== Seed Fix ===================


def seed_fix(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_fix(42)
# ==============================================


################### Model Run ###################



import albumentations as A 
import cv2

AUGMENTATION_METHOD_DESCRIPTION = "Albumentations (뒤집기/회전/이동, 색상/밝기/노이즈/블러, 드롭아웃/JPEG)"
print(f"\n{AUGMENTATION_METHOD_DESCRIPTION} 증강 파이프라인을 사용하여 모델을 훈련합니다.\n")

# --- Albumentations 커스텀 변환 정의 ---
custom_transforms_alb = [
    # --- 기하학적 변형 ---
    A.HorizontalFlip(p=0.5), # 좌우 반전
    A.Rotate(limit=60, p=0.4, border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0)), # ±60도 범위 회전
    A.ShiftScaleRotate(shift_limit=0.075, scale_limit=0.15, rotate_limit=0, p=0.4, border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0)), # 약한 이동 및 확대/축소

    # --- 색상 및 노이즈 변형 ---
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
    A.GaussNoise(p=0.1),
    A.Blur(blur_limit=3, p=0.1),
    A.MotionBlur(blur_limit=3, p=0.1),
    A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=20, p=0.3),
    A.RandomGamma(p=0.2),
    
    # --- 가려짐 및 파일 손상 시뮬레이션 ---
    A.CoarseDropout(max_holes=1, max_height=0.1, max_width=0.1, min_holes=1, fill_value=0, p=0.1), # 컷아웃
    A.JpegCompression(quality_lower=70, quality_upper=90, p=0.1), # JPEG 압축 손실
]

# W&B 초기화
wandb.init(
    project="codeit_team8",
    entity = "codeit_team8",
    config={
        "model": "yolo12m.pt",
        "data": "data/yolo/pills.yaml",
        "epochs": 50,
        "imgsz": 640,
        "conf": 0.5,
        "iou": 0.5,
        "max_det": 100,
        "augmentation_pipeline": AUGMENTATION_METHOD_DESCRIPTION # W&B config에 기록
    }
)

model = YOLO("yolo12m.pt")

model.add_callback("on_fit_epoch_end", wandb_train_logging)
model.add_callback("on_val_end", wandb_val_logging)  

data_config_path = "data/yolo/pills.yaml"
temp_data_config_path = "data/yolo/pills_for_alb_direct.yaml" # 임시 YAML 파일 경로명 변경

# 원본 pills.yaml 로드
with open(data_config_path, 'r', encoding='utf-8') as f:
    data_config = yaml.safe_load(f)

updated_data_config_for_direct_alb = data_config.copy() # 원본 복사 (클린 버전을 위해)

# 업데이트된 내용을 새로운 임시 YAML 파일로 저장
with open(temp_data_config_path, 'w', encoding='utf-8') as f:
    yaml.dump(updated_data_config_for_direct_alb, f, default_flow_style=False, allow_unicode=True) 

print(f"임시 data.yaml 생성 완료: {temp_data_config_path} (이 파일은 Albumentations 직접 인자 전달에 사용됩니다.)")

# 모델 훈련 시작
model.train(
    data=temp_data_config_path, # 순수 데이터셋 설정 YAML 파일 사용
    epochs=50,
    imgsz=640,
    project=wandb.run.project, # W&B 프로젝트와 연동
    name=wandb.run.name,       # W&B 런 이름과 연동
    # augment=False, 
    augmentations=custom_transforms_alb,
)

if hasattr(model, "trainer") and hasattr(model.trainer, "metrics"):
    metrics = model.trainer.metrics
    wandb.log({k: float(v) for k, v in metrics.items()})

results = model.predict(
    source="data/test_images/",
    imgsz=640,
    conf=0.5,
    iou=0.5,
    max_det=100,
    agnostic_nms=True,
    verbose=False
)

# YOLO index → 원본 category_id 매핑 로드
with open("data/yolo/class_mapping.json", "r", encoding="utf-8") as f:
    yoloid_to_catid = json.load(f)
    yoloid_to_catid = {int(k): int(v) for k, v in yoloid_to_catid.items()}

rows = []
annotation_id = 1

for res in results:
    img_name = os.path.basename(res.path)
    image_id = int(os.path.splitext(img_name)[0])
    boxes = res.boxes
    for box, cls, score in zip(boxes.xyxy, boxes.cls, boxes.conf):
        x1, y1, x2, y2 = box.tolist()
        bbox_x = int(x1)
        bbox_y = int(y1)
        bbox_w = int(x2 - x1)
        bbox_h = int(y2 - y1)

        yolo_idx = int(cls)
        original_category_id = yoloid_to_catid[yolo_idx]

        rows.append({
            "annotation_id": annotation_id,
            "image_id": image_id,
            "category_id": original_category_id,
            "bbox_x": bbox_x,
            "bbox_y": bbox_y,
            "bbox_w": bbox_w,
            "bbox_h": bbox_h,
            "score": float(score)
        })
        annotation_id += 1

os.makedirs("outputs/submissions", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

df = pd.DataFrame(rows)
output_path = f"outputs/submissions/submission_{timestamp}.csv"
df.to_csv(output_path, index=False)

print(f"\n✓ Submission 생성 완료: {output_path} ({len(df)}개 예측)")

wandb.finish()
