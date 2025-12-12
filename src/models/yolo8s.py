import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.notebook import tqdm
import random
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



################### Model Run ###################




# W&B 초기화
wandb.init(
    project="codeit_team8",
    entity = "codeit_team8",
    config={
        "model": "yolov8s.pt",
        "data": "data/yolo/pills.yaml",
        "epochs": 50,
        "imgsz": 640,
        "conf": 0.5,
        "iou": 0.5,
        "max_det": 100,
    }
)

model = YOLO("yolov8s.pt")

model.add_callback("on_fit_epoch_end", wandb_train_logging)
model.add_callback("on_val_end", wandb_val_logging)  

model.train(
    data="data/yolo/pills.yaml",
    epochs=50,
    imgsz=640,
    save=True,
    plots=True,
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
    verbose=False,
    save=True,
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
