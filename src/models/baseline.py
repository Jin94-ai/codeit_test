import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.notebook import tqdm
import random

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

from ultralytics import YOLO

model = YOLO("yolov8n.pt")   
model.train(
    data="data/yolo/pills.yaml",
    epochs=50,
    imgsz=640,
)

results = model.predict(source="data/test_images/", imgsz=640)




rows = []
annotation_id = 1

for res in results:
    img_name = os.path.basename(res.path)  # 예: "1.png"
    image_id = int(os.path.splitext(img_name)[0])  # 파일명 숫자 추출
    boxes = res.boxes  # Boxes 객체
    for box, cls, score in zip(boxes.xyxy, boxes.cls, boxes.conf):
        x1, y1, x2, y2 = box.tolist()
        bbox_x = int(x1)
        bbox_y = int(y1)
        bbox_w = int(x2 - x1)
        bbox_h = int(y2 - y1)
        rows.append({
            "annotation_id": annotation_id,
            "image_id": image_id,
            "category_id": int(cls)+1,  # YOLO 클래스 0-based → 캐글 1-based
            "bbox_x": bbox_x,
            "bbox_y": bbox_y,
            "bbox_w": bbox_w,
            "bbox_h": bbox_h,
            "score": float(score)
        })
        annotation_id += 1

df = pd.DataFrame(rows)
df.to_csv("submission.csv", index=False)
