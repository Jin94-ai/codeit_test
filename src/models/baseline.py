import os
import json
from datetime import datetime
import pandas as pd

import wandb
from ultralytics import YOLO

# 경고 무시
import warnings
warnings.filterwarnings('ignore')


################### Model Run ###################

# W&B 초기화
wandb.init(
    project="codeit_team8",
    entity="codeit_team8",
    config={
        "model": "yolo12m.pt",
        "data": "data/yolo/pills.yaml",
        "epochs": 50,
        "imgsz": 640,
        "batch": 16,
    }
)

# YOLOv12m 모델
model = YOLO("yolo12m.pt")

# 학습 - 기본 설정 (최소 augmentation)
model.train(
    data="data/yolo/pills.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    patience=10,
    half=True,
    rect=True,

    # 기본 augmentation만
    mosaic=1.0,
    fliplr=0.5,

    # 최적화
    optimizer="AdamW",
    lr0=0.01,
    lrf=0.01,

    # 저장
    project="runs/detect",
    name="train",
    exist_ok=False,
)

# 가장 최근 학습된 모델 찾기
import glob
train_dirs = sorted(glob.glob("runs/detect/train*"), key=os.path.getmtime)
latest_train = train_dirs[-1] if train_dirs else "runs/detect/train"
best_weights = f"{latest_train}/weights/best.pt"
print(f"\n✓ 사용 모델: {best_weights}")

# 예측
best_model = YOLO(best_weights)
results = best_model.predict(
    source="data/test_images/",
    imgsz=640,
    conf=0.3,
    iou=0.5,
    max_det=300,
    agnostic_nms=True,
    half=True,
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
