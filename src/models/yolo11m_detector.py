"""
YOLO11m Detector 학습 스크립트
- 목표: 3개 알약 이미지 138개 + 4개 알약 이미지 705개 = 총 3,234개 알약 검출
- 실행: python src/models/yolo11m_detector.py
"""
import os
import sys
from pathlib import Path

# 프로젝트 루트를 PYTHONPATH에 추가
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import json
from datetime import datetime
import numpy as np
import pandas as pd
import random
import torch
from ultralytics import YOLO

import warnings
warnings.filterwarnings('ignore')

print(f"Working directory: {os.getcwd()}")


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


# =================== Config ===================
CONFIG = {
    "model": "yolo11m.pt",  # 12m은 GPU 메모리 부족, 11m 사용
    "data": "data/yolo/pills.yaml",
    "epochs": 50,
    "imgsz": 1280,  # 640 → 1280 (작은 알약 검출 향상)
    "batch": 2,  # imgsz 1280에서 6GB GPU용
    "patience": 15,  # 15 에폭 동안 개선 없으면 중지

    # Detection thresholds (낮춰서 누락 줄이기)
    "conf": 0.3,
    "iou": 0.5,

    # Project
    "project": "runs/detect",
    "name": "yolo11m_detector_1280",  # 구분을 위해 이름 변경
}


# =================== Training ===================
def train():
    print("=" * 60)
    print("YOLO11m Detector 학습 시작")
    print("=" * 60)
    print(f"Model: {CONFIG['model']}")
    print(f"Epochs: {CONFIG['epochs']}")
    print(f"Image Size: {CONFIG['imgsz']}")
    print(f"Confidence Threshold: {CONFIG['conf']}")
    print("=" * 60)

    # 모델 로드
    model = YOLO(CONFIG["model"])

    # 학습
    model.train(
        data=CONFIG["data"],
        epochs=CONFIG["epochs"],
        imgsz=CONFIG["imgsz"],
        batch=CONFIG["batch"],
        seed=42,

        # 저장
        save=True,
        save_period=10,

        # Early stopping
        patience=CONFIG["patience"],

        # Augmentation (검출 성능 향상)
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        degrees=15,
        translate=0.1,
        scale=0.3,
        fliplr=0.5,
        flipud=0.0,
        mosaic=1.0,
        mixup=0.1,

        # 최적화
        optimizer="AdamW",
        lr0=0.01,
        lrf=0.01,

        # 프로젝트 설정
        project=CONFIG["project"],
        name=CONFIG["name"],
        exist_ok=True,
    )

    return model


# =================== Inference ===================
def inference(model, test_dir="data/test_images", conf=0.3):
    """테스트 이미지 추론 및 검출 수 확인"""
    print("\n" + "=" * 60)
    print("추론 시작")
    print("=" * 60)

    results = model.predict(
        source=test_dir,
        imgsz=CONFIG["imgsz"],
        conf=conf,
        iou=CONFIG["iou"],
        max_det=10,  # 이미지당 최대 검출 수
        agnostic_nms=True,
        verbose=False
    )

    # 검출 수 카운트
    detection_counts = {}
    total_detections = 0

    for res in results:
        img_name = os.path.basename(res.path)
        image_id = os.path.splitext(img_name)[0]
        num_boxes = len(res.boxes)
        detection_counts[image_id] = num_boxes
        total_detections += num_boxes

    # 통계
    from collections import Counter
    dist = Counter(detection_counts.values())

    print(f"\n총 이미지: {len(results)}개")
    print(f"총 검출: {total_detections}개")
    print(f"목표: 3,234개")
    print(f"\n[검출 수 분포]")
    for count in sorted(dist.keys()):
        num_images = dist[count]
        status = "✓" if count >= 3 else "⚠️"
        print(f"  {count}개: {num_images}개 이미지 {status}")

    # 3개 검출, 4개 검출 비교
    count_3 = dist.get(3, 0)
    count_4 = dist.get(4, 0)
    print(f"\n[목표 비교]")
    print(f"  3개 검출 이미지: {count_3}개 (목표: 138개)")
    print(f"  4개 검출 이미지: {count_4}개 (목표: 705개)")

    return results, detection_counts


# =================== Submission ===================
def create_submission(results, output_path=None):
    """Kaggle 제출용 CSV 생성"""
    rows = []
    annotation_id = 1

    for res in results:
        img_name = os.path.basename(res.path)
        image_id = int(os.path.splitext(img_name)[0])
        boxes = res.boxes

        for box, conf in zip(boxes.xyxy, boxes.conf):
            x1, y1, x2, y2 = box.tolist()
            rows.append({
                "annotation_id": annotation_id,
                "image_id": image_id,
                "category_id": 0,  # Detector는 단일 클래스
                "bbox_x": int(x1),
                "bbox_y": int(y1),
                "bbox_w": int(x2 - x1),
                "bbox_h": int(y2 - y1),
                "score": round(float(conf), 2)
            })
            annotation_id += 1

    os.makedirs("outputs/submissions", exist_ok=True)

    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"outputs/submissions/submission_yolo11m_{timestamp}.csv"

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\n✓ Submission 생성: {output_path} ({len(df)}개 예측)")

    return df


# =================== Main ===================
def main():
    # 학습만 수행 (추론/제출은 submit.py에서)
    model = train()

    print("\n" + "=" * 60)
    print("학습 완료!")
    print("=" * 60)
    print(f"모델 저장 위치: {CONFIG['project']}/{CONFIG['name']}/weights/best.pt")
    print(f"\n제출 파일 생성: python src/inference/submit.py")


if __name__ == "__main__":
    main()
