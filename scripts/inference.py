"""
Inference Pipeline for Object Detection
캐글 제출용 추론 파이프라인
"""
import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
from ultralytics import YOLO

# 프로젝트 루트를 sys.path에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_class_mapping(mapping_path: str = "data/yolo/class_mapping.json") -> dict:
    """YOLO index → 원본 category_id 매핑 로드"""
    with open(mapping_path, "r", encoding="utf-8") as f:
        yoloid_to_catid = json.load(f)
        return {int(k): int(v) for k, v in yoloid_to_catid.items()}


def run_inference(
    model_path: str,
    test_images_dir: str,
    output_dir: str,
    imgsz: tuple = (512, 896),
    conf: float = 0.5,
    iou: float = 0.6,
    max_det: int = 100,
    device: int = 0,
    half: bool = True,
    batch: int = 4,
) -> str:
    """
    객체 탐지 추론 실행 및 Kaggle 제출 파일 생성

    Args:
        model_path: 학습된 YOLO 모델 경로 (예: "runs/detect/train13/weights/best.pt")
        test_images_dir: 테스트 이미지 디렉토리 (예: "data/test_images/")
        output_dir: 출력 디렉토리 (예: "outputs/submissions")
        imgsz: 입력 이미지 크기
        conf: Confidence threshold
        iou: IoU threshold for NMS
        max_det: Maximum detections per image
        device: GPU device (0, 1, ... or 'cpu')
        half: Use FP16 inference
        batch: Batch size

    Returns:
        생성된 submission 파일 경로
    """

    # 1. 모델 로드
    print(f"[1/4] 모델 로딩: {model_path}")
    model = YOLO(model_path)

    # 2. 추론 실행
    print(f"[2/4] 추론 실행: {test_images_dir}")
    results = model.predict(
        source=test_images_dir,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        max_det=max_det,
        device=device,
        half=half,
        verbose=False,
        batch=batch,
        save=False  # 이미지 저장 비활성화
    )

    # 3. 클래스 매핑 로드
    print("[3/4] 클래스 매핑 로드")
    yoloid_to_catid = load_class_mapping()

    # 4. Kaggle 제출 형식으로 변환
    print("[4/4] Submission 파일 생성")
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

    # 5. CSV 저장
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = os.path.join(output_dir, f"submission_{timestamp}.csv")

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    print(f"\n✅ Submission 생성 완료: {output_path}")
    print(f"   총 예측 수: {len(df)}개")
    print(f"   고유 이미지 수: {df['image_id'].nunique()}개")
    print(f"   평균 검출 수/이미지: {len(df) / df['image_id'].nunique():.2f}개")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="YOLO 모델 추론 및 Kaggle 제출 파일 생성")

    # 필수 인자
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="학습된 모델 경로 (예: runs/detect/train13/weights/best.pt)"
    )

    # 선택 인자
    parser.add_argument("--test-images", type=str, default="data/test_images/", help="테스트 이미지 디렉토리")
    parser.add_argument("--output-dir", type=str, default="outputs/submissions", help="출력 디렉토리")
    parser.add_argument("--imgsz", type=int, nargs=2, default=[512, 896], help="이미지 크기 (width height)")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.6, help="IoU threshold for NMS")
    parser.add_argument("--max-det", type=int, default=100, help="Maximum detections per image")
    parser.add_argument("--device", type=str, default="0", help="Device (0, 1, ... or 'cpu')")
    parser.add_argument("--half", action="store_true", help="Use FP16 inference")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")

    args = parser.parse_args()

    # Device 처리
    device = args.device if args.device == "cpu" else int(args.device)

    # 추론 실행
    output_path = run_inference(
        model_path=args.model,
        test_images_dir=args.test_images,
        output_dir=args.output_dir,
        imgsz=tuple(args.imgsz),
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        device=device,
        half=args.half,
        batch=args.batch,
    )

    print(f"\n제출 방법:")
    print(f"kaggle competitions submit -c <competition-name> -f {output_path} -m 'submission message'")


if __name__ == "__main__":
    main()
