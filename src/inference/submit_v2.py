"""
Kaggle 제출용 배치 추론 V2 (YOLO + ConvNeXt)
- test_images 폴더의 모든 이미지에 대해 추론
- Object Detection 형식 CSV 저장
"""

import os
import glob
import csv
from pathlib import Path
from tqdm import tqdm

from .pill_pipeline_v2 import PillPipelineV2


def k_code_to_dl_idx(k_code: str) -> int:
    """
    K-code를 dl_idx(category_id)로 변환
    - K-001900 -> 1899 (K-code 형식)
    - 숫자 폴더명인 경우 그대로 반환
    """
    if not k_code or not isinstance(k_code, str):
        return -1

    # K-XXXXXX 형식 확인
    if k_code.startswith("K-") and len(k_code) == 8:
        try:
            return int(k_code[2:]) - 1  # K-001900 -> 1899
        except ValueError:
            pass

    # 숫자만 있는 경우 (폴더명이 숫자인 경우)
    try:
        return int(k_code)
    except ValueError:
        pass

    return -1


def get_next_submission_path(base_name: str = "submission") -> str:
    """다음 번호의 submission 파일 경로 반환"""
    i = 1
    while True:
        path = f"{base_name}_{i}.csv"
        if not os.path.exists(path):
            return path
        i += 1


def run_submission(
    test_dir: str = "data/test_images",
    output_csv: str = None,  # None이면 자동 번호 부여
    detector_path: str = "runs/detect/yolo11m_detector/weights/best.pt",  # Best Score: 0.96703
    classifier_path: str = "runs/classify/convnext/best.pt",
    detector_conf: float = 0.05,
    classifier_conf: float = 0.3,
    detector_iou: float = 0.5,
    agnostic_nms: bool = True,
    use_tta: bool = False,  # TTA 적용 여부
    max_det: int = 4,  # 이미지당 최대 검출 수
):
    """
    테스트 이미지 전체에 대해 추론 후 CSV 생성

    Args:
        test_dir: 테스트 이미지 폴더
        output_csv: 출력 CSV 경로 (None이면 자동 번호 부여)
        detector_path: YOLO Detector 경로
        classifier_path: ConvNeXt Classifier 경로
        detector_conf: Detector confidence threshold
        classifier_conf: Classifier confidence threshold
        detector_iou: NMS IoU threshold
        agnostic_nms: 클래스 무관 NMS 적용
        use_tta: Test Time Augmentation 적용 (느리지만 정확도 향상)
    """
    # 출력 파일명 자동 생성
    if output_csv is None:
        output_csv = get_next_submission_path("submission")

    # 파이프라인 초기화
    pipeline = PillPipelineV2(
        detector_path=detector_path,
        classifier_path=classifier_path,
        detector_conf=detector_conf,
        classifier_conf=classifier_conf,
        detector_iou=detector_iou,
        agnostic_nms=agnostic_nms,
        use_tta=use_tta,
    )

    # 테스트 이미지 목록
    test_images = sorted(glob.glob(os.path.join(test_dir, "*.png")))
    if not test_images:
        test_images = sorted(glob.glob(os.path.join(test_dir, "*.jpg")))

    print(f"\n테스트 이미지: {len(test_images)}개")
    print(f"출력 CSV: {output_csv}")
    print("=" * 60)

    results = []
    annotation_id = 1

    for img_path in tqdm(test_images, desc="추론 중"):
        img_name = Path(img_path).stem  # 파일명 (확장자 제외)
        image_id = int(img_name)  # 정수 변환

        # 추론
        preds = pipeline.predict(img_path)

        # Top-N만 유지 (confidence 기준 정렬)
        preds = sorted(preds, key=lambda x: x["classifier_conf"], reverse=True)[:max_det]

        for pred in preds:
            # bbox: [x1, y1, x2, y2] → [x, y, w, h]
            x1, y1, x2, y2 = pred["bbox"]
            bbox_x = int(x1)
            bbox_y = int(y1)
            bbox_w = int(x2 - x1)
            bbox_h = int(y2 - y1)

            # K-code → category_id
            k_code = pred["k_code"]
            category_id = k_code_to_dl_idx(k_code)

            # score: classifier confidence (소수점 2자리)
            score = round(pred["classifier_conf"], 2)

            results.append({
                "annotation_id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox_x": bbox_x,
                "bbox_y": bbox_y,
                "bbox_w": bbox_w,
                "bbox_h": bbox_h,
                "score": score,
            })
            annotation_id += 1

    # CSV 저장
    fieldnames = ["annotation_id", "image_id", "category_id", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "score"]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # 통계
    print("\n" + "=" * 60)
    print("결과")
    print("=" * 60)
    print(f"총 이미지: {len(test_images)}")
    print(f"총 검출: {len(results)}개 알약")
    print(f"이미지당 평균: {len(results) / len(test_images):.1f}개")
    print(f"\n저장 완료: {output_csv}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Kaggle 제출용 배치 추론 V2 (YOLO + ConvNeXt)")
    parser.add_argument("--test_dir", default="data/test_images", help="테스트 이미지 폴더")
    parser.add_argument("--output", default=None, help="출력 CSV 파일 (미지정시 자동 번호)")
    parser.add_argument("--detector", default="runs/detect/yolo11m_detector/weights/best.pt", help="Detector 모델 경로")
    parser.add_argument("--classifier", default="runs/classify/convnext/best.pt", help="Classifier 모델 경로")
    parser.add_argument("--det_conf", type=float, default=0.05, help="Detector confidence")
    parser.add_argument("--cls_conf", type=float, default=0.3, help="Classifier confidence")
    parser.add_argument("--det_iou", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument("--agnostic_nms", action="store_true", default=True, help="클래스 무관 NMS")
    parser.add_argument("--tta", action="store_true", help="TTA 적용 (느리지만 정확도 향상)")
    parser.add_argument("--max_det", type=int, default=4, help="이미지당 최대 검출 수")

    args = parser.parse_args()

    run_submission(
        test_dir=args.test_dir,
        output_csv=args.output,
        detector_path=args.detector,
        classifier_path=args.classifier,
        detector_conf=args.det_conf,
        classifier_conf=args.cls_conf,
        detector_iou=args.det_iou,
        agnostic_nms=args.agnostic_nms,
        use_tta=args.tta,
        max_det=args.max_det,
    )


if __name__ == "__main__":
    main()
