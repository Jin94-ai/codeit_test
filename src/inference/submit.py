"""
Kaggle 제출용 배치 추론
- test_images 폴더의 모든 이미지에 대해 추론
- Object Detection 형식 CSV 저장
"""

import os
import glob
import csv
from pathlib import Path
from tqdm import tqdm

from .pill_pipeline import PillPipeline


def k_code_to_dl_idx(k_code: str) -> int:
    """K-001900 -> 1899"""
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


def run_submission(
    test_dir: str = "data/test_images",
    output_csv: str = "submission.csv",
    detector_conf: float = 0.05,
    classifier_conf: float = 0.3,
):
    """
    테스트 이미지 전체에 대해 추론 후 CSV 생성

    Args:
        test_dir: 테스트 이미지 폴더
        output_csv: 출력 CSV 경로
        detector_conf: Detector confidence threshold
        classifier_conf: Classifier confidence threshold
    """
    # 파이프라인 초기화
    pipeline = PillPipeline(
        detector_conf=detector_conf,
        classifier_conf=classifier_conf,
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

        # 추론
        preds = pipeline.predict(img_path)

        for pred in preds:
            # bbox: [x1, y1, x2, y2] → [x, y, w, h]
            x1, y1, x2, y2 = pred["bbox"]
            bbox_x = x1
            bbox_y = y1
            bbox_w = x2 - x1
            bbox_h = y2 - y1

            # K-code → dl_idx
            k_code = pred["k_code"]
            dl_idx = k_code_to_dl_idx(k_code)

            # score: classifier confidence 사용
            score = pred["classifier_conf"]

            results.append({
                "annotation_id": annotation_id,
                "image_id": img_name,
                "category_id": dl_idx,
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

    parser = argparse.ArgumentParser(description="Kaggle 제출용 배치 추론")
    parser.add_argument("--test_dir", default="data/test_images", help="테스트 이미지 폴더")
    parser.add_argument("--output", default="submission.csv", help="출력 CSV 파일")
    parser.add_argument("--det_conf", type=float, default=0.05, help="Detector confidence")
    parser.add_argument("--cls_conf", type=float, default=0.3, help="Classifier confidence")

    args = parser.parse_args()

    run_submission(
        test_dir=args.test_dir,
        output_csv=args.output,
        detector_conf=args.det_conf,
        classifier_conf=args.cls_conf,
    )


if __name__ == "__main__":
    main()
