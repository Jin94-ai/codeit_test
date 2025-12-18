"""
2-Stage Pill Detection & Classification Pipeline
- Stage 1: Detector - 알약 위치 검출
- Stage 2: Classifier - 알약 종류 분류 (74개 클래스)
"""

import json
from pathlib import Path
from PIL import Image

from ultralytics import YOLO


class PillPipeline:
    """2-Stage 알약 인식 파이프라인"""

    def __init__(
        self,
        detector_path: str = "runs/detect/pill_detector/weights/best.pt",
        classifier_path: str = "runs/classify/pill_classifier/weights/best.pt",
        class_mapping_path: str = "data/yolo_cls/class_mapping.json",
        detector_conf: float = 0.1,
        classifier_conf: float = 0.5,
    ):
        """
        Args:
            detector_path: Detector 모델 경로
            classifier_path: Classifier 모델 경로
            class_mapping_path: 클래스 매핑 JSON 경로
            detector_conf: Detector confidence threshold
            classifier_conf: Classifier confidence threshold
        """
        # 모델 로드
        self.detector = YOLO(detector_path)
        self.classifier = YOLO(classifier_path)
        self.class_mapping = self._load_class_mapping(class_mapping_path)

        # Confidence thresholds
        self.detector_conf = detector_conf
        self.classifier_conf = classifier_conf

        print(f"[PillPipeline] Detector: {detector_path}")
        print(f"[PillPipeline] Classifier: {classifier_path} ({len(self.classifier.names)} classes)")

    def _load_class_mapping(self, path: str) -> dict:
        """클래스 매핑 로드 (idx → K-code)"""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # idx_to_k_code: {"0": "K-001900", "1": "K-002483", ...}
            return {int(k): v for k, v in data.get("idx_to_k_code", {}).items()}
        except Exception as e:
            print(f"클래스 매핑 로드 실패: {e}")
            return {}

    def predict(self, image_path: str, padding_ratio: float = 0.1) -> list:
        """
        이미지에서 알약 검출 및 분류

        Args:
            image_path: 입력 이미지 경로
            padding_ratio: bbox crop 시 패딩 비율

        Returns:
            list of dict: [
                {
                    "bbox": [x1, y1, x2, y2],
                    "k_code": "K-001900",
                    "detector_conf": 0.85,
                    "classifier_conf": 0.92
                },
                ...
            ]
        """
        # 이미지 로드
        img = Image.open(image_path).convert("RGB")
        img_w, img_h = img.size

        # Stage 1: Detection
        det_results = self.detector.predict(
            image_path,
            conf=self.detector_conf,
            verbose=False
        )[0]

        results = []

        # 검출된 각 bbox에 대해
        for box in det_results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            det_conf = float(box.conf[0])

            # 패딩 적용
            w = x2 - x1
            h = y2 - y1
            pad_x = w * padding_ratio
            pad_y = h * padding_ratio

            crop_x1 = max(0, int(x1 - pad_x))
            crop_y1 = max(0, int(y1 - pad_y))
            crop_x2 = min(img_w, int(x2 + pad_x))
            crop_y2 = min(img_h, int(y2 + pad_y))

            # Crop
            cropped = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))

            # Stage 2: Classification
            cls_results = self.classifier.predict(
                cropped,
                conf=self.classifier_conf,
                verbose=False
            )[0]

            # Top-1 prediction
            probs = cls_results.probs
            top1_idx = int(probs.top1)
            top1_conf = float(probs.top1conf)

            # K-code 변환 (YOLO model의 names 사용)
            if hasattr(self.classifier, 'names') and top1_idx in self.classifier.names:
                k_code = self.classifier.names[top1_idx]
            else:
                k_code = self.class_mapping.get(top1_idx, f"Unknown_{top1_idx}")

            results.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "k_code": k_code,
                "detector_conf": round(det_conf, 3),
                "classifier_conf": round(top1_conf, 3)
            })

        return results

    def predict_with_visualization(
        self,
        image_path: str,
        output_path: str = None,
        padding_ratio: float = 0.1
    ):
        """
        예측 결과를 시각화하여 저장

        Args:
            image_path: 입력 이미지 경로
            output_path: 출력 이미지 경로 (None이면 자동 생성)
            padding_ratio: bbox crop 시 패딩 비율

        Returns:
            tuple: (results, output_path)
        """
        from PIL import ImageDraw, ImageFont

        # 예측 수행
        results = self.predict(image_path, padding_ratio)

        # 이미지 로드
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        # 폰트 설정 (기본 폰트 사용)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()

        # 결과 시각화
        for i, res in enumerate(results):
            x1, y1, x2, y2 = res["bbox"]
            k_code = res["k_code"]
            cls_conf = res["classifier_conf"]

            # bbox 그리기
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

            # 라벨 그리기
            label = f"{k_code} ({cls_conf:.2f})"
            draw.text((x1, y1 - 20), label, fill="red", font=font)

        # 저장 경로
        if output_path is None:
            input_path = Path(image_path)
            output_path = str(input_path.parent / f"{input_path.stem}_result{input_path.suffix}")

        img.save(output_path)
        print(f"결과 저장: {output_path}")

        return results, output_path


def main():
    """테스트 실행"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.inference.pill_pipeline <image_path>")
        print("Example: python -m src.inference.pill_pipeline test.png")
        return

    image_path = sys.argv[1]

    # 파이프라인 초기화
    pipeline = PillPipeline()

    # 예측
    results, output_path = pipeline.predict_with_visualization(image_path)

    # 결과 출력
    print("\n" + "=" * 60)
    print("예측 결과")
    print("=" * 60)

    for i, res in enumerate(results):
        print(f"\n[알약 {i+1}]")
        print(f"  K-code: {res['k_code']}")
        print(f"  bbox: {res['bbox']}")
        print(f"  Detector conf: {res['detector_conf']}")
        print(f"  Classifier conf: {res['classifier_conf']}")

    print(f"\n총 검출: {len(results)}개 알약")
    print(f"결과 이미지: {output_path}")


if __name__ == "__main__":
    main()
