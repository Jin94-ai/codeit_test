"""
2-Stage Pill Detection & Classification Pipeline V2
- Stage 1: YOLO11s Detector - 알약 위치 검출
- Stage 2: ConvNeXt Classifier - 알약 종류 분류 (74개 클래스)
"""

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO
import timm


class PillPipelineV2:
    """2-Stage 알약 인식 파이프라인 (YOLO + ConvNeXt)"""

    def __init__(
        self,
        detector_path: str = "best_detector.pt",
        classifier_path: str = "runs/classify/convnext/best.pt",
        detector_conf: float = 0.1,
        classifier_conf: float = 0.5,
        detector_iou: float = 0.5,
        agnostic_nms: bool = True,
        use_tta: bool = False,
        device: str = None,
    ):
        """
        Args:
            detector_path: YOLO Detector 모델 경로
            classifier_path: ConvNeXt Classifier 모델 경로
            detector_conf: Detector confidence threshold
            classifier_conf: Classifier confidence threshold
            detector_iou: NMS IoU threshold (낮으면 더 많이 제거)
            agnostic_nms: 클래스 무관 NMS 적용
            use_tta: Test Time Augmentation 사용 (augment=True)
            device: 'cuda' or 'cpu' (None이면 자동 선택)
        """
        # Device 설정
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Stage 1: YOLO Detector
        self.detector = YOLO(detector_path)
        self.detector_conf = detector_conf
        self.detector_iou = detector_iou
        self.agnostic_nms = agnostic_nms
        self.use_tta = use_tta

        # Stage 2: ConvNeXt Classifier
        self.classifier, self.label2idx, self.idx2label = self._load_classifier(classifier_path)
        self.classifier_conf = classifier_conf

        # Classifier Transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        print(f"[PillPipelineV2] Detector: {detector_path}")
        print(f"[PillPipelineV2] Classifier: {classifier_path} ({len(self.idx2label)} classes)")
        print(f"[PillPipelineV2] Device: {self.device}")
        print(f"[PillPipelineV2] TTA: {'Enabled' if use_tta else 'Disabled'}")

    def _load_classifier(self, path: str):
        """ConvNeXt Classifier 로드"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        label2idx = checkpoint.get("label2idx", {})
        idx2label = checkpoint.get("idx2label", {})

        # int 키로 변환
        idx2label = {int(k): v for k, v in idx2label.items()}

        num_classes = len(label2idx)

        # 모델 생성
        model = timm.create_model("convnext_tiny", pretrained=False, num_classes=num_classes)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(self.device)
        model.eval()

        return model, label2idx, idx2label

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

        # Stage 1: Detection (with NMS tuning + TTA)
        det_results = self.detector.predict(
            image_path,
            conf=self.detector_conf,
            iou=self.detector_iou,
            agnostic_nms=self.agnostic_nms,
            augment=self.use_tta,  # TTA: 좌우반전 + 멀티스케일
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

            # Stage 2: ConvNeXt Classification
            k_code, cls_conf = self._classify(cropped)

            results.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "k_code": k_code,
                "detector_conf": round(det_conf, 3),
                "classifier_conf": round(cls_conf, 3)
            })

        return results

    def _classify(self, cropped_img: Image.Image) -> tuple:
        """
        크롭된 이미지 분류

        Args:
            cropped_img: PIL Image (크롭된 알약)

        Returns:
            tuple: (k_code, confidence)
        """
        # Transform
        input_tensor = self.transform(cropped_img).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.classifier(input_tensor)
            probs = F.softmax(outputs, dim=1)

            top1_conf, top1_idx = probs.max(1)
            top1_conf = float(top1_conf[0])
            top1_idx = int(top1_idx[0])

        # K-code 변환
        k_code = self.idx2label.get(top1_idx, f"Unknown_{top1_idx}")

        return k_code, top1_conf

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

        # 폰트 설정
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
        print("Usage: python -m src.inference.pill_pipeline_v2 <image_path>")
        print("Example: python -m src.inference.pill_pipeline_v2 test.png")
        return

    image_path = sys.argv[1]

    # 파이프라인 초기화
    pipeline = PillPipelineV2()

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
