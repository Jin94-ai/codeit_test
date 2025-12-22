"""
Detector 데이터 정제 스크립트 (COCO JSON 포맷)
- bbox 개수 검증 (2~4개 정상, Kaggle 테스트와 동일)
- bbox 중복/위치 검증
- bbox 크기/비율 검증
- 문제 샘플 시각화
"""

import json
import shutil
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from collections import Counter
import random


class DetectorDataCleaner:
    """COCO JSON Detector 데이터 정제"""

    def __init__(
        self,
        data_dir: str = "data/aihub_detector",
        backup: bool = True,
        min_bbox_count: int = 2,          # 최소 bbox 개수 (클래스 커버리지 위해 2개 포함)
        max_bbox_count: int = 4,          # 최대 bbox 개수
        min_iou_overlap: float = 0.7,     # 중복 판정 IoU (0.3→0.7 상향)
        # 추가 검증 기준 (test_images 기반)
        min_bbox_size: int = 30,          # 최소 bbox 크기 (보수적)
        max_bbox_ratio: float = 3.5,      # 최대 종횡비 (캡슐형 ~3:1)
        min_area_ratio: float = 0.003,    # 이미지 대비 최소 면적 비율 (0.3%)
        max_area_ratio: float = 0.15,     # 이미지 대비 최대 면적 비율 (15%)
    ):
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        self.annotations_dir = self.data_dir / "annotations"
        self.backup = backup

        self.min_bbox_count = min_bbox_count
        self.max_bbox_count = max_bbox_count
        self.min_iou_overlap = min_iou_overlap

        # 추가 검증 기준
        self.min_bbox_size = min_bbox_size
        self.max_bbox_ratio = max_bbox_ratio
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio

        # 문제 파일
        self.issues = {
            "wrong_bbox_count": [],    # bbox 개수 범위 밖
            "overlapping_bbox": [],    # 중복 bbox (IoU > 0.7)
            "out_of_bounds": [],       # 이미지 범위 밖 bbox
            "orphan_files": [],        # 이미지-어노테이션 불일치
            "invalid_bbox_size": [],   # bbox 크기 이상
            "invalid_bbox_ratio": [],  # bbox 비율 이상
            "missing_annotation": [],  # annotation 누락 (검출 > annotation)
        }

        self.bbox_count_dist = Counter()

    def run(self, dry_run: bool = True, verify_with_detector: bool = False, detector_path: str = None):
        """데이터 정제 실행"""
        print("=" * 60)
        print("Detector 데이터 정제")
        print(f"데이터 경로: {self.data_dir}")
        print(f"모드: {'검사만' if dry_run else '실제 삭제'}")
        print(f"bbox 개수 범위: {self.min_bbox_count}~{self.max_bbox_count}개")
        print("=" * 60)

        # 1. 고아 파일 검사
        self._check_orphan_files()

        # 2. 어노테이션 검사
        self._check_annotations()

        # 3. Detector로 annotation 누락 검증 (선택)
        if verify_with_detector and detector_path:
            self._verify_with_detector(detector_path)

        # 4. 결과 출력
        self._print_report()

        # 5. 삭제
        if not dry_run:
            self._delete_issues()

        return self.issues

    def _verify_with_detector(self, detector_path: str, conf: float = 0.3):
        """학습된 Detector로 annotation 누락 검증"""
        print("\n[3/3] Detector로 annotation 검증...")

        try:
            from ultralytics import YOLO
        except ImportError:
            print("  ⚠️ ultralytics 없음, 스킵")
            return

        if not Path(detector_path).exists():
            print(f"  ⚠️ Detector 없음: {detector_path}")
            return

        model = YOLO(detector_path)
        print(f"  Detector: {detector_path}")
        print(f"  Confidence: {conf}")

        # annotation 있는 이미지만 검사
        json_files = list(self.annotations_dir.glob("*.json"))
        mismatch_count = 0

        for ann_path in tqdm(json_files, desc="  검증"):
            img_path = self.images_dir / f"{ann_path.stem}.png"
            if not img_path.exists():
                continue

            # annotation bbox 개수
            try:
                with open(ann_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                ann_count = len(data.get("annotations", []))
            except:
                continue

            # Detector 검출 개수
            try:
                results = model.predict(str(img_path), conf=conf, verbose=False)
                det_count = len(results[0].boxes) if results else 0
            except:
                continue

            # 검출 > annotation = 누락 가능성
            if det_count > ann_count:
                self.issues["missing_annotation"].append(
                    (ann_path, f"ann={ann_count}, det={det_count}")
                )
                mismatch_count += 1

        print(f"\n  - annotation 누락 의심: {mismatch_count}개")

    def _check_orphan_files(self):
        """이미지-어노테이션 짝 확인"""
        print("\n[1/2] 고아 파일 검사...")

        images = {p.stem for p in self.images_dir.glob("*.png")}
        annotations = {p.stem for p in self.annotations_dir.glob("*.json")}

        # 불일치 파일
        orphan_ann = annotations - images
        orphan_img = images - annotations

        for name in orphan_ann:
            self.issues["orphan_files"].append(self.annotations_dir / f"{name}.json")
        for name in orphan_img:
            self.issues["orphan_files"].append(self.images_dir / f"{name}.png")

        print(f"  고아 파일: {len(self.issues['orphan_files'])}개")

    def _calculate_iou(self, box1, box2):
        """IoU 계산 (COCO: x, y, w, h)"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # xyxy로 변환
        ax1, ay1, ax2, ay2 = x1, y1, x1+w1, y1+h1
        bx1, by1, bx2, by2 = x2, y2, x2+w2, y2+h2

        # 교집합
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0

        inter = (ix2 - ix1) * (iy2 - iy1)
        union = w1*h1 + w2*h2 - inter

        return inter / union if union > 0 else 0.0

    def _check_annotations(self):
        """어노테이션 검사"""
        print("\n[2/2] 어노테이션 검사...")

        json_files = list(self.annotations_dir.glob("*.json"))

        for ann_path in tqdm(json_files, desc="  검사"):
            try:
                with open(ann_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except:
                continue

            img_info = data.get("images", [{}])[0]
            img_w = img_info.get("width", 1)
            img_h = img_info.get("height", 1)
            img_area = img_w * img_h

            annotations = data.get("annotations", [])
            bbox_count = len(annotations)

            # bbox 개수 분포 기록
            self.bbox_count_dist[bbox_count] += 1

            # bbox 개수 검사 (2~4개 정상, Kaggle 테스트와 동일)
            if not (self.min_bbox_count <= bbox_count <= self.max_bbox_count):
                self.issues["wrong_bbox_count"].append(
                    (ann_path, f"got {bbox_count}")
                )
                continue  # 개수가 틀리면 다른 검사 스킵

            # bbox 추출 및 검증
            bboxes = []
            has_size_issue = False
            has_ratio_issue = False

            for ann in annotations:
                bbox = ann.get("bbox", [])
                if len(bbox) == 4:
                    x, y, w, h = bbox
                    bboxes.append((x, y, w, h))

                    # 범위 검사
                    if x < 0 or y < 0 or x+w > img_w or y+h > img_h:
                        self.issues["out_of_bounds"].append((ann_path, f"bbox: {bbox}"))

                    # 크기 검사 (너무 작은 bbox)
                    if w < self.min_bbox_size or h < self.min_bbox_size:
                        if not has_size_issue:
                            self.issues["invalid_bbox_size"].append(
                                (ann_path, f"too small: {w}x{h}")
                            )
                            has_size_issue = True

                    # 면적 비율 검사 (이미지 대비)
                    bbox_area = w * h
                    area_ratio = bbox_area / img_area
                    if area_ratio < self.min_area_ratio or area_ratio > self.max_area_ratio:
                        if not has_size_issue:
                            self.issues["invalid_bbox_size"].append(
                                (ann_path, f"area ratio: {area_ratio:.3f}")
                            )
                            has_size_issue = True

                    # 종횡비 검사 (너무 길쭉한 bbox)
                    if w > 0 and h > 0:
                        aspect_ratio = max(w/h, h/w)
                        if aspect_ratio > self.max_bbox_ratio:
                            if not has_ratio_issue:
                                self.issues["invalid_bbox_ratio"].append(
                                    (ann_path, f"ratio: {aspect_ratio:.1f}")
                                )
                                has_ratio_issue = True

            # 중복 bbox 검사 (IoU > 0.7이면 진짜 중복)
            for i in range(len(bboxes)):
                for j in range(i+1, len(bboxes)):
                    iou = self._calculate_iou(bboxes[i], bboxes[j])
                    if iou > self.min_iou_overlap:
                        self.issues["overlapping_bbox"].append(
                            (ann_path, f"IoU={iou:.2f}")
                        )

        # 분포 출력
        print(f"\n  [bbox 개수 분포]")
        for k in sorted(self.bbox_count_dist.keys()):
            count = self.bbox_count_dist[k]
            valid = self.min_bbox_count <= k <= self.max_bbox_count
            mark = "✓" if valid else "✗"
            print(f"    {mark} {k}개: {count}개 이미지")

        print(f"\n  - bbox 개수 불일치: {len(self.issues['wrong_bbox_count'])}개")
        print(f"  - 중복 bbox (IoU>{self.min_iou_overlap}): {len(self.issues['overlapping_bbox'])}개")
        print(f"  - 범위 밖 bbox: {len(self.issues['out_of_bounds'])}개")
        print(f"  - 크기 이상 bbox: {len(self.issues['invalid_bbox_size'])}개")
        print(f"  - 비율 이상 bbox: {len(self.issues['invalid_bbox_ratio'])}개")

    def _print_report(self):
        """결과 리포트"""
        print("\n" + "=" * 60)
        print("검사 결과")
        print("=" * 60)

        total_files = sum(self.bbox_count_dist.values())
        problem_files = set()

        for category, items in self.issues.items():
            for item in items:
                if isinstance(item, tuple):
                    problem_files.add(item[0])
                else:
                    problem_files.add(item)

        print(f"총 파일: {total_files}개")
        print(f"문제 파일: {len(problem_files)}개")
        print(f"정상 파일: {total_files - len(problem_files)}개")

    def visualize_samples(self, output_dir: str = None, num_samples: int = 10):
        """문제 샘플 시각화"""
        if output_dir is None:
            output_dir = self.data_dir / "viz_samples"
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[시각화] → {output_dir}")

        categories = [
            "wrong_bbox_count",
            "overlapping_bbox",
            "out_of_bounds",
            "invalid_bbox_size",
            "invalid_bbox_ratio",
            "missing_annotation",
        ]

        for category in categories:
            items = self.issues.get(category, [])
            if not items:
                continue

            cat_dir = output_dir / category
            cat_dir.mkdir(exist_ok=True)

            samples = random.sample(items, min(num_samples, len(items)))

            for idx, item in enumerate(samples):
                ann_path = item[0] if isinstance(item, tuple) else item
                detail = item[1] if isinstance(item, tuple) else ""

                img_path = self.images_dir / f"{ann_path.stem}.png"
                if not img_path.exists():
                    continue

                try:
                    self._draw_sample(img_path, ann_path,
                                     cat_dir / f"{idx+1}_{ann_path.stem}.png",
                                     category, detail)
                except Exception as e:
                    print(f"  오류: {e}")

            print(f"  - {category}: {len(samples)}개")

        # 정상 샘플
        self._visualize_normal(output_dir / "normal", num_samples)

    def _draw_sample(self, img_path, ann_path, out_path, issue, detail):
        """샘플 시각화"""
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()

        with open(ann_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        annotations = data.get("annotations", [])
        colors = ["red", "blue", "green", "orange", "purple", "cyan", "yellow", "magenta"]

        for i, ann in enumerate(annotations):
            bbox = ann.get("bbox", [])
            if len(bbox) != 4:
                continue

            x, y, w, h = bbox
            color = colors[i % len(colors)]
            draw.rectangle([x, y, x+w, y+h], outline=color, width=3)
            draw.text((x, y-20), f"#{i+1}", fill=color, font=font)

        # 정보 표시
        draw.rectangle([5, 5, 350, 55], fill="white", outline="black")
        draw.text((10, 8), f"{issue}", fill="red", font=font)
        draw.text((10, 30), f"BBox: {len(annotations)}개 | {detail}", fill="black", font=font)

        img.save(out_path)

    def _visualize_normal(self, output_dir, num_samples):
        """정상 샘플 시각화"""
        output_dir.mkdir(exist_ok=True)

        problem_files = set()
        for items in self.issues.values():
            for item in items:
                if isinstance(item, tuple):
                    problem_files.add(item[0])
                else:
                    problem_files.add(item)

        normal_files = [
            p for p in self.annotations_dir.glob("*.json")
            if p not in problem_files
        ]

        if not normal_files:
            print(f"  - normal: 0개")
            return

        samples = random.sample(normal_files, min(num_samples, len(normal_files)))

        for idx, ann_path in enumerate(samples):
            img_path = self.images_dir / f"{ann_path.stem}.png"
            if img_path.exists():
                try:
                    self._draw_sample(img_path, ann_path,
                                     output_dir / f"{idx+1}_{ann_path.stem}.png",
                                     "NORMAL", "정상")
                except:
                    pass

        print(f"  - normal: {len(samples)}개")

    def _delete_issues(self):
        """문제 파일 삭제"""
        print("\n" + "=" * 60)
        print("삭제 중...")
        print("=" * 60)

        backup_dir = None
        if self.backup:
            backup_dir = self.data_dir / "backup_deleted"
            backup_dir.mkdir(exist_ok=True)

        # 삭제할 파일 수집
        files_to_delete = set()
        for items in self.issues.values():
            for item in items:
                if isinstance(item, tuple):
                    files_to_delete.add(item[0])
                else:
                    files_to_delete.add(item)

        deleted = 0
        for path in files_to_delete:
            if not path.exists():
                continue

            # 어노테이션이면 이미지도 같이 삭제
            if path.suffix == ".json":
                img_path = self.images_dir / f"{path.stem}.png"
                if img_path.exists():
                    if self.backup:
                        shutil.move(str(img_path), str(backup_dir / img_path.name))
                    else:
                        img_path.unlink()
                    deleted += 1

            if self.backup:
                shutil.move(str(path), str(backup_dir / path.name))
            else:
                path.unlink()
            deleted += 1

        print(f"삭제 완료: {deleted}개")
        if self.backup:
            print(f"백업: {backup_dir}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Detector 데이터 정제 (개선판)")
    parser.add_argument("--data_dir", default="data/aihub_detector")
    parser.add_argument("--delete", action="store_true", help="실제 삭제")
    parser.add_argument("--no-backup", action="store_true", help="백업 없이 삭제")
    parser.add_argument("--viz", action="store_true", help="샘플 시각화")
    # bbox 개수 (Kaggle 테스트: 3~4개, 학습용: 2~4개)
    parser.add_argument("--min-count", type=int, default=2, help="최소 bbox 개수 (기본: 2)")
    parser.add_argument("--max-count", type=int, default=4, help="최대 bbox 개수 (기본: 4)")
    # IoU (0.7로 상향 - 진짜 중복만 검출)
    parser.add_argument("--iou", type=float, default=0.7, help="중복 판정 IoU (기본: 0.7)")
    # 추가 검증 기준 (test_images 기반)
    parser.add_argument("--min-size", type=int, default=30, help="최소 bbox 크기 (기본: 30px)")
    parser.add_argument("--max-ratio", type=float, default=3.5, help="최대 종횡비 (기본: 3.5)")
    parser.add_argument("--min-area", type=float, default=0.003, help="최소 면적비 (기본: 0.3%)")
    parser.add_argument("--max-area", type=float, default=0.15, help="최대 면적비 (기본: 15%)")
    # Detector 검증 (annotation 누락 검출)
    parser.add_argument("--verify", action="store_true", help="Detector로 annotation 누락 검증")
    parser.add_argument("--detector", default="runs/detect/pill_detector/weights/best.pt",
                        help="검증용 Detector 경로")

    args = parser.parse_args()

    cleaner = DetectorDataCleaner(
        data_dir=args.data_dir,
        backup=not args.no_backup,
        min_bbox_count=args.min_count,
        max_bbox_count=args.max_count,
        min_iou_overlap=args.iou,
        min_bbox_size=args.min_size,
        max_bbox_ratio=args.max_ratio,
        min_area_ratio=args.min_area,
        max_area_ratio=args.max_area,
    )

    cleaner.run(
        dry_run=not args.delete,
        verify_with_detector=args.verify,
        detector_path=args.detector
    )

    if args.viz:
        cleaner.visualize_samples()


if __name__ == "__main__":
    main()
