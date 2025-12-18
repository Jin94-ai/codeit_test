"""
AIHub ZIP에서 원본 이미지 + bbox 추출 (크롭 없이)
- Detector 학습용 데이터 생성
- 조합 이미지 그대로 사용 + 모든 bbox 정보 유지
- extract_and_crop.py와 동일한 최적화 패턴 적용 (ZIP 캐싱, 2단계 처리)
"""

import json
import zipfile
from pathlib import Path
from io import BytesIO
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# AIHub ZIP 경로
AIHUB_BASE = PROJECT_ROOT / "data" / "166.약품식별 인공지능 개발을 위한 경구약제 이미지 데이터" / "01.데이터" / "1.Training"
AIHUB_LABEL_DIR = AIHUB_BASE / "라벨링데이터" / "경구약제조합 5000종"
AIHUB_IMAGE_DIR = AIHUB_BASE / "원천데이터" / "경구약제조합 5000종"

# 출력 경로
OUTPUT_DIR = PROJECT_ROOT / "data" / "aihub_detector"
OUTPUT_IMG_DIR = OUTPUT_DIR / "images"
OUTPUT_ANN_DIR = OUTPUT_DIR / "annotations"

# 최대 이미지 수 (None = 제한 없음)
MAX_IMAGES = 5000


def collect_metadata():
    """1단계: 메타데이터만 수집 (빠름)"""
    print("\n[1단계: 메타데이터 수집]")

    label_zips = sorted(AIHUB_LABEL_DIR.glob("TL_*_조합.zip"))
    print(f"라벨링 ZIP: {len(label_zips)}개")

    collected = []

    for label_zip_path in label_zips:
        if MAX_IMAGES and len(collected) >= MAX_IMAGES:
            break

        # 대응하는 이미지 ZIP
        zip_num = label_zip_path.name.split("_")[1]
        image_zip_path = AIHUB_IMAGE_DIR / f"TS_{zip_num}_조합.zip"

        if not image_zip_path.exists():
            print(f"  경고: 이미지 ZIP 없음 - {image_zip_path.name}")
            continue

        print(f"  스캔 중: {label_zip_path.name}")

        try:
            with zipfile.ZipFile(label_zip_path, 'r') as label_zip, \
                 zipfile.ZipFile(image_zip_path, 'r') as image_zip:

                # 이미지 파일 목록 캐시
                image_files = {Path(f).name: f for f in image_zip.namelist() if f.endswith('.png')}

                # JSON 파일 순회
                json_files = [f for f in label_zip.namelist() if f.endswith('.json')]

                for json_path in json_files:
                    if MAX_IMAGES and len(collected) >= MAX_IMAGES:
                        break

                    try:
                        with label_zip.open(json_path) as f:
                            data = json.load(f)

                        # 이미지 정보
                        img_info = data.get("images", [{}])[0]
                        img_filename = img_info.get("file_name", "")

                        if img_filename not in image_files:
                            continue

                        # 모든 bbox 수집
                        annotations = data.get("annotations", [])
                        if not annotations:
                            continue

                        bboxes = []
                        for ann in annotations:
                            bbox = ann.get("bbox", [])
                            if len(bbox) == 4:
                                bboxes.append(bbox)

                        if not bboxes:
                            continue

                        # 메타데이터 저장 (이미지는 아직 로드 안 함)
                        collected.append({
                            'image_zip': image_zip_path,
                            'img_path_in_zip': image_files[img_filename],
                            'width': img_info.get("width"),
                            'height': img_info.get("height"),
                            'bboxes': bboxes
                        })

                    except Exception:
                        continue

        except Exception as e:
            print(f"  오류: {e}")
            continue

    print(f"  수집된 메타데이터: {len(collected)}개")
    return collected


def save_images(metadata_list):
    """2단계: 이미지 저장 (ZIP 캐싱으로 빠름)"""
    print("\n[2단계: 이미지 저장]")

    # 출력 디렉토리 생성
    OUTPUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_ANN_DIR.mkdir(parents=True, exist_ok=True)

    # ZIP 캐시 (한 번 열면 재사용)
    zip_cache = {}

    total_images = 0
    total_bboxes = 0

    for idx, item in enumerate(metadata_list):
        image_id = idx + 1

        try:
            # ZIP 캐싱 - 이미 열려있으면 재사용
            zip_path = item['image_zip']
            if zip_path not in zip_cache:
                zip_cache[zip_path] = zipfile.ZipFile(zip_path, 'r')

            zf = zip_cache[zip_path]

            # 이미지 로드 및 저장
            with zf.open(item['img_path_in_zip']) as f:
                img = Image.open(BytesIO(f.read())).convert('RGB')

            img_w, img_h = img.size

            # 새 파일명 생성
            new_stem = f"aihub_{image_id:06d}"
            new_img_path = OUTPUT_IMG_DIR / f"{new_stem}.png"
            new_ann_path = OUTPUT_ANN_DIR / f"{new_stem}.json"

            # 이미지 저장
            img.save(new_img_path)

            # 어노테이션 저장 (COCO 형식)
            ann_data = {
                "images": [{
                    "id": image_id,
                    "file_name": f"{new_stem}.png",
                    "width": img_w,
                    "height": img_h
                }],
                "annotations": [],
                "categories": [{"id": 0, "name": "Pill"}]
            }

            for i, bbox in enumerate(item['bboxes']):
                ann_data["annotations"].append({
                    "id": i + 1,
                    "image_id": image_id,
                    "category_id": 0,  # 단일 클래스: Pill
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3]
                })

            with open(new_ann_path, "w", encoding="utf-8") as f:
                json.dump(ann_data, f, ensure_ascii=False, indent=2)

            total_images += 1
            total_bboxes += len(item['bboxes'])

            if total_images % 500 == 0:
                print(f"  진행: {total_images}개 이미지, {total_bboxes}개 bbox")

        except Exception:
            continue

    # ZIP 캐시 정리
    for zf in zip_cache.values():
        zf.close()

    return total_images, total_bboxes


def extract_aihub_for_detector():
    print("=" * 60)
    print("AIHub 원본 이미지 추출 (Detector용)")
    print(f"최대 이미지 수: {MAX_IMAGES}")
    print("=" * 60)

    # 1단계: 메타데이터 수집 (빠름)
    metadata = collect_metadata()

    # 2단계: 이미지 저장 (ZIP 캐싱)
    total_images, total_bboxes = save_images(metadata)

    print("\n" + "=" * 60)
    print("결과")
    print("=" * 60)
    print(f"총 이미지: {total_images}개")
    print(f"총 bbox: {total_bboxes}개")
    print(f"평균 bbox/이미지: {total_bboxes / total_images:.1f}개" if total_images > 0 else "")
    print(f"\n출력 경로: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    extract_aihub_for_detector()
