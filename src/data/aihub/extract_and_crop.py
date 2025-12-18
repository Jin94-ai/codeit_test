"""
AIHub ZIP + Kaggle 데이터에서 74개 클래스 추출 및 ROI Crop
- 클래스당 최대 210개 (잘못 크롭 대비 여유분)
- AIHub: 원본 ZIP에서 타겟 클래스만 추출
- Kaggle: train_images + train_annotations에서 추출
"""

import json
import shutil
import random
import zipfile
from pathlib import Path
from PIL import Image
from collections import defaultdict
from io import BytesIO

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# 입력 경로
AIHUB_BASE = PROJECT_ROOT / "data" / "166.약품식별 인공지능 개발을 위한 경구약제 이미지 데이터" / "01.데이터" / "1.Training"
AIHUB_LABEL_DIR = AIHUB_BASE / "라벨링데이터" / "경구약제조합 5000종"
AIHUB_IMAGE_DIR = AIHUB_BASE / "원천데이터" / "경구약제조합 5000종"

KAGGLE_IMG_DIR = PROJECT_ROOT / "data" / "train_images"
KAGGLE_ANN_DIR = PROJECT_ROOT / "data" / "train_annotations"

# 출력 경로
OUTPUT_DIR = PROJECT_ROOT / "data" / "cropped"
OUTPUT_IMG_DIR = OUTPUT_DIR / "images"
OUTPUT_ANN_DIR = OUTPUT_DIR / "annotations"

# 클래스당 목표 개수
TARGET_PER_CLASS = 210

# Crop 패딩 (bbox 크기의 비율, 0.1 = 10% 여유)
CROP_PADDING_RATIO = 0.15

# 74개 타겟 클래스 (dl_idx 기준)
KAGGLE_56_DL_IDX = {
    1899, 2482, 3350, 3482, 3543, 3742, 3831, 4542,
    12080, 12246, 12777, 13394, 13899, 16231, 16261, 16547,
    16550, 16687, 18146, 18356, 19231, 19551, 19606, 19860,
    20013, 20237, 20876, 21324, 21770, 22073, 22346, 22361,
    24849, 25366, 25437, 25468, 27732, 27776, 27925, 27992,
    28762, 29344, 29450, 29666, 30307, 31862, 31884, 32309,
    33008, 33207, 33879, 34596, 35205, 36636, 38161, 41767
}

MISSING_18_DL_IDX = {
    4377, 5093, 5885, 6191, 6562,
    10220, 10223, 12419, 18109, 21025, 22626,
    23202, 23222, 27652, 29870, 31704,
    33877, 44198
}

ALL_74_DL_IDX = KAGGLE_56_DL_IDX | MISSING_18_DL_IDX
TARGET_K_CODES = {f"K-{dl_idx + 1:06d}" for dl_idx in ALL_74_DL_IDX}


def k_code_to_dl_idx(k_code: str) -> int:
    """K-001900 -> 1899"""
    return int(k_code.split("-")[1]) - 1


def dl_idx_to_k_code(dl_idx: int) -> str:
    """1899 -> K-001900"""
    return f"K-{dl_idx + 1:06d}"


def process_aihub_zip(class_counts: dict, class_data: dict):
    """AIHub ZIP 파일에서 타겟 클래스만 추출"""
    print("\n[AIHub ZIP 처리]")

    # ZIP 파일 쌍 찾기 (TL_*_조합.zip, TS_*_조합.zip)
    label_zips = sorted(AIHUB_LABEL_DIR.glob("TL_*_조합.zip"))
    print(f"  라벨링 ZIP: {len(label_zips)}개")

    for label_zip_path in label_zips:
        # 대응하는 이미지 ZIP 찾기
        zip_num = label_zip_path.name.split("_")[1]  # TL_1_조합.zip -> 1
        image_zip_path = AIHUB_IMAGE_DIR / f"TS_{zip_num}_조합.zip"

        if not image_zip_path.exists():
            print(f"  경고: 이미지 ZIP 없음 - {image_zip_path.name}")
            continue

        print(f"  처리 중: {label_zip_path.name} + {image_zip_path.name}")

        try:
            with zipfile.ZipFile(label_zip_path, 'r') as label_zip, \
                 zipfile.ZipFile(image_zip_path, 'r') as image_zip:

                # 이미지 ZIP 파일 목록 캐시
                image_files = {Path(f).name: f for f in image_zip.namelist() if f.endswith('.png')}

                # 라벨링 JSON 파일 순회
                for json_path in label_zip.namelist():
                    if not json_path.endswith('.json'):
                        continue

                    # K-code 폴더 확인 (구조: {조합}_json/{K-code}/{filename}.json)
                    parts = json_path.split('/')
                    if len(parts) < 3:
                        continue

                    k_code_folder = parts[1]  # K-code 폴더
                    if k_code_folder not in TARGET_K_CODES:
                        continue

                    dl_idx = k_code_to_dl_idx(k_code_folder)

                    # 이미 목표 달성했으면 스킵
                    if class_counts.get(dl_idx, 0) >= TARGET_PER_CLASS:
                        continue

                    try:
                        # JSON 읽기
                        with label_zip.open(json_path) as f:
                            data = json.load(f)

                        # 이미지 파일명
                        img_info = data.get("images", [{}])[0]
                        img_filename = img_info.get("file_name", "")

                        if img_filename not in image_files:
                            continue

                        # bbox 추출 (해당 K-code의 bbox)
                        annotations = data.get("annotations", [])
                        if not annotations:
                            continue

                        bbox = annotations[0].get("bbox", [])
                        if len(bbox) != 4:
                            continue

                        x, y, w, h = bbox

                        # 데이터 저장
                        class_data[dl_idx].append({
                            'source': 'aihub_zip',
                            'image_zip': image_zip_path,
                            'img_path_in_zip': image_files[img_filename],
                            'bbox': (x, y, w, h),
                            'k_code': k_code_folder
                        })
                        class_counts[dl_idx] = class_counts.get(dl_idx, 0) + 1

                    except Exception:
                        continue

        except Exception as e:
            print(f"  오류: {label_zip_path.name} - {e}")
            continue

    total = sum(class_counts.values())
    print(f"  AIHub에서 수집: {total}개")


def process_kaggle(class_counts: dict, class_data: dict):
    """Kaggle 데이터에서 추출"""
    print("\n[Kaggle 데이터 처리]")

    ann_files = list(KAGGLE_ANN_DIR.glob("**/*.json"))
    print(f"  어노테이션 파일: {len(ann_files)}개")

    added = 0
    for ann_path in ann_files:
        if ann_path.name.startswith("._"):
            continue

        # K-code 폴더명에서 클래스 결정
        k_code_folder = ann_path.parent.name
        if not k_code_folder.startswith("K-"):
            continue

        if k_code_folder not in TARGET_K_CODES:
            continue

        dl_idx = k_code_to_dl_idx(k_code_folder)

        # 이미 목표 달성했으면 스킵
        if class_counts.get(dl_idx, 0) >= TARGET_PER_CLASS:
            continue

        try:
            with open(ann_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 이미지 파일명
            img_info = data.get("images", [{}])[0]
            img_filename = img_info.get("file_name", "")
            img_path = KAGGLE_IMG_DIR / img_filename

            if not img_path.exists():
                continue

            # bbox 추출
            annotations = data.get("annotations", [])
            if not annotations:
                continue

            bbox = annotations[0].get("bbox", [])
            if len(bbox) != 4:
                continue

            x, y, w, h = bbox

            # 데이터 저장
            class_data[dl_idx].append({
                'source': 'kaggle',
                'img_path': img_path,
                'bbox': (x, y, w, h),
                'k_code': k_code_folder
            })
            class_counts[dl_idx] = class_counts.get(dl_idx, 0) + 1
            added += 1

        except Exception:
            continue

    print(f"  Kaggle에서 추가: {added}개")


def crop_and_save(class_data: dict):
    """수집된 데이터를 crop하여 저장"""
    print("\n[ROI Crop 및 저장]")

    total_saved = 0
    class_saved = defaultdict(int)

    # ZIP 캐시
    zip_cache = {}

    for dl_idx in sorted(ALL_74_DL_IDX):
        k_code = dl_idx_to_k_code(dl_idx)
        items = class_data.get(dl_idx, [])

        # 210개로 제한 (랜덤 샘플링)
        if len(items) > TARGET_PER_CLASS:
            random.seed(42)
            items = random.sample(items, TARGET_PER_CLASS)

        for idx, item in enumerate(items):
            try:
                x, y, w, h = item['bbox']

                if item['source'] == 'aihub_zip':
                    # ZIP에서 이미지 로드
                    zip_path = item['image_zip']
                    if zip_path not in zip_cache:
                        zip_cache[zip_path] = zipfile.ZipFile(zip_path, 'r')

                    zf = zip_cache[zip_path]
                    with zf.open(item['img_path_in_zip']) as f:
                        img = Image.open(BytesIO(f.read())).convert('RGB')
                else:
                    # 로컬 파일에서 로드
                    img = Image.open(item['img_path']).convert('RGB')

                img_w, img_h = img.size

                # 패딩 추가 (bbox 크기 기준 비율)
                pad_x = int(w * CROP_PADDING_RATIO)
                pad_y = int(h * CROP_PADDING_RATIO)

                # 패딩 적용 (이미지 경계 체크)
                crop_x1 = max(0, x - pad_x)
                crop_y1 = max(0, y - pad_y)
                crop_x2 = min(img_w, x + w + pad_x)
                crop_y2 = min(img_h, y + h + pad_y)

                # ROI Crop (패딩 포함)
                cropped = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                crop_w, crop_h = cropped.size

                # 저장
                new_stem = f"{k_code}_{idx + 1:04d}"
                new_img_path = OUTPUT_IMG_DIR / f"{new_stem}.png"
                new_ann_path = OUTPUT_ANN_DIR / f"{new_stem}.json"

                cropped.save(new_img_path)

                # 어노테이션 생성
                ann_data = {
                    "images": [{
                        "id": 1,
                        "file_name": f"{new_stem}.png",
                        "width": crop_w,
                        "height": crop_h
                    }],
                    "annotations": [{
                        "id": 1,
                        "image_id": 1,
                        "category_id": dl_idx,
                        "bbox": [0, 0, crop_w, crop_h],
                        "area": crop_w * crop_h,
                        "k_code": k_code
                    }],
                    "categories": [{
                        "id": dl_idx,
                        "name": k_code
                    }]
                }

                with open(new_ann_path, "w", encoding="utf-8") as f:
                    json.dump(ann_data, f, ensure_ascii=False, indent=2)

                total_saved += 1
                class_saved[k_code] += 1

                if item['source'] != 'aihub_zip':
                    img.close()

            except Exception as e:
                continue

        if total_saved % 500 == 0 and total_saved > 0:
            print(f"  진행: {total_saved}개 저장 완료")

    # ZIP 캐시 정리
    for zf in zip_cache.values():
        zf.close()

    return total_saved, class_saved


def main():
    print("=" * 60)
    print("74개 클래스 추출 및 ROI Crop")
    print(f"클래스당 최대: {TARGET_PER_CLASS}개")
    print("=" * 60)

    # 출력 디렉토리 초기화
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_ANN_DIR.mkdir(parents=True, exist_ok=True)

    # 데이터 수집
    class_counts = {}  # {dl_idx: count}
    class_data = defaultdict(list)  # {dl_idx: [item, ...]}

    # 1. AIHub ZIP에서 수집
    process_aihub_zip(class_counts, class_data)

    # 2. Kaggle에서 추가 수집
    process_kaggle(class_counts, class_data)

    # 3. Crop 및 저장
    total_saved, class_saved = crop_and_save(class_data)

    # 결과 출력
    print("\n" + "=" * 60)
    print("결과")
    print("=" * 60)

    print(f"\n{'K-code':<12} {'개수':>8}")
    print("-" * 24)

    under_target = 0
    for dl_idx in sorted(ALL_74_DL_IDX):
        k_code = dl_idx_to_k_code(dl_idx)
        count = class_saved.get(k_code, 0)
        status = "" if count >= 200 else f" (부족)"
        print(f"{k_code:<12} {count:>8}{status}")
        if count < 200:
            under_target += 1

    print("-" * 24)
    print(f"{'합계':<12} {total_saved:>8}")

    print(f"\n총 저장: {total_saved}개")
    print(f"200개 이상: {74 - under_target}/74 클래스")
    print(f"\n출력 경로: {OUTPUT_DIR}")
    print("\n다음 단계: 수동으로 잘못 잘린 이미지 삭제 후")
    print("python src/data/aihub/cleanup_orphan_annotations.py 실행")
    print("=" * 60)


if __name__ == "__main__":
    main()
