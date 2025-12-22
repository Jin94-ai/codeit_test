"""
AIHub ZIP에서 Detector용 원본 이미지 + bbox 추출
- 74개 타겟 클래스가 포함된 이미지만 추출
- 이미지당 모든 bbox(알약) 수집
- 랜덤으로 MAX_IMAGES개 이미지 선택
"""

import json
import random
import zipfile
from pathlib import Path
from io import BytesIO
from PIL import Image
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# AIHub ZIP 경로 (여러 하위 폴더 지원)
AIHUB_BASE = PROJECT_ROOT / "data" / "166.약품식별 인공지능 개발을 위한 경구약제 이미지 데이터" / "01.데이터" / "1.Training"
AIHUB_LABEL_BASE = AIHUB_BASE / "라벨링데이터"
AIHUB_IMAGE_BASE = AIHUB_BASE / "원천데이터"

# 출력 경로
OUTPUT_DIR = PROJECT_ROOT / "data" / "aihub_detector"
OUTPUT_IMG_DIR = OUTPUT_DIR / "images"
OUTPUT_ANN_DIR = OUTPUT_DIR / "annotations"

# 최대 이미지 수 (랜덤 샘플링)
MAX_IMAGES = 7000
RANDOM_SEED = 42

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


def collect_metadata():
    """
    1단계: 메타데이터 수집
    - 모든 K-code 폴더를 스캔해서 이미지별 모든 bbox 수집
    - 74개 타겟 클래스가 포함된 이미지만 최종 선택
    - 여러 하위 폴더 탐색 (경구약제조합 5000종, 10000종 등)
    """
    print("\n[1단계: 메타데이터 수집]")
    print(f"타겟 클래스: {len(TARGET_K_CODES)}개")

    # 모든 하위 폴더에서 라벨링 ZIP 찾기
    label_zips = sorted(AIHUB_LABEL_BASE.glob("**/TL_*_조합.zip"))
    print(f"라벨링 ZIP: {len(label_zips)}개")

    # 이미지별로 데이터 수집 {이미지파일명: {메타정보}}
    image_data = defaultdict(lambda: {
        'image_zip': None,
        'img_path_in_zip': None,
        'width': None,
        'height': None,
        'bboxes': [],
        'k_codes': set()  # 포함된 K-code들 (타겟 여부 확인용)
    })

    for label_zip_path in label_zips:
        # 대응하는 이미지 ZIP 찾기 (같은 하위 폴더 구조에서)
        # 예: 라벨링데이터/경구약제조합 5000종/TL_1_조합.zip
        #  -> 원천데이터/경구약제조합 5000종/TS_1_조합.zip
        zip_num = label_zip_path.name.split("_")[1]
        subfolder = label_zip_path.parent.name  # 예: "경구약제조합 5000종"
        image_zip_path = AIHUB_IMAGE_BASE / subfolder / f"TS_{zip_num}_조합.zip"

        if not image_zip_path.exists():
            print(f"  경고: 이미지 ZIP 없음 - {image_zip_path}")
            continue

        print(f"  스캔 중: {subfolder}/{label_zip_path.name}")

        try:
            with zipfile.ZipFile(label_zip_path, 'r') as label_zip, \
                 zipfile.ZipFile(image_zip_path, 'r') as image_zip:

                # 이미지 파일 목록 캐시
                image_files = {Path(f).name: f for f in image_zip.namelist() if f.endswith('.png')}

                # JSON 파일 순회 (모든 K-code 폴더 스캔)
                json_files = [f for f in label_zip.namelist() if f.endswith('.json')]

                for json_path in json_files:
                    try:
                        # K-code 폴더 확인 (구조: {조합}_json/{K-code}/{filename}.json)
                        parts = json_path.split('/')
                        if len(parts) < 3:
                            continue

                        k_code_folder = parts[1]  # K-code 폴더

                        with label_zip.open(json_path) as f:
                            data = json.load(f)

                        # 이미지 정보
                        img_info = data.get("images", [{}])[0]
                        img_filename = img_info.get("file_name", "")

                        if not img_filename or img_filename not in image_files:
                            continue

                        # bbox 추출 (모든 K-code의 bbox 수집)
                        annotations = data.get("annotations", [])
                        if not annotations:
                            continue

                        for ann in annotations:
                            bbox = ann.get("bbox", [])
                            if len(bbox) == 4:
                                img_data = image_data[img_filename]

                                # 첫 번째 발견 시 메타정보 저장
                                if img_data['image_zip'] is None:
                                    img_data['image_zip'] = image_zip_path
                                    img_data['img_path_in_zip'] = image_files[img_filename]
                                    img_data['width'] = img_info.get("width")
                                    img_data['height'] = img_info.get("height")

                                # bbox 추가 (중복 방지)
                                if bbox not in img_data['bboxes']:
                                    img_data['bboxes'].append(bbox)

                                # K-code 기록 (타겟 클래스인 경우만)
                                if k_code_folder in TARGET_K_CODES:
                                    img_data['k_codes'].add(k_code_folder)

                    except Exception:
                        continue

        except Exception as e:
            print(f"  오류: {e}")
            continue

    # 74개 타겟 클래스가 포함된 이미지만 필터링
    print(f"\n  전체 스캔 이미지: {len(image_data)}개")
    filtered_count = 0
    for img_filename in list(image_data.keys()):
        if not image_data[img_filename]['k_codes']:  # 타겟 클래스 없음
            del image_data[img_filename]
            filtered_count += 1
    print(f"  타겟 클래스 없는 이미지 제외: {filtered_count}개")
    print(f"  타겟 클래스 포함 이미지: {len(image_data)}개")

    # 이미지 분류: bbox 개수별로 분리 (3개, 4개만 사용)
    images_by_count = {3: [], 4: []}
    bbox_filter_stats = {1: 0, 2: 0, 3: 0, 4: 0, '5+': 0}

    for img_filename, data in image_data.items():
        bbox_count = len(data['bboxes'])

        # 통계 기록
        if bbox_count >= 5:
            bbox_filter_stats['5+'] += 1
        elif bbox_count > 0:
            bbox_filter_stats[bbox_count] += 1

        # 3개, 4개만 분류 (2개 제외)
        if bbox_count in (3, 4):
            img_data = {'filename': img_filename, **data}
            images_by_count[bbox_count].append(img_data)

    print(f"\n  [사용 가능 이미지]")
    for k in [3, 4]:
        print(f"    {k}개 bbox: {len(images_by_count[k])}개")

    # 4개 우선, 3개 보충 (비율 5:1)
    random.seed(RANDOM_SEED)

    collected = []
    collected_set = set()

    # 4개 bbox 먼저 최대한 수집
    images_4 = images_by_count[4]
    random.shuffle(images_4)

    target_4 = min(len(images_4), int(MAX_IMAGES * 5 / 6))  # 5/6 = 약 5833개
    for img in images_4:
        if len(collected) >= target_4:
            break
        collected.append(img)
        collected_set.add(img['filename'])

    print(f"  4개 bbox: {len(collected)}개 선택")

    # 나머지 3개 bbox로 채움
    images_3 = images_by_count[3]
    random.shuffle(images_3)

    remaining = MAX_IMAGES - len(collected)
    count_3 = 0
    for img in images_3:
        if count_3 >= remaining:
            break
        if img['filename'] not in collected_set:
            collected.append(img)
            collected_set.add(img['filename'])
            count_3 += 1

    print(f"  3개 bbox: {count_3}개 선택")

    # 통계 출력
    bbox_counts = [len(item['bboxes']) for item in collected]
    if bbox_counts:
        from collections import Counter
        count_dist = Counter(bbox_counts)
        print(f"\n  [bbox 개수 분포]")
        for k in sorted(count_dist.keys()):
            print(f"    {k}개: {count_dist[k]}개 이미지")
        print(f"  평균 bbox/이미지: {sum(bbox_counts)/len(bbox_counts):.2f}개")

    # 클래스 커버리지 확인
    all_k_codes = set()
    for item in collected:
        all_k_codes.update(item['k_codes'])
    print(f"  포함된 클래스: {len(all_k_codes)}/74개")

    print(f"\n  수집된 이미지: {len(collected)}개")
    return collected


def save_images(metadata_list):
    """2단계: 이미지 저장"""
    print("\n[2단계: 이미지 저장]")

    # 출력 디렉토리 생성/초기화
    import shutil
    print(f"  출력 경로: {OUTPUT_DIR}")

    if OUTPUT_DIR.exists():
        print(f"  기존 디렉토리 삭제 중...")
        try:
            shutil.rmtree(OUTPUT_DIR)
            print(f"  삭제 완료")
        except Exception as e:
            print(f"  삭제 실패: {e}")
            print(f"  기존 파일 위에 덮어씁니다.")

    OUTPUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_ANN_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  디렉토리 생성 완료")

    # ZIP 캐시
    zip_cache = {}

    total_images = 0
    total_bboxes = 0
    errors = 0

    print(f"  총 {len(metadata_list)}개 이미지 저장 시작...")

    for idx, item in enumerate(metadata_list):
        image_id = idx + 1

        try:
            # ZIP 캐싱
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
                json.dump(ann_data, f, ensure_ascii=False)

            total_images += 1
            total_bboxes += len(item['bboxes'])

            # 진행 상황 출력 (500개마다)
            if total_images % 500 == 0:
                print(f"  진행: {total_images}/{len(metadata_list)} ({total_images*100//len(metadata_list)}%)")

        except Exception as e:
            errors += 1
            if errors <= 5:  # 처음 5개만 출력
                print(f"  오류 (이미지 {idx}): {e}")
            continue

    # ZIP 캐시 정리
    for zf in zip_cache.values():
        zf.close()

    if errors > 5:
        print(f"  ... 외 {errors - 5}개 오류")

    return total_images, total_bboxes


def main():
    print("=" * 60)
    print("AIHub Detector 데이터 추출 (74개 클래스 필터)")
    print(f"최대 이미지: {MAX_IMAGES}개 (랜덤 샘플링)")
    print("=" * 60)

    # 1단계: 메타데이터 수집
    metadata = collect_metadata()

    if not metadata:
        print("\n수집된 데이터가 없습니다.")
        return

    # 2단계: 이미지 저장
    total_images, total_bboxes = save_images(metadata)

    print("\n" + "=" * 60)
    print("결과")
    print("=" * 60)
    print(f"총 이미지: {total_images}개")
    print(f"총 bbox: {total_bboxes}개")
    print(f"평균 bbox/이미지: {total_bboxes / total_images:.1f}개" if total_images > 0 else "")
    print(f"\n출력 경로: {OUTPUT_DIR}")
    print("\n다음 단계:")
    print("  1. python -m src.data.cleanup_detector_data --viz")
    print("  2. python -m src.data.yolo_dataset.yolo_export_detector")
    print("=" * 60)


if __name__ == "__main__":
    main()
