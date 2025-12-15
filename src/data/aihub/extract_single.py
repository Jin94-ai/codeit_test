"""
AIHub 단일 데이터에서 클래스당 100개씩 추출

1. TL_X_단일.zip에서 라벨 JSON 읽기
2. TARGET_CLASSES만 필터링
3. 클래스당 100개까지 수집
4. 이미지를 train_images/에 추출
5. JSON을 train_annotations/에 복사 (기존 yolo_export 파이프라인 활용)

실행: python -m src.data.aihub.extract_single
또는: python src/data/aihub/extract_single.py
"""
import os
import sys
import json
import zipfile
from pathlib import Path
from collections import defaultdict

# 직접 실행 시 import 경로 추가
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    from src.data.aihub.config import (
        TARGET_CLASSES,
        MAX_PER_CLASS,
        AIHUB_LABEL_DIR,
        AIHUB_IMAGE_DIR,
        OUTPUT_IMAGE_DIR,
        OUTPUT_ANN_DIR,
    )
else:
    from .config import (
        TARGET_CLASSES,
        MAX_PER_CLASS,
        AIHUB_LABEL_DIR,
        AIHUB_IMAGE_DIR,
        OUTPUT_IMAGE_DIR,
        OUTPUT_ANN_DIR,
    )


def scan_tl_zips():
    """TL zip 파일 목록 반환"""
    if not os.path.exists(AIHUB_LABEL_DIR):
        print(f"오류: 라벨 디렉토리가 없습니다: {AIHUB_LABEL_DIR}")
        return []

    return sorted([
        os.path.join(AIHUB_LABEL_DIR, f)
        for f in os.listdir(AIHUB_LABEL_DIR)
        if f.endswith('.zip') and '단일' in f
    ])


def extract_from_tl_zip(zip_path: str, class_counts: dict, collected: dict):
    """
    TL zip 파일에서 TARGET 클래스 데이터 수집

    Args:
        zip_path: TL_X_단일.zip 경로
        class_counts: 클래스별 현재 수집 개수
        collected: 수집된 데이터 {file_name: {'json_data': ..., 'ts_zip': ...}}
    """
    tl_name = Path(zip_path).stem  # TL_X_단일
    ts_zip_name = tl_name.replace("TL_", "TS_") + ".zip"  # TS_X_단일.zip

    print(f"\n처리 중: {tl_name}")
    added = 0

    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            json_files = [f for f in zf.namelist() if f.endswith('.json')]

            for json_file in json_files:
                # 이미 모든 클래스가 충분하면 스킵
                if all(class_counts.get(c, 0) >= MAX_PER_CLASS for c in TARGET_CLASSES):
                    break

                try:
                    with zf.open(json_file) as f:
                        data = json.load(f)
                except:
                    continue

                # 이미지 정보 확인
                if not data.get('images') or not data.get('annotations'):
                    continue

                img_info = data['images'][0]
                file_name = img_info.get('file_name', '')

                if not file_name:
                    continue

                # 이미 수집된 파일이면 스킵
                if file_name in collected:
                    continue

                # dl_idx로 클래스 확인 (AIHub 단일 이미지는 category_id=1이고 실제 클래스는 dl_idx)
                dl_idx = str(img_info.get('dl_idx', ''))

                # TARGET 클래스가 아니면 스킵
                if dl_idx not in TARGET_CLASSES:
                    continue

                # 이미 충분하면 스킵
                if class_counts.get(dl_idx, 0) >= MAX_PER_CLASS:
                    continue

                # annotation이 없으면 스킵
                annotations = data.get('annotations', [])
                if not annotations:
                    continue

                # 수집
                collected[file_name] = {
                    'json_data': data,
                    'ts_zip': ts_zip_name,
                    'json_path': json_file,
                    'dl_idx': dl_idx
                }

                # 클래스 카운트 업데이트
                class_counts[dl_idx] = class_counts.get(dl_idx, 0) + 1

                added += 1

        print(f"  추가: {added}개")

    except Exception as e:
        print(f"  오류: {e}")


def extract_to_train_dirs(collected: dict):
    """
    수집된 데이터를 train_images/, train_annotations/에 추출
    """
    os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_ANN_DIR, exist_ok=True)

    # ts_zip별로 그룹화
    ts_to_files = defaultdict(list)
    for file_name, info in collected.items():
        ts_to_files[info['ts_zip']].append((file_name, info))

    images_extracted = 0
    jsons_created = 0

    for ts_zip_name, files in ts_to_files.items():
        ts_zip_path = os.path.join(AIHUB_IMAGE_DIR, ts_zip_name)

        if not os.path.exists(ts_zip_path):
            print(f"  경고: {ts_zip_name} 없음")
            continue

        print(f"\n{ts_zip_name}에서 {len(files)}개 이미지 추출 중...")

        try:
            with zipfile.ZipFile(ts_zip_path, 'r') as zf:
                # zip 내 파일 목록 (한번만 읽기)
                zip_files_list = zf.namelist()
                zip_files_map = {os.path.basename(f): f for f in zip_files_list}

                for file_name, info in files:
                    # 이미지 추출
                    dst_img = os.path.join(OUTPUT_IMAGE_DIR, file_name)

                    if not os.path.exists(dst_img):
                        # zip 내에서 파일 찾기
                        if file_name in zip_files_map:
                            try:
                                with zf.open(zip_files_map[file_name]) as src:
                                    with open(dst_img, 'wb') as dst:
                                        dst.write(src.read())
                                images_extracted += 1
                            except Exception as e:
                                print(f"    이미지 추출 실패: {file_name} - {e}")
                                continue
                        else:
                            print(f"    이미지 없음: {file_name}")
                            continue

                    # JSON 저장 (간단한 구조로)
                    # dl_idx별 폴더로 저장
                    json_data = info['json_data']
                    dl_idx = info.get('dl_idx', 'unknown')

                    json_dir = os.path.join(OUTPUT_ANN_DIR, f"dl_{dl_idx}")
                    os.makedirs(json_dir, exist_ok=True)

                    json_path = os.path.join(json_dir, Path(file_name).stem + ".json")

                    if not os.path.exists(json_path):
                        with open(json_path, 'w', encoding='utf-8') as f:
                            json.dump(json_data, f, ensure_ascii=False, indent=2)
                        jsons_created += 1

        except Exception as e:
            print(f"  오류: {ts_zip_name} - {e}")

    return images_extracted, jsons_created


def main():
    print("=" * 60)
    print("AIHub 단일 데이터 추출 (클래스당 100개)")
    print("=" * 60)

    # 1. TL zip 파일 목록
    zip_files = scan_tl_zips()
    if not zip_files:
        return

    print(f"TL zip 파일 수: {len(zip_files)}")

    # 2. 클래스별 카운트 및 수집 데이터
    class_counts = defaultdict(int)
    collected = {}

    # 3. 각 zip 파일 처리
    for zip_path in zip_files:
        extract_from_tl_zip(zip_path, class_counts, collected)

        # 모든 클래스가 충분하면 중단
        if all(class_counts.get(c, 0) >= MAX_PER_CLASS for c in TARGET_CLASSES):
            print("\n모든 클래스가 100개 이상 수집됨!")
            break

    print(f"\n총 수집: {len(collected)}개 이미지")

    # 4. train_images, train_annotations에 추출
    print("\n" + "=" * 60)
    print("데이터 추출 중...")
    print("=" * 60)

    images_extracted, jsons_created = extract_to_train_dirs(collected)

    # 5. 결과 출력
    print("\n" + "=" * 60)
    print("추출 완료")
    print("=" * 60)
    print(f"이미지 추출: {images_extracted}개")
    print(f"JSON 생성: {jsons_created}개")

    print(f"\n[클래스별 수집 현황]")
    sorted_cats = sorted(TARGET_CLASSES, key=int)
    for cat_id in sorted_cats:
        count = class_counts.get(cat_id, 0)
        status = "✓" if count >= MAX_PER_CLASS else f"({count}/{MAX_PER_CLASS})"
        print(f"  {cat_id}: {count}개 {status}")

    insufficient = [c for c in TARGET_CLASSES if class_counts.get(c, 0) < MAX_PER_CLASS]
    if insufficient:
        print(f"\n경고: {len(insufficient)}개 클래스가 100개 미만")

    print(f"\n출력 경로:")
    print(f"  이미지: {OUTPUT_IMAGE_DIR}")
    print(f"  라벨: {OUTPUT_ANN_DIR}/AIHub_single/")
    print(f"\n다음 단계: python -m src.data.yolo_dataset.yolo_export")


if __name__ == "__main__":
    main()
