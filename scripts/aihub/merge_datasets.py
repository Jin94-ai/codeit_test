#!/usr/bin/env python3
"""AIHub 데이터를 Competition 데이터셋에 병합"""
import shutil
import json
from pathlib import Path

def merge_data():
    print("=" * 70)
    print("AIHub → Competition 데이터 병합")
    print("=" * 70)

    # 경로 설정
    aihub_img = Path("data/aihub_integrated/images")
    aihub_anno = Path("data/aihub_integrated/annotations")
    comp_img = Path("data/train")
    comp_anno = Path("data/train_annotations")

    # 폴더 확인
    if not aihub_img.exists() or not aihub_anno.exists():
        print("❌ AIHub 데이터 없음")
        return

    comp_img.mkdir(parents=True, exist_ok=True)
    comp_anno.mkdir(parents=True, exist_ok=True)

    print(f"\n[1/2] 이미지 병합")
    img_copied = 0
    img_skipped = 0

    for img_path in aihub_img.glob("*.png"):
        dest = comp_img / img_path.name
        if dest.exists():
            img_skipped += 1
        else:
            shutil.copy2(img_path, dest)
            img_copied += 1
            if img_copied % 100 == 0:
                print(f"  복사 중: {img_copied}개...")

    print(f"  ✅ 이미지 복사: {img_copied}개")
    if img_skipped > 0:
        print(f"  ⚠️ 이미지 건너뜀: {img_skipped}개 (이미 존재)")

    print(f"\n[2/2] JSON 병합")
    json_copied = 0
    json_skipped = 0
    json_no_image = 0

    for json_path in aihub_anno.glob("*.json"):
        # 이미지 확인
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if 'images' not in data or not data['images']:
                json_no_image += 1
                continue

            img_file = data['images'][0]['file_name']
            if not (comp_img / img_file).exists():
                json_no_image += 1
                continue
        except:
            json_no_image += 1
            continue

        # JSON 복사
        dest = comp_anno / json_path.name
        if dest.exists():
            json_skipped += 1
        else:
            shutil.copy2(json_path, dest)
            json_copied += 1
            if json_copied % 100 == 0:
                print(f"  복사 중: {json_copied}개...")

    print(f"  ✅ JSON 복사: {json_copied}개")
    if json_skipped > 0:
        print(f"  ⚠️ JSON 건너뜀: {json_skipped}개 (이미 존재)")
    if json_no_image > 0:
        print(f"  ⚠️ JSON 제외: {json_no_image}개 (이미지 없음)")

    print("\n" + "=" * 70)
    print(f"병합 완료")
    print(f"  이미지: {img_copied}개 추가")
    print(f"  JSON: {json_copied}개 추가")
    print("=" * 70)

if __name__ == "__main__":
    merge_data()
