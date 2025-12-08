"""
232개 유효 데이터만 필터링해서 저장

실행:
    python src/data/filter_valid_data.py

결과:
    data/processed/train_valid.json  (232개 이미지 + 763개 어노테이션)
"""

import os
import json
from pathlib import Path
from tqdm import tqdm

# 경로 설정
DATA_ROOT = Path('data/raw')
TRAIN_IMG_DIR = DATA_ROOT / 'train_images'
TRAIN_ANNO_DIR = DATA_ROOT / 'train_annotations'
OUTPUT_FILE = Path('data/processed/train_valid.json')

# 출력 디렉토리 생성
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("JSON 파일 파싱 중...")
print("=" * 60)

# JSON 파일 찾기
json_files = []
for root, dirs, files in os.walk(TRAIN_ANNO_DIR):
    for file in files:
        if file.endswith('.json'):
            json_files.append(Path(root) / file)

print(f"총 JSON 파일: {len(json_files)}개")

# 데이터 수집
all_images = []
all_annotations = []
all_categories = []

parsed_filenames = set()
parsed_anno_ids = set()
parsed_category_pairs = set()

for json_path in tqdm(json_files, desc="파싱"):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except:
        continue

    # Images: file_name 중복 제거
    if 'images' in data:
        for img in data['images']:
            if img['file_name'] not in parsed_filenames:
                all_images.append(img)
                parsed_filenames.add(img['file_name'])

    # Annotations: bbox 유효성 + id 중복 제거
    if 'annotations' in data:
        for anno in data['annotations']:
            if ('bbox' in anno and anno['bbox'] and
                len(anno['bbox']) == 4 and
                anno['id'] not in parsed_anno_ids):
                all_annotations.append(anno)
                parsed_anno_ids.add(anno['id'])

    # Categories
    if 'categories' in data:
        for cat in data['categories']:
            cat_pair = (cat['id'], cat['name'])
            if cat_pair not in parsed_category_pairs:
                all_categories.append(cat)
                parsed_category_pairs.add(cat_pair)

print(f"\n파싱 완료: {len(all_images)}개 이미지, {len(all_annotations)}개 어노테이션")

# 이미지-JSON 매칭 (나연님 로직)
print("\n" + "=" * 60)
print("이미지-JSON 매칭 검증...")
print("=" * 60)

actual_images = set([f.name for f in TRAIN_IMG_DIR.glob('*.png')])
json_images = set([img['file_name'] for img in all_images])
valid_images = actual_images.intersection(json_images)

print(f"정상 매칭:     {len(valid_images)}개")
print(f"JSON 없음:    {len(actual_images - valid_images)}개")
print(f"이미지 없음:  {len(json_images - valid_images)}개")

# 유효한 데이터만 필터링
filtered_images = [img for img in all_images if img['file_name'] in valid_images]
valid_image_ids = set([img['id'] for img in filtered_images])
filtered_annotations = [anno for anno in all_annotations if anno['image_id'] in valid_image_ids]

print(f"\n 최종 데이터:")
print(f"   이미지:       {len(filtered_images)}개")
print(f"   어노테이션:   {len(filtered_annotations)}개")
print(f"   클래스:       {len(all_categories)}개")

# JSON 저장
output_data = {
    'images': filtered_images,
    'annotations': filtered_annotations,
    'categories': all_categories
}

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

print(f"\n 저장 완료: {OUTPUT_FILE}")
print(f"   파일 크기: {OUTPUT_FILE.stat().st_size / 1024 / 1024:.2f} MB")
print("\n다음 단계: 이 JSON 파일을 팀원에게 공유")
print("=" * 60)
