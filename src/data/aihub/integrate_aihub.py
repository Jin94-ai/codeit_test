#!/usr/bin/env python3
"""
AIHub 데이터 통합 스크립트

사용법:
1. AIHub에서 TL 데이터셋 다운로드 (aihubshell 사용)
2. 이 스크립트 실행: python integrate_aihub.py
3. 자동으로 압축 해제 → 필터링 → JSON 수정
4. 결과: data/aihub_integrated/annotations/
"""
import json
import zipfile
from pathlib import Path
from collections import defaultdict

TARGET_CLASSES = {
    '1899', '2482', '3350', '3482', '3543', '3742', '3831', '4542',
    '12080', '12246', '12777', '13394', '13899', '16231', '16261', '16547',
    '16550', '16687', '18146', '18356', '19231', '19551', '19606', '19860',
    '20013', '20237', '20876', '21324', '21770', '22073', '22346', '22361',
    '24849', '25366', '25437', '25468', '27732', '27776', '27925', '27992',
    '28762', '29344', '29450', '29666', '30307', '31862', '31884', '32309',
    '33008', '33207', '33879', '34596', '35205', '36636', '38161', '41767'
}

MAX_PER_CLASS = 200  # 클래스당 최대 개수


def analyze_competition():
    """Competition 데이터 분석"""
    print("\n[1/3] Competition 데이터 분석")
    comp_anno = Path("data/train_annotations")
    counts = defaultdict(int)

    if comp_anno.exists():
        for json_path in comp_anno.glob("*.json"):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if 'categories' in data and data['categories']:
                    cat_id = str(data['categories'][0]['id'])
                    if cat_id in TARGET_CLASSES:
                        counts[cat_id] += 1
            except:
                continue

    lacking = sum(1 for c in TARGET_CLASSES if counts.get(c, 0) < 10)
    print(f"  Competition 클래스: {len(counts)}/56")
    print(f"  10개 미만 클래스: {lacking}개")
    return counts


def extract_zips():
    """다운로드된 ZIP 파일 압축 해제"""
    print("\n[2/3] ZIP 파일 압축 해제")

    base_dir = Path("data/166.약품식별 인공지능 개발을 위한 경구약제 이미지 데이터/01.데이터/1.Training/라벨링데이터")
    extract_base = Path("data/aihub_downloads")
    extract_base.mkdir(parents=True, exist_ok=True)

    # 조합만 처리
    zip_dirs = [
        (base_dir / "경구약제조합 5000종", "_조합", "_combo"),
    ]

    total_extracted = 0

    for zip_dir, suffix, folder_suffix in zip_dirs:
        if not zip_dir.exists():
            print(f"  ⚠️ {zip_dir.name} 폴더 없음")
            continue

        for zip_file in zip_dir.glob(f"TL_*{suffix}.zip"):
            name = zip_file.stem.replace(suffix, "")
            extract_dir = extract_base / f"{name}{folder_suffix}"

            if extract_dir.exists() and any(extract_dir.rglob("*.json")):
                print(f"  ✓ {name}{suffix} 이미 압축 해제됨")
                continue

            try:
                print(f"  압축 해제 중: {zip_file.name}")
                with zipfile.ZipFile(zip_file, 'r') as zf:
                    zf.extractall(extract_dir)
                print(f"  ✅ {name}{suffix} 완료")
                total_extracted += 1
            except Exception as e:
                print(f"  ❌ {name}{suffix} 실패: {e}")

    if total_extracted == 0:
        print("  (모두 이미 압축 해제됨)")


def integrate_aihub(comp_counts):
    """AIHub 데이터 필터링 및 통합"""
    print("\n[3/3] AIHub 데이터 통합")

    output_dir = Path("data/aihub_integrated/annotations")
    output_dir.mkdir(parents=True, exist_ok=True)

    aihub_counts = defaultdict(int)
    total = 0

    # TL combo + single 데이터 처리
    aihub_dir = Path("data/aihub_downloads")
    if not aihub_dir.exists():
        print("  ⚠️ AIHub 데이터 없음")
        return aihub_counts

    # 조합만 처리
    for tl_dir in sorted(aihub_dir.glob("TL_*_combo")):
        print(f"  처리 중: {tl_dir.name}")

        # 조기 종료: 모든 클래스가 MAX_PER_CLASS 채워졌는지 확인
        all_filled = all(
            comp_counts.get(c, 0) + aihub_counts.get(c, 0) >= MAX_PER_CLASS
            for c in TARGET_CLASSES
        )
        if all_filled:
            print("  ✅ 모든 클래스 200개 채워짐, 조기 종료")
            break

        processed_in_dir = 0
        for json_path in tl_dir.rglob("*.json"):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if 'images' not in data or not data['images']:
                    continue

                img_info = data['images'][0]
                dl_idx = str(img_info.get('dl_idx', ''))

                if dl_idx not in TARGET_CLASSES:
                    continue

                # 클래스당 MAX_PER_CLASS개까지만
                current = comp_counts.get(dl_idx, 0) + aihub_counts.get(dl_idx, 0)
                if current >= MAX_PER_CLASS:
                    continue

                # JSON 수정
                dl_name = img_info.get('dl_name', 'Drug')
                cat_id = int(dl_idx)

                data['categories'] = [{
                    'supercategory': 'pill',
                    'id': cat_id,
                    'name': dl_name
                }]

                for anno in data['annotations']:
                    anno['category_id'] = cat_id

                # 저장
                out_name = f"{dl_idx}_{aihub_counts[dl_idx]:04d}.json"
                with open(output_dir / out_name, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

                aihub_counts[dl_idx] += 1
                total += 1
                processed_in_dir += 1

                if total % 100 == 0:
                    print(f"    진행: {total}개 (현재: {len(aihub_counts)}개 클래스)")

            except:
                continue

        print(f"    → {tl_dir.name}에서 {processed_in_dir}개 추가")

    print(f"  ✅ {total}개 파일 통합 완료")
    return aihub_counts


def print_summary(comp_counts, aihub_counts):
    """결과 요약"""
    print("\n" + "=" * 70)
    print("통합 결과")
    print("=" * 70)

    total_img = sum(comp_counts.values()) + sum(aihub_counts.values())
    avg = total_img / 56 if total_img > 0 else 0
    lacking = sum(1 for c in TARGET_CLASSES
                  if comp_counts.get(c, 0) + aihub_counts.get(c, 0) < 10)

    print(f"Competition: {sum(comp_counts.values())}개")
    print(f"AIHub 추가: {sum(aihub_counts.values())}개")
    print(f"총 이미지: {total_img}개 (평균 {avg:.1f}개/클래스)")
    print(f"10개 미만 클래스: {lacking}개")

    if lacking > 0:
        print(f"\n추가 다운로드 필요: TL_11 이상 데이터셋")
    else:
        print(f"\n✅ 모든 클래스 10개 이상 확보")

    print(f"\n통합 데이터: data/aihub_integrated/annotations/")
    print("=" * 70)


def main():
    print("=" * 70)
    print("AIHub 데이터 통합")
    print("=" * 70)

    comp_counts = analyze_competition()
    extract_zips()
    aihub_counts = integrate_aihub(comp_counts)
    print_summary(comp_counts, aihub_counts)


if __name__ == "__main__":
    main()
