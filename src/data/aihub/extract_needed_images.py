#!/usr/bin/env python3
"""
combo 데이터의 필요한 이미지만 선택적으로 추출
"""
import json
import zipfile
from pathlib import Path
from collections import defaultdict

def collect_needed_images():
    """통합된 JSON에서 필요한 이미지 파일명 수집"""
    anno_dir = Path("data/aihub_integrated/annotations")

    if not anno_dir.exists():
        print("❌ 먼저 integrate_aihub.py를 실행하세요")
        return {}

    # TL별로 필요한 이미지 분류
    needed_by_tl = defaultdict(set)  # {TL_name: {image_files}}

    print("JSON 파일에서 이미지 파일명 수집 중...")

    for json_path in anno_dir.glob("*.json"):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if 'images' not in data or not data['images']:
                continue

            img_file = data['images'][0]['file_name']

            # 파일명 패턴: K-001900-004543-010224-016551_0_2_0_2_70_000_200.png
            # 이 파일은 여러 TL에 걸쳐 있을 수 있음
            # 일단 모든 combo TL에서 찾도록
            needed_by_tl['all'].add(img_file)

        except Exception as e:
            continue

    total = len(needed_by_tl['all'])
    print(f"✅ 필요한 이미지: {total}개")

    return needed_by_tl

def extract_from_zip(zip_path: Path, needed_files: set, output_dir: Path):
    """ZIP에서 필요한 파일만 선택적으로 추출"""

    if not zip_path.exists():
        print(f"  ⚠️ ZIP 없음: {zip_path.name}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    extracted = 0

    size_gb = zip_path.stat().st_size / 1024**3
    print(f"  처리 중: {zip_path.name} ({size_gb:.1f}GB)")

    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # ZIP 내부 파일 목록
            all_files = zf.namelist()

            # PNG 파일만 필터
            image_files = [f for f in all_files if f.endswith('.png')]

            print(f"    ZIP 내부 이미지: {len(image_files)}개")

            # 필요한 파일만 추출
            for zip_file in image_files:
                # 파일명만 추출
                filename = Path(zip_file).name

                if filename in needed_files:
                    try:
                        # 파일명만으로 저장 (폴더 구조 무시)
                        data = zf.read(zip_file)
                        (output_dir / filename).write_bytes(data)
                        extracted += 1

                        if extracted % 50 == 0:
                            print(f"    추출: {extracted}개...")

                    except Exception as e:
                        continue

    except Exception as e:
        print(f"  ❌ 오류: {e}")

    return extracted

def main():
    print("=" * 70)
    print("combo 데이터 이미지 선택적 추출")
    print("=" * 70)

    # 1. 필요한 이미지 파일명 수집
    print("\n[1/2] 필요한 이미지 파일명 수집")
    needed = collect_needed_images()

    if not needed or not needed['all']:
        print("❌ 필요한 이미지 없음")
        return

    needed_files = needed['all']
    print(f"\n대상 이미지: {len(needed_files)}개")

    # 2. ZIP에서 선택적 추출
    print("\n[2/2] 이미지 ZIP에서 추출")

    zip_base = Path("data/166.약품식별 인공지능 개발을 위한 경구약제 이미지 데이터/01.데이터/1.Training/원천데이터/경구약제조합 5000종")
    output_base = Path("data/aihub_integrated/images")
    output_base.mkdir(parents=True, exist_ok=True)

    if not zip_base.exists():
        print("⚠️ 이미지 ZIP 폴더 없음")
        print(f"경로: {zip_base}")
        print("\n수동 다운로드 필요:")
        print("  - aihubshell로 원천데이터(이미지) 다운로드")
        print("  - TL_X_조합.zip 파일을 위 경로에 배치")
        return

    total_extracted = 0

    # TL 또는 TS combo ZIP에서 필요한 파일만 추출
    zip_files = list(zip_base.glob("TL_*_조합.zip")) + list(zip_base.glob("TS_*_조합.zip"))

    if not zip_files:
        print("  ⚠️ 조합 이미지 ZIP 파일 없음")
        print(f"  경로: {zip_base}")
        return

    for zip_file in sorted(zip_files):
        name = zip_file.stem.replace("_조합", "")
        extracted = extract_from_zip(zip_file, needed_files, output_base)
        total_extracted += extracted

        if extracted > 0:
            print(f"  ✅ {name}: {extracted}개 추출")

    print("\n" + "=" * 70)
    print(f"총 추출: {total_extracted}/{len(needed_files)}개")

    if total_extracted < len(needed_files):
        missing = len(needed_files) - total_extracted
        print(f"⚠️ 누락: {missing}개 (더 많은 TL 이미지 필요)")
    else:
        print("✅ 모든 이미지 추출 완료")

    print(f"\n저장 위치: {output_base}/")
    print("=" * 70)

if __name__ == "__main__":
    main()
