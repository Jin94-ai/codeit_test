"""
고아 어노테이션 정리
- 이미지가 없는 JSON 파일 삭제
- 수동으로 잘못된 이미지 삭제 후 실행
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# 경로 설정
CROPPED_IMG_DIR = PROJECT_ROOT / "data" / "cropped" / "images"
CROPPED_ANN_DIR = PROJECT_ROOT / "data" / "cropped" / "annotations"


def cleanup_orphan_annotations():
    """이미지 없는 어노테이션 삭제"""
    print("=" * 60)
    print("고아 어노테이션 정리")
    print("=" * 60)

    # 존재하는 이미지 파일명 (확장자 제외)
    existing_images = set()
    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        for img_path in CROPPED_IMG_DIR.glob(ext):
            existing_images.add(img_path.stem)

    print(f"이미지 수: {len(existing_images)}")

    # 어노테이션 파일 확인
    deleted = 0
    kept = 0

    for json_path in CROPPED_ANN_DIR.glob("*.json"):
        stem = json_path.stem

        if stem not in existing_images:
            # 이미지 없음 → 삭제
            json_path.unlink()
            deleted += 1
        else:
            kept += 1

    print(f"\n삭제된 어노테이션: {deleted}")
    print(f"유지된 어노테이션: {kept}")
    print("=" * 60)


if __name__ == "__main__":
    cleanup_orphan_annotations()
