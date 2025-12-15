import json
import shutil
from pathlib import Path
import copy

from src.data.yolo_dataset.config import (
    ADD_ANN_DIR,
    ADD_IMG_DIR,
    TRAIN_ANN_DIR,
    TRAIN_IMG_DIR,
    ADDED_TRAIN_ANN_DIR
)


def diff_summary(before: dict, after: dict) -> dict:
    """
    수정 전/후 핵심 변경 사항 요약
    """
    return {
        "category_id": (
            before["categories"][0]["id"],
            after["categories"][0]["id"],
        ),
        "category_name": (
            before["categories"][0]["name"],
            after["categories"][0]["name"],
        ),
    }


def update_category_from_image_info(json_path: Path):
    """
    categories.id   <- images.dl_idx
    categories.name <- images.dl_name
    annotations.category_id 동기화
    """

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    original = copy.deepcopy(data)

    images = data.get("images", [])
    categories = data.get("categories", [])
    annotations = data.get("annotations", [])

    if not images or not categories:
        return False, None

    dl_idx = images[0].get("dl_idx")
    dl_name = images[0].get("dl_name")

    if dl_idx is None or dl_name is None:
        return False, None

    categories[0]["id"] = int(dl_idx)
    categories[0]["name"] = dl_name

    for ann in annotations:
        ann["category_id"] = int(dl_idx)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    diff = diff_summary(original, data)
    return True, diff


def move_annotation(json_path: Path):
    dst = Path(TRAIN_ANN_DIR) / json_path.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(json_path), TRAIN_ANN_DIR)
    return dst


def move_image(image_name: str):
    src = Path(ADD_IMG_DIR) / image_name
    if not src.exists():
        return False

    dst = Path(TRAIN_IMG_DIR) / image_name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    return True


def process_add_dataset():
    updated_json_count = 0
    moved_ann_count = 0
    moved_img_count = 0

    ann_root = Path(ADD_ANN_DIR)

    for sub_dir in ann_root.iterdir():
        if not sub_dir.is_dir():
            continue

        for json_file in sub_dir.glob("*.json"):
            success, diff = update_category_from_image_info(json_file)

            if not success:
                continue

            updated_json_count += 1

            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            image_name = data["images"][0]["file_name"]

            # annotation 이동
            move_annotation(json_file)
            moved_ann_count += 1

            # image 이동 (폴더 ❌, 파일만)
            if move_image(image_name):
                moved_img_count += 1

            # diff 출력 (검증용)
            print(
                f"[DIFF] {json_file.name} | "
                f"id: {diff['category_id'][0]} → {diff['category_id'][1]}, "
                f"name: {diff['category_name'][0]} → {diff['category_name'][1]}"
            )

    print("\n===== ADD DATASET PROCESS SUMMARY =====")
    print(f"✓ 수정 완료 json 수          : {updated_json_count}")
    print(f"✓ 이동 완료 annotation 수    : {moved_ann_count}")
    print(f"✓ 이동 완료 image 수         : {moved_img_count}")
    print("======================================")


if __name__ == "__main__":
    process_add_dataset()