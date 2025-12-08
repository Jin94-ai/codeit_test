import os
import glob
import json
import pandas as pd
from typing import Tuple, List
from config import TRAIN_ANN_DIR

def load_coco_tables() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    train_annotations 폴더의 모든 COCO 스타일 JSON을 읽어
    images_df, annotations_df, categories_df를 반환한다.
    """
    json_files: List[str] = glob.glob(os.path.join(TRAIN_ANN_DIR, "**/*.json"), recursive=True)
    print(f"[coco_parser] JSON 파일 수: {len(json_files)}")

    all_images = []
    all_annotations = []
    all_categories = []

    parsed_image_ids = set()
    parsed_anno_ids = set()
    parsed_cat_pairs = set()

    for jp in json_files:
        try:
            with open(jp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[coco_parser] 경고: JSON 파싱 실패: {jp} - {e}")
            continue

        # images
        for img in data.get("images", []):
            if img["id"] not in parsed_image_ids:
                all_images.append(img)
                parsed_image_ids.add(img["id"])

        # annotations
        for ann in data.get("annotations", []):
            bbox = ann.get("bbox")
            if (
                ann["id"] not in parsed_anno_ids
                and isinstance(bbox, list)
                and len(bbox) == 4
            ):
                all_annotations.append(ann)
                parsed_anno_ids.add(ann["id"])

        # categories
        for cat in data.get("categories", []):
            key = (cat["id"], cat.get("name", ""))
            if key not in parsed_cat_pairs:
                all_categories.append(cat)
                parsed_cat_pairs.add(key)

    images_df = pd.DataFrame(all_images)
    annotations_df = pd.DataFrame(all_annotations)
    categories_df = pd.DataFrame(all_categories)

    print(f"[coco_parser] images_df: {len(images_df)}")
    print(f"[coco_parser] annotations_df: {len(annotations_df)}")
    print(f"[coco_parser] categories_df: {len(categories_df)}")

    return images_df, annotations_df, categories_df


if __name__ == "__main__":
    # 테스트 실행용
    imgs, anns, cats = load_coco_tables()
    print("샘플 images_df:\n", imgs.head())
    print("샘플 annotations_df:\n", anns.head())
    print("샘플 categories_df:\n", cats.head())
