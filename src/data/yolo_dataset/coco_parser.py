import os
import glob
import json
import pandas as pd
from typing import Tuple, List
from .config import TRAIN_ANN_DIR, TRAIN_IMG_DIR, AIHUB_ANN_DIR, AIHUB_IMG_DIR


def load_coco_tables_with_consistency() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    1) train_annotations + aihub_single/annotations 아래 모든 JSON을 파싱
    2) 폴더(train_images + aihub_single/images)와 JSON 양쪽에 모두 존재하는 file_name만 필터링
       → 최종적으로 정합성이 보장된 images_df, annotations_df, categories_df 반환
    """
    # 1. JSON 파싱 (캐글 + AIHub)
    json_files: List[str] = glob.glob(os.path.join(TRAIN_ANN_DIR, "**/*.json"), recursive=True)
    aihub_json_files: List[str] = glob.glob(os.path.join(AIHUB_ANN_DIR, "**/*.json"), recursive=True)

    print(f"[coco_parser] 캐글 JSON 파일 수: {len(json_files)}")
    print(f"[coco_parser] AIHub JSON 파일 수: {len(aihub_json_files)}")

    json_files.extend(aihub_json_files)
    print(f"[coco_parser] 총 JSON 파일 수: {len(json_files)}")

    all_images_meta = []
    all_annotations = []
    all_category_mappings = []

    parsed_image_filenames_set = set()
    parsed_category_id_name_pairs_set = set()

    # file_name -> unique_image_id 매핑 (AIHub ID 충돌 해결)
    filename_to_image_id = {}
    next_image_id = 100000  # 캐글 ID와 충돌 방지
    next_anno_id = 100000

    for json_path in json_files:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[coco_parser] 경고: {json_path} 파싱 중 오류 발생: {e} (건너뜀)")
            continue

        # AIHub 데이터 여부 확인 (dl_idx가 있으면 AIHub)
        dl_idx = None
        if data.get("images"):
            dl_idx = data["images"][0].get("dl_idx")
        is_aihub = dl_idx is not None

        # images 섹션
        for img_info in data.get("images", []):
            fn = img_info.get("file_name")
            if fn and fn not in parsed_image_filenames_set:
                img_info = img_info.copy()

                if is_aihub:
                    # AIHub: 고유 image_id 생성
                    img_info["id"] = next_image_id
                    filename_to_image_id[fn] = next_image_id
                    next_image_id += 1
                else:
                    # 캐글: 기존 ID 유지
                    filename_to_image_id[fn] = img_info["id"]

                all_images_meta.append(img_info)
                parsed_image_filenames_set.add(fn)

        # annotations 섹션
        for anno_info in data.get("annotations", []):
            bbox = anno_info.get("bbox")
            if not (isinstance(bbox, list) and len(bbox) == 4):
                continue

            anno_info = anno_info.copy()

            if is_aihub:
                # AIHub: 고유 annotation_id 생성 + category_id 변환
                anno_info["id"] = next_anno_id
                anno_info["category_id"] = int(dl_idx)
                # image_id도 새로 생성된 ID로 업데이트
                fn = data["images"][0].get("file_name")
                if fn in filename_to_image_id:
                    anno_info["image_id"] = filename_to_image_id[fn]
                next_anno_id += 1

            all_annotations.append(anno_info)

        # categories 섹션
        if is_aihub and dl_idx:
            # AIHub: dl_idx를 category_id로, dl_name을 name으로 사용
            dl_name = data["images"][0].get("dl_name", f"AIHub_{dl_idx}")
            cat_id_name_pair = (int(dl_idx), dl_name)
            if cat_id_name_pair not in parsed_category_id_name_pairs_set:
                all_category_mappings.append({"id": int(dl_idx), "name": dl_name})
                parsed_category_id_name_pairs_set.add(cat_id_name_pair)
        else:
            # 캐글: 기존 방식
            for cat_info in data.get("categories", []):
                cat_id_name_pair = (cat_info["id"], cat_info.get("name", ""))
                if cat_id_name_pair not in parsed_category_id_name_pairs_set:
                    all_category_mappings.append(cat_info)
                    parsed_category_id_name_pairs_set.add(cat_id_name_pair)

    images_df = pd.DataFrame(all_images_meta)
    annotations_df = pd.DataFrame(all_annotations)
    categories_df = pd.DataFrame(all_category_mappings)

    print(f"[coco_parser] JSON 기준 이미지 메타데이터 개수: {len(images_df)}")
    print(f"[coco_parser] JSON 기준 어노테이션 개수: {len(annotations_df)}")
    print(f"[coco_parser] 고유 카테고리 개수: {len(categories_df)}")

    # 2. 폴더(train_images + aihub_single/images)와 JSON 간 정합성 검사 → 교집합만 사용
    train_images = set(os.path.basename(p) for p in glob.glob(os.path.join(TRAIN_IMG_DIR, "*.png")))
    aihub_images = set(os.path.basename(p) for p in glob.glob(os.path.join(AIHUB_IMG_DIR, "*.png")))
    all_train_image_filenames = train_images.union(aihub_images)

    print(f"[coco_parser] 캐글 이미지 수: {len(train_images)}")
    print(f"[coco_parser] AIHub 이미지 수: {len(aihub_images)}")
    parsed_image_filenames_from_json = set(images_df["file_name"].tolist())

    valid_image_filenames = all_train_image_filenames.intersection(parsed_image_filenames_from_json)

    unmatched_images_in_folder = all_train_image_filenames - valid_image_filenames
    unmatched_images_in_json = parsed_image_filenames_from_json - valid_image_filenames

    print("\n[coco_parser] 데이터 정합성 검사 결과")
    print(f"- 폴더에는 있지만 JSON에 없는 이미지 수: {len(unmatched_images_in_folder)}")
    print(f"- JSON에는 있지만 폴더에 없는 이미지 수: {len(unmatched_images_in_json)}")

    # 3. 교집합(유효한 file_name)만 남기기 → 여기서 232개로 줄어듦
    images_df = images_df[images_df["file_name"].isin(valid_image_filenames)].reset_index(drop=True)

    # annotations_df는 image_id를 통해 이미지와 연결되므로, 먼저 image_id 집합을 좁힌 뒤 필터링
    valid_image_ids = set(images_df["id"].tolist())
    annotations_df = annotations_df[annotations_df["image_id"].isin(valid_image_ids)].reset_index(drop=True)

    print(f"\n[coco_parser] 최종 필터링 후 이미지 개수: {len(images_df)}")
    print(f"[coco_parser] 최종 필터링 후 어노테이션 개수: {len(annotations_df)}")

    return images_df, annotations_df, categories_df


if __name__ == "__main__":
    imgs, anns, cats = load_coco_tables_with_consistency()
    print(imgs.head())
    print(anns.head())
    print(cats.head())
