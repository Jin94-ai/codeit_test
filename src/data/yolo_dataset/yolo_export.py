import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import TRAIN_IMG_DIR, YOLO_ROOT, VAL_RATIO, SPLIT_SEED
from .coco_parser import load_coco_tables_with_consistency


def build_image_level_df(images_df: pd.DataFrame, annotations_df: pd.DataFrame) -> pd.DataFrame:
    """
    이미지 단위 DataFrame 생성 + 각 이미지당 대표 category_id(rep_category) 부여.
    """
    first_cat_per_image = (
        annotations_df
        .sort_values("id")
        .groupby("image_id")["category_id"]
        .first()
        .rename("rep_category")
        .reset_index()
    )

    img_level_df = images_df.merge(
        first_cat_per_image,
        left_on="id",
        right_on="image_id",
        how="inner"
    )

    return img_level_df


def stratified_train_val_split(
    img_level_df: pd.DataFrame,
    annotations_df: pd.DataFrame,
    val_ratio: float,
    seed: int
):
    """
    대표 class(rep_category)를 기준으로 stratified 8:2 split.
    - 단, 이미지가 1장뿐인 클래스는 stratify에서 제외하고 전부 train에 넣는다.
    """
    X_all = img_level_df["id"].values
    y_all = img_level_df["rep_category"].values

    # 클래스별 이미지 수
    class_counts = img_level_df["rep_category"].value_counts()

    # 2장 이상 있는 클래스만 stratify 대상으로 사용
    valid_classes = class_counts[class_counts >= 2].index.tolist()

    mask_valid = img_level_df["rep_category"].isin(valid_classes)
    mask_single = ~mask_valid  # 1장짜리 클래스들

    X_valid = img_level_df.loc[mask_valid, "id"].values
    y_valid = img_level_df.loc[mask_valid, "rep_category"].values

    # stratify 가능한 부분에만 train_test_split 적용
    train_ids_valid, val_ids_valid = train_test_split(
        X_valid,
        test_size=val_ratio,
        random_state=seed,
        stratify=y_valid
    )

    train_ids_valid = set(train_ids_valid.tolist())
    val_ids_valid = set(val_ids_valid.tolist())

    # 1장짜리 클래스들은 모두 train에 보냄
    single_class_ids = set(img_level_df.loc[mask_single, "id"].values)

    train_ids = train_ids_valid.union(single_class_ids)
    val_ids = val_ids_valid  # val에는 single-class 이미지 없음

    train_images_df = img_level_df[img_level_df["id"].isin(train_ids)].copy()
    val_images_df = img_level_df[img_level_df["id"].isin(val_ids)].copy()

    train_ann_df = annotations_df[annotations_df["image_id"].isin(train_ids)].copy()
    val_ann_df = annotations_df[annotations_df["image_id"].isin(val_ids)].copy()

    print(f"[split] Train 이미지 수: {len(train_images_df)}, Val 이미지 수: {len(val_images_df)}")
    print(f"[split] Train 어노테이션 수: {len(train_ann_df)}, Val 어노테이션 수: {len(val_ann_df)}")
    print(f"[split] rep_category 1장짜리 클래스 수: {len(class_counts[class_counts == 1])} (전부 train에 포함)")

    return train_images_df, val_images_df, train_ann_df, val_ann_df


def setup_yolo_dirs(root: str):
    images_train_dir = Path(root) / "images" / "train"
    images_val_dir = Path(root) / "images" / "val"
    labels_train_dir = Path(root) / "labels" / "train"
    labels_val_dir = Path(root) / "labels" / "val"

    for d in [images_train_dir, images_val_dir, labels_train_dir, labels_val_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"[yolo_export] YOLO 디렉터리 생성 완료: {root}")
    return images_train_dir, images_val_dir, labels_train_dir, labels_val_dir


def coco_to_yolo_bbox(bbox, img_w, img_h):
    """
    COCO [x, y, w, h] -> YOLO [x_center, y_center, w, h], 0~1 정규화.
    """
    x, y, w, h = bbox
    x_c = x + w / 2.0
    y_c = y + h / 2.0
    return [x_c / img_w, y_c / img_h, w / img_w, h / img_h]


def export_split(
    split_images_df: pd.DataFrame,
    split_ann_df: pd.DataFrame,
    img_src_root: str,
    img_dst_dir: Path,
    label_dst_dir: Path,
    catid_to_yoloid: dict
):
    """
    단일 split(train 또는 val)에 대해:
    - 이미지를 img_dst_dir로 복사
    - YOLO txt 라벨을 label_dst_dir에 생성
    """
    ann_group = (
        split_ann_df
        .groupby("image_id")
        .apply(lambda df: df.to_dict(orient="records"))
        .to_dict()
    )

    for _, row in split_images_df.iterrows():
        img_id = row["id"]
        file_name = row["file_name"]
        img_w = row["width"]
        img_h = row["height"]

        src_path = Path(img_src_root) / file_name
        dst_img_path = img_dst_dir / file_name

        if not dst_img_path.exists():
            try:
                shutil.copy2(src_path, dst_img_path)
            except Exception as e:
                print(f"[yolo_export] 이미지 복사 실패: {src_path} - {e}")
                continue

        ann_list = ann_group.get(img_id, [])
        label_path = label_dst_dir / (Path(file_name).stem + ".txt")

        with open(label_path, "w", encoding="utf-8") as f:
            for ann in ann_list:
                cat_id = ann["category_id"]
                if cat_id not in catid_to_yoloid:
                    continue
                yolo_cls = catid_to_yoloid[cat_id]
                x_c, y_c, w, h = coco_to_yolo_bbox(ann["bbox"], img_w, img_h)
                f.write(f"{yolo_cls} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")


def write_data_yaml(yolo_root: str, categories_df: pd.DataFrame, unique_cat_ids: list):
    data_yaml_path = Path(yolo_root) / "pills.yaml"
    names = []
    for cid in unique_cat_ids:
        name = categories_df[categories_df["id"] == cid]["name"].iloc[0]
        names.append(name)

    yaml_text = f"""
path: {yolo_root}
train: images/train
val: images/val

nc: {len(unique_cat_ids)}
names: {names}
"""

    with open(data_yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_text.strip() + "\n")

    print(f"[yolo_export] data.yaml 생성 완료: {data_yaml_path}")


def main():
    # 1) COCO 테이블 로딩 + 정합성 필터링 (232 이미지 / 763 어노테이션)
    images_df, annotations_df, categories_df = load_coco_tables_with_consistency()

    # 2) 이미지 레벨 DF 생성 후 stratified 8:2 split (방법 1)
    img_level_df = build_image_level_df(images_df, annotations_df)
    train_images_df, val_images_df, train_ann_df, val_ann_df = stratified_train_val_split(
        img_level_df,
        annotations_df,
        val_ratio=VAL_RATIO,
        seed=SPLIT_SEED,
    )

    # 3) YOLO 디렉터리 생성
    images_train_dir, images_val_dir, labels_train_dir, labels_val_dir = setup_yolo_dirs(YOLO_ROOT)

    # 4) category id → 0..nc-1 매핑
    unique_cat_ids = sorted(categories_df["id"].unique().tolist())
    catid_to_yoloid = {cid: idx for idx, cid in enumerate(unique_cat_ids)}
    print(f"[yolo_export] 클래스 수: {len(unique_cat_ids)}")

    # 5) Train/Val 변환
    export_split(train_images_df, train_ann_df, TRAIN_IMG_DIR, images_train_dir, labels_train_dir, catid_to_yoloid)
    export_split(val_images_df, val_ann_df, TRAIN_IMG_DIR, images_val_dir, labels_val_dir, catid_to_yoloid)

    # 6) data.yaml 생성
    write_data_yaml(YOLO_ROOT, categories_df, unique_cat_ids)

    print("[yolo_export] YOLOv8용 데이터셋 변환 완료")


if __name__ == "__main__":
    main()