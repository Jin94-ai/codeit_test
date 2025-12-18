"""
Stage 1: 단일 클래스 Pill Detector용 YOLO 데이터셋 생성
- 모든 알약을 'Pill' 단일 클래스로 통합
- 원본 이미지 + 원본 bbox 사용
"""

import os
import glob
import json
import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    TRAIN_IMG_DIR, TRAIN_ANN_DIR,
    AIHUB_DETECTOR_IMG_DIR, AIHUB_DETECTOR_ANN_DIR,
    YOLO_ROOT, VAL_RATIO, SPLIT_SEED
)


def load_kaggle_annotations():
    """Kaggle annotation JSON 파일들을 로드"""
    json_files = glob.glob(os.path.join(TRAIN_ANN_DIR, "**/*.json"), recursive=True)
    print(f"[Kaggle] JSON 파일 수: {len(json_files)}")

    all_images = []
    all_annotations = []

    for json_path in json_files:
        if os.path.basename(json_path).startswith("._"):
            continue

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        # images 섹션
        for img_info in data.get("images", []):
            fn = img_info.get("file_name")
            if fn:
                all_images.append({
                    "id": img_info.get("id"),
                    "file_name": fn,
                    "width": img_info.get("width"),
                    "height": img_info.get("height"),
                    "source": "kaggle"
                })

        # annotations 섹션 - 모든 bbox를 Pill(class 0)로 통합
        for ann in data.get("annotations", []):
            bbox = ann.get("bbox")
            if isinstance(bbox, list) and len(bbox) == 4:
                all_annotations.append({
                    "id": ann.get("id"),
                    "image_id": ann.get("image_id"),
                    "bbox": bbox,
                    "category_id": 0  # 단일 클래스: Pill
                })

    images_df = pd.DataFrame(all_images).drop_duplicates(subset=["file_name"]) if all_images else pd.DataFrame()
    annotations_df = pd.DataFrame(all_annotations) if all_annotations else pd.DataFrame()

    print(f"[Kaggle] 이미지 수: {len(images_df)}")
    print(f"[Kaggle] 어노테이션 수: {len(annotations_df)}")

    return images_df, annotations_df


def load_aihub_annotations():
    """AIHub detector annotation JSON 파일들을 로드"""
    if not os.path.exists(AIHUB_DETECTOR_ANN_DIR):
        print(f"[AIHub] 디렉토리 없음: {AIHUB_DETECTOR_ANN_DIR}")
        return pd.DataFrame(), pd.DataFrame()

    json_files = glob.glob(os.path.join(AIHUB_DETECTOR_ANN_DIR, "*.json"))
    print(f"[AIHub] JSON 파일 수: {len(json_files)}")

    all_images = []
    all_annotations = []

    # AIHub ID offset (Kaggle과 충돌 방지)
    id_offset = 1000000

    for json_path in json_files:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        # images 섹션
        for img_info in data.get("images", []):
            fn = img_info.get("file_name")
            if fn:
                all_images.append({
                    "id": img_info.get("id") + id_offset,
                    "file_name": fn,
                    "width": img_info.get("width"),
                    "height": img_info.get("height"),
                    "source": "aihub"
                })

        # annotations 섹션
        for ann in data.get("annotations", []):
            bbox = ann.get("bbox")
            if isinstance(bbox, list) and len(bbox) == 4:
                all_annotations.append({
                    "id": ann.get("id") + id_offset,
                    "image_id": ann.get("image_id") + id_offset,
                    "bbox": bbox,
                    "category_id": 0  # 단일 클래스: Pill
                })

    images_df = pd.DataFrame(all_images).drop_duplicates(subset=["file_name"]) if all_images else pd.DataFrame()
    annotations_df = pd.DataFrame(all_annotations) if all_annotations else pd.DataFrame()

    print(f"[AIHub] 이미지 수: {len(images_df)}")
    print(f"[AIHub] 어노테이션 수: {len(annotations_df)}")

    return images_df, annotations_df


def filter_existing_images(images_df, annotations_df):
    """실제 존재하는 이미지만 필터링 (source별로 다른 디렉토리 확인)"""
    if images_df.empty:
        return images_df, annotations_df

    # Kaggle 이미지 확인
    kaggle_existing = set(os.path.basename(p) for p in glob.glob(os.path.join(TRAIN_IMG_DIR, "*.png")))
    print(f"[Kaggle] 폴더 내 이미지 수: {len(kaggle_existing)}")

    # AIHub 이미지 확인
    aihub_existing = set()
    if os.path.exists(AIHUB_DETECTOR_IMG_DIR):
        aihub_existing = set(os.path.basename(p) for p in glob.glob(os.path.join(AIHUB_DETECTOR_IMG_DIR, "*.png")))
    print(f"[AIHub] 폴더 내 이미지 수: {len(aihub_existing)}")

    # source별로 필터링
    valid_mask = []
    for _, row in images_df.iterrows():
        fn = row["file_name"]
        source = row.get("source", "kaggle")
        if source == "kaggle":
            valid_mask.append(fn in kaggle_existing)
        else:
            valid_mask.append(fn in aihub_existing)

    images_df = images_df[valid_mask]

    valid_ids = set(images_df["id"])
    annotations_df = annotations_df[annotations_df["image_id"].isin(valid_ids)]

    print(f"[전체] 필터링 후 이미지 수: {len(images_df)}")
    print(f"[전체] 필터링 후 어노테이션 수: {len(annotations_df)}")

    return images_df, annotations_df


def train_val_split(images_df, annotations_df):
    """Train/Val 분할"""
    train_ids, val_ids = train_test_split(
        images_df["id"].values,
        test_size=VAL_RATIO,
        random_state=SPLIT_SEED
    )

    train_ids = set(train_ids)
    val_ids = set(val_ids)

    train_images = images_df[images_df["id"].isin(train_ids)]
    val_images = images_df[images_df["id"].isin(val_ids)]
    train_anns = annotations_df[annotations_df["image_id"].isin(train_ids)]
    val_anns = annotations_df[annotations_df["image_id"].isin(val_ids)]

    print(f"[detector] Train: {len(train_images)} 이미지, {len(train_anns)} bbox")
    print(f"[detector] Val: {len(val_images)} 이미지, {len(val_anns)} bbox")

    return train_images, val_images, train_anns, val_anns


def setup_yolo_dirs():
    """YOLO 디렉토리 생성"""
    yolo_root = Path(YOLO_ROOT)

    # 기존 데이터 삭제
    if yolo_root.exists():
        shutil.rmtree(yolo_root)

    dirs = {
        "images_train": yolo_root / "images" / "train",
        "images_val": yolo_root / "images" / "val",
        "labels_train": yolo_root / "labels" / "train",
        "labels_val": yolo_root / "labels" / "val",
    }

    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    print(f"[detector] YOLO 디렉토리 생성: {yolo_root}")
    return dirs


def coco_to_yolo_bbox(bbox, img_w, img_h):
    """COCO [x, y, w, h] → YOLO [x_center, y_center, w, h] (0~1 정규화)"""
    x, y, w, h = bbox
    x_c = (x + w / 2.0) / img_w
    y_c = (y + h / 2.0) / img_h
    w_n = w / img_w
    h_n = h / img_h
    return x_c, y_c, w_n, h_n


def export_split(images_df, annotations_df, img_dst_dir, label_dst_dir):
    """이미지 복사 + YOLO 라벨 생성 (source별로 다른 디렉토리에서)"""
    ann_by_image = annotations_df.groupby("image_id").apply(
        lambda df: df.to_dict(orient="records")
    ).to_dict()

    copied = 0
    for _, row in images_df.iterrows():
        img_id = row["id"]
        fn = row["file_name"]
        img_w = row["width"]
        img_h = row["height"]
        source = row.get("source", "kaggle")

        # source별 이미지 경로
        if source == "kaggle":
            src = Path(TRAIN_IMG_DIR) / fn
        else:
            src = Path(AIHUB_DETECTOR_IMG_DIR) / fn

        dst = img_dst_dir / fn

        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            copied += 1

        # YOLO 라벨 생성
        anns = ann_by_image.get(img_id, [])
        label_path = label_dst_dir / (Path(fn).stem + ".txt")

        with open(label_path, "w", encoding="utf-8") as f:
            for ann in anns:
                x_c, y_c, w_n, h_n = coco_to_yolo_bbox(ann["bbox"], img_w, img_h)
                f.write(f"0 {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}\n")

    print(f"[detector] 복사된 이미지: {copied}")


def write_yaml():
    """pills.yaml 생성 (단일 클래스)"""
    yaml_path = Path(YOLO_ROOT) / "pills.yaml"
    yaml_content = f"""path: {YOLO_ROOT}
train: images/train
val: images/val

nc: 1
names: ['Pill']
"""
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"[detector] YAML 생성: {yaml_path}")


def main():
    print("=" * 60)
    print("Stage 1: Pill Detector YOLO 데이터셋 생성")
    print("=" * 60)

    # 1. Kaggle 데이터 로드
    print("\n[Kaggle 데이터 로드]")
    kaggle_images, kaggle_anns = load_kaggle_annotations()

    # 2. AIHub 데이터 로드
    print("\n[AIHub 데이터 로드]")
    aihub_images, aihub_anns = load_aihub_annotations()

    # 3. 데이터 결합
    print("\n[데이터 결합]")
    dfs_to_concat_img = [df for df in [kaggle_images, aihub_images] if not df.empty]
    dfs_to_concat_ann = [df for df in [kaggle_anns, aihub_anns] if not df.empty]

    if dfs_to_concat_img:
        images_df = pd.concat(dfs_to_concat_img, ignore_index=True)
    else:
        images_df = pd.DataFrame()

    if dfs_to_concat_ann:
        annotations_df = pd.concat(dfs_to_concat_ann, ignore_index=True)
    else:
        annotations_df = pd.DataFrame()

    print(f"결합 후 총 이미지: {len(images_df)}")
    print(f"결합 후 총 어노테이션: {len(annotations_df)}")

    # 4. 존재하는 이미지만 필터링
    print("\n[이미지 필터링]")
    images_df, annotations_df = filter_existing_images(images_df, annotations_df)

    # 5. Train/Val 분할
    train_images, val_images, train_anns, val_anns = train_val_split(images_df, annotations_df)

    # 6. YOLO 디렉토리 생성
    dirs = setup_yolo_dirs()

    # 7. Export
    print("\n[Train 데이터 export]")
    export_split(train_images, train_anns, dirs["images_train"], dirs["labels_train"])

    print("\n[Val 데이터 export]")
    export_split(val_images, val_anns, dirs["images_val"], dirs["labels_val"])

    # 8. YAML 생성
    write_yaml()

    print("\n" + "=" * 60)
    print("완료! 다음 단계: Pill Detector 학습")
    print("=" * 60)


if __name__ == "__main__":
    main()
