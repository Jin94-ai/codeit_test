import shutil
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import numpy as np 

from .config import TRAIN_IMG_DIR, YOLO_ROOT, VAL_RATIO, SPLIT_SEED
from .coco_parser import load_coco_tables_with_consistency


def build_image_level_df(images_df: pd.DataFrame, annotations_df: pd.DataFrame) -> pd.DataFrame:
    """
    이미지 단위 DataFrame 생성 + 각 이미지당 대표 category_id(rep_category) 부여.
    로딩된 이미지/어노테이션 DF가 비어있으면 빈 DF 반환.
    """
    if images_df.empty or annotations_df.empty:
        print("[build_image_level_df] 경고: images_df 또는 annotations_df가 비어 있어 이미지 레벨 DF를 생성할 수 없습니다.")
        return pd.DataFrame(columns=['id', 'file_name', 'width', 'height', 'rep_category']) # 예상 컬럼으로 빈 DF 반환

    first_cat_per_image = (
        annotations_df
        .sort_values("id")
        .groupby("image_id")["category_id"]
        .first()
        .rename("rep_category")
        .reset_index()
    )
    
    # first_cat_per_image가 비어있으면 매칭될게 없으니 빈 DF 반환
    if first_cat_per_image.empty:
        print("[build_image_level_df] 경고: annotations_df에서 대표 카테고리를 추출할 수 없어 이미지 레벨 DF를 생성할 수 없습니다.")
        return pd.DataFrame(columns=['id', 'file_name', 'width', 'height', 'rep_category'])

    # annotations_df의 image_id와 images_df의 id를 inner join
    img_level_df = images_df.merge(
        first_cat_per_image,
        left_on="id",
        right_on="image_id",
        how="inner"
    ).drop(columns=["image_id"]) # 병합에 사용된 image_id 컬럼은 제거

    return img_level_df

def stratified_train_val_split(
    img_level_df: pd.DataFrame,
    annotations_df: pd.DataFrame,
    val_ratio: float,
    seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    대표 class(rep_category)를 기준으로 stratified 8:2 split.
    - 단, 이미지가 1장뿐인 클래스는 stratify에서 제외하고 전부 train에 넣음
    """
    if img_level_df.empty:
        print("[split] 경고: img_level_df가 비어 있어 훈련/검증 분할을 수행할 수 없습니다. 빈 데이터프레임을 반환합니다.")
        empty_df = pd.DataFrame()
        return empty_df, empty_df, empty_df, empty_df

    # 디버그: img_level_df의 크기 확인
    print(f"[split-DEBUG] img_level_df 총 행 수: {len(img_level_df)}")
    print(f"[split-DEBUG] img_level_df rep_category 고유값: {img_level_df['rep_category'].nunique()}")

    # 클래스별 이미지 수
    class_counts = img_level_df["rep_category"].value_counts()
    print(f"[split-DEBUG] rep_category 별 카운트:\n{class_counts.to_string()}") # to_string()으로 모든 행 출력

    # 2장 이상 있는 클래스만 stratify 대상으로 사용
    valid_classes = class_counts[class_counts >= 2].index.tolist()
    print(f"[split-DEBUG] 2개 이상 샘플을 가진 유효 클래스 (stratify 대상): {valid_classes}")

    mask_valid = img_level_df["rep_category"].isin(valid_classes)
    mask_single = ~mask_valid  # 1장짜리 클래스들 또는 2개 미만 클래스들 (stratify 불가)

    X_valid = img_level_df.loc[mask_valid, "id"].values
    y_valid = img_level_df.loc[mask_valid, "rep_category"].values

    # 디버그: X_valid와 y_valid의 크기 확인
    print(f"[split-DEBUG] X_valid 행 수 (stratify 대상): {len(X_valid)}")
    print(f"[split-DEBUG] y_valid 행 수 (stratify 대상): {len(y_valid)}")
    
    train_ids = set()
    val_ids = set()

    if len(X_valid) > 0:
        print(f"[split-DEBUG] train_test_split 호출 시작 (X_valid_len={len(X_valid)})")
        train_ids_valid, val_ids_valid = train_test_split(
            X_valid,
            test_size=val_ratio,
            random_state=seed,
            stratify=y_valid
        )
        train_ids = set(train_ids_valid.tolist())
        val_ids = set(val_ids_valid.tolist())
        print(f"[split-DEBUG] train_test_split 호출 완료. train_ids_valid_len={len(train_ids_valid)}, val_ids_valid_len={len(val_ids_valid)}")
    else:
        print("[split-DEBUG] X_valid가 비어있으므로 train_test_split를 건너뜁니다.")
        print("[split] 경고: stratify 할 수 있는 클래스가 충분하지 않습니다 (모든 클래스가 샘플 2개 미만). 모든 이미지를 훈련 세트에 할당합니다.")
        train_ids = set(img_level_df["id"].values.tolist())
        val_ids = set()

    # 1장짜리 또는 2개 미만 클래스들은 모두 train에 보냄
    single_class_ids = set(img_level_df.loc[mask_single, "id"].values)
    train_ids = train_ids.union(single_class_ids)

    # 최종 훈련/검증 이미지 및 어노테이션 데이터프레임 생성
    train_images_df = img_level_df[img_level_df["id"].isin(train_ids)].copy()
    val_images_df = img_level_df[img_level_df["id"].isin(val_ids)].copy()

    # 원본 annotations_df에서 분할된 이미지 ID에 해당하는 어노테이션만 가져옴
    train_ann_df = annotations_df[annotations_df["image_id"].isin(train_ids)].copy()
    val_ann_df = annotations_df[annotations_df["image_id"].isin(val_ids)].copy()

    print(f"[split] Train 이미지 수: {len(train_images_df)}, Val 이미지 수: {len(val_images_df)}")
    print(f"[split] Train 어노테이션 수: {len(train_ann_df)}, Val 어노테이션 수: {len(val_ann_df)}")
    print(f"[split] rep_category 1장짜리 클래스 수 (또는 2개 미만 클래스): {len(class_counts[class_counts < 2])} (전부 train에 포함)")

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
    # 바운딩 박스 값이 음수이거나 잘못된 경우를 대비한 유효성 검사
    if w <= 0 or h <= 0 or img_w <= 0 or img_h <= 0:
        return [0, 0, 0, 0] # 유효하지 않은 경우 0으로 반환

    x_c = x + w / 2.0
    y_c = y + h / 2.0
    
    # 0~1 범위 벗어나는 값 보정 (YOLO 포맷의 요구사항)
    x_c = np.clip(x_c / img_w, 0.0, 1.0)
    y_c = np.clip(y_c / img_h, 0.0, 1.0)
    w_norm = np.clip(w / img_w, 0.0, 1.0)
    h_norm = np.clip(h / img_h, 0.0, 1.0)
    
    return [x_c, y_c, w_norm, h_norm]


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
    if split_images_df.empty:
        print(f"[yolo_export] 경고: {img_dst_dir.name} 스플릿의 이미지 데이터가 비어 있습니다. 내보내기 작업을 건너뜁니다.")
        return

    # annotations_df가 비어있을 경우 groupby에서 에러가 날 수 있으므로 조건부 처리
    ann_group = {}
    if not split_ann_df.empty:
        # FutureWarning 해결: include_groups=False 추가
        ann_group = (
            split_ann_df
            .groupby("image_id")
            .apply(lambda df: df.to_dict(orient="records"), include_groups=False) 
            .to_dict()
        )

    for idx, row in split_images_df.iterrows(): # .iterrows()를 사용하여 각 이미지에 대한 처리
        img_id = row["id"]
        file_name = row["file_name"] # file_name이 float 또는 NaN일 수 있으므로 str로 캐스팅 필요
        img_w = row["width"]
        img_h = row["height"]

        file_name_str = str(file_name) 

        src_path = Path(img_src_root) / file_name_str
        dst_img_path = img_dst_dir / file_name_str

        # 이미지 복사 (이미 대상 경로에 존재하면 건너뜀)
        if not dst_img_path.exists():
            if src_path.exists():
                try:
                    shutil.copy2(src_path, dst_img_path)
                except Exception as e:
                    print(f"[yolo_export] 이미지 복사 실패: {src_path} -> {dst_img_path} - {e}")
                    continue
            else:
                # print(f"[yolo_export] 원본 이미지 파일을 찾을 수 없습니다. 스킵: {src_path}") # 너무 많은 로그 출력 방지
                continue # 이미지가 없으면 해당 이미지는 처리하지 않고 넘어감

        ann_list = ann_group.get(img_id, [])
        label_path = label_dst_dir / (Path(file_name_str).stem + ".txt") # file_name_str 사용

        # 어노테이션이 없으면 라벨 파일 생성 불필요
        if not ann_list:
            # print(f"[yolo_export] 이미지 {file_name_str} 에 대한 어노테이션이 없습니다. 라벨 파일을 생성하지 않습니다.") # 너무 많은 로그 출력 방지
            continue

        try:
            with open(label_path, "w", encoding="utf-8") as f:
                for ann in ann_list:
                    cat_id = ann["category_id"]
                    if cat_id not in catid_to_yoloid:
                        print(f"[yolo_export] 경고: 알 수 없는 카테고리 ID {cat_id}를 가진 어노테이션이 발견되었습니다. 스킵: {file_name_str}")
                        continue
                    yolo_cls = catid_to_yoloid[cat_id]
                    # coco_to_yolo_bbox는 이제 이미 정규화된 w_norm, h_norm을 반환
                    x_c, y_c, w_norm, h_norm = coco_to_yolo_bbox(ann["bbox"], img_w, img_h) 

                    # 유효한 바운딩 박스만 출력
                    # coco_to_yolo_bbox에서 [0,0,0,0]으로 반환했다면 유효하지 않음을 의미
                    if w_norm > 0 and h_norm > 0:
                        f.write(f"{yolo_cls} {x_c:.6f} {y_c:.6f} {w_norm:.6f} {h_norm:.6f}\n") 
                    else:
                        print(f"[yolo_export] 경고: 이미지 {file_name_str}의 어노테이션 {ann['id']}에 유효하지 않은 바운딩 박스 ({ann['bbox']})가 포함되어 스킵합니다.")
        except Exception as e:
            print(f"[yolo_export] 라벨 파일 생성 실패: {label_path} - {e}")


def write_data_yaml(yolo_root: str, categories_df: pd.DataFrame, unique_cat_ids: list):
    """
    YOLO 학습을 위한 data.yaml 파일을 생성합니다.
    """
    data_yaml_path = Path(yolo_root) / "pills.yaml"
    names = []
    
    # unique_cat_ids가 비어있거나 categories_df가 비어있으면 data.yaml 생성 불필요
    if not unique_cat_ids or categories_df.empty:
        print("[yolo_export] 경고: 유효한 카테고리 ID 또는 카테고리 정보가 없어 data.yaml을 생성할 수 없습니다.")
        return

    # unique_cat_ids를 기반으로 categories_df에서 이름을 찾아 목록 생성
    if 'id' not in categories_df.columns or 'name' not in categories_df.columns:
        print("[yolo_export] 오류: categories_df에 'id' 또는 'name' 컬럼이 없어 data.yaml을 생성할 수 없습니다.")
        return
        
    category_map = categories_df.set_index('id')['name'].to_dict()

    for cid in unique_cat_ids:
        name = category_map.get(cid, "unknown_category_id") # 존재하지 않는 ID에 대비
        names.append(name)
    
    # YOLO 학습에 필요한 경로들이 실제 존재하는지 확인 후 반영
    # YOLOv8의 data.yaml에서는 path는 데이터셋의 루트 경로를, train/val은 그 path로부터의 상대경로를 지정
    train_relative_path = Path("images") / "train"
    val_relative_path = Path("images") / "val"

    yaml_text = f"""
path: {str(Path(YOLO_ROOT).absolute())}
train: {train_relative_path}
val: {val_relative_path}

nc: {len(unique_cat_ids)}
names: {names}
"""

    try:
        with open(data_yaml_path, "w", encoding="utf-8") as f:
            f.write(yaml_text.strip() + "\n")
        print(f"[yolo_export] data.yaml 생성 완료: {data_yaml_path}")
    except Exception as e:
        print(f"[yolo_export] data.yaml 생성 실패: {e}")


def main():
    # 1) COCO 테이블 로딩 + 정합성 필터링
    print("[디버그] COCO 테이블 로딩 시작...")
    images_df, annotations_df, categories_df = load_coco_tables_with_consistency()
    
    print(f"[디버그] load_coco_tables_with_consistency() 결과 - images_df 행: {len(images_df)}")
    print(f"[디버그] load_coco_tables_with_consistency() 결과 - annotations_df 행: {len(annotations_df)}")
    print(f"[디버그] load_coco_tables_with_consistency() 결과 - categories_df 행: {len(categories_df)}")

    # 1.5) 로딩된 데이터의 유효성 검사
    if images_df.empty or annotations_df.empty or categories_df.empty:
        print("[오류] COCO 데이터 로딩 실패. images_df, annotations_df, 또는 categories_df 중 하나 이상이 비어 있습니다. 원본 JSON 파일 및 coco_parser를 확인하세요.")
        return

    # 2) 이미지 레벨 DF 생성 후 stratified 8:2 split
    img_level_df = build_image_level_df(images_df, annotations_df)

    # 2.5) 이미지 레벨 DF 유효성 검사
    if img_level_df.empty:
        print("[오류] 생성된 이미지 레벨 데이터프레임이 비어있습니다. 원본 데이터(이미지 또는 어노테이션)의 매핑 관계를 확인해 주세요.")
        return

    train_images_df, val_images_df, train_ann_df, val_ann_df = stratified_train_val_split(
        img_level_df,
        annotations_df, # 원본 어노테이션 DF를 넘겨줌 (SMOTE 후에도 이걸로 필터링하기 위해)
        val_ratio=VAL_RATIO,
        seed=SPLIT_SEED,
    )
    
    # 2.6) 분할된 훈련 데이터 유효성 검사
    if train_images_df.empty:
        print("[오류] `stratified_train_val_split` 후 훈련 데이터셋이 비어 있습니다. 이후 과정을 진행할 수 없습니다.")
        return

    # 3) 훈련 데이터셋에 SMOTE 적용
    print("\n[SMOTE] 훈련 데이터셋에 SMOTE 적용 시작...")
    
    # SMOTE를 적용하기 위해 이미지 ID와 대표 카테고리 준비
    X_train_original = train_images_df[['id']].values
    y_train_original = train_images_df['rep_category'].values
    
    # 각 클래스별 샘플 수 계산
    class_counts_in_train = train_images_df['rep_category'].value_counts()
    print(f"[SMOTE] 훈련 데이터셋의 클래스별 샘플 수:\n{class_counts_in_train.to_string()}")

    target_k_neighbors = 5 
    
    # SMOTE가 최소한의 이웃을 찾을 수 있는 클래스 (샘플이 2개 이상)
    eligible_for_smote = class_counts_in_train[class_counts_in_train >= 2].index.tolist()
    
    # SMOTE에 입력될 데이터 (클래스별 샘플이 2개 이상인 이미지들만)
    smote_input_df = train_images_df[train_images_df['rep_category'].isin(eligible_for_smote)].copy()
    
    X_smote_input = smote_input_df[['id']].values
    y_smote_input = smote_input_df['rep_category'].values
    
    # SMOTE 적용 후 최종적으로 구성될 이미지 데이터프레임과 어노테이션 데이터프레임을 위한 임시 변수
    final_train_images_df_after_smote = None

    if len(X_smote_input) > 0:
        min_samples_in_smote_input_classes = pd.Series(y_smote_input).value_counts().min()
        
        if min_samples_in_smote_input_classes > 1: # 샘플이 1개 초과인 경우에만 이웃을 찾을 수 있음
            actual_k_neighbors_for_smote = min(target_k_neighbors, int(min_samples_in_smote_input_classes - 1))
        else:
            print(f"[SMOTE] 경고: SMOTE 입력 데이터에 샘플 수가 1개인 클래스가 있어 k_neighbors를 1로 조정합니다.")
            actual_k_neighbors_for_smote = 1 # 최소 k_neighbors는 1로 설정

        print(f"[SMOTE] SMOTE를 {len(eligible_for_smote)}개 클래스에 적용 시도 (조정된 k_neighbors={actual_k_neighbors_for_smote})")
        
        smote_sampler = SMOTE(
            random_state=SPLIT_SEED, 
            sampling_strategy='auto', 
            k_neighbors=actual_k_neighbors_for_smote 
        )
        
        X_resampled, y_resampled = smote_sampler.fit_resample(X_smote_input, y_smote_input)
        
        smote_resampled_ids = X_resampled.flatten()
        smote_resampled_labels = y_resampled
        print(f"[SMOTE] SMOTE 적용 후 대상 클래스 샘플 수: {len(smote_resampled_ids)}")
        
        # SMOTE 증강된 데이터 (복제된 ID 포함)
        smote_augmented_data = pd.DataFrame({'id': smote_resampled_ids, 'rep_category': smote_resampled_labels})

        original_image_metadata = train_images_df[['id', 'file_name', 'width', 'height']].drop_duplicates(subset=['id'])

        # SMOTE 결과 + 이미지 메타데이터
        smote_full_df = smote_augmented_data.merge(original_image_metadata, on='id', how='left')
        
        # SMOTE이 적용되지 않은 원본 데이터 (eligible_for_smote에 포함되지 않았던 이미지들)
        # 즉, train_images_df 에서 smote_input_df에 포함되지 않았던 이미지들 (주로 샘플 1개 클래스)
        original_non_smote_data = train_images_df[~train_images_df['id'].isin(smote_input_df['id'].unique())].copy()

        # 최종 훈련 이미지 DataFrame은 SMOTE으로 증강된 부분 + SMOTE 적용 제외된 원본 데이터
        final_train_images_df_after_smote = pd.concat([smote_full_df, original_non_smote_data], ignore_index=True)
        final_train_images_df_after_smote.index = pd.RangeIndex(len(final_train_images_df_after_smote.index)) # 인덱스 재설정
        
    else: # len(X_smote_input) == 0
        print("[SMOTE] SMOTE를 적용할 대상 클래스 데이터가 없거나 샘플 수가 너무 적어 건너뜁니다. 기존 훈련 데이터 유지.")
        final_train_images_df_after_smote = train_images_df.copy() # 원본 train_images_df를 그대로 사용

    # 최종 train_images_df 갱신
    train_images_df = final_train_images_df_after_smote

    # === SMOTE 적용 후 최종 train_images_df 점검 ===
    # `export_split`에서 'file_name' NaN으로 인한 에러 방지
    if train_images_df['file_name'].isnull().any():
        print("[오류] train_images_df에 'file_name'이 NaN인 행이 존재합니다. 해당 행들을 제거합니다.")
        train_images_df = train_images_df.dropna(subset=['file_name']).copy() # .copy() 추가하여 SettingWithCopyWarning 방지
        # `id`도 float이 될 수 있으므로 int로 변환
        train_images_df['id'] = train_images_df['id'].astype(int)
        print(f"[SMOTE] NaN 'file_name' 제거 후 Train 이미지 샘플 수: {len(train_images_df)}")
    # ======================================================

    # SMOTE 적용 후 최종 train_ann_df 구성
    unique_train_image_ids_after_smote = train_images_df['id'].unique()
    train_ann_df = annotations_df[annotations_df['image_id'].isin(unique_train_image_ids_after_smote)].copy()

    print(f"[SMOTE] SMOTE 최종 적용 후 Train 이미지 샘플 수 (데이터 프레임 행 수): {len(train_images_df)}")
    print(f"[SMOTE] SMOTE 최종 적용 후 Train 어노테이션 샘플 수: {len(train_ann_df)}")
    print("[SMOTE] 훈련 데이터셋에 SMOTE 적용 완료!")

    # 4) YOLO 디렉터리 생성
    images_train_dir, images_val_dir, labels_train_dir, labels_val_dir = setup_yolo_dirs(YOLO_ROOT)

    # 5) category id → 0..nc-1 매핑
    # train_images_df를 사용하기 전, 원본 categories_df를 기준으로 매핑
    unique_cat_ids = sorted(categories_df["id"].unique().tolist())
    catid_to_yoloid = {cid: idx for idx, cid in enumerate(unique_cat_ids)}
    print(f"\n[yolo_export] 클래스 수: {len(unique_cat_ids)}")

    # 6) Train/Val 변환
    export_split(train_images_df, train_ann_df, TRAIN_IMG_DIR, images_train_dir, labels_train_dir, catid_to_yoloid)
    export_split(val_images_df, val_ann_df, TRAIN_IMG_DIR, images_val_dir, labels_val_dir, catid_to_yoloid)

    # 7) data.yaml 생성
    write_data_yaml(YOLO_ROOT, categories_df, unique_cat_ids)

    # 8) YOLO index → 원본 category_id 매핑 저장
    # `baseline.py`에서 {YOLO_idx (int): original_cat_id (int)} 형태로 사용
    yoloid_to_catid_final = {idx: cid for cid, idx in catid_to_yoloid.items()}
    mapping_path = Path(YOLO_ROOT) / "class_mapping.json"
    try:
        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump(yoloid_to_catid_final, f, indent=2, ensure_ascii=False)
        print(f"[yolo_export] 클래스 매핑 저장: {mapping_path}")
    except Exception as e:
        print(f"[yolo_export] 클래스 매핑 저장 실패: {e}")

    print("\n[yolo_export] YOLOv8용 데이터셋 변환 완료!")


if __name__ == "__main__":
    main()