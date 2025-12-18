"""
Stage 2: Pill Classifier용 YOLO Classification 데이터셋 생성
- 74개 클래스 분류
- 크롭된 이미지 사용 (이미지 전체 = 알약 1개)
- YOLO Classification 폴더 구조로 변환
"""

import os
import glob
import json
import shutil
from pathlib import Path
from collections import defaultdict

from sklearn.model_selection import train_test_split

from .config import CROPPED_IMG_DIR, CROPPED_ANN_DIR, VAL_RATIO, SPLIT_SEED

# Classification 데이터셋 경로
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
YOLO_CLS_ROOT = PROJECT_ROOT / "data" / "yolo_cls"


def load_cropped_data():
    """크롭된 데이터 로드"""
    json_files = glob.glob(os.path.join(CROPPED_ANN_DIR, "*.json"))
    print(f"[Classifier] JSON 파일 수: {len(json_files)}")

    data_list = []

    for json_path in json_files:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 이미지 정보
            img_info = data.get("images", [{}])[0]
            fn = img_info.get("file_name", "")

            # 어노테이션에서 클래스 정보
            ann = data.get("annotations", [{}])[0]
            k_code = ann.get("k_code", "")
            category_id = ann.get("category_id")

            if fn and k_code:
                img_path = Path(CROPPED_IMG_DIR) / fn
                if img_path.exists():
                    data_list.append({
                        "file_name": fn,
                        "img_path": img_path,
                        "k_code": k_code,
                        "category_id": category_id
                    })

        except Exception:
            continue

    print(f"[Classifier] 유효한 이미지 수: {len(data_list)}")

    # 클래스별 통계
    class_counts = defaultdict(int)
    for item in data_list:
        class_counts[item["k_code"]] += 1

    print(f"[Classifier] 클래스 수: {len(class_counts)}")

    return data_list, class_counts


def setup_yolo_cls_dirs():
    """YOLO Classification 디렉토리 생성"""
    # 기존 데이터 삭제
    if YOLO_CLS_ROOT.exists():
        shutil.rmtree(YOLO_CLS_ROOT)

    train_dir = YOLO_CLS_ROOT / "train"
    val_dir = YOLO_CLS_ROOT / "val"

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Classifier] YOLO Classification 디렉토리: {YOLO_CLS_ROOT}")

    return train_dir, val_dir


def export_classification_dataset(data_list, train_dir, val_dir):
    """Classification 데이터셋 생성 (클래스별 폴더 구조)"""

    # Train/Val 분할 (stratified by class)
    k_codes = [item["k_code"] for item in data_list]

    train_indices, val_indices = train_test_split(
        range(len(data_list)),
        test_size=VAL_RATIO,
        random_state=SPLIT_SEED,
        stratify=k_codes
    )

    train_data = [data_list[i] for i in train_indices]
    val_data = [data_list[i] for i in val_indices]

    print(f"[Classifier] Train: {len(train_data)}개")
    print(f"[Classifier] Val: {len(val_data)}개")

    # Train 데이터 복사
    train_class_counts = defaultdict(int)
    for item in train_data:
        k_code = item["k_code"]
        class_dir = train_dir / k_code
        class_dir.mkdir(exist_ok=True)

        dst = class_dir / item["file_name"]
        if not dst.exists():
            shutil.copy2(item["img_path"], dst)
        train_class_counts[k_code] += 1

    # Val 데이터 복사
    val_class_counts = defaultdict(int)
    for item in val_data:
        k_code = item["k_code"]
        class_dir = val_dir / k_code
        class_dir.mkdir(exist_ok=True)

        dst = class_dir / item["file_name"]
        if not dst.exists():
            shutil.copy2(item["img_path"], dst)
        val_class_counts[k_code] += 1

    print(f"[Classifier] Train 클래스 수: {len(train_class_counts)}")
    print(f"[Classifier] Val 클래스 수: {len(val_class_counts)}")

    return train_class_counts, val_class_counts


def write_class_mapping(class_counts):
    """클래스 매핑 파일 생성"""
    # K-code 정렬하여 인덱스 부여
    sorted_k_codes = sorted(class_counts.keys())

    # K-code → YOLO index 매핑
    k_code_to_idx = {k_code: idx for idx, k_code in enumerate(sorted_k_codes)}

    # 매핑 파일 저장
    mapping_path = YOLO_CLS_ROOT / "class_mapping.json"
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump({
            "k_code_to_idx": k_code_to_idx,
            "idx_to_k_code": {v: k for k, v in k_code_to_idx.items()}
        }, f, ensure_ascii=False, indent=2)

    print(f"[Classifier] 클래스 매핑 저장: {mapping_path}")

    return k_code_to_idx


def main():
    print("=" * 60)
    print("Stage 2: Pill Classifier YOLO 데이터셋 생성")
    print("=" * 60)

    # 1. 크롭 데이터 로드
    print("\n[데이터 로드]")
    data_list, class_counts = load_cropped_data()

    if not data_list:
        print("오류: 크롭 데이터가 없습니다.")
        return

    # 2. 디렉토리 생성
    print("\n[디렉토리 생성]")
    train_dir, val_dir = setup_yolo_cls_dirs()

    # 3. 데이터셋 생성
    print("\n[데이터셋 생성]")
    train_counts, val_counts = export_classification_dataset(data_list, train_dir, val_dir)

    # 4. 클래스 매핑 저장
    print("\n[클래스 매핑]")
    k_code_to_idx = write_class_mapping(class_counts)

    # 결과 출력
    print("\n" + "=" * 60)
    print("결과")
    print("=" * 60)
    print(f"총 클래스: {len(class_counts)}개")
    print(f"총 이미지: {len(data_list)}개")
    print(f"Train: {sum(train_counts.values())}개")
    print(f"Val: {sum(val_counts.values())}개")
    print(f"\n출력 경로: {YOLO_CLS_ROOT}")
    print("\n다음 단계: Pill Classifier 학습")
    print("=" * 60)


if __name__ == "__main__":
    main()
