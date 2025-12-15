"""
AIHub Single 데이터 통합을 위한 설정
"""

# 56개 TARGET 클래스 (dl_idx 기준)
TARGET_CLASSES = {
    '1899', '2482', '3350', '3482', '3543', '3742', '3831', '4542',
    '12080', '12246', '12777', '13394', '13899', '16231', '16261', '16547',
    '16550', '16687', '18146', '18356', '19231', '19551', '19606', '19860',
    '20013', '20237', '20876', '21324', '21770', '22073', '22346', '22361',
    '24849', '25366', '25437', '25468', '27732', '27776', '27925', '27992',
    '28762', '29344', '29450', '29666', '30307', '31862', '31884', '32309',
    '33008', '33207', '33879', '34596', '35205', '36636', '38161', '41767'
}

# dl_idx → K-코드 매핑 (dl_idx + 1 = K-코드 숫자)
# 예: dl_idx 2482 → K-002483
def dl_idx_to_k_code(dl_idx: str) -> str:
    return f"K-{int(dl_idx) + 1:06d}"

# 경로 설정
AIHUB_SINGLE_DIR = "data/aihub_single"  # 다운로드된 single 데이터
TRAIN_IMG_DIR = "data/train_images"
TRAIN_ANN_DIR = "data/train_annotations"

# 클래스당 최대 샘플 수
MAX_PER_CLASS = 100

# AIHub 다운로드 경로 (D: 드라이브)
AIHUB_BASE_DIR = "D:/166.약품식별 인공지능 개발을 위한 경구약제 이미지 데이터/01.데이터/1.Training"
AIHUB_LABEL_DIR = f"{AIHUB_BASE_DIR}/라벨링데이터/단일경구약제 5000종"
AIHUB_IMAGE_DIR = f"{AIHUB_BASE_DIR}/원천데이터/단일경구약제 5000종"

# 출력 경로 (AIHub 데이터 별도 폴더)
OUTPUT_IMAGE_DIR = "data/aihub_single/images"
OUTPUT_ANN_DIR = "data/aihub_single/annotations"
