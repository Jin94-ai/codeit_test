# add_filtered_zip_mover.py
# /data 폴더 내에 AI HUB에서 한번에 TS(TL) 1, 3, 4 데이터를 선택하여 다운로드하고 해당 모듈을 실행한다.
# 압축 해제를 진행하지 않고 /data 폴더 내에 '166.약품식별 인공지능 개발을 위한 경구약제 이미지 데이터' 폴더를 둔다.

# 예상 실행 결과
# ✓ JSON 이동 완료: 14735개
# ✓ 이미지 이동 완료: 4912개

import zipfile
import shutil
from pathlib import Path
import re

from .config import PROJECT_ROOT, BASE_DIR, TRAIN_IMG_DIR, TRAIN_ANN_DIR

# ==============================
# 설정
# ==============================
LABEL_ZIP_ROOT = BASE_DIR / "166.약품식별 인공지능 개발을 위한 경구약제 이미지 데이터/01.데이터/1.Training/라벨링데이터/경구약제조합 5000종"
IMAGE_ZIP_ROOT = BASE_DIR / "166.약품식별 인공지능 개발을 위한 경구약제 이미지 데이터/01.데이터/1.Training/원천데이터/경구약제조합 5000종"

ADDED_ANN_DIR = Path(TRAIN_ANN_DIR) / "added_data"
Path(TRAIN_IMG_DIR).mkdir(parents=True, exist_ok=True)
ADDED_ANN_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_IMG_CATEGORY_LIST = {
    1900, 2483, 3351, 3483, 3544, 3743, 3832, 4378, 4543, 5094, 5886, 6192,
    6563, 10221, 10224, 12081, 12247, 12420, 12778, 13395, 13900, 16232,
    16262, 16548, 16551, 16688, 18110, 18147, 18357, 19232, 19552, 19607,
    19861, 20014, 20238, 20877, 21026, 21325, 21771, 22074, 22347, 22362,
    22627, 23203, 23223, 24850, 25367, 25438, 25469, 27653, 27733, 27777,
    27926, 27993, 28763, 29345, 29451, 29667, 29871, 30308, 31705, 31863,
    31885, 32310, 33009, 33208, 33878, 33880, 34597, 35206, 36637, 38162,
    41768, 44199
}
# 노트북에서 추출한 dl_idx 값 + 1 한 리스트(파일명은 dl_idx 보다 값이 1 높음)

# 파일명에서 K-000250-000573-002483-006192 추출
FILENAME_PATTERN = re.compile(r"K-(\d+)-(\d+)-(\d+)-(\d+)")

json_count = 0
img_count = 0

# ==============================
# JSON ZIP 처리
# ==============================
for zip_path in LABEL_ZIP_ROOT.glob("*.zip"):
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            if not member.endswith(".json"):
                continue

            filename = Path(member).name
            match = FILENAME_PATTERN.search(filename)
            if not match:
                continue

            dl_ids = {int(x) for x in match.groups()}
            if not dl_ids & TRAIN_IMG_CATEGORY_LIST:
                continue

            target_path = ADDED_ANN_DIR / filename
            with zf.open(member) as src, open(target_path, "wb") as dst:
                shutil.copyfileobj(src, dst)

            json_count += 1

# ==============================
# IMAGE ZIP 처리
# ==============================
for zip_path in IMAGE_ZIP_ROOT.glob("*.zip"):
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            if not member.endswith(".png"):
                continue

            filename = Path(member).name
            match = FILENAME_PATTERN.search(filename)
            if not match:
                continue

            dl_ids = {int(x) for x in match.groups()}
            if not dl_ids & TRAIN_IMG_CATEGORY_LIST:
                continue

            target_path = Path(TRAIN_IMG_DIR) / filename
            with zf.open(member) as src, open(target_path, "wb") as dst:
                shutil.copyfileobj(src, dst)

            img_count += 1

# ==============================
# 결과 출력
# ==============================
print(f"✓ JSON 이동 완료: {json_count}개")
print(f"✓ 이미지 이동 완료: {img_count}개")