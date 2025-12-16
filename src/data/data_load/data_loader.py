import sys
import subprocess
import os
import json

# ---------------------------------------------------------
# 1. 경로 설정
# ---------------------------------------------------------
# 현재 dir에서 .. 이동 후 data 폴더 지정
project_root = os.getcwd()
target_dir = os.path.join(project_root, "data")


os.makedirs(target_dir, exist_ok=True)

# kaggle.json 위치
kaggle_dir = os.path.join(target_dir, ".kaggle")
os.makedirs(kaggle_dir, exist_ok=True)

kaggle_json_path = os.path.join(kaggle_dir, "kaggle.json")

# ---------------------------------------------------------
# 2. Kaggle username, key 입력받기
# ---------------------------------------------------------


username = input("Kaggle username을 입력하세요: ").strip()
key = input("Kaggle API key를 입력하세요: ").strip()


# ---------------------------------------------------------
# 3. kaggle.json 생성
# ---------------------------------------------------------
kaggle_dict = {
    "username": username,
    "key": key
}

with open(kaggle_json_path, "w") as f:
    json.dump(kaggle_dict, f)


# ---------------------------------------------------------
# 4. 권한 설정 (chmod 600)
# ---------------------------------------------------------
subprocess.run(["chmod", "600", kaggle_json_path])


# ---------------------------------------------------------
# 5. Kaggle API 버전 확인
# ---------------------------------------------------------
subprocess.run(["kaggle", "--version"])


# ---------------------------------------------------------
# 6. Kaggle 데이터 다운로드 (이미 있으면 스킵)
# ---------------------------------------------------------
zip_path = os.path.join(target_dir, "ai06-level1-project.zip")

if not os.path.exists(zip_path):
    print("[INFO] ZIP 파일이 없어서 Kaggle에서 다운로드합니다.")
    subprocess.run([
        "kaggle", "competitions", "download",
        "-c", "ai06-level1-project",
        "-p", target_dir
    ])
else:
    print("[INFO] ZIP 파일이 이미 존재하여 다운로드를 생략합니다.")


# ---------------------------------------------------------
# 7. 압축 해제 (이미 폴더가 있으면 스킵)
# ---------------------------------------------------------
extract_marker = os.path.join(target_dir, "train_images")  # train_images 폴더 존재 여부 확인

if not os.path.exists(extract_marker):
    print("[INFO] 압축 파일을 해제합니다.")
    subprocess.run(["unzip", "-o", zip_path, "-d", target_dir])
else:
    print("[INFO] 데이터가 이미 압축 해제되어 있어 스킵합니다.")



# ---------------------------------------------------------
# 8. 추가 데이터
# ---------------------------------------------------------

data_path = os.path.join(target_dir, "aihub_single")

if not os.path.exists(data_path):
    print("[INFO] 추가 데이터 파일이 없어서 Gdrive에서 다운로드합니다.")
    subprocess.run([
        "gdown",
        "--folder",
        "1wfrysUZwosthpMUCgi2ECvvCOjn4ziXh",
        "-O",
        target_dir
    ])

else:
    print("[INFO] 추가 데이터 파일이 이미 존재하여 생략합니다.")