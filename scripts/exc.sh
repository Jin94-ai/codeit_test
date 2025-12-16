#!/bin/bash

# 현재 실행 위치 = 프로젝트 루트
PROJECT_ROOT="$(pwd)"
SCRIPT_DIR="$PROJECT_ROOT/scripts"

# run.sh 생성
cat << 'EOF' > "$SCRIPT_DIR/run.sh"
#!/bin/bash

PY=python3
export KAGGLE_CONFIG_DIR=./data/.kaggle

# 필요한 패키지 설치
$PY -m pip install gdown
$PY -m pip install --upgrade pip
$PY -m pip install albumentations
$PY -m pip install ultralytics==8.3.235
$PY -m pip install kaggle==1.7.4.5
$PY -m pip install matplotlib
$PY -m pip install seaborn

# 기본 실행 모델
MODEL_FILE=${1:-baseline.py}

# 상대경로 실행 (프로젝트 루트 기준)
$PY -m src.data.data_load.data_loader
$PY -m src.data.yolo_dataset.yolo_export
$PY -m src.models.$(basename $MODEL_FILE .py)
EOF


chmod +x "$SCRIPT_DIR/run.sh"

# 기존 alias 제거
sed -i '/alias exc_pip=/d' ~/.bashrc

# alias 등록 (항상 프로젝트 루트에서 실행하는 전제)
echo 'alias exc_pip="bash ./scripts/run.sh \"\$@\""' >> ~/.bashrc

# 적용
source ~/.bashrc

echo "exc_pip 명령어가 등록되었습니다."
