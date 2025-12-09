#!/bin/bash

# 현재 실행 위치 = 프로젝트 루트
PROJECT_ROOT="$(pwd)"
SCRIPT_DIR="$PROJECT_ROOT/scripts"

# run.sh 생성
cat << EOF > "$SCRIPT_DIR/run.sh"
#!/bin/bash

PY=python3
export KAGGLE_CONFIG_DIR=./data/.kaggle

# 필요한 패키지 설치
\$PY -m pip install --upgrade pip
\$PY -m pip install ultralytics==8.3.235
\$PY -m pip install kaggle==1.7.4.5
\$PY -m pip install matplotlib
\$PY -m pip install seaborn

# 모델 실행
MODEL_FILE=\${1:-baseline.py}

# 상대경로 실행 (프로젝트 루트 기준)
\$PY src/data/data_load/data_loader.py
\$PY src/data/yolo_dataset/yolo_export.py
\$PY src/models/\$MODEL_FILE
EOF

chmod +x "$SCRIPT_DIR/run.sh"

# alias 등록 (절대경로 대신 PROJECT_ROOT 기준 상대경로)
if ! grep -q "alias exc_pip=" ~/.bashrc; then
    echo "alias exc_pip=\"bash $PROJECT_ROOT/scripts/run.sh \\\"\\\$@\\\"\"" >> ~/.bashrc
fi

source ~/.bashrc

echo "exc_pip 명령어가 등록되었습니다."
