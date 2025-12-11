#!/bin/bash

PY=python3
export KAGGLE_CONFIG_DIR=./data/.kaggle

# 필요한 패키지 설치
$PY -m pip install --upgrade pip
$PY -m pip install ultralytics==8.3.235
$PY -m pip install kaggle==1.7.4.5
$PY -m pip install matplotlib
$PY -m pip install seaborn
$PY -m pip install scikit-learn pandas numpy wandb

# 기본 실행 모델
MODEL_FILE=${1:-baseline.py}

# 상대경로 실행 (프로젝트 루트 기준)
$PY -m src.data.data_load.data_loader
$PY -m src.data.yolo_dataset.yolo_export
$PY -m src.models.$(basename $MODEL_FILE .py)
