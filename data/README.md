# data/

데이터 저장 폴더 (gitignore에 포함됨)

## 구조

```
data/
├── train_images/          # 학습용 이미지 (651개 원본)
├── train_annotations/     # COCO JSON 파일 (1001개)
└── test_images/           # 테스트 이미지 (843개)
```

## 데이터 현황

### Train 데이터
- **원본 이미지**: 651개 PNG 파일
- **유효 이미지**: 232개 (JSON 어노테이션 존재)
- **어노테이션**: 763개 객체
- **클래스**: 56개

### Test 데이터
- **이미지**: 843개 PNG 파일
- **클래스**: 40개 (Train과 불일치)

## 사용법

### 1. 원본 데이터 다운로드
```bash
# Kaggle Competition에서 데이터 다운로드 후
# data/ 폴더에 압축 해제
```

### 2. YOLO 포맷 변환
```bash
# 232개 필터링 + YOLO 변환
python -m src.data.yolo_dataset.yolo_export

# 결과: datasets/pills/ 폴더에 YOLO 포맷 생성
# - images/train/ : 학습 이미지
# - images/val/   : 검증 이미지
# - labels/train/ : 학습 라벨 (.txt)
# - labels/val/   : 검증 라벨 (.txt)
# - pills.yaml    : YOLOv8 설정 파일
```

## 주의사항

- **이 폴더는 Git에 커밋되지 않습니다** (.gitignore)
- 팀원 각자 로컬에서 데이터 다운로드 필요
- 큰 파일은 절대 Git에 올리지 마세요

## 담당

- Data Engineer: 김민우, 김나연
