# data/

데이터 저장 폴더 (gitignore에 포함됨)

## 구조

```
data/
├── raw/           # 원본 데이터 (Kaggle에서 다운로드)
└── processed/     # 전처리된 데이터
```

## 사용법

### 1. 원본 데이터 다운로드
```bash
# Kaggle Competition에서 데이터 다운로드 후
# data/raw/ 폴더에 압축 해제
```

### 2. 전처리된 데이터
- `src/data/preprocessing.py`로 생성
- `data/processed/` 폴더에 저장

## 주의사항

- **이 폴더는 Git에 커밋되지 않습니다** (.gitignore)
- 팀원 각자 로컬에서 데이터 다운로드 필요
- 큰 파일은 절대 Git에 올리지 마세요

## 담당

- Data Engineer: 김민우, 김나연
