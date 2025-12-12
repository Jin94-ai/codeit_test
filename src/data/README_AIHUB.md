# AI Hub 데이터 통합 가이드

Competition 56개 클래스의 데이터 불균형을 해소하기 위한 AI Hub 추가 데이터셋 통합 스크립트입니다.

## 제약사항

- **TS2, TL2 데이터셋 사용 금지**
- 나머지 모든 TL/TS 데이터셋 사용 가능 (TL_1, TL_3, TL_4, ..., TL_81, TS1, TS3, ...)

## 주요 기능

### 1. 현재 데이터 분석

```bash
python src/data/aihub_integration.py --analyze-only
```

Competition 데이터의 56개 클래스별 이미지 개수를 분석합니다.

### 2. API로 데이터 다운로드

#### 2.1 파일 목록 조회

```bash
python src/data/aihub_integration.py --list-files
```

AI Hub 의약품 데이터셋(#576)의 파일 목록과 filekey를 조회합니다.

#### 2.2 특정 파일 다운로드 (수동)

현재는 filekey를 수동으로 지정해야 합니다:

```python
from src.data.aihub_integration import AIHubDownloader

downloader = AIHubDownloader('.', 'YOUR_API_KEY')
downloader.install_aihubshell()
downloader.download_dataset_file('12345', 'TL_1')  # filekey와 데이터셋명
```

### 3. 로컬 데이터셋 통합

#### 3.1 data/added/ 준비

다운로드한 AI Hub 데이터를 다음 구조로 배치:

```
data/
└── added/
    ├── tl1/
    │   ├── train_annotations/  # JSON 파일들
    │   └── train_images/       # 이미지 파일들
    ├── tl3/
    │   ├── train_annotations/
    │   └── train_images/
    ├── tl4/
    │   ├── train_annotations/
    │   └── train_images/
    └── ...
```

#### 3.2 모든 데이터셋 자동 통합

```bash
python src/data/aihub_integration.py
```

- `data/added/` 하위의 모든 TL*/TS* 데이터셋을 자동 탐색
- TS2, TL2는 자동으로 제외
- 10개 미만인 클래스에만 데이터 추가
- **클래스당 최대 개수 제한 없음** (기본값)

#### 3.3 특정 데이터셋만 처리

```bash
python src/data/aihub_integration.py --datasets TL_1 TL_3 TL_4
```

#### 3.4 파라미터 조정

```bash
# min-samples 조정: 5개 미만인 클래스만
python src/data/aihub_integration.py --min-samples 5

# max-samples-per-class 제한: 클래스당 최대 100개
python src/data/aihub_integration.py --max-samples-per-class 100

# 제한 없이 모두 추가 (기본값)
python src/data/aihub_integration.py
```

## 작동 방식

### 1단계: 현재 데이터 분석

Competition 데이터에서 56개 클래스별 이미지 개수 파악

### 2단계: AI Hub JSON 형식 변환

AI Hub 원본:
```json
{
  "categories": [{"id": 1, "name": "Drug"}],
  "images": [{"dl_idx": 1899, "dl_name": "실제약품명"}]
}
```

Competition 형식으로 변환:
```json
{
  "categories": [{"id": 1899, "name": "실제약품명"}],
  "images": [{"dl_idx": 1899, "dl_name": "실제약품명"}]
}
```

### 3단계: 56개 클래스 필터링

- `dl_idx`가 56개 TARGET_CLASSES에 있는지 확인
- 없으면 스킵

### 4단계: 이미지 존재 확인

JSON에 해당하는 이미지 파일이 실제로 있는지 검증

### 5단계: 선별적 추가

- 현재 이미지 개수가 min_samples(기본 10개) 미만인 클래스만
- max_samples_per_class가 지정되면 클래스당 최대 개수 제한 (기본: 무제한)

### 6단계: 통합

- 수정된 JSON을 `data/train_annotations_integrated/`로 복사
- 해당 이미지를 `data/train_images_integrated/`로 복사

## 출력 디렉토리

- `data/added/*/annotations_fixed/`: 수정된 JSON (중간 결과)
- `data/train_annotations_integrated/`: 최종 통합 JSON
- `data/train_images_integrated/`: 최종 통합 이미지

## API 키 설정

`.env.aihub` 파일에 저장:

```bash
AIHUB_API_KEY=C418F868-33C7-4EEE-A17F-AB10BBFD7CAF
```

또는 명령줄 옵션으로 전달:

```bash
python src/data/aihub_integration.py --list-files --api-key YOUR_KEY
```

## 전체 워크플로우 예시

```bash
# 1. 현재 데이터 상태 확인
python src/data/aihub_integration.py --analyze-only

# 2. API 파일 목록 조회
python src/data/aihub_integration.py --list-files

# 3. (수동) AI Hub에서 데이터 다운로드 후 data/added/에 배치

# 4. 자동 통합 실행
python src/data/aihub_integration.py

# 5. 결과 확인
ls data/train_annotations_integrated/ | wc -l
ls data/train_images_integrated/ | wc -l
```

## 주의사항

1. **용량**: 데이터셋 압축 해제 시 원본의 2~3배 용량 필요
2. **TS2/TL2**: 자동으로 제외되므로 신경 쓰지 않아도 됨
3. **중복 방지**: 이미 존재하는 이미지는 복사하지 않음
4. **최대 개수**: 기본적으로 제한 없음, 필요시 `--max-samples-per-class` 사용

## 트러블슈팅

### "경고: data/added/ 디렉토리가 없습니다"

→ `data/added/tl1/`, `data/added/tl3/` 등의 구조로 데이터 배치

### "사용 가능한 데이터셋이 없습니다"

→ 각 데이터셋 폴더에 `train_annotations/`와 `train_images/` 디렉토리 확인

### "aihubshell 설치 실패"

→ curl이 설치되어 있는지 확인: `which curl`
