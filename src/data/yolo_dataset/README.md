## ADD 데이터셋 Category 매핑 및 병합 모듈 안내
- 본 모듈은 **ADD 데이터셋의 COCO JSON 구조를 기존 학습 파이프라인과 호환되도록 수정**하고,
- 수정된 어노테이션과 이미지를 **기존 train 데이터셋에 안전하게 병합**하기 위한 전처리 단계이다.
- YOLO 변환 이전에 **반드시 1회 실행**해야 한다.

### 목적

- ADD 데이터셋의 COCO JSON 내부 `categories` 정보를 이미지 메타데이터 기준으로 동적 매핑
- 수정 완료된 JSON을 기존 학습용 annotation 디렉터리 하위로 이동
- ADD 이미지 파일을 기존 학습용 image 디렉터리로 이동
- 이후 **기존 YOLO 변환 파이프라인을 그대로 재사용 가능**

---

### 수행 작업 요약

본 모듈은 다음 작업을 자동으로 수행한다.

1. **JSON 내부 category 수정**

   * `categories[0].id`
     → `images[0].dl_idx` 값으로 치환
   * `categories[0].name`
     → `images[0].dl_name` 값으로 치환
   * `annotations[*].category_id` 는 수정하지 않음
     (기존 YOLO 파이프라인에서 중복 허용)

2. **JSON 수정 검증**

   * 수정 전 / 수정 후 category 값 diff 검사
   * 필수 키 누락 시 해당 JSON 스킵

3. **파일 이동**

   * 수정 완료된 JSON
     → `data/train_annotations/added_train_annotations/`
   * ADD 이미지 파일
     → `data/train_images/`
   * 대상 디렉터리가 없으면 자동 생성

4. **실행 결과 요약 출력**

   * 수정된 JSON 개수
   * 이동된 annotation 파일 수
   * 이동된 image 파일 수

### ADD 데이터셋 전제 구조

```text
data/
 └─ add/
    ├─ annotations/
    │   ├─ dl_13899/
    │   │   └─ xxx.json
    │   ├─ dl_16231/
    │   │   └─ xxx.json
    │   └─ ...
    └─ images/
        ├─ xxx.png
        ├─ yyy.png
        └─ ...
```

* `annotations` 내부에는 **여러 하위 폴더**가 존재
* 각 하위 폴더에는 **1개 이상의 COCO JSON 파일**
* `images`는 폴더 단위가 아니라 **이미지 파일 단위로 이동**

### 실행 위치 및 방법

* 모듈 위치

  * `src/data/yolo_dataset/add_json_category_mapper.py`

* 실행 방법 (WSL / Linux / Mac 공통)

```bash
python -m src.data.yolo_dataset.add_json_category_mapper
```

⚠️ **파일 직접 실행 금지**

```bash
python add_json_category_mapper.py  # ❌
```

이후 과정은 **아래 YOLOv8 변환 모듈 사용 방법을 그대로 따르면 된다.**

---

### YOLOv8 변환 모듈 사용 방법

- 디렉터리 준비  
  - 프로젝트 루트 폴더를 하나 만든다.  
  - 그 안에 다음 구조로 데이터 폴더를 둔다.  
    - `data/train_images/` : 학습용 원본 이미지  
    - `data/test_images/`  : 테스트용 이미지 (있으면 유지, 없어도 무방)  
    - `data/train_annotations/` : COCO 형식 JSON들이 들어있는 폴더 (현재 사용 중인 구조 그대로)  

- 모듈 파일 위치
  - `config.py`, `coco_parser.py`, `yolo_export.py` 세 파일이 `src/data/yolo_dataset/` 안에 있다.
  - 최종 구조 예시
    - `project_root/src/data/yolo_dataset/config.py`
    - `project_root/src/data/yolo_dataset/coco_parser.py`
    - `project_root/src/data/yolo_dataset/yolo_export.py`
    - `project_root/data/...`

- 실행 방법
  - 터미널(또는 Anaconda Prompt)에서 프로젝트 루트로 이동한다.
    - 예: `cd project_root`
  - 다음 명령으로 메인 모듈을 실행한다.
    - `python -m src.data.yolo_dataset.yolo_export`
  - 또는 직접 실행:
    - `cd src/data/yolo_dataset && python yolo_export.py`  

- 실행 중 확인할 로그  
  - COCO 파싱 및 정합성 검사 요약  
    - `JSON 파일 수: 1001`  
    - `JSON 기준 이미지 메타데이터 개수: 369`  
    - `데이터 정합성 검사 결과`  
      - `폴더에는 있지만 JSON에 없는 이미지 수: ...`  
      - `JSON에는 있지만 폴더에 없는 이미지 수: ...`  
    - `최종 필터링 후 이미지 개수: 232`  
    - `최종 필터링 후 어노테이션 개수: 763`  
  - Train/Val 분할 결과 (stratified split)  
    - `[split] Train 이미지 수: ..., Val 이미지 수: ...`  
    - `[split] Train 어노테이션 수: ..., Val 어노테이션 수: ...`  
    - `rep_category 1장짜리 클래스 수: ... (전부 train에 포함)`  
  - YOLO 디렉터리 생성 메시지  
    - `YOLO 디렉터리 생성 완료: project_root/data/yolo`  
  - 클래스 수 및 data.yaml 생성  
    - `클래스 수: ...`  
    - `data.yaml 생성 완료: project_root/data/yolo/pills.yaml`  
  - 이미지 복사 실패가 있으면 파일명과 함께 경고가 출력된다.  

- 실행 후 결과 확인  
  - `project_root/data/yolo/` 아래 구조를 확인한다.  
    - `project_root/data/yolo/images/train/` : 학습용 이미지  
    - `project_root/data/yolo/images/val/`   : 검증용 이미지  
    - `project_root/data/yolo/labels/train/` : 각 이미지와 동일한 이름의 `.txt` 라벨  
    - `project_root/data/yolo/labels/val/`   : 검증용 라벨  
    - `project_root/data/yolo/pills.yaml`    : YOLOv8 학습용 설정 파일  
  - 각 이미지에 대해 같은 이름의 `.txt`가 생성되었는지, txt 내용이  
    - `class_id x_center y_center width height` (0~1 사이 실수) 형식인지 확인한다.  

- YOLOv8에서 학습 시작 예시  
  - Ultralytics YOLOv8 환경이 준비되어 있다고 가정하면, 다음과 같이 학습을 시작할 수 있다.  
    - `yolo detect train data=project_root/data/yolo/pills.yaml model=yolov8n.pt imgsz=640 epochs=50`