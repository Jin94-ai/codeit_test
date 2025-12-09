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
    - `YOLO 디렉터리 생성 완료: datasets/pills`  
  - 클래스 수 및 data.yaml 생성  
    - `클래스 수: ...`  
    - `data.yaml 생성 완료: datasets/pills/pills.yaml`  
  - 이미지 복사 실패가 있으면 파일명과 함께 경고가 출력된다.  

- 실행 후 결과 확인  
  - `datasets/pills/` 아래 구조를 확인한다.  
    - `datasets/pills/images/train/` : 학습용 이미지  
    - `datasets/pills/images/val/`   : 검증용 이미지  
    - `datasets/pills/labels/train/` : 각 이미지와 동일한 이름의 `.txt` 라벨  
    - `datasets/pills/labels/val/`   : 검증용 라벨  
    - `datasets/pills/pills.yaml`    : YOLOv8 학습용 설정 파일  
  - 각 이미지에 대해 같은 이름의 `.txt`가 생성되었는지, txt 내용이  
    - `class_id x_center y_center width height` (0~1 사이 실수) 형식인지 확인한다.  

- YOLOv8에서 학습 시작 예시  
  - Ultralytics YOLOv8 환경이 준비되어 있다고 가정하면, 다음과 같이 학습을 시작할 수 있다.  
    - `yolo detect train data=datasets/pills/pills.yaml model=yolov8n.pt imgsz=640 epochs=50`

---

### Albumentations 기반 YOLO 증강 모듈 사용 방법

- 파일 구성  
  - `src/data/yolo_dataset/yolo_augmentation.py`  
  - `src/data/yolo_dataset/yolo_dataset.py`  
  - `notebooks/visualize_aug.ipynb` (시각화용 노트북, 위치는 프로젝트 구조에 맞게 조정 가능)  

- 학습 코드에서 사용  
  - 프로젝트 루트에서 실행하는 것을 기준으로 한다.  
  - 예시 (train 스크립트 내):  
```python
from src.data.yolo_dataset.yolo_dataset import get_train_loader

train_loader, train_dataset = get_train_loader(
    img_dir="datasets/pills/images/train",
    label_dir="datasets/pills/labels/train",
    batch_size=4,
    num_workers=0,
)
```  
  - `get_train_loader`는 내부에서  
    - 시드 고정(`set_global_seed`)  
    - Albumentations 증강 파이프라인(`get_train_transform`)  
    을 설정한 뒤, on-the-fly 증강이 적용된 `DataLoader`를 반환한다.

- 증강 결과 시각화(visualize_aug.ipynb)  
  - Jupyter Notebook에서 `notebooks/visualize_aug.ipynb`를 열고, 커널의 작업 디렉터리를 프로젝트 루트로 맞춘다.  
  - 셀을 순서대로 실행하면  
    - `get_train_loader`로 한 배치를 불러오고  
    - 정규화를 되돌린 뒤  
    - YOLO 포맷 bbox를 픽셀 좌표로 변환해 이미지 위에 그려준다.  
  - 여러 샘플을 한 번에 표시하므로  
    - 밝기/대비·색조·회전 증강이 예상대로 적용되는지  
    - bbox가 여전히 알약을 잘 둘러싸는지  
    를 눈으로 확인할 수 있다.

- 주의 사항  
  - YOLO 라벨 파일(.txt)은 `class x_center y_center width height` 형식이고, 모든 값이 0~1 범위여야 한다.  
  - `datasets/pills/images/train`와 `datasets/pills/labels/train` 경로가 실제 YOLOv8 변환 결과와 일치해야 한다.  
  - 노트북/스크립트 실행 시 항상 프로젝트 루트에서 실행하거나, `PYTHONPATH`에 루트를 추가해서 `src.data.yolo_dataset...` 임포트가 가능하도록 설정한다.

---

### 학습에서 원본과 증강본을 같이 사용하는 방법

- 원본·증강용 Dataset 구성  
  - 원본 이미지만 사용하는 Dataset과, Albumentations on-the-fly 증강을 적용하는 Dataset을 각각 만든 뒤, 합쳐서 사용한다.  
  - 예시 아이디어  
    - `orig_dataset` : `transforms=None` 으로 원본만 반환  
    - `aug_dataset`  : `transforms=get_train_transform()` 으로 증강본만 반환  
    - `CombinedDataset`에서  
      - 앞부분 인덱스는 `orig_dataset[idx]`  
      - 뒷부분 인덱스는 `aug_dataset[idx - N]`  
      로 반환해, 총 `2N` 개 샘플(원본 N + 증강본 N)을 한 DataLoader에서 섞어서 학습에 사용한다.

- 증강 데이터 양과 epoch 관계  
  - on-the-fly 증강은 디스크에 새로운 이미지를 저장하지 않고, **각 epoch마다 원본을 불러올 때마다 매번 다른 변형본을 생성**한다.  
  - 증강용 Dataset의 크기(예: N장)를 고정해 두더라도,  
    - epoch를 여러 번 돌면 매 epoch마다 N개의 “새로운 증강본”을 보게 되어,  
    - 결과적으로 epoch 수에 비례해 모델이 보게 되는 증강 샘플의 양을 늘릴 수 있다.