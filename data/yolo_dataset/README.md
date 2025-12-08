### YOLOv8 변환 모듈 사용 방법

- 디렉터리 준비  
  - 프로젝트 루트 폴더를 하나 만든다.  
  - 그 안에 다음 구조로 데이터 폴더를 둔다.  
    - `data/train_images/` : 학습용 원본 이미지  
    - `data/test_images/`  : 테스트용 이미지 (있으면 유지, 없어도 무방)  
    - `data/train_annotations/` : COCO 형식 JSON들이 들어있는 폴더(현재 사용 중인 구조 그대로)  

- 모듈 파일 위치  
  - `config.py`, `coco_parser.py`, `yolo_export.py` 세 파일을 모두 **프로젝트 루트** 바로 아래에 둔다.  
  - 최종 구조 예시  
    - `project_root/config.py`  
    - `project_root/coco_parser.py`  
    - `project_root/yolo_export.py`  
    - `project_root/data/...`  

- 실행 방법  
  - 터미널(또는 Anaconda Prompt)에서 작업 디렉터리를 프로젝트 루트로 이동한다.  
    - 예: `cd project_root`  
  - 다음 명령으로 메인 모듈을 실행한다.  
    - `python yolo_export.py`  

- 실행 중 확인할 로그  
  - COCO 파싱 요약  
    - `images_df: …`, `annotations_df: …`, `categories_df: …`  
  - Train/Val 분할 결과  
    - `Train 이미지: …, Val 이미지: …`  
  - YOLO 디렉터리 생성 메시지  
    - `YOLO 디렉터리 생성 완료: datasets/pills`  
  - 클래스 수  
    - `클래스 수: …`  
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