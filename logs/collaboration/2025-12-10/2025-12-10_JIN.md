# Daily 협업일지

### [1] 오늘 날짜 / 이름 / 팀명

- 날짜: 2025-12-10
- 이름: 이진석 (JIN)
- 팀명: 코드잇 8팀

---

### [2] 오늘 맡은 역할 및 구체적인 작업 내용

✍️ 답변:

Leader & Integration Specialist 역할:

1. **Kaggle 제출 형식 수정 (핵심 작업)**
   - category_id 매핑 오류 발견: YOLO 0-based index를 단순 +1 처리 → 점수 0점 원인
   - class_mapping.json 생성으로 원본 category_id 보존 (1899, 3350 등)
   - yolo_export.py 수정: YOLO index → 원본 category_id 매핑 저장
   - baseline.py 수정: 매핑 로드하여 정확한 submission 생성

2. **파이프라인 안정성 개선**
   - run.sh 필수 패키지 누락 해결 (scikit-learn, pandas, numpy, wandb)
   - 근본 원인 분석: yolo_export가 패키지 없어서 실패 → pills.yaml 미생성
   - NMS 타임아웃 문제 해결 시도 (WSL2 /mnt/c/ 경로 I/O 병목 확인)

3. **실험 추적 시스템 구축**
   - W&B 통합 (project: codeit_team8)
   - 타임스탬프 기반 submission 파일 자동 생성
   - outputs/ 폴더 .gitignore 추가

4. **코드 통합 및 커밋**
   - JIN 브랜치에 모든 수정사항 커밋 및 푸시
   - 3개 파일 수정: run.sh, yolo_export.py, baseline.py

---

### [3] 오늘 작업 완료도 체크 (하나만 체크)

- [ ]  🔴 0% (시작 못함)
- [ ]  🟠 25% (시작은 했지만 진척 없음)
- [ ]  🟡 50% (진행 중, 절반 이하)
- [x]  🔵 75% (거의 완료됨)
- [ ]  🟢 100% (완료 및 점검까지 완료)

📌 간단한 근거:

- ✅ Kaggle 제출 형식 수정 완료 (category_id 매핑 정상화)
- ✅ 파이프라인 필수 패키지 문제 해결
- ✅ W&B 통합 및 submission 자동 생성
- ⚠️ NMS 타임아웃 문제 완전 해결은 못함 (WSL2 환경 한계)
- 🔄 실제 Kaggle 제출 및 점수 확인 필요

---

### [4] 오늘 협업 중 제안하거나 피드백한 내용이 있다면?

✍️ 답변:

**팀원 간 논의 내용:**

1. **캐글 제출 횟수 확인 (민우님)**
   - 공식: 팀 합산 5회/일
   - 실제: 10회로 표시됨 (1인당 1개씩 제출 가능한 듯)

2. **제출 파일 관리 방식 (유민님 제안)**
   - outputs/submissions/ 폴더에 타임스탬프로 자동 수집
   - .gitignore 처리로 깃허브에는 업로드 안 됨

3. **NMS 에러 해결 (강사님 조언)**
   - time_limit 파라미터 조정
   - agnostic_nms=False 시도
   - → 근본 원인은 WSL2 /mnt/c/ 경로의 느린 I/O

4. **W&B 통합 (유민님)**
   - 원래 보윤님 버전 기반으로 진행 예정
   - 제가 먼저 codeit_team8 프로젝트로 통합 완료

---

### [5] 오늘 분석/실험 중 얻은 인사이트나 발견한 문제점은?

✍️ 답변:

**주요 인사이트:**

1. **Category ID 매핑 문제 발견**
   - YOLO는 내부적으로 0-based 연속 인덱스 사용
   - Kaggle은 원본 COCO category_id 필요 (1899, 2482, 3350 등)
   - 단순 +1 처리는 완전히 잘못된 접근

2. **파이프라인 의존성 문제**
   - run.sh가 일부 패키지만 설치 → yolo_export 실패 → pills.yaml 미생성
   - 근본 원인: scikit-learn, pandas 누락
   - 매번 수동 설치가 필요했던 이유 파악

3. **WSL2 환경 병목**
   - /mnt/c/ 경로는 Windows 파일시스템 마운트
   - 파일 I/O가 10-100배 느림
   - NMS 자체 문제가 아닌 저장 속도 문제

**발견한 문제점:**

1. 제출용 최종 파라미터 결정 필요 (imgsz, conf, iou 등)
2. 학습된 모델 경로 하드코딩 (runs/detect/train13/)
3. GPU 관련 파라미터 (device=0, half=True) 환경 의존성

---

### [6] 일정 지연이나 협업 중 어려웠던 점이 있다면?

✍️ 답변:

1. **환경 문제로 인한 시행착오**
   - WSL2 환경 특성상 NMS 타임아웃 반복 발생
   - 파라미터 조정만으로 해결 안 됨
   - 최종적으로 imgsz 축소 + save=False로 우회 해결

2. **Kaggle 0점 원인 파악**
   - 처음엔 제출 형식 문제로 추정
   - 실제로는 category_id 매핑 오류였음
   - 근본 원인 찾는 데 시간 소요

---

### [7] 오늘 발표 준비나 커뮤니케이션에서 기여한 부분은?

✍️ 답변:

1. **기술적 문제 해결 및 공유**
   - Category_id 매핑 문제 해결 방법 공유
   - 파이프라인 안정성 개선

2. **제출 시스템 자동화**
   - outputs/submissions/ 폴더 자동 생성
   - 타임스탬프 기반 파일명으로 히스토리 보관

3. **실험 추적 기반 마련**
   - W&B 프로젝트 통합 (codeit_team8)
   - 팀원들이 실험 결과 공유 가능하도록 설정

---

### [8] 내일 목표 / 할 일

✍️ 답변:

1. **실제 Kaggle 제출 및 검증**
   - 수정된 submission 파일로 제출
   - 점수 정상 확인 (0점 해결 여부)

2. **Inference 파이프라인 분리**
   - scripts/inference.py 생성
   - 학습과 추론 분리로 재사용성 향상

3. **파라미터 최적화 실험**
   - conf, iou, max_det 등 튜닝
   - W&B로 실험 결과 추적

4. **팀원 코드 리뷰 및 통합 지원**
   - 유민님 W&B PR 리뷰
   - 민우님 데이터 추가 작업 확인

---
