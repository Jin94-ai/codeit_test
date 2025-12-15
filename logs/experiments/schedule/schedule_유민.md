#황유민

실험ID: exp2XX_ym_Best1

| 실험 번호  | 데이터셋     | 증강 기법                         | 하이퍼 파라미터               | 입력 크기 |
| ------ | -------- | ----------------------------- | ---------------------- | ----- |
| exp001 | pills v2 | 증강 없음(baseline 완료됨)           | epochs 50              | 640   |
| exp002 | pills v2 | 팀 실험 top-5 중 1위 Aug           | epochs 60 / multi-seed | 640   |
| exp003 | pills v2 | 팀 실험 top-5 중 2위 Aug           | epochs 60 / multi-seed | 640   |
| exp004 | pills v2 | exp002 설정 유지 + imgsz 768      | epochs 60              | 768   |
| exp005 | pills v2 | exp002 설정 유지 + imgsz 960(조건부) | epochs 60              | 960   |