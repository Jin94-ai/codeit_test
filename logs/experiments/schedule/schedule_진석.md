# 이진석

expID: exp2XX_js_imgsz_512

| 실험 번호  | 데이터셋     | 증강 기법           | 하이퍼파라미터                     | 입력 크기             |
| ------ | -------- | --------------- | --------------------------- | ----------------- |
| exp2XX_inputsz_512 | pills v2 | 증강 없음(baseline) | epochs 50 / lr 0.01         | 640               |
| exp2XX_inputsz_512 | pills v2 | 기본 Aug          | epochs 50                   | 512               |
| exp2XX_inputsz_640 | pills v2 | 기본 Aug          | epochs 50                   | 640               |
| exp2XX_inputsz_768 | pills v2 | 기본 Aug          | epochs 50                   | 768               |
| exp2XX_inputsz_960 | pills v2 | 기본 Aug          | epochs 고정 / imgsz 변화 민감도 분석 | 512·640·768 요약 분석 |
