# final0706 US 슬리브 경로 틸트(path_gate) 사전 스크리닝

- 유니버스: VGT, XLF, IYH, FENY, FIDU, FSTA, IYC, XLU, XLB, FREL, FCOM (final0706 US 슬리브 top_k=3 후보 11종)
- 방법: `yoon_profile_path_study.py`와 동일 방법론(NW t + Bonferroni/BH-FDR + WF 3사이클)을 거래량 프로파일 percentile 버킷 축(1차원)에 이식. 사전 기하 가설이 없어 5개 버킷 전부를 family에 포함(m=30, 더 보수적인 다중비교 보정).
- vel = cumulative_percentile.diff(10) (사전고정, 스윕 금지). arrival: up=vel≥0 / down=vel<0.
- 풀링: 종목별 NW(mean,se,n) 산출 후 표본가중 결합(종목 간 독립 가정 — 한계로 보고, 실제로는 섹터 ETF 간 일부 상관 존재).

## 버킷×horizon delta(up−down) 판정

| bucket(base weight) | horizon | delta(up-down) | t | FDR유의 | WF 부호일치 | 판정 |
| --- | ---: | ---: | ---: | --- | --- | --- |
| B[0,20) | 20 | 0.33%p | 0.63 | False | 2/3 | **기각** |
| B[0,20) | 60 | -1.01%p | -0.84 | False | 3/3 | **기각** |
| B[20,40) | 20 | 0.13%p | 0.40 | False | 2/3 | **기각** |
| B[20,40) | 60 | -0.63%p | -0.85 | False | 2/3 | **기각** |
| B[40,60) | 20 | -0.06%p | -0.23 | False | 2/3 | **기각** |
| B[40,60) | 60 | -0.38%p | -0.67 | False | 3/3 | **기각** |
| B[60,80) | 20 | -0.18%p | -0.89 | False | 1/3 | **기각** |
| B[60,80) | 60 | -0.69%p | -1.31 | False | 2/3 | **기각** |
| B[80,100] | 20 | -0.60%p | -2.71 | False | 2/3 | **보류(방향 일치)** |
| B[80,100] | 60 | -1.14%p | -2.25 | False | 2/3 | **보류(방향 일치)** |

## 파일

- JSON: `reports/profile_sizing/final0706_us_path_study.json`
