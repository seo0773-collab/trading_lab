# final0706 vs SPY/QQQ/QLD buy&hold 비교

> **[정정 2026-07-07]** 이 리포트의 CAGR은 봉 수 기준 연환산 버그로 과대표기됨(.KS 데이터 갭 2007-02~2009-04, 예: 전체구간 16.84%→달력 기준 13.96%). Sharpe·MDD·순위는 유효. 정정판: `final0706_benchmark_compare_v2.md`, 진단: `final0706_data_gap_diagnosis.md`.

- Period: `2007-01-29~2026-07-02` (4084 bars, 원/달러 KRW 기준)
- final0706 equity는 공식 파이프라인(`Yoon1kHandler.build_artifacts`)이 만든 값을 그대로 사용(재구현 없음).
- `_KRW_bnh` = 원화환산 buy&hold(원화 투자자 기준, final0706과 동일 통화 조건), `_USD_bnh` = 참고용 달러 그대로.

## all (2007-01-29~2026-07-02)

| candidate | CAGR | MDD | Sharpe | Total Return |
| --- | ---: | ---: | ---: | ---: |
| final0706 | 16.84% | -16.40% | 1.242 | 1146.37% |
| SPY_KRW_bnh | 16.91% | -30.15% | 0.847 | 1157.85% |
| SPY_USD_bnh | 13.24% | -39.65% | 0.732 | 649.69% |
| QQQ_KRW_bnh | 23.85% | -30.33% | 1.010 | 3100.63% |
| QQQ_USD_bnh | 19.95% | -35.12% | 0.943 | 1807.60% |
| QLD_KRW_bnh | 34.88% | -60.94% | 0.899 | 12670.59% |
| QLD_USD_bnh | 30.65% | -63.68% | 0.844 | 7511.37% |

## test (2022-12-15~2026-07-02)

| candidate | CAGR | MDD | Sharpe | Total Return |
| --- | ---: | ---: | ---: | ---: |
| final0706 | 38.56% | -12.00% | 2.021 | 187.90% |
| SPY_KRW_bnh | 31.09% | -17.67% | 1.503 | 140.53% |
| SPY_USD_bnh | 24.00% | -18.76% | 1.432 | 100.83% |
| QQQ_KRW_bnh | 42.50% | -21.29% | 1.610 | 215.28% |
| QQQ_USD_bnh | 34.79% | -22.77% | 1.525 | 163.25% |
| QLD_KRW_bnh | 71.81% | -40.86% | 1.461 | 478.12% |
| QLD_USD_bnh | 62.51% | -42.29% | 1.368 | 382.70% |

## recent_3y (2023-07-07~2026-07-02)

| candidate | CAGR | MDD | Sharpe | Total Return |
| --- | ---: | ---: | ---: | ---: |
| final0706 | 42.50% | -12.00% | 2.094 | 164.10% |
| SPY_KRW_bnh | 30.83% | -17.67% | 1.478 | 108.94% |
| SPY_USD_bnh | 22.97% | -18.76% | 1.369 | 76.28% |
| QQQ_KRW_bnh | 36.49% | -21.29% | 1.416 | 134.65% |
| QQQ_USD_bnh | 28.28% | -22.77% | 1.284 | 97.97% |
| QLD_KRW_bnh | 56.79% | -40.86% | 1.244 | 243.20% |
| QLD_USD_bnh | 47.36% | -42.29% | 1.132 | 189.55% |

## 페어드 부트스트랩 (final0706 vs 원화환산 buy&hold, 8000 iters, block=20)

| vs | P(Sharpe 개선) | P(CAGR 개선) | P(MDD 더 얕음) |
| --- | ---: | ---: | ---: |
| final0706_vs_SPY_KRW | 97.2% | 42.2% | 94.7% |
| final0706_vs_QQQ_KRW | 88.6% | 5.5% | 97.9% |
| final0706_vs_QLD_KRW | 91.1% | 2.3% | 100.0% |

## 파일

- JSON: `reports/profile_sizing/final0706_benchmark_compare.json`
