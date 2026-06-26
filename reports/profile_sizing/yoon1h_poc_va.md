# yoon1h — POC/VA 매물대 위치 사이징 vs yoon1b percentile

yoon1b의 사이징 입력(cumulative_percentile)을 VA 매물대 대비 연속 위치(va_position: VAL→0·POC→0.5·VAH→1, 밖은 외삽+클립)로 교체. 같은 엔진·유니버스·bucket을 통과하는 순수 A/B. 규율: **holdout(test) Sharpe ≥ yoon1b** 우선.


## phase=validation (val(선정))

| 변형 | 노출 | CAGR | Sharpe | MDD |
| --- | ---: | ---: | ---: | ---: |
| yoon1b (percentile) | 77% | +11.9% | 0.932 | -19.5% |
| yoon1h va_pct0.60 | 75% | +10.6% | 0.864 | -18.1% |
| yoon1h va_pct0.70 | 75% | +10.7% | 0.868 | -18.1% |
| yoon1h va_pct0.80 | 75% | +10.7% | 0.867 | -18.1% |
| SPY | 100% | +2.7% | 0.234 | -55.2% |

## phase=test (holdout(OOS))

| 변형 | 노출 | CAGR | Sharpe | MDD |
| --- | ---: | ---: | ---: | ---: |
| yoon1b (percentile) | 91% | +19.2% | 1.331 | -22.9% |
| yoon1h va_pct0.60 | 90% | +18.7% | 1.317 | -23.0% |
| yoon1h va_pct0.70 | 90% | +18.6% | 1.309 | -23.0% |
| yoon1h va_pct0.80 | 90% | +18.7% | 1.313 | -23.0% |
| SPY | 100% | +14.0% | 0.855 | -33.7% |

## phase=all (full-cycle)

| 변형 | 노출 | CAGR | Sharpe | MDD |
| --- | ---: | ---: | ---: | ---: |
| yoon1b (percentile) | 85% | +15.6% | 1.142 | -32.0% |
| yoon1h va_pct0.60 | 82% | +15.1% | 1.132 | -32.0% |
| yoon1h va_pct0.70 | 83% | +15.1% | 1.128 | -32.0% |
| yoon1h va_pct0.80 | 83% | +15.2% | 1.131 | -32.0% |
| SPY | 100% | +10.8% | 0.644 | -55.2% |

## 규율 판정 (holdout test Sharpe 대비 yoon1b)

- yoon1h va_pct0.60: test ΔSharpe -0.014 → **기각**
- yoon1h va_pct0.70: test ΔSharpe -0.022 → **기각**
- yoon1h va_pct0.80: test ΔSharpe -0.019 → **기각**
