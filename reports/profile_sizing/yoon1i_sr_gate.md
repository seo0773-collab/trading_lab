# yoon1i — HVN 지지/저항 기대값 게이트 vs yoon1b

yoon1b 점수 × EV=(저항−종가)/(저항−지지) 게이트[g_min,1]. 지지근처=매수기대(열림)·저항근처=매도기대(억제). 규율: **holdout(test) Sharpe ≥ yoon1b**.


## phase=validation (val(선정))

| 변형 | 노출 | CAGR | Sharpe | MDD |
| --- | ---: | ---: | ---: | ---: |
| yoon1b (게이트 off) | 77% | +11.9% | 0.932 | -19.5% |
| yoon1i SR g0.3 | 68% | +9.8% | 0.899 | -15.0% |
| yoon1i SR g0.5 | 71% | +10.3% | 0.898 | -16.3% |
| yoon1i SR g0.7 | 74% | +11.1% | 0.916 | -17.7% |
| SPY | 100% | +2.7% | 0.234 | -55.2% |

## phase=test (holdout(OOS))

| 변형 | 노출 | CAGR | Sharpe | MDD |
| --- | ---: | ---: | ---: | ---: |
| yoon1b (게이트 off) | 91% | +19.2% | 1.331 | -22.9% |
| yoon1i SR g0.3 | 85% | +15.5% | 1.206 | -20.3% |
| yoon1i SR g0.5 | 87% | +16.5% | 1.235 | -21.2% |
| yoon1i SR g0.7 | 89% | +17.4% | 1.261 | -22.2% |
| SPY | 100% | +14.0% | 0.855 | -33.7% |

## phase=all (full-cycle)

| 변형 | 노출 | CAGR | Sharpe | MDD |
| --- | ---: | ---: | ---: | ---: |
| yoon1b (게이트 off) | 85% | +15.6% | 1.142 | -32.0% |
| yoon1i SR g0.3 | 72% | +12.3% | 1.056 | -31.8% |
| yoon1i SR g0.5 | 77% | +13.4% | 1.086 | -32.3% |
| yoon1i SR g0.7 | 81% | +14.4% | 1.112 | -32.1% |
| SPY | 100% | +10.8% | 0.644 | -55.2% |

## 규율 판정 (holdout test Sharpe 대비 yoon1b)

- yoon1i SR g0.3: test ΔSharpe -0.125 → **기각**
- yoon1i SR g0.5: test ΔSharpe -0.096 → **기각**
- yoon1i SR g0.7: test ΔSharpe -0.070 → **기각**
