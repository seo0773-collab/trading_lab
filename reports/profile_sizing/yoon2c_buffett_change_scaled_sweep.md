# yoon2c 후보: 버핏지수 변화율 revert 강도 스윕

버핏지수 변화율의 revert 해석이 50/50보다 CAGR/총수익을 높였지만 Sharpe가 낮아졌기 때문에,
raw 비중을 그대로 쓰지 않고 50/50 주변으로 완화하는 `k`를 스윕했다.

- 기간: 2007-01-29 ~ 2026-06-29 (4081봉)
- 버핏지수: World Bank `CM.MKT.LCAP.GD.ZS` = `Stock market capitalization to GDP (%)`
- raw revert: `raw_w_KRW = US_growth / (KR_growth + US_growth)`
- 스윕 공식: `w_KRW = clip(0.5 + k * (raw_w_KRW - 0.5), [0.30, 0.70])`
- `k=0`: 기존 yoon2 50/50, `k=1`: 이전 revert change와 동일.

## 결과

| k | all CAGR | all MDD | all Sharpe | all TotalRet | val Sharpe | test MDD |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 (base) | 14.10% | -14.46% | 1.151 | 747.18% | 1.083 | -10.26% |
| 0.25 | 14.17% | -14.46% | 1.148 | 755.24% | 1.086 | -10.25% |
| 0.50 | 14.24% | -14.46% | 1.145 | 763.18% | 1.088 | -10.24% |
| 0.75 | 14.30% | -14.46% | 1.141 | 771.01% | 1.090 | -10.23% |
| 1.00 | 14.36% | -14.46% | 1.135 | 778.72% | 1.092 | -10.80% |
| 1.25 | 14.42% | -14.46% | 1.129 | 786.31% | 1.094 | -11.42% |
| 1.50 | 14.48% | -14.46% | 1.122 | 793.78% | 1.095 | -12.04% |

## 판정

실용 후보는 **k=1.25**다.

- base 대비 all CAGR: 14.10% → 14.42%
- base 대비 all TotalRet: 747.18% → 786.31%
- base 대비 all MDD: -14.46% → -14.46%
- base 대비 all Sharpe: 1.151 → 1.129

해석은 명확하다. `k`를 키울수록 수익률은 올라가지만 Sharpe가 점진적으로 내려간다.
MDD는 전구간 기준으로 거의 유지되므로, yoon2의 목적함수를 Sharpe 최우선이 아니라
`MDD 유지 + CAGR 개선`으로 둔다면 scaled revert는 채택 가능한 후보가 된다.

최신 확정 변화율 기준(2024년 관측치) raw revert KRW 비중은
60.8%이고, k=1.25 적용 비중은
63.4%다.

## 결론

`k=1.50`은 수익은 가장 많이 올리지만 Sharpe 하락과 test MDD 악화가 더 크다. 균형형 yoon2c 후보로는
**k=1.25 scaled revert**를 우선 검토하는 편이 낫다. 다음 단계는 이 후보를
정수주/DCA 실행 조건에서도 같은 방향으로 유지되는지 확인하는 것이다.

## 파일

- 스크립트: `scripts/profile_sizing/yoon2c_buffett_change_scaled_sweep.py`
- 결과 CSV: `reports/profile_sizing/yoon2c_buffett_change_scaled_sweep.csv`
- 비중 CSV: `reports/profile_sizing/yoon2c_buffett_change_scaled_weights.csv`
- 그래프 PNG: `reports/profile_sizing/yoon2c_buffett_change_scaled_sweep.png`
