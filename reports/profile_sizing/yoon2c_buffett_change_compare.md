# yoon2c 후보: 버핏지수 변화율 기반 통화 조절

yoon2의 미국/한국 슬리브는 유지하고, 월간 통화 리밸런싱 비중만 버핏지수 레벨이 아니라
전년 대비 변화율로 바꿔 검증했다.

- 기간: 2007-01-29 ~ 2026-06-29 (4081봉)
- 버핏지수: World Bank `CM.MKT.LCAP.GD.ZS` = `Stock market capitalization to GDP (%)`
- 변화율: `growth = Buffett_t / Buffett_(t-1)`; 단순 YoY 변화율은 `growth - 1`
- direct change: `w_KRW = KR_growth / (KR_growth + US_growth)`
- revert change: `w_KRW = US_growth / (KR_growth + US_growth)`; 버핏지수가 더 빨리 오르는 시장을 줄이는 반대 해석
- 무누수 처리: 관측연도 Y의 변화율은 다음 해 1월 1일부터 사용. 2025년 이후는 2024년 확정치 forward-fill.

## 판정: 변화율 기준도 yoon2 50/50 대체 실패

direct change는 train Sharpe와 MDD가 좋아지지만 validation Sharpe가 낮고 전구간 총수익도 줄어든다.
revert change는 전구간 CAGR/총수익은 올라가지만 Sharpe가 낮아지고 test 낙폭이 깊어진다.
둘 다 안정적인 위험조정 개선으로 보기 어렵다.

| 변형 | phase | CAGR | MDD | Sharpe | Vol | TotalRet |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| **base 50/50** | train | 8.95% | -12.09% | 0.788 | 11.72% | 129.96% |
|  | validation | 11.44% | -14.46% | 1.083 | 10.52% | 42.03% |
|  | test | 34.15% | -10.26% | 2.115 | 14.39% | 159.24% |
|  | all | 14.10% | -14.46% | 1.151 | 12.10% | 747.25% |
| direct change | train | 9.42% | -11.94% | 0.815 | 11.89% | 139.73% |
|  | validation | 11.44% | -14.45% | 1.071 | 10.65% | 42.01% |
|  | test | 30.82% | -10.30% | 2.142 | 12.94% | 138.95% |
|  | all | 13.82% | -14.45% | 1.148 | 11.89% | 714.01% |
| revert change | train | 8.48% | -12.23% | 0.756 | 11.63% | 120.47% |
|  | validation | 11.45% | -14.46% | 1.092 | 10.43% | 42.04% |
|  | test | 37.45% | -10.80% | 2.037 | 16.28% | 180.45% |
|  | all | 14.36% | -14.46% | 1.135 | 12.50% | 778.78% |

## 해석

- direct all: CAGR 14.10% → 13.82%, Sharpe 1.151 → 1.148, MDD -14.46% → -14.45%.
- revert all: CAGR 14.10% → 14.36%, Sharpe 1.151 → 1.135, MDD -14.46% → -14.46%.
- validation만 보면 revert change Sharpe 1.092가 기준 1.083보다 약간 높지만, 전구간 Sharpe가 낮고 test MDD가 깊어 채택 근거가 약하다.
- 최신 확정 변화율 기준(2024년 관측치): Korea YoY -22.2%, US YoY 20.5%, direct KRW 비중 39.2%, revert KRW 비중 60.8%.

## 결론

버핏지수의 레벨 비율보다 변화율 기준은 비중이 50% 근처에서 움직여 더 완만하지만,
성과 개선은 나오지 않았다. 현재 기준에서는 yoon2의 고정 50/50 통화 리밸런싱을 유지하는 쪽이 낫다.

## 파일

- 스크립트: `scripts/profile_sizing/yoon2c_buffett_change_compare.py`
- 결과 CSV: `reports/profile_sizing/yoon2c_buffett_change_compare.csv`
- 변화율/비중 CSV: `reports/profile_sizing/yoon2c_buffett_change_weights.csv`
- 그래프 PNG: `reports/profile_sizing/yoon2c_buffett_change_compare.png`
