# yoon2c 후보: 한국/미국 버핏지수 비율 통화 조절

yoon2의 미국 슬리브(`yoon1j_lp`)와 한국 슬리브(`yoon1j_kr`)는 그대로 두고,
월간 통화 리밸런싱 비중만 고정 50/50에서 World Bank 버핏지수 비율로 바꿔 비교했다.

- 기간: 2007-01-29 ~ 2026-06-29 (4081봉)
- 버핏지수: World Bank `CM.MKT.LCAP.GD.ZS` = `Stock market capitalization to GDP (%)`
- 무누수 처리: 관측연도 Y 값은 다음 해 1월 1일부터 사용. 2025년 이후 구간은 최신 확정치 forward-fill.
- direct: `w_KRW = KoreaBuffett / (KoreaBuffett + USBuffett)`
- inverse value: `w_KRW = (1/KoreaBuffett) / (1/KoreaBuffett + 1/USBuffett)`; 밸류에이션이 낮은 시장을 더 사는 민감도 비교.

## 판정: yoon2c direct는 보류

직접비율 방식은 train 구간 CAGR과 MDD는 개선되지만 validation과 전구간 Sharpe가 기준 50/50보다 낮다.
가치투자식 역비율도 test 구간 수익률은 좋아 보이나 train·validation Sharpe와 MDD가 기준보다 나빠 연구 게이트를 통과하지 못한다.

| 변형 | phase | CAGR | MDD | Sharpe | Vol | TotalRet |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| **base 50/50** | train | 8.95% | -12.09% | 0.788 | 11.72% | 129.96% |
|  | validation | 11.44% | -14.46% | 1.083 | 10.52% | 42.03% |
|  | test | 34.15% | -10.26% | 2.115 | 14.39% | 159.24% |
|  | all | 14.10% | -14.46% | 1.151 | 12.10% | 747.24% |
| yoon2c direct | train | 10.14% | -10.97% | 0.787 | 13.34% | 155.45% |
|  | validation | 11.98% | -14.72% | 1.071 | 11.14% | 44.24% |
|  | test | 27.82% | -9.44% | 2.134 | 11.84% | 121.62% |
|  | all | 13.85% | -14.72% | 1.088 | 12.64% | 717.07% |
| inverse value | train | 7.73% | -13.69% | 0.756 | 10.57% | 106.09% |
|  | validation | 10.88% | -14.96% | 1.028 | 10.59% | 39.71% |
|  | test | 40.24% | -13.45% | 1.878 | 18.99% | 199.37% |
|  | all | 14.23% | -16.58% | 1.108 | 12.74% | 762.47% |

## 해석

- direct all: CAGR 14.10% → 13.85%, Sharpe 1.151 → 1.088, MDD -14.46% → -14.72%.
- direct validation: Sharpe 1.083 → 1.071, MDD -14.46% → -14.72%. 기준보다 위험효율이 낮다.
- inverse value all: CAGR 14.23%, Sharpe 1.108; 전구간은 좋아도 validation Sharpe 1.028로 기준 1.083보다 낮다.
- 최신 확정 버핏지수 기준(2024년 관측치): Korea 83.05%, US 216.29%, direct KRW 비중 27.7%, inverse KRW 비중 72.3%.

## 결론

현재 데이터와 게이트 기준에서는 yoon2의 단순 50/50 통화 리밸런싱을 대체할 만큼
버핏지수 비율 신호가 안정적이지 않다. yoon2c를 만든다면 direct 비율보다는
inverse value 쪽이 후보지만, validation 통과 전까지는 실전 전략 등록을 보류하는 게 맞다.

## 파일

- 스크립트: `scripts/profile_sizing/yoon2c_buffett_ratio_compare.py`
- 결과 CSV: `reports/profile_sizing/yoon2c_buffett_ratio_compare.csv`
- 버핏지수/비중 CSV: `reports/profile_sizing/yoon2c_buffett_ratio_weights.csv`
- 그래프 PNG: `reports/profile_sizing/yoon2c_buffett_ratio_compare.png`
