# yoon1e 추천 파라미터 백테스트 리포트

목적: 직전 탐색에서 후보로 나온 `top20/monthly/trail12`를 기준 전략(yoon1b), 현재 yoon1e
기본값(`top30/daily/trail5`), 인접 후보와 비교한다.

추천 후보:
- `top_k=20`
- `rebalance_freq=monthly`
- `short_hedge.hedge_ratio=0.5`
- `short_hedge.max_short=0.25`
- `short_hedge.trailing_take_profit.trail_pct=0.12`

비교 구간:
- validation: 2000-08-25~2013-07-23
- holdout(test): 2013-07-24~2026-06-17
- all: 1962-01-02~2026-06-17

## Validation

| 전략 | CAGR | MDD | Sharpe | 총수익 | 평균 롱 | 평균 숏 | 평균 순노출 | 롱 거래 | 숏 거래 | 트레일링 익절 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| yoon1b 기준 | 11.9% | -19.5% | 0.936 | 326.8% | 76.8% | 0.0% | 76.8% | 353 | 0 | 0 |
| 추천: top20 monthly trail12 | 12.6% | -17.3% | 1.017 | 362.4% | 76.8% | 4.7% | 72.1% | 353 | 11 | 5 |
| 비교: top20 monthly trail8 | 12.1% | -17.3% | 0.968 | 336.2% | 76.8% | 3.4% | 73.4% | 353 | 12 | 8 |
| 비교: top20 monthly no trailing | 13.0% | -17.3% | 1.078 | 382.9% | 76.8% | 9.0% | 67.8% | 353 | 8 | 0 |
| 현재 yoon1e: top30 daily trail5 | 7.9% | -23.4% | 0.748 | 166.9% | 72.4% | 2.6% | 69.9% | 4 | 53 | 10 |

## Holdout(Test)

| 전략 | CAGR | MDD | Sharpe | 총수익 | 평균 롱 | 평균 숏 | 평균 순노출 | 롱 거래 | 숏 거래 | 트레일링 익절 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| yoon1b 기준 | 19.3% | -22.9% | 1.340 | 872.5% | 90.8% | 0.0% | 90.8% | 412 | 0 | 0 |
| 추천: top20 monthly trail12 | 18.7% | -17.7% | 1.354 | 805.0% | 90.8% | 3.0% | 87.9% | 412 | 11 | 6 |
| 비교: top20 monthly trail8 | 19.0% | -21.0% | 1.359 | 837.4% | 90.8% | 2.2% | 88.6% | 412 | 11 | 8 |
| 비교: top20 monthly no trailing | 18.1% | -17.7% | 1.331 | 754.5% | 90.9% | 4.1% | 86.8% | 412 | 9 | 0 |
| 현재 yoon1e: top30 daily trail5 | 15.0% | -18.2% | 1.258 | 502.9% | 86.2% | 1.8% | 84.4% | 0 | 32 | 11 |

## All

| 전략 | CAGR | MDD | Sharpe | 총수익 | 평균 롱 | 평균 숏 | 평균 순노출 | 롱 거래 | 숏 거래 | 트레일링 익절 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| yoon1b 기준 | 15.7% | -32.0% | 1.144 | 1176417.3% | 84.9% | 0.0% | 84.9% | 960 | 0 | 0 |
| 추천: top20 monthly trail12 | 15.5% | -32.0% | 1.153 | 1089519.8% | 84.9% | 1.8% | 83.1% | 960 | 26 | 11 |
| 비교: top20 monthly trail8 | 15.5% | -32.0% | 1.145 | 1064466.3% | 84.9% | 1.4% | 83.5% | 960 | 27 | 18 |
| 비교: top20 monthly no trailing | 15.5% | -32.0% | 1.160 | 1074237.8% | 84.9% | 2.9% | 82.0% | 960 | 21 | 0 |
| 현재 yoon1e: top30 daily trail5 | 12.7% | -33.3% | 1.030 | 214330.1% | 82.9% | 1.1% | 81.8% | 29 | 107 | 24 |

## 숏 거래 품질

| 구간 | 전략 | 숏 평균수익 | 숏 승률 | 숏 거래 | 트레일링 익절 |
| --- | --- | ---: | ---: | ---: | ---: |
| validation | 추천: top20 monthly trail12 | 4.2% | 36.4% | 11 | 5 |
| test | 추천: top20 monthly trail12 | -2.0% | 18.2% | 11 | 6 |
| all | 추천: top20 monthly trail12 | -0.4% | 23.1% | 26 | 11 |
| validation | 비교: top20 monthly trail8 | 1.4% | 41.7% | 12 | 8 |
| test | 비교: top20 monthly trail8 | -1.0% | 45.5% | 11 | 8 |
| all | 비교: top20 monthly trail8 | -1.0% | 37.0% | 27 | 18 |
| validation | 비교: top20 monthly no trailing | 9.1% | 37.5% | 8 | 0 |
| test | 비교: top20 monthly no trailing | -5.2% | 0.0% | 9 | 0 |
| all | 비교: top20 monthly no trailing | -0.3% | 14.3% | 21 | 0 |
| validation | 현재 yoon1e: top30 daily trail5 | -0.5% | 34.0% | 53 | 10 |
| test | 현재 yoon1e: top30 daily trail5 | -0.6% | 31.2% | 32 | 11 |
| all | 현재 yoon1e: top30 daily trail5 | -0.8% | 29.0% | 107 | 24 |

## 분석

`top20/monthly/trail12`는 validation과 holdout에서 모두 yoon1b 대비 MDD와 Sharpe를 개선했다.
holdout 기준 MDD는 -22.9%에서 -17.7%로 5.2%p 개선됐고, Sharpe는 1.340에서 1.354로 소폭
올랐다. 대신 CAGR은 19.3%에서 18.7%로 0.6%p 낮아졌다.

`trail12`의 의미는 숏 헤지를 오래 끌고 가는 쪽이다. validation에서는 숏 평균수익이 +4.2%로
좋았지만, holdout에서는 -2.0%로 손실이었다. 그런데도 포트폴리오 Sharpe와 MDD가 개선된 이유는
숏 자체가 알파라기보다 약세 구간의 변동성 완충 장치로 작동했기 때문이다.

`trail8`은 다른 성격의 후보이다. holdout에서 Sharpe 1.359로 가장 높고 CAGR 손실도 더 작지만,
MDD 개선폭은 -21.0%로 trail12보다 작다. 따라서 선택 기준은 명확하다:
- MDD 방어 우선: `trail12`
- CAGR/Sharpe 균형 우선: `trail8`

`no trailing`은 validation에서는 가장 좋지만 holdout 숏 승률이 0%이고 숏 평균수익도 -5.2%라
채택하기 어렵다. 최근 구간에서 SPY가 200일선 아래로 내려간 뒤 빠르게 반등하는 패턴에 취약하다.

`top30/daily/trail5`는 폐기하는 게 맞다. top30은 랭킹 효과를 희석하고, daily는 리밸런싱 노이즈와
숏 거래 빈도만 키웠다.

## 결론

추천 파라미터 `top20/monthly/trail12`는 방어형 후보로 유효하다. 다만 yoon1b를 완전히 대체할
정도는 아니다. CAGR을 일부 포기하고 MDD를 낮추는 목적이면 채택 가능하다.

다음 후보는 `top20/monthly/trail8`이다. 이 후보는 holdout에서 Sharpe가 가장 높았고 CAGR 손실이
작다. 최종 선택은 운용 목적에 따라 나누는 것이 맞다:
- 방어형 yoon1e: `trail12`
- 균형형 yoon1e: `trail8`
- 공격형 기준: yoon1b 유지
