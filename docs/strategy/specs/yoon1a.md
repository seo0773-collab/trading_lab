# 전략 명세: yoon1a

> `yoon2` 방어형 코어에 USD(2x Semiconductors) 레버리지 ETF 슬리브를 상위 계층에서 혼합한 공격형 파생 전략.
> 목표는 50/50 패시브 B&H와 비슷한 MDD를 감수하면서 CAGR을 크게 높이는 것이다.

## 1. 구조

| 항목 | 값 |
| --- | --- |
| strategy_id | yoon1a |
| 코어 | yoon2 65% |
| 레버리지 슬리브 | USD(2x Semiconductors) 35% |
| 리밸런싱 | 월간 |
| 기준 통화 | KRW, USD 슬리브는 USD/KRW 환산 |
| 벤치마크 | 50/50 SPY원화+KODEX200 |

## 2. 탐색 근거

`reports/profile_sizing/yoon2_leverage_etf_sweep.md` 기준, `yoon2 65% + USD 35%`는
Near 조건 최고 CAGR 후보였다.

| 조합 | CAGR | MDD | B&H CAGR | B&H MDD | Sharpe |
| --- | ---: | ---: | ---: | ---: | ---: |
| yoon2 65% + USD 35% | 26.93% | -32.38% | 16.68% | -31.35% | 1.04 |

공통기간 적립식 비교(2010-03-11~2026-06-26, 초기 200만원 + 월 20만원)에서는
최종평가액 739,890,100원, 계좌 MDD -31.69%였다.

## 3. 주의

- USD는 반도체 2x 일일 리셋 ETF라 섹터 집중도와 변동성 손실이 크다.
- Near 조건 후보라 MDD가 50/50 B&H보다 약 1%p 깊어질 수 있다.
- 환전비용, 세금, 실거래 정수주 제약은 별도 운영 레이어에서 확인해야 한다.
