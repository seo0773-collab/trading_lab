# 전략 명세: yoon2c

> `yoon2`의 하위 슬리브는 그대로 두고, 상위 원/달러 통화 배분만
> World Bank 버핏지수 변화율 기준으로 동적 조절하는 파생 전략.

## 1. 식별 정보

| 항목 | 값 |
| --- | --- |
| strategy_id | yoon2c |
| 기반 전략 | yoon2 |
| 구조 | 미국 yoon1j_lp + 한국 yoon1j_kr |
| 동적 배분 | 버핏지수 변화율 revert scaled |
| 채택값 | k=1.25, KRW 비중 밴드 30~70% |
| 리밸런싱 | 월간 |
| 기준 통화 | KRW, USD 슬리브는 USD/KRW 환산 |
| 핸들러 | src/trading_lab/strategies/yoon1k.py (`Yoon1kHandler`) |
| config | configs/strategies/yoon2c.json |

## 2. 통화 배분 규칙

버핏지수는 World Bank `CM.MKT.LCAP.GD.ZS`
(`Stock market capitalization to GDP (%)`)를 사용한다.

무누수 처리는 보수적으로 둔다. 관측연도 `Y`의 값과 전년 대비 변화율은 다음 해
1월 1일부터 사용할 수 있다고 본다. 예를 들어 2024년 값은 2025-01-01 이후 월간
리밸런싱부터 쓴다.

```
US_growth = US_Buffett_t / US_Buffett_(t-1)
KR_growth = KR_Buffett_t / KR_Buffett_(t-1)

raw_w_KRW = US_growth / (KR_growth + US_growth)
w_KRW = clip(0.5 + 1.25 * (raw_w_KRW - 0.5), 0.30, 0.70)
w_USD = 1 - w_KRW
```

해석: 한국 버핏지수가 미국보다 더 빨리 오르면 한국 비중을 줄이고, 미국 버핏지수가
더 빨리 오르면 한국 비중을 늘린다. 즉 레벨이 아니라 "비싸지는 속도"에 대한
평균회귀형 통화/시장 배분이다.

## 3. 채택 근거

`reports/profile_sizing/yoon2c_buffett_change_scaled_sweep.md`에서 `k`를 스윕했다.

| k | CAGR | MDD | Sharpe | TotalRet |
| ---: | ---: | ---: | ---: | ---: |
| 0.00, yoon2 50/50 | 14.10% | -14.46% | 1.151 | 747.18% |
| 0.50 | 14.24% | -14.46% | 1.145 | 763.17% |
| 1.00 | 14.36% | -14.46% | 1.135 | 778.75% |
| **1.25, 채택** | **14.42%** | **-14.46%** | **1.129** | **786.31%** |
| 1.50 | 14.48% | -14.46% | 1.122 | 793.78% |

`k`를 키울수록 CAGR/총수익은 올라가지만 Sharpe가 점진적으로 낮아진다. yoon2c는
Sharpe 최우선 전략이 아니라, yoon2의 MDD를 거의 유지하면서 CAGR/총수익을 개선하는
후보로 정의한다. 이 기준에서 `k=1.25`를 채택했다. `k=1.50`은 수익률은 더 높지만
Sharpe 하락과 test 구간 MDD 악화가 더 커 균형 후보에서 제외했다.

## 4. 구조

```
[상위] yoon2c — 월간 동적 통화 리밸런싱
   ├─ USD 슬리브 → yoon1j_lp
   └─ KRW 슬리브 → yoon1j_kr
```

하위 슬리브의 종목 선정/방어 로직은 yoon2와 동일하다. yoon2c가 바꾸는 것은 월간
상위 슬리브 목표비중뿐이다.

## 5. 주의

- World Bank 연간 데이터라 신호 갱신 빈도는 낮다. 2025년 이후 구간은 최신 확정치가
  나올 때까지 2024년 관측치를 forward-fill한다.
- 실거래/live 레이어는 아직 동적 통화비중을 자동 계산하지 않는다. 따라서
  `live_eligible=False`를 유지한다.
- Sharpe는 yoon2보다 낮다. 목표가 "위험조정 최강"이면 yoon2가 기준이고,
  목표가 "MDD 유사 + CAGR 개선"이면 yoon2c가 후보가 된다.
