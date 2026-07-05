# 전략 명세: yoon2_final_fx

> `yoon2_final`의 동결 스냅샷은 유지하고, 상위 원/달러 비중 조정 레이어만
> 월간 동적으로 바꾼 검증 후보.

## 1. 식별 정보

| 항목 | 값 |
| --- | --- |
| strategy_id | yoon2_final_fx |
| 기반 전략 | yoon2_final |
| 변경 범위 | 상위 USD/KRW 목표비중 레이어만 변경 |
| 하위 슬리브 | yoon1j_lp(VGT 포함 top3) + yoon1j_kr(top3) |
| 리밸런싱 | 월간 |
| 기준 통화 | KRW, USD 슬리브는 USD/KRW 환산 |
| 핸들러 | `src/trading_lab/strategies/yoon1k.py` (`Yoon1kHandler`) |
| config | `configs/strategies/yoon2_final_fx.json` |

## 2. 통화 배분 규칙

기존 `yoon2_final`은 상위 통화 비중을 50/50으로 고정한다. 이 파생은 같은 월간
리밸런스 날짜에만 목표비중을 바꾸되, 밴드를 40~60%로 제한한다.

```
usd_base = SPY * USDKRW
krw_base = KODEX200

score =
    63d_ret(usd_base)
  - 63d_ret(krw_base)

w_USD = clip(0.5 + 0.125 * tanh(score / 0.10), 0.40, 0.60)
w_KRW = 1 - w_USD
```

신호는 1거래일 shift해서 리밸런스 당일 종가 누수를 피한다. `SPY`, `069500.KS`,
`KRW=X` 중 하나라도 없으면 기존 50/50 정적 비중으로 후퇴한다.

별도 USD/KRW 추세 가산은 초기 스윕에서 MDD를 키우는 경향이 있어 0으로 둔다. 환율
정보는 `SPY * USDKRW` 원화환산 수익률에 이미 포함된다.

## 3. 의도와 검증 기준

- 의도: 한국/미국 슬리브의 종목선정 로직을 건드리지 않고, 큰 통화·시장 상대 추세만
  상위 레이어에 반영한다.
- 기대 효과: 강한 달러/미국 우위 구간에서 USD 비중을 늘리고, 반대 구간에서는 KRW
  비중을 늘려 50/50 고정의 기회비용을 줄인다.
- 폐기 기준: `yoon2_final` 대비 MDD 방어 확률 또는 Sharpe가 유의하게 훼손되면 폐기한다.

## 4. 주의

- 아직 검증 후보이며 `live_eligible=False`다.
- 실거래 리밸런싱 스크립트가 이 동적 목표비중을 자동 계산하도록 연결되기 전까지는
  운영 전략으로 승격하지 않는다.

## 5. 하위 슬리브 처닝(churn) 방지 — `hold_buffer`

하위 슬리브(yoon1j_lp/yoon1j_kr top3)는 매 리밸런스마다 점수 순위로 엄격히
top_k를 골라, 컷오프 경계(3위/4위) 점수가 노이즈로 흔들릴 때 같은 종목이
편출 후 몇 달 안에 재편입되는 처닝이 잦았다(전체 거래의 40~45%가 3개월 내
재편입, 이 거래들의 평균 수익률은 안정 보유 거래 대비 1/6~3/4 수준). 이를 줄이려
`scripts/profile_sizing/portfolio.py`의 `_target_weights`/`simulate_portfolio`에
opt-in `hold_buffer` 파라미터를 추가했다: 신규 편입은 top_k 엄격, 이미 보유 중인
종목은 순위가 top_k+hold_buffer 안이면 유지한다. 노출 레벨(=mean top_k 점수)은
버퍼와 무관하게 항상 엄격 top_k 기준으로 계산해 레짐 사이징 자체는 불변이다.
기본값 0(비활성)이라 다른 전략에는 영향이 없다.

이 config는 양쪽 슬리브에 `hold_buffer: 2`를 적용한다. 개별 슬리브 스윕 결과
([0,1,2,3,4] 버퍼 비교, `PYTHONPATH=src:scripts python3` 임시 스크립트로 검증)
US는 buffer=2에서 거래 174→100건(-42%), 3개월내 재편입 40.8%→21.0%, CAGR
7.71%→8.35%, Sharpe 0.678→0.749로 개선. KR은 buffer=2에서 거래 141→87건
(-38%), 재편입 44.7%→31.0%, CAGR 12.52%→12.03%(소폭 하락), MDD -40.33%→
-37.55%(개선). 통화 레이어까지 합친 전체 포트폴리오 기준(버퍼 0/0 대비 2/2):

| | CAGR | MDD | Sharpe |
| --- | ---: | ---: | ---: |
| buffer 0/0(이전) | 16.46% | -18.46% | 1.187 |
| buffer 2/2(현재) | 16.27% | -17.13% | 1.189 |

MDD가 뚜렷이 개선되고 Sharpe는 사실상 동일, CAGR만 미미하게 하락한다. 회전율
감소로 인한 (모델링되지 않는) 실거래 비용·슬리피지 여유도 커진다. `reports/
profile_sizing/yoon2_final_fx_validation.md`, `yoon2_final_fx_2_overlay_report.md`가
이 새 베이스라인으로 갱신되어 있다.
