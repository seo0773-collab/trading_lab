# final0706 위험게이트: 신규 후보 신호 탐색 (2026-07-09)

사용자 질문: "신용스프레드 보면서 cash로 돌리는 국면 있는데, 또 다른 유망한 게
있는지 서치해봐"에 대한 실측 응답.

- 방법: `yoon2_final_fx_leading_signals.py`(기존 채택된 credit_spread를 VIX·추세형
  게이트와 비교했던 스크립트)와 동일한 방법론(22개 드로다운 에피소드 리드타임
  진단 + top-level 노출 0.5배 축소 오버레이 백테스트 + 페어드 부트스트랩) 재사용.
- 기간: 2007-01-29~2026-07-09, 드로다운 에피소드(-5% 이상) 22개.
- 스크립트: `scripts/profile_sizing/final0706_gate_new_signal_search.py`.
- final0706.json/yoon2_final_fx.json 등 실제 config는 건드리지 않은 순수 리서치.

## 테스트한 신규 후보

| 신호 | 근거 |
| --- | --- |
| `credit_spread_hyg_ief` | 기존 채택(비교 기준선), HYG(회사채)/IEF(국채) — 신용스프레드+듀레이션차 혼합 |
| `credit_hyg_lqd` | HYG/LQD — 둘 다 회사채(등급만 다름)라 듀레이션 오염이 적은 "순수 신용스프레드" 가설 |
| `small_cap_relative` | IWM/SPY — 소형주가 대형주보다 먼저 꺾이는 표준적 시장폭 조기경보 |
| `dollar_strength` | UUP 추세 상향 — 안전자산 선호 시 달러 강세(플라이트투퀄리티) |
| `copper_gold` | CPER/GLD — "Dr. Copper"(경기민감) vs 금(안전자산) 심리 비율 |

## Part A. 리드타임 진단 (peak 대비 오프셋, 거래일)

| 신호 | 적중 | 선행 발동 | 평균 오프셋 | 중앙값 오프셋 |
| --- | ---: | ---: | ---: | ---: |
| credit_spread_hyg_ief(기존) | 18/22 | 14 | -43.7일 | -48.0일 |
| credit_hyg_lqd | 15/22 | 12 | -33.0일 | -43.0일 |
| **small_cap_relative** | **21/22** | **21** | **-73.9일** | **-80.0일** |
| dollar_strength | 19/22 | 16 | -47.4일 | -51.0일 |
| **copper_gold** | **20/22** | **20** | **-80.9일** | **-87.0일** |

`small_cap_relative`와 `copper_gold`가 적중률·리드타임 모두 기존 신용스프레드보다
뚜렷이 우수(거의 모든 에피소드에서 발동, 30일 이상 더 일찍).

## Part B. 오버레이 백테스트 (트리거 시 top-level 노출 0.5배)

| 구성 | CAGR | MDD | Sharpe |
| --- | ---: | ---: | ---: |
| current(게이트 없음) | 13.16% | -17.13% | 1.167 |
| credit_spread_hyg_ief(기존) | 13.13% | -16.40% | 1.190 |
| credit_hyg_lqd | 13.31% | -16.40% | 1.199 |
| small_cap_relative | 12.91% | -17.13% | 1.204 |
| dollar_strength | 12.57% | -17.13% | 1.139 |
| copper_gold | 12.30% | -16.40% | 1.213 |
| **credit_or_small_cap**(기존 OR 소형주) | **12.87%** | **-11.20%** | **1.228** |

## 페어드 부트스트랩 (vs current, 5000회, block=20)

| 신호 | P(Sharpe 개선) | P(CAGR 개선) | P(MDD 더 얕음) |
| --- | ---: | ---: | ---: |
| credit_spread_hyg_ief(기존) | 69.6% | 43.6% | 72.7% |
| credit_hyg_lqd | 78.4% | 60.1% | 74.0% |
| small_cap_relative | 78.5% | 35.1% | 87.0% |
| dollar_strength | 9.7% | 1.3% | 48.2% |
| copper_gold | 79.7% | 12.0% | 87.3% |
| **credit_or_small_cap** | **83.8%** | 33.5% | **94.3%** |

## 해석

1. **`dollar_strength`(UUP)는 기각** — 모든 지표에서 current보다 나쁨(P(Sharpe>)
   9.7%). "안전자산 선호=달러강세"라는 통념이 이 시장(US+KR 배분)에서는 오탐이
   많아 실효가 없다.
2. **`credit_hyg_lqd`(순수 신용스프레드)는 기존 HYG/IEF보다 소폭 우수**
   (부트스트랩 P(Sharpe>) 78.4% vs 69.6%, P(CAGR>) 60.1% vs 43.6%) — 리드타임
   적중률은 오히려 낮지만(15/22 vs 18/22) 신호가 덜 잦은 만큼 비용(회전율)이
   줄어 순효과가 더 나은 것으로 보인다. 기존 게이트를 이걸로 교체하는 것도
   검토할 만하나 단독으로는 극적이지 않음.
3. **`small_cap_relative`(IWM/SPY)가 가장 눈에 띔** — 리드타임 median -80일
   (기존 대비 32일 더 빠름), 적중 21/22(거의 모든 에피소드). 단독 게이트로는
   MDD 포인트추정 개선이 없지만(-17.13%로 동일 — 특정 한 에피소드가 off_scale=0.5로
   못 막힘) 부트스트랩상 MDD 방어는 뚜렷(P(더 얕음) 87.0%).
4. **`copper_gold`도 강력한 리드타임**(median -87일, 20/22)이지만 CAGR 희생이
   큼(P(CAGR>) 12.0%) — 신호가 너무 자주/오래 켜져 과잉방어(보험료 과다) 성향.
5. **최유망: `credit_or_small_cap`(기존 신용스프레드 OR 소형주 상대강도)** —
   두 신호를 OR로 묶으면(하나라도 발동하면 노출 축소) MDD가 -17.13%→**-11.20%**로
   뚜렷이 개선되고 Sharpe도 전체 후보 중 최고(1.228), 부트스트랩 P(MDD 더 얕음)
   94.3%·P(Sharpe 개선) 83.8%로 가장 강건. CAGR은 다소 양보(P(CAGR>) 33.5%,
   current 대비 -0.29%p)하지만 이건 기존 credit_spread_hyg_ief 단독(43.6%)보다도
   나은 방어-수익 균형이다. **참고로 이전 연구(`yoon2_final_fx_leading_signals.md`)의
   `vix_or_credit_gate`(P(Sharpe>) 33.8%, P(MDD얕음) 78.8%)보다 확실히 우수** —
   VIX 대신 소형주 상대강도를 보완신호로 쓰는 게 더 낫다.

## 결론 및 제안

`credit_or_small_cap`이 지금까지 이 랩이 시도한 모든 게이트 변형(추세형, VIX,
VIX AND 신용, VIX OR 신용) 중 MDD 방어와 Sharpe 양쪽에서 가장 강력하다. 다만:

- 아직 final0706의 실제 리스크게이트 코드(`_relative_momentum_fx2_weights`,
  `source in ("credit_spread","credit_and_vix")`만 지원)에는 OR 결합 소스가
  구현돼 있지 않다 — 채택하려면 `source="credit_or_small_cap"`(가칭) 분기 추가
  또는 범용 다중신호 OR/AND 결합으로 게이트 로직을 일반화해야 한다.
  이는 final0706 자체가 아니라 파생(`final0706_gate2`류)으로 먼저 검증하는 게
  이 랩의 관례(config 동결 정책)에 맞다.
- 이 결과는 point-estimate 1회 + 부트스트랩 검증까지만 거쳤다 — 롤링
  워크포워드(과최적화 여부)와 회전율/비용 민감도(신호가 더 잦아지면 왕복비용이
  커질 수 있음, `yoon2_final_fx_credit_gate_regrid_cost.md`가 기존 게이트에서
  이 문제를 다룬 선례)는 아직 검증하지 않았다 — 채택 전 필수.

## 파일

- JSON: `reports/profile_sizing/final0706_gate_new_signal_search.json`
- 스크립트: `scripts/profile_sizing/final0706_gate_new_signal_search.py`
