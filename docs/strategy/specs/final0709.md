# 전략 명세: final0709

> `final0706`과 구조가 완전히 동일한 파생. 유일한 차이는 위험게이트 신호 —
> 신용스프레드(HYG/IEF) 단독에서 **신용스프레드 OR 소형주 상대강도(IWM/SPY)**
> 결합으로 확장했다.

## 1. 식별 정보

| 항목 | 값 |
| --- | --- |
| strategy_id | final0709 |
| 기반 | [[final0706]] (US yoon1j_lp+KR yoon1j_kr, 63일 통화틸트, safe_sleeve.profile_scored) — 구조 100% 동일 |
| 유일한 차이 | `currency_allocation.risk_gate.source`: `credit_spread` → `credit_or_small_cap` |
| config | `configs/strategies/final0709.json` |
| 핸들러 | `Yoon1kHandler`(`src/trading_lab/strategies/yoon1k.py`) — 신규 게이트 소스 `credit_or_small_cap`만 추가, 기존 `credit_spread`/`credit_and_vix`/`trend` 경로는 불변 |

## 2. 배경 — 사용자 질문에서 시작

사용자: "무슨 스프레드 보면서 cash로 돌리는 국면 있잖아. 또 다른 유망한 게
있는지 서치해봐." final0706의 신용스프레드 게이트는 이미 VIX 백워데이션·
추세확인형 게이트와 비교돼 채택된 상태였다(`yoon2_final_fx_leading_signals.md`).
이 질문에 답하기 위해 **아직 검토 안 된 후보 4개**(HYG/LQD 순수신용스프레드,
IWM/SPY 소형주 상대강도, UUP 달러강세, CPER/GLD 구리금비율)를 같은 방법론
(22개 드로다운 에피소드 리드타임 진단 + 오버레이 백테스트 + 페어드 부트스트랩)
으로 실측했다(`scripts/profile_sizing/final0706_gate_new_signal_search.py`,
`reports/profile_sizing/final0706_gate_new_signal_search.md`).

## 3. 리서치 요약 (진입점)

| 신호 | 리드타임 중앙값 | 적중 | 부트스트랩 P(Sharpe개선) |
| --- | ---: | ---: | ---: |
| credit_spread(HYG/IEF, 기존) | -48일 | 18/22 | 69.6% |
| dollar_strength(UUP) | -51일 | 19/22 | **9.7%(기각)** |
| credit_hyg_lqd | -43일 | 15/22 | 78.4% |
| small_cap_relative(IWM/SPY) | **-80일** | **21/22** | 78.5% |
| copper_gold(CPER/GLD) | -87일 | 20/22 | 79.7%(CAGR 희생 큼) |
| **credit_or_small_cap** | - | - | **83.8%** |

`credit_or_small_cap`(기존 신용스프레드 OR 소형주 상대강도)이 오버레이
point-estimate에서 MDD -17.13%→-11.20%, Sharpe 1.167→1.228로 가장 우수했고,
이전에 기각됐던 `vix_or_credit_gate`(P(Sharpe>) 33.8%, P(MDD얕음) 78.8%,
`yoon2_final_fx_leading_signals.md`)보다도 확실히 나았다 — 이 결과를 실제
config로 옮겨 공식 파이프라인(`Yoon1kHandler.build_artifacts`)으로 재검증한
것이 이 전략이다.

## 4. 게이트 구현

`_relative_momentum_fx2_weights`(`yoon1k.py`)에 `source == "credit_or_small_cap"`
분기 추가:

```python
credit_weak = _credit_spread_severity(raw, idx, gate) > 0.0        # 기존 로직 재사용
small_cap_weak = _ratio_weak(raw, idx, small_cap_symbols,           # 신규 범용 헬퍼
                              lookback, sma_lookback, threshold)
weak = credit_weak | small_cap_weak
```

- `_ratio_weak`는 `_credit_spread_severity`의 이진판(강도 없이 True/False만)을
  범용화한 헬퍼 — 신용스프레드 전용 로직을 건드리지 않고 별도 추가.
- config의 `small_cap_symbols`(기본 `["IWM","SPY"]`), `small_cap_gate`(lookback/
  sma_lookback/threshold 개별 오버라이드, 미지정 시 신용스프레드와 동일 파라미터
  공유)로 제어.
- `severity`(연속형 강도)는 계산하지 않음 — 리서치에서 검증한 것과 동일하게
  이진 발동+off_scale=0.5 고정 방식만 채택(continuous 블록은 미사용).
- min_consecutive=2(기존과 동일 노이즈필터), safe_asset=IEF(기존과 동일) 유지.

## 5. 실측 (공식 파이프라인, `scripts/profile_sizing/final0709_validate.py`)

권위구간(2011-01~, `Yoon1kHandler.build_artifacts` equity 그대로):

| 후보 | CAGR | MDD | Sharpe |
| --- | ---: | ---: | ---: |
| final0706 | 14.04% | -16.15% | 1.148 |
| **final0709** | **14.38%** | **-11.15%** | **1.229** |

MDD가 31% 상대개선(-16.15%→-11.15%)되면서 Sharpe·CAGR도 동시에 소폭 개선 —
뚜렷한 트레이드오프가 거의 없다. 2025~ 랠리 구간·holdout에서는 CAGR이 소폭
낮음(발동빈도 증가에 따른 통상적 방어형 보험료).

페어드 부트스트랩(권위구간, 8000회, block=20):

| vs | P(Sharpe 개선) | P(CAGR 개선) | P(MDD 더 얕음) |
| --- | ---: | ---: | ---: |
| final0706 | 89.6% | 64.7% | 88.8% |
| SPY_KRW_bnh | 96.2% | 32.1% | 97.0% |
| QQQ_KRW_bnh | 90.0% | 6.5% | 99.2% |

final0706 대비 P(Sharpe>) 89.6%·P(MDD얕음) 88.8%는 final0706 자신의 원 게이트
채택 근거(P(Sharpe>) 88%·P(MDD얕음) 75%, `yoon2_final_fx_leading_signals.md`)
보다도 강하다. 상세: `reports/profile_sizing/final0709_validate.md`.

## 6. 롤링 3년 안정성 (16개 윈도우, 2009~2026)

**final0709가 이긴 횟수: Sharpe 13/16, CAGR 12/16, MDD 더 얕음 15/16.**

MDD 개선이 15/16 윈도우에 걸쳐 있어, final0706 원 게이트가 겪었던 "개선분이
2020-02 코로나 1건에 집중" 같은 표본 편중이 이 파생에서는 나타나지 않는다.

## 7. 회전율/비용 민감도

게이트 발동빈도: credit_only(기존) 294일/71에피소드 → credit_or_small_cap(신규)
677일/156에피소드로 2배 이상 증가(OR 결합이라 소형주 신호가 자주 단독 발동).

왕복비용별 Sharpe(manual context, `yoon2_final_fx_credit_gate_regrid_cost.py`
방법론 재사용):

| 비용 | final0706 Sharpe | final0709 Sharpe |
| --- | ---: | ---: |
| 0bp | 1.220 | 1.295 |
| 10bp | 1.206 | 1.264 |
| 25bp(이 랩의 실제 가정) | 1.185 | **1.217** |
| 50bp(공격적 가정) | 1.149 | 1.138 |

25bp(KIS 실조건 기준 가정 비용)에서도 final0709가 여전히 우위 — 발동빈도가
2배 이상 늘어도 비용에 잡아먹히지 않는다. 50bp라는 공격적 가정에서만 거의
동률로 수렴하지만 그 지점에서도 MDD는 final0709가 훨씬 얕다(-12.40% vs -16.50%).

## 8. 알려진 한계

- ~~위기 표본 편중~~ — §6 롤링 3년 검증(16개 윈도우 중 MDD 15/16 우위)으로
  final0706 원 게이트가 겪던 "개선이 코로나 1건에 집중" 패턴은 이 파생에서는
  재현되지 않음을 확인. 다만 강세장(2025~) CAGR 희생은 구조적(발동빈도 2배↑에
  따른 보험료)이라 계속 관찰 필요.
- IWM(러셀2000 ETF)은 2000년 상장이라 표본 제약은 없으나, 소형주 상대강도
  신호가 최근 시장구조(빅테크 쏠림 등)에서도 여전히 유효한지는 별도 하위표본
  검증이 필요(아직 미실시).
- 비용 민감도는 manual context(빠른 재계산용, `regrid_cost.py` 방법론)로
  검증했고 공식 파이프라인으로 직접 재현하지는 않았음 — 25bp까지는 우위
  유지를 확인했으나 정밀치는 근사.
- live_eligible=false — final0706 자체도 아직 페이퍼 검증 전이라 이 파생은
  그 이전 단계.

## 9. 승격 정책

`final0706`이 승격되기 전까지는 `live_eligible=false` 유지. **공식 파이프라인
point-estimate·부트스트랩·롤링 3년·비용 민감도 4가지 검증을 모두 통과**(§5~7,
2026-07-09) — `final0706_safe_scored`가 base로 승격됐던 선례와 비교해도 개선폭이
더 크고 일관적이라 **차기 base 후보로 유력**하다. 다만 즉시 승격하지 않고
final0706과 함께 관찰을 이어간다: final0706이 아직 페이퍼 전진검증 중(판정
2026-10-07+)이므로, 그 판정과 함께 final0709로의 교체 여부를 사용자가
결정할 시점에 재논의한다.
