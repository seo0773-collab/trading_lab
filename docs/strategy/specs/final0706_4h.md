# 전략 명세: final0706_4h

> final0706(일봉, US+KR 2조 통화틸트+신용스프레드 게이트)을 4시간봉으로 압축한 실험 파생.
> 2026-07-06 등록, `live_eligible=false`.

## 1. 식별 정보

| 항목 | 값 |
| --- | --- |
| strategy_id | final0706_4h |
| 기반 전략 | final0706 |
| 등록일 | 2026-07-06 |
| 핸들러 | `src/trading_lab/strategies/yoon1k.py` (`Yoon1kHandler`) |
| config | `configs/strategies/final0706_4h.json` |

## 2. 배경 — 왜 4시간봉인가

final0706의 percentile 버킷 스코어(컨트래리언, base_cycle=SMA200)가 타임프레임을
낮춰도 경향성을 유지하는지 조사하는 과정에서(이 세션의 여러 스크립트,
`reports/profile_sizing/final0706_tf_robustness*.md`, `final0706_tf_cycle_sweep.md`
참고) 다음이 확인됐다:

- 5m/15m/30m: yfinance 데이터 한도(60일, 단일 레짐)로 검증 자체가 불가능.
- 1h/4h: 캐시 누적으로 2~3년 확보 가능, **SMA 길이가 몇 주~몇 달의 wall-clock
  메모리를 담아야** 컨트래리언 경향성이 재현된다(SMA를 봉 수만 많고 실제 기간이
  짧게 압축하면 신호가 사라짐).
- US 슬리브 단독으로 cycle_len(SMA)×rebalance 2D 그리드서치(`final0706_4h_optimize.md`)
  결과 **cycle=100봉(≈3개월)/rebal=42봉(≈1개월, 원본 월간과 거의 동일 실빈도)**에서
  Sharpe=1.450 — 동일 3년 구간 일봉 원본(1.380) 대비 **105%**.
- 리밸런싱 주기까지 SMA와 같은 비율로 줄이면(예: 매 1~2봉) 수수료(왕복 ~25bp)가
  고빈도 거래에 복리로 작용해 **Sharpe가 -700까지 붕괴**함을 실측
  (`final0706_period_scale_cross_tf_v2.md`) — 그래서 SMA는 압축하되 리밸런싱은
  원본과 비슷한 실빈도(월 1회 안팎)로 독립적으로 유지해야 한다.

## 3. US 단독 → US+KR 2조 확장에서 드러난 문제

US 슬리브 단독 최적치(cycle=100,rebal=42)를 원본 final0706의 US+KR 2조 구조
전체(통화틸트+신용게이트 포함)에 그대로 적용했더니 **Sharpe=0.713**으로 급락했다
(`final0706_4h_backtest.md`, 최초 시도). KR 슬리브 결합이 성과를 크게 깎는다는 뜻이라,
2조 구조 레벨에서 cycle_len×rebalance를 다시 그리드서치했다(1h/4h,
`final0706_4h_full_cross_sweep.md`).

**버그 발견 및 수정**: 최초 시도에서 `IndexError`로 크래시했다. 원인은 서로 다른
거래소의 인트라데이 봉은 실제 봉 시각이 겹치지 않기 때문 — 미국 4h봉은 UTC
13:30/17:30대(현지 개장 기준), 한국 4h봉은 UTC 00:00/04:00대(KST 개장 기준)라
`Yoon1kHandler`의 기존 교집합(intersection) 기반 슬리브 결합 인덱스가 완전히
비어버린다. `src/trading_lab/strategies/yoon1k.py`의 결합 로직을
**union(합집합) + ffill**로 일반 수정했다 — 일봉에서는 사실상 기존 동작과 동일하고
(같은 날짜면 이미 겹침), 인트라데이에서는 "그 시장이 쉬는 동안 그 슬리브 NAV는
마지막 값에서 안 움직인다"는 올바른 의미가 된다. 전체 176개 테스트로 회귀 없음 확인.

## 4. 2조 구조 그리드서치 결과 (`final0706_4h_full_cross_sweep.md`)

기준 = 그 조합의 equity가 덮는 정확히 같은 구간을 자른 일봉·무스케일 원본
final0706 Sharpe(≈2.17, 최근 강세장 구간이라 US 단독 스윕 때 기준 1.38보다 높음 —
표본 구간이 다름에 유의).

| TF | 최선 조합 | Sharpe | 기준 대비 |
| --- | --- | ---: | ---: |
| 1h | cycle=125, rebal=26 | 1.303 | 60% |
| **4h(채택)** | **cycle=100, rebal=31** | **1.129** | **52%** |

1h가 근소하게 더 나았지만 이 config는 4h로 커밋한다(1h는 `final0706_1h` 후보로 별도
검토). **어느 조합도 기준의 100%에 도달하지 못했다** — US 단독에서는 105%까지
나왔던 것과 달리, 2조 구조 전체를 인트라데이로 압축하면 아직 완전한 신호 재현에는
미치지 못한다.

## 5. 채택 파라미터 (`final0706_4h.json`)

| 파라미터 | 원본(일봉) | 4h(ratio=0.5) |
| --- | ---: | ---: |
| base_cycle.length | 200 | 100 |
| profile.rolling_window | 126 | 63 |
| market_filter/safe_sleeve.ma_len | 100 | 50 |
| regime_cap.recovery_stage_bars | 21 | 11 |
| 통화틸트 lookback | 63 | 32 |
| 신용게이트 lookback/sma_lookback | 10/20 | 5/10 |
| 리밸런싱(봉수 기준, `every_N`) | monthly(≈21거래일) | every_31(≈19거래일, 독립 재탐색값) |

`hold_buffer`(2), `min_consecutive`(2), `signal_lag`(1)은 이벤트/카운트 단위라
스케일하지 않고 원본 그대로.

## 6. 실데이터 백테스트 (`final0706_4h_backtest.md`, 최종 config 기준)

| | 값 |
| --- | --- |
| 구간 | 2024-07-08~2026-07-06 (1,957봉, 4h) — KR측 데이터가 병목이라 실사용 구간은 약 2년 |
| Sharpe | **1.121** |
| CAGR | 11.48% |
| MDD | -17.89% |
| Total Return | 68.07% |

(Sharpe/CAGR은 4h 실제 연간 거래봉수로 직접 계산 — `engine.py` 기본 `_ann_factor`는
1d/1h/1wk/1mo만 인식해 4h를 252로 잘못 폴백하므로 신뢰하지 않는다.)

## 7. 알려진 한계

- **표본이 짧다**(yfinance 4h 캐시 한도로 실사용 약 2년, 단일 강세장 위주 레짐) —
  일봉 원본의 19년·다중 사이클 검증과는 표본의 질이 다르다.
- 어느 TF/파라미터 조합도 동일 구간 일봉 기준의 100%에 도달하지 못했다 — "타임프레임을
  낮춰도 성과가 보존된다"는 확정이 아니라 "부분적으로만 재현된다"는 결론이다.
- 1h가 4h보다 근소하게 나은 결과를 보여, 이 config가 최적 TF라는 보장은 없다.
- `union+ffill` 결합 수정은 일반적이고 일봉 회귀는 없음을 확인했으나, 인트라데이
  다중거래소 결합이라는 새로운 사용 패턴 자체의 실전 검증(슬리피지·체결가정 등)은
  아직 없다.

## 8. 승격 정책

`live_eligible=false`. 승격 전 필요: ① 이 파생 자체의 out-of-sample(추가 캘린더
누적) 재검증, ② final0706 공통 게이트(페이퍼 전진검증 3개월 이상, KIS 실수수료·
환전우대 확정 — 특히 4h 빈도 리밸런싱에서 실제 체결 슬리피지가 가정한 왕복 25bp 안에
드는지 확인 필요).
