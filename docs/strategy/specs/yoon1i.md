# 전략 명세: yoon1i — HVN 지지/저항 기대값 게이트 (다종목 포트폴리오)

## 1. 아이디어

yoon1b와 동일한 포트폴리오 엔진을 쓰되, 종목 점수(매수 비중)에 **heatmap2의 HVN
지지/저항 기대값**을 곱한다(블렌드). yoon1h(percentile→VA위치 *교체*)와 달리,
percentile 점수를 유지하고 지지/저항 기대값으로 *미세조정*한다(yoon3 게이트와 동형).

가설: 가격이 매물대 지지 근처면 상방여지가 커 매수 기대↑, 저항 근처면 상방이 막혀
매도 기대(=비중 억제)↑. 단일 VA가 아니라 현재가에 가장 가까운 HVN을 본다.

## 2. 게이트 산출 (lookahead 없음)

heatmap2 `rolling_sr_levels`로 각 t에서 현재가 인접 지지(`val`)/저항(`vah`)을 구하고:

    EV = (저항 − 종가) / (저항 − 지지)  ∈ [0, 1]
    gate = g_min + (1 − g_min) × EV      ∈ [g_min, 1]
    score' = score × gate

- 지지 근처 → EV→1 → gate→1(비중 유지, 매수 기대)
- 저항 근처 → EV→0 → gate→g_min(비중 억제, 매도 기대)
- 한쪽 노드만: 위 막힘 없음=상방 무제한(EV=1), 아래 받침 없음(EV=0)
- warmup·노드 부재 → gate=1.0(불변)

무누수: `rolling_sr_levels`는 t 이하만 사용 + 엔진 `simulate_portfolio`가 점수 shift(1).

## 3. 파라미터 (yoon1b 대비 추가분 `sr_gate` 블록)

| 키 | 의미 | 기본 |
| --- | --- | --- |
| `enabled` | 게이트 on | true(yoon1i) |
| `g_min` | 게이트 하한(저항 근처 최소 배수) | 0.5 |
| `lookback` | 프로파일 윈도우(봉) | 120 |
| `profile_bins` | 가격 bin 수 | 80 |
| `price_scale` | log / linear | log |
| `node_top_n` / `node_min_strength` / `node_min_gap_bins` | HVN 추출 | 4 / 0.3 / 3 |
| `va_pct` | POC 폴백용 VA 비율 | 0.70 |

나머지는 yoon1b와 동일. `sr_gate.enabled=false`면 yoon1b와 완전 동일(additive).

## 4. 산출 (StrategyArtifacts)

`ProfilePortfolioHandler` 공유 → forecast/trades/equity/metrics 및 portfolio_wave·
perf_vs_bnh(SPY)·top_contributors 계약 동일. 게이트는 compute_universe에서 점수에만
곱해진다.

## 5. 검증

- `test_strategy_contract.py` 자동 포함.
- 실데이터 A/B: yoon1b vs yoon1i(g_min 0.3/0.5/0.7) val/test/all. 규율: holdout(test)
  Sharpe ≥ yoon1b. 위험: HVN/percentile이 같은 프로파일에서 나와 정보가 겹칠 수 있음
  (yoon1h 기각·yoon3 방어다이얼 전례). 블렌드+비대칭 기대값이 차별점.
