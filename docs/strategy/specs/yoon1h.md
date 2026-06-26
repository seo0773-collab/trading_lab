# 전략 명세: yoon1h — POC/VA 매물대 위치 사이징 (다종목 포트폴리오)

## 1. 아이디어

yoon1b와 **동일한 다종목 포트폴리오 엔진**(상위 K 추종 + 개별 방어 합산 현금화 +
SPY 200MA 시장필터 + regime cap + 트렌드 오버레이)을 쓰되, 개별 종목의 사이징
입력만 바꾼다.

- yoon1b: `cumulative_percentile`(누적 분포에서 현재가의 하위 누적비율) → bucket 가중.
- **yoon1h**: 같은 볼륨 프로파일에서 **POC/VAH/VAL 매물대 레벨**(heatmap1 계보)을 뽑아
  현재가의 **Value Area 대비 위치** `va_position`(0~1)을 만들고 → **동일 bucket 가중**.

percentile은 "분포상 위치"라 둔하다. VA는 *실제 거래가 쌓인 지지/저항 밴드*라
"싸다/비싸다"의 근거(POC 회귀)가 분명하다는 가설.

## 2. 레벨·위치 산출 (lookahead 없음)

각 종목·각 봉 t에서 **rolling 볼륨 프로파일**(최근 `rolling_window`봉, cycle_multiple
공간)로부터:

| 값 | 정의 |
| --- | --- |
| `poc` | volume 최대 bin 중심 multiple (Point of Control) |
| `vah` / `val` | POC에서 좌우로 *더 큰 이웃*을 흡수하며 누적 volume이 total×`va_pct`(기본 0.70)에 도달할 때까지 확장한 구간의 상·하단 edge |
| `va_position` | 현재가(`cm_close`)의 VA 대비 위치. POC=0.5, VAL=0(싸다), VAH=1(비싸다) |

`va_position`은 **연속 외삽**(설계 선택 B): VA 안은 POC 기준 상·하 반폭으로 각각
선형, VA 밖 이탈은 같은 기울기로 외삽 후 `[0,1]` 클립 → 이탈 강도까지 반영.
POC가 VA 가운데가 아닐 수 있어 상·하 반폭(`vah-poc`, `poc-val`)을 따로 정규화한다.

프로파일은 t 이하 데이터만 누적 → **룩어헤드 없음**. warmup/빈 분포는 NaN → 비중 0.

## 3. 사이징 (yoon1b와 동일)

`va_position`(0~1)을 percentile 자리에 넣어 **같은 `weight_model="bucket"`**을 통과:
position이 낮을(VAL 근처/아래=싸다)수록 비중↑. 이후 trend overlay·regime cap·
점진 rebalance·포트폴리오 노출 배분은 yoon1b 그대로.

## 4. 파라미터 (yoon1b 대비 추가분만)

| 키 | 의미 | 기본 |
| --- | --- | --- |
| `position_source` | `percentile`(기존) / `poc_va`(yoon1h) | yoon1h=`poc_va` |
| `profile.compute_va` | rolling profile에서 POC/VA 산출 on | yoon1h=true |
| `profile.va_pct` | Value Area 비율 | 0.70 |

나머지(universe, top_k, exposure_gain=1.25, market_filter, regime_cap, trend_overlay,
buckets 등)는 yoon1b와 동일.

## 5. 산출 (StrategyArtifacts)

포트폴리오 핸들러(`ProfilePortfolioHandler`)를 그대로 재사용 → forecast(포트폴리오
노출/현금/보유수)·trades·equity·metrics·portfolio_wave·perf_vs_bnh(주 벤치마크 SPY)·
top_contributors 계약 동일. 종목별 `poc/vah/val/va_position`은 종목 파이프라인의
forecast 컬럼으로 산출되며, 포트폴리오 run에서는 사이징 입력으로만 소비된다.

## 6. 검증

- 무누수: t의 POC/VA가 t 이하 rolling profile로만 산출(시프트 점검).
- `test_strategy_contract.py`가 yoon1h를 합성 데이터로 자동 포함 → 계약 검증.
- 실데이터 A/B: yoon1b(percentile) vs yoon1h(POC/VA) holdout(validation→test) 비교.
  같은 엔진·같은 유니버스라 "위치 측정법"만의 순수 비교가 된다.
