# 전략 명세: profile-sizing-v1

> Cumulative Profile Percentile로 현재 가격 위치를 판단하고, 국면(regime)별 비중
> 상한과 점진 rebalance로 **목표 주식 비중**을 관리하는 long-only 사이징 전략.
> 가격 예측 모델이 아니라 "저가권 매수 · 하락장 생존 · 단계적 회복" 포지션 관리기다.
> 근거 계획: `profile_plan.txt`.

## 1. 식별 정보

| 항목 | 값 |
| --- | --- |
| strategy_id | profile-sizing-v1 |
| version | 1 |
| 설명 | profile percentile 기반 국면별 목표비중 사이징(long-only) |
| 작성일 | 2026-06-16 |
| 상태 | validation (핸들러 등록·계약/단위 테스트 통과, 30종목 B&H 배치 완료) |
| config 경로 | configs/strategies/profile_sizing_v1.json |
| 결과 단위 | 단일 종목 (1 run = 1 종목). 유니버스 비교는 scripts/profile_sizing/batch.py |

## 2. 가설과 엣지

- **시장 가설**: 장기 누적 거래분포(cumulative profile) 안에서 현재 가격이 차지하는
  percentile은 "싸다/비싸다"의 위치 신호다. 싼 위치(낮은 percentile)일수록 기대수익이
  높으므로 비중을 늘리고, 비싼 위치일수록 줄인다(평균회귀형 사이징).
- **하락장 생존**: 최근 분포 중심(rolling_mid_50)이 장기 분포 중심(cumulative_mid_50)
  아래로 내려가고 가격이 base_cycle 아래면 DEFENSE로 보고 **저가권이어도 물타기 금지**,
  보유 비중을 cap까지 축소한다. 회복이 확인되면 단계적으로만 비중을 복구한다.
- **엣지의 성격**: 이 전략의 메리트는 절대 수익 극대화가 아니라 **낙폭(drawdown) 축소와
  생존**이다. 따라서 buy & hold 대비 평가는 총수익률뿐 아니라 MDD·Sharpe를 함께 본다.
- **무효화 조건** (*결과 보기 전 고정*): validation에서 (1) MDD 개선이 없고
  (2) Sharpe가 B&H보다 유의하게 낮으면 사이징 규칙(버킷/cap)을 재설계한다.

## 3. 신호 정의 (profile_plan §3~§9)

- **base_cycle** = MA(close, length, type) × scale. type ∈ {SMA, EMA, RMA, WMA, VWMA}.
- **cycle_multiple_x** = x / base_cycle. base_cycle가 NaN/≤0이면 그 봉은 profile 제외.
- **profile**: cycle_multiple을 [min_mult, max_mult] bin에 누적. rolling(최근 N봉) /
  cumulative(시작~현재) 두 종류. weight_mode ∈ {time, volume, volume_fallback},
  accumulation_mode ∈ {range_uniform, ohlc, range_close}.
- **percentile**: cumulative_percentile(현재 cm_close의 하위 누적비율),
  cumulative/rolling_mid_50(weighted median), cumulative_lower/upper_percentile.
- **regime**: NORMAL / CAUTION / DEFENSE / RECOVERY (상태기계, 무누수).
- **base_target_weight**: weight_model ∈ {bucket, exponential}. bucket은 percentile
  구간별 고정 비중(저가권↑). exponential은 w = max_w·exp(−k·percentile).
- **regime_cap**: 국면별 최대 비중(NORMAL 1.0 / CAUTION 0.6 / DEFENSE 0.3 / RECOVERY 0.5,
  회복 지속 봉수로 0.5→0.7→1.0 단계 상향).
- **final_target_weight** = min(base, cap). DEFENSE면 min(현재비중, cap)으로 증액 금지.

## 4. 실행 규칙 (profile_plan §10~§12)

| 항목 | 값 |
| --- | --- |
| 비중 조정 | rebalance threshold(0.03) 미만 차이는 무거래, 봉당 최대 변화 0.20 |
| 방어장 | DEFENSE에서 신규 매수(증액) 금지(defense_buy_allowed=false) |
| 무누수 | 봉 t에서 결정한 비중을 t+1 수익률에 적용(weight 1봉 지연) |
| 계좌 평가 | equity = 비중×자산수익률 누적(평가자산). buy & hold는 상시 100% 보유 |
| 거래 매핑 | 비중 증감을 FIFO lot으로 분해 → 대시보드 trades(진입/청산/순수익) |
| 청산 사유 | rebalance(목표 축소) / defense_cut(방어 축소) / end_of_data |

## 5. 비용 모델

| 항목 | 값 |
| --- | --- |
| 수수료 | fee_bps_per_side (기본 5bp) |
| 슬리피지 | slippage_bps (기본 5bp) |
| 적용 | 비중이 바뀐 봉의 회전율(|Δw|)에 비례 차감 |

## 6. 검증 설계

| 항목 | 값 |
| --- | --- |
| 분할 | train(0.6)/validation(0.2)/test(0.2) 시간순. phase로 슬라이스 |
| 파라미터 적합 | **없음**(규칙 기반·고정 config). 따라서 과적합 우려가 없어 phase=all 비교가 유효 |
| 무누수 검증 | cumulative profile은 과거·현재만 사용(test_no_lookahead_cumulative_profile) |
| 벤치마크 | 동일 구간 buy & hold (총수익률·CAGR·Sharpe·MDD) |

## 7. 합격선 (결과 확인 전 고정)

| 지표 | 합격선 |
| --- | --- |
| MDD 개선 | 전략 MDD가 B&H보다 얕은 종목 비율 ≥ 70% |
| Sharpe | B&H 평균 대비 −0.15 이내 |
| 총수익률 | (참고) B&H 대비 열위 허용 — 본 전략은 낙폭 축소가 목적 |

## 8. 산출물과 등록

- [x] `scripts/profile_sizing/` 모듈(config·indicators·profile·regime·sizing·engine·synthetic·run·batch)
- [x] `configs/strategies/profile_sizing_v1.json`
- [x] `src/trading_lab/strategies/profile_sizing.py` 핸들러 (StrategyArtifacts 매핑)
- [x] `registry.py` 등록(enabled=True), `presentation.EXIT_REASON_LABELS`에 defense_cut 추가
- [x] `tests/test_profile_sizing.py` + 계약 테스트 + 전체 90개 통과
- [x] 30종목 B&H 배치: `reports/profile_sizing/perf_batch.md`
- [ ] 사이징 규칙 튜닝(익스포저↑ 또는 trend 필터) 후 Sharpe·총수익 재평가
