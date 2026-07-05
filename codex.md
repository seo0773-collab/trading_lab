목표:
TradingView Pine v6 지표의 Rolling Profile 개념을 Python 전략으로 옮기고 싶다. rolling window를 63 bars로 설정했을 때, 매 bar마다 profile 분포가 어떻게 증가/감소하는지를 이용해 smooth profile distribution을 만들고, 현재 가격이 그 분포의 상단/하단 어디에 위치하는지로 매수/매도 기대값을 추정하는 전략을 설계해줘.

현재 Pine 지표 핵심 구조:
- baseCalcTf 기준으로 계산한다. 기본은 D.
- cycle = smooth(close, baseLen, baseType) * baseScale
- 각 가격은 raw price가 아니라 price / cycle 로 normalize한 multiple 공간에서 계산한다.
- profileMinMult ~ profileMaxMult 범위를 profileBinCount개 bin으로 나눈다.
- 각 bar의 contribution을 profile histogram에 누적한다.
- rolling update는 다음 구조다:

    H_t = H_{t-1} + contribution(current_bar) - contribution(bar_63_bars_ago)

- 즉 rollingProfileBars = 63이면, 최신 63개 base bar의 profile만 유지한다.
- profile accumulation modes:
    1. Range Uniform: low~high 구간을 겹치는 bin에 균등 분배
    2. OHLC: open/high/low/close에 각각 25% 분배
    3. Range + Close Weight: range 분배에 close bin 가중치를 추가
- weight modes:
    1. Time: 모든 bar weight = 1
    2. Volume: volume 사용
    3. Volume With 1 Fallback: volume이 있으면 volume, 없으면 1

전략화 아이디어:
1. Python에서 각 bar를 contribution vector로 변환한다.
2. 최근 63개 contribution vector를 deque에 보관한다.
3. 매 bar마다 새 contribution을 더하고, 63개 초과 시 가장 오래된 contribution을 뺀다.
4. 이 rolling histogram을 local kernel smoothing 또는 KDE로 매끄러운 분포로 근사한다. 전역 대칭 Gaussian 분포를 fit하지 않는다.
5. smooth profile의 CDF를 계산한다.
6. 현재 close multiple이 profile에서 몇 percentile에 위치하는지 계산한다.

기본 점수화:
- m_t = close_t / cycle_t
- F_t = smooth_profile CDF at m_t
- mean reversion score:

    score = 2 * (0.5 - F_t)

예:
- F_t = 0.05 -> score ≈ +0.90, 강한 매수 후보
- F_t = 0.50 -> score ≈ 0.00, 중립
- F_t = 0.95 -> score ≈ -0.90, 강한 매도 후보

주의점:
- profile 상단이라고 무조건 매도하면 안 된다.
- 상단 위치는 “과포화/되돌림”일 수도 있지만, “breakout/acceptance”일 수도 있다.
- 따라서 profile 위치 점수는 raw alpha가 아니라 상태 변수로 보고, 과거 데이터에서 미래수익률로 기대값을 보정해야 한다.

검증 방식:
- 각 bar에서 profile score를 계산한다.
- 미래 수익률을 여러 horizon으로 측정한다:

    H = 1, 3, 5, 10, 20 bars

- score 구간별로 다음을 계산한다:
    - mean future return
    - win rate
    - avg win
    - avg loss
    - expectancy = P(win) * avg_win - P(loss) * avg_loss - trading_cost
    - MFE / MAE
    - sample count

필요 feature 후보:
- close_percentile: 현재 close의 profile CDF percentile
- candle_percentile: OHLC 또는 range 평균 percentile
- density_at_price: 현재 가격 위치의 smooth profile density
- POC distance: 현재가와 profile peak 간 거리
- lower/upper percentile band
- profile_width: upper - lower
- profile_skew: 상단 질량 / 하단 질량
- profile_delta: H_t - H_{t-1}
- positive_delta_mass / negative_delta_mass
- profile stability: 최근 profile 변화량이 작은지
- JS divergence 또는 cosine distance로 profile 변화 강도 측정

라이브 tick 처리 주의:
- 현재 진행 중인 봉을 tick마다 hist에 계속 더하면 중복 누적된다.
- live에서는 closed_hist와 current live contribution을 분리해야 한다.

구조:
    closed_hist = 확정된 최근 봉들의 rolling histogram
    live_contribution = 현재 진행 중인 봉의 임시 contribution
    live_profile = closed_hist + live_contribution

봉이 닫힐 때만 live_contribution을 deque에 확정 저장하고, 63개 초과분을 제거한다.

요청:
- 위 아이디어를 Python 전략/백테스트 구조로 설계해줘.
- 가능하면 모듈 구조를 제안해줘.
- contribution 계산, rolling histogram, smoothing, CDF percentile score, expectancy table, walk-forward 검증까지 구현 가능한 형태로 정리해줘.
- 전략 신호는 단순 threshold가 아니라, score 구간별 historical expectancy 기반으로 판단하는 방식이 좋다.

중요: “profile delta를 보고 분포가 이동했다”고 해석하기보다는, bar별 contribution의 유입/유출로 해석하고, 최종 매매 판단은 smooth profile percentile과 과거 미래수익률 검증으로 기대값화해줘.

---

우리 환경 적용안:

## 1. 전략 이름과 위치

- strategy_id: `rolling-profile-expectancy-v1`
- 목적: Pine의 63봉 rolling profile을 multiple 공간의 smooth 근사분포로 만들고, 그 분포 위에서 현재 multiple-cycle 캔들(OHLC/range/close)이 차지하는 위치가 과거에 어떤 미래수익률 기대값을 만들었는지 학습해 long-only 목표비중을 정한다.
- 기존 전략과의 관계:
    - `profile-sizing-v1`은 cumulative percentile 기반 생존/사이징 전략이다.
    - 이번 전략은 rolling smooth distribution + multiple-cycle candle location + expectancy table 기반 예측/사이징 전략이다.
    - 기존 `profile-sizing-v1` 성과와 계약을 보존하기 위해 별도 config/spec/handler로 시작한다.

핵심 목표를 수식으로 쓰면 다음과 같다.

```text
raw OHLCV
  -> base_cycle_t
  -> multiple candle C_t = (open/cycle, high/cycle, low/cycle, close/cycle)
  -> recent 63-bar contribution histogram H_t
  -> smooth distribution P_t(m), CDF F_t(m)
  -> candle location features L_t = location(C_t on P_t, F_t)
  -> E[r_{t+h} | bucket(L_t)]
  -> target_weight_t
```

따라서 `H_t - H_{t-1}` 자체가 매수/매도 신호가 아니다. 변화량은 분포 안정성/상태 필터고, 직접 기대값을 만드는 입력은 `P_t` 위의 캔들 위치다.

권장 파일 배치:

```text
scripts/profile_sizing/profile.py
    기존 contribution matrix와 rolling profile 계산을 확장한다.

scripts/profile_sizing/rolling_expectancy.py
    smoothing, rolling CDF percentile, feature table, forward return bucket expectancy,
    walk-forward expectancy fitting을 담당한다.

src/trading_lab/strategies/rolling_profile_expectancy.py
    StrategyHandler. raw OHLCV를 받아 feature/expectancy/weight/equity/trades를
    StrategyArtifacts로 변환한다.

configs/strategies/rolling_profile_expectancy_v1.json
    Pine 입력값과 검증 파라미터를 저장한다.

docs/strategy/specs/rolling-profile-expectancy-v1.md
docs/strategy/checklists/rolling-profile-expectancy-v1.md
    Gate 0~5 문서.
```

## 2. 현재 코드에서 재사용할 것

이미 구현된 재사용 지점:

- `scripts/profile_sizing/indicators.py::compute_cycle`
    - base cycle SMA/EMA/RMA/WMA/VWMA, scale, OHLC multiple 계산.
- `scripts/profile_sizing/profile.py`
    - `_bar_weights`: Time / Volume / Volume With 1 Fallback 대응.
    - `_contrib_matrix`: Range Uniform / OHLC / Range + Close 비슷한 구조 대응.
    - rolling profile 계산: `cum_profile[t] - cum_profile[t-window]`.
- `scripts/profile_sizing/engine.py`
    - weight 1봉 지연, 거래비용, equity, FIFO lot trades.
- `src/trading_lab/strategies/profile_sizing.py`
    - StrategyArtifacts 매핑 패턴.

따라서 첫 구현은 새 엔진을 처음부터 만들지 말고, 위 모듈을 확장한다.

## 3. Pine 입력값의 우리 config 매핑

기본값은 Pine 원본보다는 이번 요청의 63 bars 실험을 우선한다.

```json
{
  "version": "rolling-profile-expectancy-v1",
  "interval": "1d",
  "period": "max",
  "base_cycle": {
    "type": "SMA",
    "length": 200,
    "scale": 1.0
  },
  "profile": {
    "min_mult": 0.0,
    "max_mult": 5.0,
    "bin_count": 120,
    "rolling_window": 63,
    "weight_mode": "volume_fallback",
    "accumulation_mode": "range_uniform",
    "close_weight_max_share": 0.70,
    "close_weight_full_range_bins": 12.0
  },
  "smoothing": {
    "method": "gaussian_kernel",
    "sigma_bins": 2.0
  },
  "expectancy": {
    "horizons": [1, 3, 5, 10, 20],
    "primary_horizon": 5,
    "score_bins": [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "min_samples": 80,
    "train_window": 756,
    "refit_every": 21,
    "cost_bps_roundtrip": 20.0
  },
  "sizing": {
    "long_only": true,
    "max_weight": 1.0,
    "neutral_weight": 0.0,
    "expectancy_scale_bps": 50.0,
    "min_positive_expectancy_bps": 5.0,
    "rebalance_threshold": 0.05,
    "max_trade_weight_per_bar": 0.25
  },
  "train_frac": 0.6,
  "validation_frac": 0.2
}
```

## 4. 구현해야 할 계산 계약

### 4.1 Contribution

`_contrib_matrix(cycle, cfg, weights)`는 그대로 재사용하되 `range_close`는 Pine 수식에 맞춘다.

- 현재 구현: range 50% + close 50% 고정.
- Pine 원본: candle range가 `closeWeightFullRangeBins`에 가까워질수록 close share가 0에서 `closeWeightMaxShare`까지 증가.
- 수정 규칙:

```text
range_in_bins = (high_mult - low_mult) / bin_width
range_ratio = min(range_in_bins / close_weight_full_range_bins, 1)
close_share = close_weight_max_share * range_ratio
range_share = 1 - close_share
contribution = range_uniform * range_share + close_bin * close_share
```

### 4.2 Rolling smooth profile

새 public 함수는 다음 형태가 적당하다.

```python
compute_rolling_profile_features(cycle, raw, cfg) -> tuple[pd.DataFrame, np.ndarray]
```

반환 feature 컬럼:

- `rolling_percentile`: smoothed rolling profile CDF에서 `cm_close` 위치.
- `profile_score`: `2 * (0.5 - rolling_percentile)`.
- `open_percentile`, `high_percentile`, `low_percentile`, `close_percentile`: multiple candle OHLC가 smoothed CDF에서 차지하는 위치.
- `candle_mid_percentile`: `(high_percentile + low_percentile) / 2`.
- `body_mid_percentile`: `(open_percentile + close_percentile) / 2`.
- `range_percentile_width`: `high_percentile - low_percentile`.
- `close_in_candle_percentile`: 캔들 range 안에서 close가 어느 쪽에 닫혔는지.
- `density_at_price`: smoothed density에서 현재 close bin 값 / 총 질량.
- `poc_mult`: smoothed profile 최대 bin 중심.
- `poc_distance`: `cm_close - poc_mult`.
- `lower_band`, `upper_band`: 20/80% 또는 config percentile band.
- `profile_width`: `upper_band - lower_band`.
- `profile_skew`: upper half mass / lower half mass.
- `profile_turnover`: `sum(abs(contrib_in - contrib_out)) / rolling_total`.
- `profile_jsd`: 직전 smoothed profile과 현재 smoothed profile의 Jensen-Shannon distance.

중요 해석:

- `profile_delta`, `profile_turnover`, `profile_jsd`는 "분포가 예측적으로 이동했다"는 신호가 아니다.
- 이것들은 최신 봉이 들어오고 63봉 전 봉이 빠진 유입/유출 강도, 즉 상태 안정성/불안정성 필터로만 쓴다.

### 4.3 Smoothing

첫 버전은 KDE 대신 bin histogram에 local Gaussian kernel convolution을 적용한다.

- 이유: Pine histogram 구조와 가장 가깝고 빠르며, walk-forward에서 반복 계산 비용이 작다.
- 주의: 여기서 Gaussian은 smoothing kernel일 뿐이다. profile 전체를 평균/분산 하나의 대칭 Gaussian으로 근사하지 않으므로, 비대칭·fat-tail·multi-modal 분포 형태를 그대로 허용한다.
- 구현: `scipy.ndimage.gaussian_filter1d`가 있으면 사용하고, 없으면 numpy kernel convolution fallback.
- smoothing 후에는 음수 방지와 재정규화:

```text
smooth = max(smooth, 0)
prob = smooth / smooth.sum()
cdf = cumsum(prob)
```

## 5. Expectancy 모델

feature score를 직접 threshold로 쓰지 않는다. 각 시점 t에서 과거 학습 구간만 사용해 "multiple-cycle 캔들 위치 bucket"별 미래수익률 통계를 만든 뒤, t의 bucket에 해당하는 기대값을 조회한다.

기본 bucket key는 첫 버전에서 단순하게 유지한다.

```text
location_score = 2 * (0.5 - close_percentile)
location_bucket = bucket(location_score)
```

두 번째 단계에서 아래 조건을 추가해 세분화한다.

```text
close_zone = bucket(close_percentile)
density_zone = low / mid / high density_at_price
range_zone = narrow / normal / wide range_percentile_width
stability_zone = stable / unstable profile_jsd
```

최종적으로 학습하는 값은 다음 조건부 기대값이다.

```text
E[future_return_h | close_zone, density_zone, range_zone, stability_zone]
```

forward return:

```text
future_ret_h[t] = close[t+h] / close[t] - 1
net_ret_h[t] = future_ret_h[t] - roundtrip_cost
```

bucket별 산출:

- `n`
- `mean_future_return`
- `win_rate`
- `avg_win`
- `avg_loss`
- `expectancy = win_rate * avg_win - (1 - win_rate) * abs(avg_loss) - trading_cost`
- `mfe`
- `mae`

walk-forward 규칙:

```text
for t:
    train = [t - train_window, t - primary_horizon)
    label이 확정된 과거 샘플만 사용
    t가 속한 score_bucket의 expectancy를 조회
    n < min_samples이면 neutral 처리
```

추천 초기 매매 규칙:

```text
if expectancy_bps <= min_positive_expectancy_bps:
    target_weight = neutral_weight
else:
    target_weight = min(max_weight, expectancy_bps / expectancy_scale_bps)
```

long-only 첫 버전에서는 음수 expectancy를 short으로 연결하지 않는다. 음수 bucket은 현금/축소 신호로만 쓴다.

## 6. 백테스트와 대시보드 계약

전략 핸들러는 `ProfileSizingHandler` 패턴을 따른다.

`forecast` 필수/권장 컬럼:

- `close`
- `base_cycle`
- `cm_close`
- `rolling_percentile`
- `profile_score`
- `expectancy_bps`
- `expectancy_samples`
- `target_weight`
- `actual_weight`
- `density_at_price`
- `poc_mult`
- `poc_distance`
- `profile_width`
- `profile_jsd`

성과 계산:

- `scripts/profile_sizing/engine.py::portfolio_returns`
- `equity_from_returns`
- `buy_hold_equity`
- `lot_trades`
- `performance`

무누수 규칙:

- profile은 t 봉까지의 OHLCV로 만들 수 있다.
- expectancy table은 t 시점에서 label이 확정된 과거 샘플만 사용한다. horizon=5면 최소 t-5 이전의 샘플까지만 학습에 포함한다.
- t에서 결정한 weight는 기존 엔진처럼 t+1 수익률에 적용한다.

## 7. 검증 순서

1. Unit test
    - contribution 행 합계가 weight와 일치하는지 확인.
    - rolling window=63에서 `roll[t] = contrib[t-62:t].sum(axis=0)`과 일치하는지 확인.
    - smoothed CDF가 단조 증가하고 0~1 범위인지 확인.
    - expectancy 학습이 미래 label을 사용하지 않는지 synthetic 데이터로 확인.

2. 단일 자산 smoke

```bash
source .venv/bin/activate
trading-lab backtest --strategy rolling-profile-expectancy-v1 --symbol SPY --phase validation
trading-lab backtest --strategy rolling-profile-expectancy-v1 --symbol QQQ --phase validation
```

3. 일반화 배치

```bash
python scripts/profile_sizing/batch.py --strategy rolling-profile-expectancy-v1 --phase validation
```

4. 리포트

```text
reports/profile_sizing/rolling_profile_expectancy_v1.md
```

필수 표:

- horizon별 score bucket expectancy table
- validation 자산별 전략 vs buy & hold: CAGR, Sharpe, MDD, total return
- bucket sample count 부족 구간
- parameter sensitivity: `rolling_window` 42/63/126, `sigma_bins` 1/2/3, `primary_horizon` 3/5/10

## 8. Gate 0 합격/폐기 기준 초안

결과를 보기 전에 아래 기준을 명세에 고정한다.

- validation에서 전체 자산 평균 `expectancy_bps`가 비용 차감 후 양수여야 한다.
- score bucket이 단조일 필요는 없지만, 최소 하나의 long bucket이 `n >= min_samples` 및 `expectancy_bps > 5`를 만족해야 한다.
- buy & hold 대비 Sharpe가 평균 -0.10보다 더 나쁘면 폐기 또는 v2로 재설계한다.
- MDD가 buy & hold보다 깊어진 자산 비율이 50%를 넘으면 long-only sizing 규칙을 재설계한다.
- `rolling_window`, `sigma_bins`, `primary_horizon`의 작은 변화에서 성과 방향이 뒤집히면 과적합으로 보고 폐기한다.

## 9. 라이브 적용 메모

실시간 KIS/paper 적용은 validation 통과 전까지 하지 않는다. 나중에 붙일 경우 구조는 다음과 같다.

```text
closed_contrib_deque: 확정 봉 contribution만 저장, 최대 63개
closed_hist: 확정 봉 rolling histogram
current_live_contribution: 진행 중 봉의 임시 contribution
live_hist = closed_hist + current_live_contribution
```

틱마다 `current_live_contribution`만 재계산하고, deque에는 봉 마감 시 한 번만 append한다. 이 규칙을 어기면 진행 중 봉이 중복 누적되어 profile percentile이 왜곡된다.

## 10. 1차 구현 우선순위

1. `profile.py`에 Pine 호환 `range_close` close share를 추가한다.
2. `rolling_expectancy.py`에 smoothing/CDF/features/expectancy table을 구현한다.
3. synthetic unit test로 무누수와 rolling update를 먼저 잠근다.
4. `rolling-profile-expectancy-v1` config/spec/checklist/registry/handler를 추가한다.
5. SPY/QQQ validation smoke 후 generalization 배치를 돌린다.

---

## 참고: yoon_profile_3 타겟 방침 (2026-07-04)

**yoon_profile_3(3×3 상태 필드 + path_tilt)의 타겟 자산은 크립토로 한정하지
않는다.** 멀티에셋 검증(`reports/profile_sizing/yoon_profile_3_multiasset_eval.md`)
에서 ETF(SPY/QQQ/GLD)가 기각된 것은 **BTC 튜닝 state_weights를 무수정 이식**한
조건이지 구조 자체가 아니다 — 필드 구조(대각선 궤적·하락 종착지·노출 정합)는
자산·타임프레임을 가로질러 자기유사함이 확인됐다
(`state_field_trend_analysis.md` §5).

타 자산(주식·ETF·금 등)에 적용할 때의 전제:

1. 해당 자산 데이터로 `scripts/profile_sizing/yoon_profile_study.py`
   (state_weights 재추정) 재실행.
2. `scripts/profile_sizing/yoon_profile_path_study.py`(path_tilt 검정) 재실행 —
   path_tilt는 BTC에서만 +기여가 확인된 상태라 자산별 재검정 없이 켜지 않는다.
3. 현 가중치를 그대로 쓸 거면 base yoon_profile(tilt off)을 사용한다
   (예: SOL은 base가 우위).

아래는 파인스크립트 원본이야. 

//@version=6
indicator(
     "평면차트 Light - Rolling Profile Modes + Anchor Scale",
     shorttitle = "CycleMult Light Anchor Scale",
     overlay = false,
     max_lines_count = 300,
     max_labels_count = 50,
     max_polylines_count = 80,
     max_bars_back = 5000
)

//==================================================
// Inputs
//==================================================
groupCore = "Core"

baseCalcTf = input.timeframe(
     "D",
     "Base Calculation Timeframe",
     group = groupCore,
     tooltip = "cycle / rolling profile / percentile을 계산할 고정 TF입니다. 기본값 D."
)

baseLen = input.int(
     200,
     "Base Cycle Length",
     minval = 10,
     maxval = 1000,
     group = groupCore
)

baseType = input.string(
     "SMA",
     "Base Cycle Type",
     options = ["SMA", "EMA", "RMA", "WMA", "VWMA"],
     group = groupCore
)

baseScale = input.float(
     1.0,
     "Base Cycle Scale",
     minval = 0.1,
     maxval = 10.0,
     step = 0.05,
     group = groupCore
)

// ---- Rolling Profile ----
groupProfile = "Rolling Profile"

rollingProfileBars = input.int(
     126,
     "Rolling Window Bars",
     minval = 20,
     maxval = 1000,
     group = groupProfile,
     tooltip = "baseCalcTf 기준 rolling profile 길이입니다. D 기준 126은 대략 6개월 거래일, 코인은 180 전후를 사용할 수 있습니다."
)

profileBinCount = input.int(
     120,
     "Profile Bin Count",
     minval = 20,
     maxval = 250,
     group = groupProfile,
     tooltip = "가볍게 보기 위한 기본값 120. 이전 400보다 훨씬 가볍습니다."
)

profilePercentileValue = input.float(
     20.0,
     "Upper / Lower Profile Percent",
     minval = 1.0,
     maxval = 49.0,
     step = 1.0,
     group = groupProfile
)

profilePercentileSmoothLen = input.int(
     5,
     "Percentile EMA Smoothing",
     minval = 1,
     maxval = 100,
     group = groupProfile
)

profileMinMult = input.float(
     0.0,
     "Profile Min Multiple",
     minval = 0.0,
     maxval = 5.0,
     step = 0.1,
     group = groupProfile
)

profileMaxMult = input.float(
     5.0,
     "Profile Max Multiple",
     minval = 0.5,
     maxval = 10.0,
     step = 0.1,
     group = groupProfile
)

profileWeightMode = input.string(
     "Volume With 1 Fallback",
     "Profile Weight Mode",
     options = ["Volume With 1 Fallback", "Volume", "Time"],
     group = groupProfile
)

profileAccumulationMode = input.string(
     "Range Uniform",
     "Profile Accumulation Mode",
     options = ["Range Uniform", "OHLC", "Range + Close Weight"],
     group = groupProfile,
     tooltip = "Range Uniform: 기존 방식. 캔들의 low~high 구간을 겹친 bin에 균등 분배합니다. OHLC: 시가/고가/저가/종가에 25%씩 분배합니다. Range + Close Weight: 기본 range 분배에 더해, 캔들이 길어질수록 종가 bin 비중을 높입니다."
)

closeWeightMaxShare = input.float(
     70.0,
     "Close Weight Max Share (%)",
     minval = 10.0,
     maxval = 95.0,
     step = 5.0,
     group = groupProfile,
     tooltip = "Range + Close Weight 모드에서 장대캔들이 될수록 종가 bin에 배정할 최대 비중입니다."
)

closeWeightFullRangeBins = input.float(
     12.0,
     "Close Weight Full Range (bins)",
     minval = 1.0,
     maxval = 100.0,
     step = 1.0,
     group = groupProfile,
     tooltip = "캔들 길이가 bin 몇 개를 넘으면 Close Weight Max Share에 도달할지 정합니다."
)

// ---- 6M Snapshots ----
groupSnapshot = "6M Rolling Profile Snapshots"

showRollingSnapshots = input.bool(
     true,
     "Show 6M Snapshots",
     group = groupSnapshot,
     tooltip = "Jan / Jul 전환 시점마다 직전 rolling window profile을 보조창에 스냅샷으로 표시합니다."
)

snapshotYearsToKeep = input.int(
     4,
     "Years To Keep",
     minval = 1,
     maxval = 10,
     group = groupSnapshot
)

snapshotWidthDays = input.int(
     90,
     "Snapshot Width Days",
     minval = 10,
     maxval = 240,
     group = groupSnapshot
)

snapshotColor = input.color(
     color.silver,
     "Snapshot Color",
     group = groupSnapshot
)

snapshotTransparency = input.int(
     82,
     "Snapshot Transparency",
     minval = 0,
     maxval = 100,
     group = groupSnapshot
)

snapshotOutlineWidth = input.int(
     1,
     "Snapshot Outline Width",
     minval = 1,
     maxval = 4,
     group = groupSnapshot
)

showCurrentRollingProfile = input.bool(
     true,
     "Show Current Rolling Profile",
     group = groupSnapshot,
     tooltip = "마지막 봉 오른쪽에 현재 진행 중인 rolling profile을 표시합니다."
)

currentProfileRightOffsetBars = input.int(
     6,
     "Current Profile Right Offset Bars",
     minval = 1,
     maxval = 100,
     group = groupSnapshot
)

currentProfileMaxWidthBars = input.int(
     70,
     "Current Profile Max Width Bars",
     minval = 10,
     maxval = 300,
     group = groupSnapshot
)

currentProfileColor = input.color(
     color.gray,
     "Current Profile Color",
     group = groupSnapshot
)

currentProfileTransparency = input.int(
     72,
     "Current Profile Transparency",
     minval = 0,
     maxval = 100,
     group = groupSnapshot
)

currentProfileOutlineWidth = input.int(
     1,
     "Current Profile Outline Width",
     minval = 1,
     maxval = 4,
     group = groupSnapshot
)

showAnchorRollingProfile = input.bool(
     true,
     "Show Draggable Anchor Profile",
     group = groupSnapshot,
     tooltip = "드래그 가능한 Anchor Time 세로선 기준의 rolling profile을 세로선 오른쪽에 표시합니다. Anchor Time은 설정창 또는 차트 위 입력 마커를 드래그해서 조정합니다."
)

anchorProfileTime = input.time(
     1704067200000,
     "Anchor Time",
     confirm = true,
     group = groupSnapshot,
     tooltip = "이 시간에 해당하는 baseCalcTf 봉에서 rolling window profile을 저장하고, 해당 세로선 오른쪽에 표시합니다."
)

anchorProfileRightOffsetDays = input.int(
     3,
     "Anchor Profile Right Offset Days",
     minval = 0,
     maxval = 120,
     group = groupSnapshot
)

anchorProfileMaxWidthDays = input.int(
     70,
     "Anchor Profile Max Width Days",
     minval = 5,
     maxval = 240,
     group = groupSnapshot
)

anchorLineColor = input.color(
     color.orange,
     "Anchor Line Color",
     group = groupSnapshot
)

anchorLineTransparency = input.int(
     20,
     "Anchor Line Transparency",
     minval = 0,
     maxval = 100,
     group = groupSnapshot
)

anchorProfileColor = input.color(
     color.orange,
     "Anchor Profile Color",
     group = groupSnapshot
)

anchorProfileTransparency = input.int(
     76,
     "Anchor Profile Transparency",
     minval = 0,
     maxval = 100,
     group = groupSnapshot
)

anchorProfileOutlineWidth = input.int(
     1,
     "Anchor Profile Outline Width",
     minval = 1,
     maxval = 4,
     group = groupSnapshot
)

profileDisplayScaleMode = input.string(
     "Pane Scale",
     "Profile Display Scale",
     options = ["Pane Scale", "Full Profile Range"],
     group = groupSnapshot,
     tooltip = "Pane Scale: 오른쪽 cycle-multiple 축/가이드 범위에 맞춰 profile y범위를 제한합니다. Full Profile Range: Profile Min/Max Multiple 전체 범위로 표시합니다."
)

// ---- Display ----
groupDisplay = "Display"

showPlaneCandles = input.bool(true, "Cycle Multiple Candles", group = groupDisplay)
showOverlayCycleLines = input.bool(true, "Cycle Multiple Lines (price)", group = groupDisplay)
showPaneCycleLines = input.bool(true, "Cycle Multiple Guide Lines (pane)", group = groupDisplay)
showPercentilePriceLines = input.bool(true, "Upper / Lower Percentile Price Lines", group = groupDisplay)
limitLinesToTouched = input.bool(true, "Limit Cycle Lines To Highest Touched", group = groupDisplay)

// ---- Style ----
groupStyle = "Style"

upColor = input.color(color.lime, "Up Candle", group = groupStyle)
downColor = input.color(color.red, "Down Candle", group = groupStyle)
wickColor = input.color(color.gray, "Wick", group = groupStyle)
cycleLineColor = input.color(color.gray, "Cycle Line Color", group = groupStyle)
cycleLineTransparency = input.int(72, "Cycle Line Transparency", minval = 0, maxval = 100, group = groupStyle)
lowerLineColor = input.color(color.red, "Lower Percentile Price", group = groupStyle)
upperLineColor = input.color(color.lime, "Upper Percentile Price", group = groupStyle)
percentileLineWidth = input.int(2, "Percentile Price Line Width", minval = 1, maxval = 4, group = groupStyle)

// Fixed performance settings
profileCalcBars = 5000
paneGuideMinMult = 0.5
paneGuideMaxMult = 5.0
paneGuideStepMult = 0.5
paneGuideBackBars = 5000
paneGuideForwardBars = 120

//==================================================
// Functions
//==================================================
f_smooth(src, len, mode) =>
    mode == "SMA" ? ta.sma(src, len) :
     mode == "EMA" ? ta.ema(src, len) :
     mode == "RMA" ? ta.rma(src, len) :
     mode == "WMA" ? ta.wma(src, len) :
     mode == "VWMA" ? ta.vwma(src, len) :
     ta.sma(src, len)

f_add_point_to_profile(float[] _vp, float _mult, float _weight, float _sign, float _binWidth) =>
    if not na(_mult) and not na(_weight) and _weight > 0 and not na(_binWidth) and _binWidth > 0 and array.size(_vp) == profileBinCount
        if _mult >= profileMinMult and _mult <= profileMaxMult
            idx = int(math.floor((_mult - profileMinMult) / _binWidth))
            idx := math.max(0, math.min(idx, profileBinCount - 1))
            oldValue = array.get(_vp, idx)
            array.set(_vp, idx, math.max(oldValue + _sign * _weight, 0.0))

f_add_range_to_profile(float[] _vp, float _lowMult, float _highMult, float _weight, float _sign, float _binWidth) =>
    if not na(_lowMult) and not na(_highMult) and not na(_weight) and _weight > 0 and not na(_binWidth) and _binWidth > 0 and array.size(_vp) == profileBinCount
        barLowMult = math.min(_lowMult, _highMult)
        barHighMult = math.max(_lowMult, _highMult)

        clippedLow = math.max(barLowMult, profileMinMult)
        clippedHigh = math.min(barHighMult, profileMaxMult)

        if not na(clippedLow) and not na(clippedHigh) and clippedHigh >= clippedLow
            if clippedHigh == clippedLow
                singleIdx = int(math.floor((clippedLow - profileMinMult) / _binWidth))
                singleIdx := math.max(0, math.min(singleIdx, profileBinCount - 1))
                oldValue = array.get(_vp, singleIdx)
                array.set(_vp, singleIdx, math.max(oldValue + _sign * _weight, 0.0))
            else
                lowIdx = int(math.floor((clippedLow - profileMinMult) / _binWidth))
                highIdx = int(math.floor((clippedHigh - profileMinMult) / _binWidth))
                lowIdx := math.max(0, math.min(lowIdx, profileBinCount - 1))
                highIdx := math.max(0, math.min(highIdx, profileBinCount - 1))

                barRange = clippedHigh - clippedLow

                if barRange > 0
                    for b = lowIdx to highIdx
                        binLow = profileMinMult + b * _binWidth
                        binHigh = binLow + _binWidth
                        overlapLow = math.max(clippedLow, binLow)
                        overlapHigh = math.min(clippedHigh, binHigh)
                        overlap = math.max(overlapHigh - overlapLow, 0.0)
                        addValue = _weight * overlap / barRange

                        if addValue > 0
                            oldBinValue = array.get(_vp, b)
                            array.set(_vp, b, math.max(oldBinValue + _sign * addValue, 0.0))

f_add_bar_to_profile(float[] _vp, float _cycleSeries, int _barsAgo, float _sign, float _binWidth) =>
    rawVol = nz(volume[_barsAgo])
    barWeight =
         profileWeightMode == "Time" ? 1.0 :
         profileWeightMode == "Volume" ? rawVol :
         rawVol > 0 ? rawVol : 1.0

    cyc = _cycleSeries[_barsAgo]
    validCycle = not na(cyc) and cyc > 0

    if validCycle and barWeight > 0 and not na(_binWidth) and _binWidth > 0 and array.size(_vp) == profileBinCount
        openMult = open[_barsAgo] / cyc
        highMult = high[_barsAgo] / cyc
        lowMult = low[_barsAgo] / cyc
        closeMult = close[_barsAgo] / cyc

        if profileAccumulationMode == "OHLC"
            pointWeight = barWeight / 4.0
            f_add_point_to_profile(_vp, openMult, pointWeight, _sign, _binWidth)
            f_add_point_to_profile(_vp, highMult, pointWeight, _sign, _binWidth)
            f_add_point_to_profile(_vp, lowMult, pointWeight, _sign, _binWidth)
            f_add_point_to_profile(_vp, closeMult, pointWeight, _sign, _binWidth)
        else if profileAccumulationMode == "Range + Close Weight"
            rawLowMult = math.min(lowMult, highMult)
            rawHighMult = math.max(lowMult, highMult)
            candleRange = rawHighMult - rawLowMult
            rangeInBins = _binWidth > 0 ? candleRange / _binWidth : 0.0
            rangeRatio = closeWeightFullRangeBins > 0 ? math.min(rangeInBins / closeWeightFullRangeBins, 1.0) : 1.0
            maxCloseShare = math.max(0.0, math.min(closeWeightMaxShare / 100.0, 0.95))
            closeShare = maxCloseShare * rangeRatio
            closeShare := math.max(0.0, math.min(closeShare, 0.95))
            rangeShare = 1.0 - closeShare

            if rangeShare > 0
                f_add_range_to_profile(_vp, lowMult, highMult, barWeight * rangeShare, _sign, _binWidth)

            if closeShare > 0
                f_add_point_to_profile(_vp, closeMult, barWeight * closeShare, _sign, _binWidth)
        else
            f_add_range_to_profile(_vp, lowMult, highMult, barWeight, _sign, _binWidth)

f_calc_profile_stats(float[] _vp, float _binWidth) =>
    totalValue = 0.0
    maxValue = 0.0

    for b = 0 to profileBinCount - 1
        v = array.get(_vp, b)
        totalValue += v
        maxValue := math.max(maxValue, v)

    float lowerRaw = na
    float upperRaw = na

    if totalValue > 0 and not na(_binWidth) and _binWidth > 0
        targetValue = totalValue * profilePercentileValue / 100.0

        lowerCum = 0.0
        for b = 0 to profileBinCount - 1
            v = array.get(_vp, b)
            before = lowerCum
            after = before + v

            if na(lowerRaw) and after >= targetValue
                hitRatio = v > 0 ? (targetValue - before) / v : 1.0
                hitRatio := math.max(0.0, math.min(hitRatio, 1.0))
                lowerRaw := profileMinMult + b * _binWidth + hitRatio * _binWidth

            lowerCum := after

        upperCum = 0.0
        for j = 0 to profileBinCount - 1
            b = profileBinCount - 1 - j
            v = array.get(_vp, b)
            before = upperCum
            after = before + v

            if na(upperRaw) and after >= targetValue
                hitRatio = v > 0 ? (targetValue - before) / v : 1.0
                hitRatio := math.max(0.0, math.min(hitRatio, 1.0))
                binHigh = profileMinMult + (b + 1) * _binWidth
                upperRaw := binHigh - hitRatio * _binWidth

            upperCum := after

    [lowerRaw, upperRaw, totalValue, maxValue]

f_visible_profile_max(float[] _bins, int _offset, float _binWidth, float _displayMin, float _displayMax) =>
    visibleMax = 0.0
    binsOk = array.size(_bins) >= _offset + profileBinCount

    if binsOk and not na(_binWidth) and _binWidth > 0 and not na(_displayMin) and not na(_displayMax)
        lowDisplay = math.min(_displayMin, _displayMax)
        highDisplay = math.max(_displayMin, _displayMax)

        for b = 0 to profileBinCount - 1
            midMult = profileMinMult + (b + 0.5) * _binWidth
            if midMult >= lowDisplay and midMult <= highDisplay
                visibleMax := math.max(visibleMax, array.get(_bins, _offset + b))

    visibleMax

f_engine() =>
    var float[] vp = array.new_float()
    var int[] snapTimes = array.new_int()
    var float[] snapFlat = array.new_float()
    var float[] snapMax = array.new_float()
    var float[] anchorBins = array.new_float()
    var float anchorMax = na
    var int anchorHitTime = na
    var float[] emptyFloatArr = array.new_float()
    var int[] emptyIntArr = array.new_int()

    cycle = f_smooth(close, baseLen, baseType) * baseScale

    validRange = profileMaxMult > profileMinMult
    binWidth = validRange ? (profileMaxMult - profileMinMult) / profileBinCount : na

    if barstate.isfirst
        array.clear(vp)
        array.clear(anchorBins)
        anchorMax := na
        anchorHitTime := na
        for i = 0 to profileBinCount - 1
            array.push(vp, 0.0)

    // Rolling profile update: add current bar, subtract bar that left the window.
    if validRange and not na(binWidth) and binWidth > 0 and array.size(vp) == profileBinCount
        f_add_bar_to_profile(vp, cycle, 0, 1.0, binWidth)

        if bar_index >= rollingProfileBars
            f_add_bar_to_profile(vp, cycle, rollingProfileBars, -1.0, binWidth)

    [lowerRaw, upperRaw, totalProfileValue, maxProfileValue] = f_calc_profile_stats(vp, binWidth)

    // Capture the rolling profile ending at the draggable anchor time.
    // If the selected time is between two base-TF bars, the first bar at or after the anchor time is used.
    isAnchorHit =
         showAnchorRollingProfile and
         not na(anchorProfileTime) and
         time >= anchorProfileTime and
         (na(time[1]) or time[1] < anchorProfileTime)

    if isAnchorHit and maxProfileValue > 0 and totalProfileValue > 0 and array.size(vp) == profileBinCount
        array.clear(anchorBins)

        for b = 0 to profileBinCount - 1
            array.push(anchorBins, array.get(vp, b))

        anchorMax := maxProfileValue
        anchorHitTime := time

    lowerSmooth = profilePercentileSmoothLen > 1 ? ta.ema(lowerRaw, profilePercentileSmoothLen) : lowerRaw
    upperSmooth = profilePercentileSmoothLen > 1 ? ta.ema(upperRaw, profilePercentileSmoothLen) : upperRaw

    // Store 6-month snapshots at Jan / Jul month transitions.
    prevBarTime = time[1]
    isMonthTransition = not na(prevBarTime) and (year(time) != year(prevBarTime) or month(time) != month(prevBarTime))
    isSixMonthReport = isMonthTransition and (month(prevBarTime) == 1 or month(prevBarTime) == 7)

    if isSixMonthReport and maxProfileValue > 0 and totalProfileValue > 0 and array.size(vp) == profileBinCount
        array.push(snapTimes, prevBarTime)
        array.push(snapMax, maxProfileValue)

        for b = 0 to profileBinCount - 1
            array.push(snapFlat, array.get(vp, b))

        maxSnapshots = snapshotYearsToKeep * 2

        while array.size(snapTimes) > maxSnapshots
            array.shift(snapTimes)
            array.shift(snapMax)
            for b = 0 to profileBinCount - 1
                if array.size(snapFlat) > 0
                    array.shift(snapFlat)

    float[] outCurrentBins = emptyFloatArr
    currentMaxValue = na(maxProfileValue) ? na : maxProfileValue
    float[] outAnchorBins = emptyFloatArr
    anchorMaxValue = anchorMax
    anchorCapturedTime = anchorHitTime
    float[] outSnapFlat = emptyFloatArr
    float[] outSnapMax = emptyFloatArr
    int[] outSnapTimes = emptyIntArr

    if barstate.islast
        outCurrentBins := array.copy(vp)
        outAnchorBins := array.copy(anchorBins)
        outSnapFlat := array.copy(snapFlat)
        outSnapMax := array.copy(snapMax)
        outSnapTimes := array.copy(snapTimes)

    [cycle, lowerSmooth, upperSmooth, binWidth, outCurrentBins, currentMaxValue, outAnchorBins, anchorMaxValue, anchorCapturedTime, outSnapTimes, outSnapFlat, outSnapMax]

//==================================================
// Base TF engine
//==================================================
[baseCycle, lowerProfileCutMult, upperProfileCutMult, fixedProfileBinWidth, currentProfileBins, currentProfileMaxValue, anchorProfileBins, anchorProfileMaxValue, anchorProfileCapturedTime, snapshotTimes, snapshotFlatBins, snapshotMaxValues] =
     request.security(
         syminfo.tickerid,
         baseCalcTf,
         f_engine(),
         barmerge.gaps_off,
         barmerge.lookahead_off,
         calc_bars_count = profileCalcBars
     )

validBaseCycle = not na(baseCycle) and baseCycle > 0

lowerProfileCutPrice = validBaseCycle and not na(lowerProfileCutMult) ? lowerProfileCutMult * baseCycle : na
upperProfileCutPrice = validBaseCycle and not na(upperProfileCutMult) ? upperProfileCutMult * baseCycle : na

//==================================================
// Cycle multiple candles
//==================================================
cmOpen = validBaseCycle ? open / baseCycle : na
cmHigh = validBaseCycle ? high / baseCycle : na
cmLow = validBaseCycle ? low / baseCycle : na
cmClose = validBaseCycle ? close / baseCycle : na

baseCandleColor = cmClose >= cmOpen ? upColor : downColor

plotcandle(
     showPlaneCandles ? cmOpen : na,
     showPlaneCandles ? cmHigh : na,
     showPlaneCandles ? cmLow : na,
     showPlaneCandles ? cmClose : na,
     title = "Cycle Multiple Candles",
     color = baseCandleColor,
     wickcolor = wickColor,
     bordercolor = baseCandleColor
)

// No cycle close line, no POC line.

//==================================================
// Upper / Lower percentile price lines
//==================================================
plot(
     showPercentilePriceLines ? lowerProfileCutPrice : na,
     title = "Lower Rolling Profile Percentile Price",
     color = color.new(lowerLineColor, 0),
     linewidth = percentileLineWidth,
     force_overlay = true
)

plot(
     showPercentilePriceLines ? upperProfileCutPrice : na,
     title = "Upper Rolling Profile Percentile Price",
     color = color.new(upperLineColor, 0),
     linewidth = percentileLineWidth,
     force_overlay = true
)

//==================================================
// Cycle multiple price lines
//==================================================
var float maxTouchedMult = na
if not na(cmHigh)
    maxTouchedMult := na(maxTouchedMult) ? cmHigh : math.max(maxTouchedMult, cmHigh)

f_show_mult(_mult) =>
    not limitLinesToTouched or na(maxTouchedMult) or _mult <= maxTouchedMult + 0.05

paneScaleTop = limitLinesToTouched and not na(maxTouchedMult) ? math.min(paneGuideMaxMult, math.max(paneGuideMinMult, maxTouchedMult + 0.05)) : paneGuideMaxMult
profileDisplayMinMult = profileDisplayScaleMode == "Pane Scale" ? paneGuideMinMult : profileMinMult
rawProfileDisplayMaxMult = profileDisplayScaleMode == "Pane Scale" ? paneScaleTop : profileMaxMult
profileDisplayStep = not na(fixedProfileBinWidth) and fixedProfileBinWidth > 0 ? fixedProfileBinWidth : 0.01
profileDisplayMaxMult = math.max(rawProfileDisplayMaxMult, profileDisplayMinMult + profileDisplayStep)

cycleLineCol = color.new(cycleLineColor, cycleLineTransparency)

plot(showOverlayCycleLines and f_show_mult(0.5) ? baseCycle * 0.5 : na, title = "Cycle 0.5x Price", color = cycleLineCol, linewidth = 1, force_overlay = true)
plot(showOverlayCycleLines and f_show_mult(1.0) ? baseCycle * 1.0 : na, title = "Cycle 1.0x Price", color = color.new(cycleLineColor, 55), linewidth = 1, force_overlay = true)
plot(showOverlayCycleLines and f_show_mult(1.5) ? baseCycle * 1.5 : na, title = "Cycle 1.5x Price", color = cycleLineCol, linewidth = 1, force_overlay = true)
plot(showOverlayCycleLines and f_show_mult(2.0) ? baseCycle * 2.0 : na, title = "Cycle 2.0x Price", color = cycleLineCol, linewidth = 1, force_overlay = true)
plot(showOverlayCycleLines and f_show_mult(2.5) ? baseCycle * 2.5 : na, title = "Cycle 2.5x Price", color = cycleLineCol, linewidth = 1, force_overlay = true)
plot(showOverlayCycleLines and f_show_mult(3.0) ? baseCycle * 3.0 : na, title = "Cycle 3.0x Price", color = cycleLineCol, linewidth = 1, force_overlay = true)
plot(showOverlayCycleLines and f_show_mult(3.5) ? baseCycle * 3.5 : na, title = "Cycle 3.5x Price", color = cycleLineCol, linewidth = 1, force_overlay = true)
plot(showOverlayCycleLines and f_show_mult(4.0) ? baseCycle * 4.0 : na, title = "Cycle 4.0x Price", color = cycleLineCol, linewidth = 1, force_overlay = true)
plot(showOverlayCycleLines and f_show_mult(4.5) ? baseCycle * 4.5 : na, title = "Cycle 4.5x Price", color = cycleLineCol, linewidth = 1, force_overlay = true)
plot(showOverlayCycleLines and f_show_mult(5.0) ? baseCycle * 5.0 : na, title = "Cycle 5.0x Price", color = cycleLineCol, linewidth = 1, force_overlay = true)

//==================================================
// Pane guide lines
//==================================================
var line[] paneGuideLines = array.new_line()
var label[] paneGuideLabels = array.new_label()
var int lastGuideBar = na

if barstate.islast and showPaneCycleLines and (na(lastGuideBar) or lastGuideBar != bar_index)
    lastGuideBar := bar_index

    paneGuideLineCount = array.size(paneGuideLines)
    if paneGuideLineCount > 0
        for i = 0 to paneGuideLineCount - 1
            line.delete(array.get(paneGuideLines, i))
    array.clear(paneGuideLines)

    paneGuideLabelCount = array.size(paneGuideLabels)
    if paneGuideLabelCount > 0
        for i = 0 to paneGuideLabelCount - 1
            label.delete(array.get(paneGuideLabels, i))
    array.clear(paneGuideLabels)

    guideMax = limitLinesToTouched and not na(maxTouchedMult) ? math.min(paneGuideMaxMult, maxTouchedMult + 0.05) : paneGuideMaxMult
    guideCount = int(math.floor((guideMax - paneGuideMinMult) / paneGuideStepMult))
    guideCount := math.max(guideCount, 0)

    leftX = math.max(0, bar_index - paneGuideBackBars)
    rightX = bar_index + paneGuideForwardBars

    for i = 0 to guideCount
        mult = paneGuideMinMult + i * paneGuideStepMult
        isWhole = math.abs(mult - math.round(mult)) < 0.001
        ln = line.new(
             x1 = leftX,
             y1 = mult,
             x2 = rightX,
             y2 = mult,
             xloc = xloc.bar_index,
             extend = extend.none,
             color = color.new(cycleLineColor, isWhole ? 55 : 78),
             style = isWhole ? line.style_solid : line.style_dotted,
             width = isWhole ? 2 : 1
        )
        array.push(paneGuideLines, ln)

        lb = label.new(
             x = rightX,
             y = mult,
             text = str.tostring(mult, "#.0") + "x",
             xloc = xloc.bar_index,
             style = label.style_label_left,
             textcolor = color.white,
             color = color.new(color.black, 85),
             size = size.tiny
        )
        array.push(paneGuideLabels, lb)

if barstate.islast and not showPaneCycleLines
    paneGuideLineCount = array.size(paneGuideLines)
    if paneGuideLineCount > 0
        for i = 0 to paneGuideLineCount - 1
            line.delete(array.get(paneGuideLines, i))
    array.clear(paneGuideLines)

    paneGuideLabelCount = array.size(paneGuideLabels)
    if paneGuideLabelCount > 0
        for i = 0 to paneGuideLabelCount - 1
            label.delete(array.get(paneGuideLabels, i))
    array.clear(paneGuideLabels)

//==================================================
// Draggable anchor line + rolling profile at anchor time
//==================================================
var line anchorTimeLine = na
var polyline anchorProfilePolyline = na
var label anchorProfileLabel = na

if barstate.islast
    if not na(anchorTimeLine)
        line.delete(anchorTimeLine)
        anchorTimeLine := na

    if not na(anchorProfilePolyline)
        polyline.delete(anchorProfilePolyline)
        anchorProfilePolyline := na

    if not na(anchorProfileLabel)
        label.delete(anchorProfileLabel)
        anchorProfileLabel := na

    if showAnchorRollingProfile and not na(anchorProfileTime)
        anchorTimeLine := line.new(
             x1 = anchorProfileTime,
             y1 = profileDisplayMinMult,
             x2 = anchorProfileTime,
             y2 = profileDisplayMaxMult,
             xloc = xloc.bar_time,
             extend = extend.none,
             color = color.new(anchorLineColor, anchorLineTransparency),
             style = line.style_dashed,
             width = 1
        )

        anchorProfileReady =
             array.size(anchorProfileBins) == profileBinCount and
             not na(anchorProfileMaxValue) and
             anchorProfileMaxValue > 0 and
             not na(fixedProfileBinWidth) and
             fixedProfileBinWidth > 0

        if anchorProfileReady
            anchorOffsetMs = anchorProfileRightOffsetDays * 86400000
            anchorWidthMs = anchorProfileMaxWidthDays * 86400000
            baseTime = anchorProfileTime + anchorOffsetMs
            ptsAnchor = array.new<chart.point>()

            anchorVisibleMax = f_visible_profile_max(anchorProfileBins, 0, fixedProfileBinWidth, profileDisplayMinMult, profileDisplayMaxMult)

            array.push(ptsAnchor, chart.point.from_time(baseTime, profileDisplayMinMult))

            for b = 0 to profileBinCount - 1
                anchorValue = array.get(anchorProfileBins, b)
                midMult = profileMinMult + (b + 0.5) * fixedProfileBinWidth
                if midMult >= profileDisplayMinMult and midMult <= profileDisplayMaxMult
                    widthMs = anchorVisibleMax > 0 ? int(math.round(anchorWidthMs * anchorValue / anchorVisibleMax)) : 0
                    widthMs := math.max(widthMs, 0)
                    array.push(ptsAnchor, chart.point.from_time(baseTime + widthMs, midMult))

            array.push(ptsAnchor, chart.point.from_time(baseTime, profileDisplayMaxMult))

            if array.size(ptsAnchor) >= 3
                fillTransp = anchorProfileTransparency
                lineTransp = int(math.max(0.0, fillTransp - 18.0))

                anchorProfilePolyline := polyline.new(
                     points = ptsAnchor,
                     curved = false,
                     closed = true,
                     xloc = xloc.bar_time,
                     line_color = color.new(anchorProfileColor, lineTransp),
                     fill_color = color.new(anchorProfileColor, fillTransp),
                     line_width = anchorProfileOutlineWidth
                )

                anchorProfileLabel := label.new(
                     x = baseTime,
                     y = profileDisplayMaxMult,
                     text = "Anchor rolling profile",
                     xloc = xloc.bar_time,
                     style = label.style_label_left,
                     textcolor = color.white,
                     color = color.new(color.black, 82),
                     size = size.tiny
                )
        else
            anchorProfileLabel := label.new(
                 x = anchorProfileTime,
                 y = profileDisplayMaxMult,
                 text = "Anchor profile not loaded",
                 xloc = xloc.bar_time,
                 style = label.style_label_down,
                 textcolor = color.white,
                 color = color.new(color.black, 82),
                 size = size.tiny
            )

//==================================================
// Current rolling profile on the right
//==================================================
var polyline currentProfilePolyline = na

if barstate.islast
    if not na(currentProfilePolyline)
        polyline.delete(currentProfilePolyline)
        currentProfilePolyline := na

    currentProfileReady =
         showCurrentRollingProfile and
         array.size(currentProfileBins) == profileBinCount and
         not na(currentProfileMaxValue) and
         currentProfileMaxValue > 0 and
         not na(fixedProfileBinWidth) and
         fixedProfileBinWidth > 0

    if currentProfileReady
        baseX = bar_index + currentProfileRightOffsetBars
        ptsCurrent = array.new<chart.point>()

        currentVisibleMax = f_visible_profile_max(currentProfileBins, 0, fixedProfileBinWidth, profileDisplayMinMult, profileDisplayMaxMult)

        array.push(ptsCurrent, chart.point.from_index(baseX, profileDisplayMinMult))

        for b = 0 to profileBinCount - 1
            currValue = array.get(currentProfileBins, b)
            midMult = profileMinMult + (b + 0.5) * fixedProfileBinWidth
            if midMult >= profileDisplayMinMult and midMult <= profileDisplayMaxMult
                widthBars = currentVisibleMax > 0 ? int(math.round(currentProfileMaxWidthBars * currValue / currentVisibleMax)) : 0
                widthBars := math.max(widthBars, 0)
                array.push(ptsCurrent, chart.point.from_index(baseX + widthBars, midMult))

        array.push(ptsCurrent, chart.point.from_index(baseX, profileDisplayMaxMult))

        if array.size(ptsCurrent) >= 3
            fillTransp = currentProfileTransparency
            lineTransp = int(math.max(0.0, fillTransp - 18.0))

            currentProfilePolyline := polyline.new(
                 points = ptsCurrent,
                 curved = false,
                 closed = true,
                 xloc = xloc.bar_index,
                 line_color = color.new(currentProfileColor, lineTransp),
                 fill_color = color.new(currentProfileColor, fillTransp),
                 line_width = currentProfileOutlineWidth
            )

//==================================================
// 6M rolling profile snapshots
//==================================================
var polyline[] snapshotPolylines = array.new<polyline>()

if barstate.islast
    snapshotPolylineCount = array.size(snapshotPolylines)
    if snapshotPolylineCount > 0
        for i = 0 to snapshotPolylineCount - 1
            polyline.delete(array.get(snapshotPolylines, i))
    array.clear(snapshotPolylines)

    snapCount = array.size(snapshotTimes)
    flatOk = snapCount > 0 and array.size(snapshotFlatBins) >= snapCount * profileBinCount and array.size(snapshotMaxValues) >= snapCount

    if showRollingSnapshots and flatOk and not na(fixedProfileBinWidth) and fixedProfileBinWidth > 0
        snapshotWidthMs = snapshotWidthDays * 86400000

        for s = 0 to snapCount - 1
            snapTime = array.get(snapshotTimes, s)
            snapMax = array.get(snapshotMaxValues, s)
            snapOffset = s * profileBinCount

            if snapMax > 0
                pts = array.new<chart.point>()
                snapVisibleMax = f_visible_profile_max(snapshotFlatBins, snapOffset, fixedProfileBinWidth, profileDisplayMinMult, profileDisplayMaxMult)

                array.push(pts, chart.point.from_time(snapTime, profileDisplayMinMult))

                for b = 0 to profileBinCount - 1
                    snapValue = array.get(snapshotFlatBins, snapOffset + b)
                    midMult = profileMinMult + (b + 0.5) * fixedProfileBinWidth
                    if midMult >= profileDisplayMinMult and midMult <= profileDisplayMaxMult
                        widthMs = snapVisibleMax > 0 ? int(math.round(snapshotWidthMs * snapValue / snapVisibleMax)) : 0
                        widthMs := math.max(widthMs, 0)
                        array.push(pts, chart.point.from_time(snapTime + widthMs, midMult))

                array.push(pts, chart.point.from_time(snapTime, profileDisplayMaxMult))

                recentRatio = snapCount > 1 ? s * 1.0 / (snapCount - 1) : 1.0
                ageExtra = int(math.round(12.0 * (1.0 - recentRatio)))
                fillTransp = int(math.min(100.0, snapshotTransparency + ageExtra))
                lineTransp = int(math.max(0.0, fillTransp - 18.0))

                poly = polyline.new(
                     points = pts,
                     curved = false,
                     closed = true,
                     xloc = xloc.bar_time,
                     line_color = color.new(snapshotColor, lineTransp),
                     fill_color = color.new(snapshotColor, fillTransp),
                     line_width = snapshotOutlineWidth
                )
                array.push(snapshotPolylines, poly)
