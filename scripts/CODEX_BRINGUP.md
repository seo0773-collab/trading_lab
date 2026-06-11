# Kalman / Flat-Chart Strategy Bring-Up

Updated: 2026-06-11 UTC

## Start Here

Workspace:

```bash
cd ~/trading_lab/scripts
```

Read these files first:

- `CODEX_BRINGUP.md`
- `CODEX_GENERALIZATION_PLAN.md`
- `run_kalman_pipeline.py`
- `conf_filter_backtest.py`
- `kalman.py`
- `flat_chart.py`

The working tree is currently untracked/dirty. Do not discard files or overwrite
unrelated user changes. The active implementations are the files without `_1`.

## Current Goal

Determine whether the Kalman forecast in flat-chart multiple space can become a
tradable price forecast after accounting for the base-cycle slope.

The current leading candidate is:

```text
BTC-USD, 1h, horizon=72, PRICE direction, q=0.85
```

This is a research result, not yet a production-ready strategy. It has only
been tested on one BTC sample with 71 completed trades.

## Implemented Changes

### `run_kalman_pipeline.py`

- Forecast horizons expanded from `(1, 4, 24)` to `(1, 4, 24, 48, 72)`.
- Forecast CSV now includes `mhat`, `sig`, `pup`, and price-band columns for
  h=48 and h=72.
- Probability plot includes all five horizons.
- Imports intentionally use local `scripts/flat_chart.py` and
  `scripts/kalman.py`. The repository's `indicators.flat_chart` and
  `indicators.kalman` modules expose different APIs and cannot replace them.

### `conf_filter_backtest.py`

- Added expected-edge entry filter.
- Added long/short trade counts and hit rates.
- Edge filtering applies only to entry/re-entry, not opposite-signal exits.
- Added MULT, PRICE, CYCLE, ALIGNED, CYCLE-WEAK, and TREND-ALIGNED diagnostics.
- Added MULT/PRICE direction-conflict diagnostics using actual future
  horizon returns.
- Added PRICE edge-filter grid.
- Equity charts include PRICE and ALIGNED and use fee-specific filenames.

Signal definitions:

```python
mult_ret = mhat_h / mult_close - 1
price_ret = price_mid_h / close - 1
cycle_ret = price_ret - mult_ret
```

- `MULT`: direction of `mult_ret`
- `PRICE`: direction of `price_ret`
- `CYCLE`: direction of `cycle_ret`
- `ALIGNED`: trade only when MULT and PRICE agree
- `CYCLE-WEAK`: MULT only when absolute cycle contribution is below its
  walk-forward rolling median
- `TREND-ALIGNED`: MULT only when MULT and cycle directions agree

## Forecast Regeneration

Command used:

```bash
../.venv/bin/python run_kalman_pipeline.py \
  --symbol BTC-USD --interval 1h --period 720d --name BTCUSD_V1
```

Downloaded sample:

```text
17239 bars
2024-06-22 00:00 UTC through 2026-06-11 15:00 UTC
```

Sigma calibration factors:

```text
h=1:  x14.9
h=4:  x11.0
h=24: x3.6
h=48: x2.1
h=72: x1.7
```

The decreasing factor with longer horizons matches the expected pattern.

Key OOS validation:

| Horizon | Direction hit | Top-30% confidence hit | OU naive |
|---|---:|---:|---:|
| 48 | 59.1% | 64.9% | 58.8% |
| 72 | 59.0% | 64.4% | 57.9% |

Forecast output:

```text
../reports/BTCUSD_V1_forecast.csv
```

The CSV contains the required h=48/h=72 columns.

## Backtest Commands

```bash
../.venv/bin/python conf_filter_backtest.py \
  --forecast ../reports/BTCUSD_V1_forecast.csv --horizon 48 --fee-bps 10

../.venv/bin/python conf_filter_backtest.py \
  --forecast ../reports/BTCUSD_V1_forecast.csv --horizon 72 --fee-bps 10

../.venv/bin/python conf_filter_backtest.py \
  --forecast ../reports/BTCUSD_V1_forecast.csv --horizon 48 --fee-bps 2

../.venv/bin/python conf_filter_backtest.py \
  --forecast ../reports/BTCUSD_V1_forecast.csv --horizon 72 --fee-bps 2
```

`fee-bps` is per side. Thus 10 means 20bp round-trip and 2 means 4bp
round-trip.

## Core Results

Average net return per completed trade:

| Strategy | h=48, 10bp | h=72, 10bp | h=48, 2bp | h=72, 2bp |
|---|---:|---:|---:|---:|
| MULT | -57.6bp | -85.1bp | -41.6bp | -69.1bp |
| PRICE | -47.9bp | **+32.1bp** | -31.9bp | **+48.1bp** |
| CYCLE-only | -3.8bp | -13.4bp | +12.2bp | +2.6bp |
| ALIGNED | -58.1bp | -84.1bp | -42.1bp | -68.1bp |
| CYCLE-WEAK | -2.3bp | -21.7bp | +13.7bp | -5.7bp |

Best current result, h=72 PRICE:

| Fee | Trades | Hit rate | Avg net | Total return | Sharpe | MDD |
|---|---:|---:|---:|---:|---:|---:|
| 10bp/side | 71 | 56.3% | +32.1bp | +19.3% | 0.65 | -23.9% |
| 2bp/side | 71 | 56.3% | +48.1bp | +33.7% | 0.96 | -21.3% |

Long/short split for h=72 PRICE:

```text
Long:  32 trades, 59.4% hit
Short: 39 trades, 53.8% hit
```

Gross edge is approximately 52.1bp/trade:

```text
32.1bp net + 20bp round-trip = 52.1bp gross
48.1bp net +  4bp round-trip = 52.1bp gross
```

## Main Diagnosis

Price is:

```text
price = multiple * cycle
```

The original strategy traded the direction of the forecast multiple and
ignored cases where the cycle slope reversed the resulting price direction.

### h=48

MULT and PRICE agreed on 95.6% of high-confidence signals. Only 4.4% conflicted.

Conflict subset:

```text
MULT direction:  36.2% hit, -110.3bp gross average
PRICE direction: 63.8% hit, +110.3bp gross average
```

PRICE conversion helps, but too few directions change to rescue h=48.
CYCLE-WEAK is near break-even at 10bp and positive at 2bp, but remains a weak
and sample-sensitive candidate.

### h=72

MULT and PRICE agreed on 86.1% of signals; 13.9% conflicted.

```text
Direction-agree subset using MULT:
  56.9% hit, +9.8bp gross average

Direction-conflict subset using MULT:
  31.7% hit, -201.0bp gross average

Same conflict subset using PRICE:
  68.3% hit, +201.0bp gross average
```

The small conflict subset caused much of the original MULT loss. PRICE does
not merely remove those trades; it reverses their direction.

ALIGNED fails because it discards this valuable conflict subset. Do not treat
agreement as the preferred filter.

CYCLE-only is not profitable at 10bp, so h=72 PRICE is not simply a cycle
trend-following strategy. The useful signal comes from combining the Kalman
multiple forecast with the extrapolated cycle in price space.

## Price Forecast Construction

There is no look-ahead in the current `price_mid_h` calculation.

At each current bar:

```python
cycle_slope = cycle.diff().ewm(span=24).mean()
cycle_k = cycle_now + cycle_slope_now * horizon
price_mid_h = mhat_h * cycle_k
```

Only information available at the current bar is used. However, constant
linear slope extrapolation over 72 hours may be unstable or overfit. This is
the main model-risk item for the next round.

## PRICE Edge Filter Result

For h=72 at 10bp/side:

```text
lambda 0:  +32.1bp/trade, Sharpe 0.65, 71 trades
lambda 10: +33.3bp/trade, Sharpe 0.67, 71 trades
lambda 15:  +0.4bp/trade, Sharpe 0.01, 55 trades
```

Higher predicted price edge did not monotonically produce better realized
returns. Do not optimize lambda aggressively on this sample.

At 2bp, lambda 0 through 30 accepts the same h=72 trades because the threshold
is too small relative to forecast price-edge magnitudes.

## Artifacts

```text
../reports/BTCUSD_V1_report.txt
../reports/BTCUSD_V1_forecast.csv
../reports/BTCUSD_V1_plots.png
../reports/BTCUSD_V1_strategy_h48_fee10bp.png
../reports/BTCUSD_V1_strategy_h48_fee2bp.png
../reports/BTCUSD_V1_strategy_h72_fee10bp.png
../reports/BTCUSD_V1_strategy_h72_fee2bp.png
```

The older `BTCUSD_V1_strategy_h48.png` and `h72.png` predate fee-specific chart
filenames and should not be used for the latest comparison.

## Verification Completed

```bash
../.venv/bin/python -m py_compile \
  conf_filter_backtest.py run_kalman_pipeline.py

git diff --check -- conf_filter_backtest.py run_kalman_pipeline.py
```

Both completed successfully.

## Next Work

Priority order:

1. Validate h=72 PRICE on multiple liquid assets without tuning parameters.
2. Use separate chronological train/validation/test periods; the current
   result is one OOS segment but strategy selection was informed by it.
3. Compare cycle-slope estimators:
   - current EWM span 24
   - slower EWM spans
   - damped slope extrapolation
   - capped horizon cycle return
4. Test rolling subperiod stability and report performance by year/regime.
5. Add adverse-selection assumptions before accepting the 2bp maker case.
6. Check execution timing explicitly: signal at close currently enters at the
   same close. A next-bar-open execution test is required.
7. Add bootstrap confidence intervals for 71-trade hit rate and mean return.

Do not conclude that the strategy is deployable until multi-asset,
next-bar-execution, and untouched holdout tests pass.

The implementation order, frozen rules, validation gates, artifact schema, and
next-session prompt are defined in `CODEX_GENERALIZATION_PLAN.md`.

## Suggested Next-Session Prompt

```text
~/trading_lab/scripts/CODEX_BRINGUP.md를 먼저 읽고 현재
코드와 산출물을 확인해줘. 기존 파일이나 사용자 변경을 되돌리지 말고,
Next Work 우선순위에 따라 h=72 PRICE 전략의 다중 자산 및 next-bar-open
검증부터 구현하고 실행한 뒤 결과를 리포트해줘.
```
