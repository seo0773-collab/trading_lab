# Generalization First-Session Report

Generated: 2026-06-11T16:36:25.392506+00:00

## Status

- Phase: `validation`
- Config: `h72-price-v1`
- Gate 0 baseline reproduction: **PASS**
- Gate 1 execution correctness: **PASS**
- Gate 2 validation feasibility: **PASS**

## Validation Dry Run

- Assets completed: 8
- Crypto/non-crypto assets: 4/4
- Pooled trades: 201
- Asset-equal average net: +11.1bp
- Trade-pooled average net: +2.3bp

| trades | hit_rate | avg_gross_bps | avg_net_bps | median_net_bps | total_return | sharpe | max_drawdown | long_trades | short_trades | symbol | name | asset_class | phase | bars | phase_bars | phase_start | phase_end | entry_after_signal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 37 | 0.5405405405405406 | 17.238530063406042 | -2.7614699365939765 | -12.961680185838969 | -0.03218826783509254 | -0.034399440611425136 | -0.15107675260864517 | 24 | 13 | BTC-USD | BTCUSD_GEN | crypto | validation | 17041 | 5112 | 2025-04-10 08:00:00+00:00 | 2025-11-09 18:00:00+00:00 | True |
| 31 | 0.4838709677419355 | 20.737580090349297 | 0.7375800903492776 | -28.27803176335351 | -0.04898743231731062 | 0.0050515070688079075 | -0.22096611313589642 | 20 | 11 | ETH-USD | ETHUSD_GEN | crypto | validation | 17038 | 5111 | 2025-04-10 07:00:00+00:00 | 2025-11-09 19:00:00+00:00 | True |
| 34 | 0.47058823529411764 | 24.20648995420536 | 4.206489954205342 | -100.74567051996107 | -0.06872143324612834 | 0.024677609777947092 | -0.22140319903605254 | 26 | 8 | SOL-USD | SOLUSD_GEN | crypto | validation | 17040 | 5112 | 2025-04-10 08:00:00+00:00 | 2025-11-09 19:00:00+00:00 | True |
| 28 | 0.42857142857142855 | 2.6387270184022316 | -17.361272981597786 | -95.52123485084327 | -0.10653527712232136 | -0.09646029552770159 | -0.26382531500057416 | 16 | 12 | XRP-USD | XRPUSD_GEN | crypto | validation | 17041 | 5112 | 2025-04-10 08:00:00+00:00 | 2025-11-09 18:00:00+00:00 | True |
| 14 | 0.5 | 78.47212534887713 | 58.47212534887712 | 38.65256564466848 | 0.07533013356634144 | 0.3466761709999397 | -0.09569218363617249 | 9 | 5 | SPY | SPY_GEN | etf | validation | 4800 | 1440 | 2024-10-11 15:30:00-04:00 | 2025-08-13 10:30:00-04:00 | True |
| 12 | 0.5833333333333334 | 66.51743849926542 | 46.51743849926541 | 2.1872989455901277 | 0.04649638683986801 | 0.22488217423405776 | -0.10729533204162955 | 7 | 5 | QQQ | QQQ_GEN | etf | validation | 4801 | 1440 | 2024-10-11 15:30:00-04:00 | 2025-08-13 10:30:00-04:00 | True |
| 8 | 0.5 | 35.70914902613298 | 15.709149026132962 | -45.277338467603606 | 0.01117398812829351 | 0.13545230403343192 | -0.03795453032542184 | 8 | 0 | GLD | GLD_GEN | etf | validation | 4799 | 1440 | 2024-10-11 15:30:00-04:00 | 2025-08-13 10:30:00-04:00 | True |
| 37 | 0.5135135135135135 | 3.2929884902815747 | -16.707011509718427 | -18.849215659160183 | -0.06107343153131939 | -0.7677255995437609 | -0.06632582712527779 | 20 | 17 | EURUSD=X | EURUSD_GEN | fx | validation | 16796 | 5039 | 2024-10-21 23:00:00+01:00 | 2025-08-15 01:00:00+01:00 | True |

## Data Quality

| symbol | name | raw_bars | forecast_bars | start | end | duplicate_index | source | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC-USD | BTCUSD_GEN |  | 17041 | 2024-06-30 07:00:00+00:00 | 2026-06-11 16:00:00+00:00 | 0 | reused | ok |
| ETH-USD | ETHUSD_GEN |  | 17038 | 2024-06-30 07:00:00+00:00 | 2026-06-11 16:00:00+00:00 | 0 | reused | ok |
| SOL-USD | SOLUSD_GEN |  | 17040 | 2024-06-30 07:00:00+00:00 | 2026-06-11 16:00:00+00:00 | 0 | reused | ok |
| XRP-USD | XRPUSD_GEN |  | 17041 | 2024-06-30 07:00:00+00:00 | 2026-06-11 16:00:00+00:00 | 0 | reused | ok |
| SPY | SPY_GEN |  | 4800 | 2023-09-08 12:30:00-04:00 | 2026-06-11 12:30:00-04:00 | 0 | reused | ok |
| QQQ | QQQ_GEN |  | 4801 | 2023-09-08 12:30:00-04:00 | 2026-06-11 12:30:00-04:00 | 0 | reused | ok |
| GLD | GLD_GEN |  | 4799 | 2023-09-08 13:30:00-04:00 | 2026-06-11 12:30:00-04:00 | 0 | reused | ok |
| EURUSD=X | EURUSD_GEN |  | 16796 | 2023-09-20 09:00:00+01:00 | 2026-06-11 17:00:00+01:00 | 0 | reused | ok |

## Scope Limits

- BTC is previously observed and is not untouched evidence.
- All results are validation evidence, not final holdout evidence.
- Random/placebo, robustness, bootstrap, and Gate 3 remain pending.
- Test-role assets were not opened.

## Tests

- Baseline regression, execution timing, fee, opposite exit, and look-ahead invariance tests passed.
- Kalman, flat-chart, vectorbt, Backtrader, and data smoke tests passed.

## Next Exact Step

Run random/placebo controls and leave-one-asset-out analysis before evaluating Gate 3.
