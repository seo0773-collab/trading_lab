# Expanded Multi-Asset Validation Report

## Scope

- Frozen strategy: h=72 PRICE, q=0.85, next-bar-open.
- Cost: 10bp per side, 20bp round trip.
- Assets: 4 crypto, 3 ETF, 1 FX.
- Test-role holdout assets were not opened.
- Crypto uses about 720 days of source data; ETF hourly data is limited by the provider to about 5,000 bars.
- For ETFs, 72 bars represent roughly 11 trading days, not 72 clock hours.

## Gate Status

- Gate 0 baseline reproduction: **PASS**
- Gate 1 execution correctness: **PASS**
- Gate 2 feasibility: **PASS**
- Gate 2 evidence: 4 crypto, 4 non-crypto, 201 trades.
- Gate 3 evidence: **NOT EVALUATED**

## Portfolio-Level Result

- Positive assets: 5/8
- Asset-equal average net: +11.1bp/trade
- Trade-pooled average net: +2.3bp/trade
- Net profitable trades: 43.3%
- Average winning trade: +4.21%
- Average losing trade: -3.17%
- Payoff ratio: 1.33:1
- Profit factor: 1.013

The pooled edge is only +2.3bp per trade after 20bp round-trip cost. It is economically thin and not yet statistically validated.

## Asset Results

| asset | class | trades | gross_hit_pct | avg_net_bps | compound_pct | sharpe | mdd_pct | long | short | validation_start | validation_end |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC-USD | crypto | 37 | 54.1 | -2.8 | -3.2 | -0.03 | -15.1 | 24 | 13 | 2025-04-10 08:00:00+00:00 | 2025-11-09 18:00:00+00:00 |
| ETH-USD | crypto | 31 | 48.4 | 0.7 | -4.9 | 0.01 | -22.1 | 20 | 11 | 2025-04-10 07:00:00+00:00 | 2025-11-09 19:00:00+00:00 |
| SOL-USD | crypto | 34 | 47.1 | 4.2 | -6.9 | 0.02 | -22.1 | 26 | 8 | 2025-04-10 08:00:00+00:00 | 2025-11-09 19:00:00+00:00 |
| XRP-USD | crypto | 28 | 42.9 | -17.4 | -10.7 | -0.1 | -26.4 | 16 | 12 | 2025-04-10 08:00:00+00:00 | 2025-11-09 18:00:00+00:00 |
| SPY | etf | 14 | 50.0 | 58.5 | 7.5 | 0.35 | -9.6 | 9 | 5 | 2024-10-11 15:30:00-04:00 | 2025-08-13 10:30:00-04:00 |
| QQQ | etf | 12 | 58.3 | 46.5 | 4.6 | 0.22 | -10.7 | 7 | 5 | 2024-10-11 15:30:00-04:00 | 2025-08-13 10:30:00-04:00 |
| GLD | etf | 8 | 50.0 | 15.7 | 1.1 | 0.14 | -3.8 | 8 | 0 | 2024-10-11 15:30:00-04:00 | 2025-08-13 10:30:00-04:00 |
| EURUSD=X | fx | 37 | 51.4 | -16.7 | -6.1 | -0.77 | -6.6 | 20 | 17 | 2024-10-21 23:00:00+01:00 | 2025-08-15 01:00:00+01:00 |

## Long Versus Short

| side | trades | net_win_pct | avg_net_bps | compound_pct |
| --- | --- | --- | --- | --- |
| LONG | 130 | 46.9 | 36.0 | 37.9 |
| SHORT | 71 | 36.6 | -59.3 | -40.7 |

Longs produced the observed edge. Baseline shorts lost money in six of seven assets that generated short trades.

## Short PnL By Holding Time

| hour | avg_pnl_pct | median_pnl_pct | positive_pct |
| --- | --- | --- | --- |
| 4 | 0.13 | -0.08 | 45.07 |
| 12 | 0.34 | -0.04 | 47.89 |
| 24 | 0.12 | -0.13 | 43.66 |
| 48 | -0.41 | -0.3 | 42.25 |
| 72 | -0.59 | -0.49 | 36.62 |

Short performance was positive through 24 bars on average, then deteriorated sharply by 48-72 bars.

## Conservative Candidate

| candidate | assets | trades | avg_asset_net_bps | positive_assets | short_assets | positive_short_assets |
| --- | --- | --- | --- | --- | --- | --- |
| BASELINE | 8 | 201 | 11.1 | 5 | 7 | 1 |
| LONG_ONLY_ENTRY | 8 | 130 | 38.2 | 7 | 0 | 0 |
| SHORT_Q90_AGREE_24H_25PCT | 8 | 240 | 21.6 | 7 | 7 | 3 |

The 25%-sized q=0.90/agreement/24-bar short candidate improved combined strategy results, but its short leg was positive in only 3 of 7 assets. It fails the predefined majority-of-assets promotion rule.

## Decision

- **Baseline h72 PRICE:** feasible but weak; continue research.
- **Long-only entry:** strongest and most consistent validation variant.
- **Shorts:** keep disabled or paper-only.
- **Holdout:** remain locked until placebo, bootstrap, and leave-one-asset-out tests complete.
