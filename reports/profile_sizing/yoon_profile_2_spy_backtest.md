# yoon_profile_2 SPY Backtest

- data: SPY 1d, 1993-01-29 ~ 2026-07-02
- cost: fee 5bp/side + slippage 5bp
- intent: stock-style `yoon_profile`; no short, 85~100% long exposure, 100% warmup exposure

## Results

| Strategy | Phase | Period | Total | CAGR | MDD | Sharpe | Avg Weight |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| yoon_profile_2 | all | 1993-01-29~2026-07-02 | +2215.7% | 9.87% | -50.0% | 0.648 | 93.2% |
| SPY B&H | all | 1993-01-29~2026-07-02 | +2988.7% | 10.82% | -55.2% | 0.646 | 100.0% |
| yoon_profile_2 | train | 1993-01-29~2013-02-12 | +352.6% | 7.83% | -50.0% | 0.518 | 93.0% |
| SPY B&H | train | 1993-01-29~2013-02-12 | +400.7% | 8.37% | -55.2% | 0.508 | 100.0% |
| yoon_profile_2 | validation | 2013-02-13~2019-10-18 | +108.4% | 11.62% | -17.5% | 0.975 | 93.8% |
| SPY B&H | validation | 2013-02-13~2019-10-18 | +123.7% | 12.81% | -19.3% | 0.993 | 100.0% |
| yoon_profile_2 | test | 2019-10-21~2026-07-02 | +143.8% | 14.27% | -31.0% | 0.829 | 93.0% |
| SPY B&H | test | 2019-10-21~2026-07-02 | +173.7% | 16.27% | -33.7% | 0.853 | 100.0% |

## Verdict

`yoon_profile_2` fixes the stock-market failure mode of `yoon_profile_1`: average
exposure rises from roughly one-third to roughly 93%, and the strategy no longer shorts
equity drawdowns. It reduces MDD and volatility versus SPY B&H, and all-period Sharpe is
roughly tied with B&H.

It does not beat SPY B&H on CAGR or total return. Treat it as a lower-drawdown SPY timing
variant, not as a superior replacement for passive SPY.
