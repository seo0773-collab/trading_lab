# SPY Cycle Reversion Backtest Report

Generated: 2026-06-11 UTC

## Scope
- Period: 2020-01-02 to 2026-06-10 (1,618 daily bars)
- Initial cash: $10,000
- Kalman length 100, profile bins 200, partial target 50%
- Fee 0.1% and slippage 0.1% per order

## Results
| Metric | Strategy | Buy and hold |
|---|---:|---:|
| Ending value | $9,445.43 | $22,329.86 |
| Total return | -5.55% | 123.30% |
| CAGR | -0.88% | 13.29% |
| Maximum drawdown | 16.30% | 34.10% |
| Sharpe | -0.095 | 0.716 |
| Exposure | 47.53% | 100.00% |

## Trading Statistics
- Orders: 115
- Closed/open trade legs: 76/1
- Win rate: 46.05%
- Profit factor: 0.897
- Total fees: $716.75
- POC: 0.9875; lower/upper: 0.9875/1.0125
- Gaussian mu/sigma: 0.9981/0.0233

## Cost Sensitivity
| Cost assumption | Ending value | Total return | Max drawdown |
|---|---:|---:|---:|
| Fee 0.1% + slippage 0.1% | $9,445.43 | -5.55% | 16.30% |
| No costs | $11,029.15 | 10.29% | 16.03% |

## Calendar Returns
| Year | Strategy % | BuyHold % |
| --- | --- | --- |
| 2020 | -10.33 | 15.09 |
| 2021 | 5.71 | 27.04 |
| 2022 | -5.59 | -19.48 |
| 2023 | -0.02 | 24.29 |
| 2024 | 8.41 | 23.30 |
| 2025 | 1.07 | 16.35 |
| 2026 | -3.65 | 6.38 |

## Largest Losing Trade Legs
| Entry | Exit | PnL $ | Return % |
| --- | --- | --- | --- |
| 2020-03-04 | 2020-03-24 | -1134.89 | -22.61 |
| 2022-09-19 | 2022-10-03 | -282.97 | -6.03 |
| 2022-11-30 | 2023-01-06 | -240.37 | -5.19 |
| 2025-02-28 | 2025-03-17 | -240.22 | -4.94 |
| 2025-04-09 | 2025-04-10 | -223.76 | -4.77 |

## Largest Winning Trade Legs
| Entry | Exit | PnL $ | Return % |
| --- | --- | --- | --- |
| 2025-05-27 | 2025-10-27 | 697.49 | 15.47 |
| 2023-12-07 | 2024-04-26 | 460.48 | 10.49 |
| 2021-05-14 | 2021-10-07 | 224.63 | 4.88 |
| 2020-04-22 | 2020-04-29 | 202.27 | 4.64 |
| 2022-11-04 | 2022-11-10 | 201.35 | 4.46 |

## Conclusion
The strategy failed this baseline: negative net return, profit factor below 1, and severe underperformance versus SPY buy-and-hold. Turnover costs are material, and the February-March 2020 loss shows that percentile recovery alone does not prevent entries into a continuing decline.

This is not an out-of-sample result. The profile thresholds use the complete test interval, creating look-ahead bias. A valid next version must use rolling or expanding profiles based only on prior bars, then separate training and evaluation periods.
