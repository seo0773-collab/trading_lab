# Adaptive 3x3 field profile backtest

- Calibration: first 756 valid 3x3 bars after indicator warmup
- Calibration stats exclude the last 60 bars to avoid forward-return leakage
- OOS: fixed selected profile, no refit
- Costs: fee 5bp + slippage 5bp per turnover, same engine convention as yoon_profile

## Selection

| symbol | kind | selected | reason | OOS start |
| --- | --- | --- | --- | --- |
| SPY | etf | balanced | 혼합형: M0A0 +13.2%, M1A1 +10.2%, M2A2 +4.3% | 1998-11-11 |
| QQQ | etf | balanced | 혼합형: M0A0 +19.1%, M1A1 -11.8%, M2A2 +6.1% | 2004-12-31 |
| GLD | etf | balanced | 혼합형: M0A0 +7.2%, M1A1 +5.6%, M2A2 +5.5% | 2010-09-10 |
| XLK | etf | balanced | 혼합형: M0A0 +15.9%, M1A1 -14.4%, M2A2 +6.2% | 2004-10-18 |
| XLV | etf | drift_soft | M0A0 양수(+8.3%) + 대부분 셀 양수(7/9) | 2004-10-18 |
| XLF | etf | balanced | 혼합형: M0A0 +3.3%, M1A1 -1.0%, M2A2 +4.1% | 2004-10-18 |
| XLY | etf | balanced | 혼합형: M0A0 +3.9%, M1A1 -5.5%, M2A2 +2.0% | 2004-10-18 |
| XLC | etf | falling_knife_defense | M0A0 음수(-5.1%)이나 M2A2 우위 약함 | 2024-04-11 |
| XLI | etf | crypto_trend | M0A0 음수(-1.2%) + M2A2 모멘텀 우위(+5.5%) | 2004-10-18 |
| XLP | etf | falling_knife_defense | M0A0 음수(-2.4%)이나 M2A2 우위 약함 | 2004-10-18 |
| XLE | etf | balanced | 혼합형: M0A0 +1.9%, M1A1 -16.6%, M2A2 +6.2% | 2004-10-18 |
| XLU | etf | falling_knife_defense | M0A0 음수(-3.0%)이나 M2A2 우위 약함 | 2004-10-18 |
| XLB | etf | balanced | 혼합형: M0A0 +6.5%, M1A1 +7.4%, M2A2 +2.1% | 2004-10-18 |
| XLRE | etf | drift_soft | M0A0 양수(+7.1%) + 대부분 셀 양수(7/9) | 2021-07-30 |
| BTC | crypto | balanced | 혼합형: M0A0 +nan%, M1A1 +9.6%, M2A2 +36.8% | 2021-08-18 |
| ETH | crypto | falling_knife_defense | M0A0 음수(-6.5%)이나 M2A2 우위 약함 | 2024-01-02 |
| SOL | crypto | balanced | 혼합형: M0A0 +7.3%, M1A1 +26.1%, M2A2 +56.9% | 2024-08-12 |

## OOS performance

| symbol | variant | profile | total | CAGR | Sharpe | MDD | avg exp |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| SPY | adaptive_profile | balanced | +413.1% | +6.1% | 0.533 | -35.8% | +75.1% |
| SPY | calib_tilt | calib_tilt | +764.4% | +8.1% | 0.543 | -50.0% | +91.7% |
| SPY | class_baseline | drift_soft | +752.3% | +8.1% | 0.539 | -50.0% | +92.2% |
| SPY | bnh | buy_hold | +973.8% | +9.0% | 0.543 | -55.2% | +100.0% |
| QQQ | adaptive_profile | balanced | +993.1% | +11.8% | 0.733 | -37.4% | +85.9% |
| QQQ | calib_tilt | calib_tilt | +1510.6% | +13.8% | 0.741 | -49.7% | +94.8% |
| QQQ | class_baseline | drift_soft | +1566.1% | +14.0% | 0.752 | -48.9% | +95.1% |
| QQQ | bnh | buy_hold | +1997.5% | +15.2% | 0.764 | -53.4% | +100.0% |
| GLD | adaptive_profile | balanced | +182.4% | +6.8% | 0.542 | -32.5% | +77.5% |
| GLD | calib_tilt | calib_tilt | +206.5% | +7.4% | 0.529 | -41.9% | +93.7% |
| GLD | class_baseline | drift_soft | +199.8% | +7.2% | 0.521 | -42.3% | +93.5% |
| GLD | bnh | buy_hold | +210.6% | +7.4% | 0.515 | -45.6% | +100.0% |
| XLK | adaptive_profile | balanced | +1322.3% | +13.0% | 0.764 | -37.5% | +87.5% |
| XLK | calib_tilt | calib_tilt | +1844.9% | +14.7% | 0.755 | -49.2% | +95.4% |
| XLK | class_baseline | drift_soft | +1907.4% | +14.8% | 0.764 | -48.7% | +95.8% |
| XLK | bnh | buy_hold | +2349.1% | +15.9% | 0.769 | -53.0% | +100.0% |
| XLV | adaptive_profile | drift_soft | +616.3% | +9.5% | 0.663 | -35.1% | +95.2% |
| XLV | calib_tilt | calib_tilt | +610.9% | +9.5% | 0.662 | -35.6% | +93.8% |
| XLV | class_baseline | drift_soft | +616.3% | +9.5% | 0.663 | -35.1% | +95.2% |
| XLV | bnh | buy_hold | +745.1% | +10.4% | 0.675 | -39.2% | +100.0% |
| XLF | adaptive_profile | balanced | +251.4% | +6.0% | 0.408 | -57.0% | +79.8% |
| XLF | calib_tilt | calib_tilt | +270.3% | +6.2% | 0.365 | -76.7% | +93.2% |
| XLF | class_baseline | drift_soft | +269.8% | +6.2% | 0.364 | -76.8% | +93.4% |
| XLF | bnh | buy_hold | +275.6% | +6.3% | 0.356 | -82.7% | +100.0% |
| XLY | adaptive_profile | balanced | +474.5% | +8.4% | 0.593 | -39.7% | +81.7% |
| XLY | calib_tilt | calib_tilt | +706.5% | +10.1% | 0.581 | -54.7% | +93.9% |
| XLY | class_baseline | drift_soft | +713.8% | +10.2% | 0.588 | -53.8% | +94.1% |
| XLY | bnh | buy_hold | +867.0% | +11.0% | 0.589 | -59.0% | +100.0% |
| XLC | adaptive_profile | falling_knife_defense | +20.2% | +8.7% | 0.813 | -10.5% | +61.5% |
| XLC | calib_tilt | calib_tilt | +32.2% | +13.4% | 0.864 | -17.3% | +94.7% |
| XLC | class_baseline | drift_soft | +32.2% | +13.4% | 0.864 | -17.3% | +94.7% |
| XLC | bnh | buy_hold | +36.1% | +14.9% | 0.913 | -18.0% | +100.0% |
| XLI | adaptive_profile | crypto_trend | +301.2% | +6.6% | 0.607 | -26.3% | +66.3% |
| XLI | calib_tilt | calib_tilt | +743.2% | +10.3% | 0.619 | -57.5% | +94.1% |
| XLI | class_baseline | drift_soft | +750.2% | +10.4% | 0.620 | -57.5% | +94.2% |
| XLI | bnh | buy_hold | +883.5% | +11.1% | 0.612 | -62.3% | +100.0% |
| XLP | adaptive_profile | falling_knife_defense | +60.7% | +2.2% | 0.326 | -16.0% | +53.4% |
| XLP | calib_tilt | calib_tilt | +465.2% | +8.3% | 0.679 | -30.2% | +94.0% |
| XLP | class_baseline | drift_soft | +466.2% | +8.3% | 0.678 | -30.1% | +94.3% |
| XLP | bnh | buy_hold | +591.3% | +9.3% | 0.704 | -32.4% | +100.0% |
| XLE | adaptive_profile | balanced | +363.6% | +7.3% | 0.438 | -50.5% | +78.3% |
| XLE | calib_tilt | calib_tilt | +428.0% | +8.0% | 0.421 | -66.2% | +92.7% |
| XLE | class_baseline | drift_soft | +435.4% | +8.0% | 0.422 | -66.0% | +93.7% |
| XLE | bnh | buy_hold | +466.6% | +8.3% | 0.419 | -71.3% | +100.0% |
| XLU | adaptive_profile | falling_knife_defense | +109.3% | +3.5% | 0.411 | -25.7% | +51.2% |
| XLU | calib_tilt | calib_tilt | +549.6% | +9.0% | 0.591 | -42.3% | +94.3% |
| XLU | class_baseline | drift_soft | +546.2% | +9.0% | 0.590 | -42.4% | +94.0% |
| XLU | bnh | buy_hold | +660.5% | +9.8% | 0.597 | -46.5% | +100.0% |
| XLB | adaptive_profile | balanced | +206.3% | +5.3% | 0.398 | -42.0% | +78.3% |
| XLB | calib_tilt | calib_tilt | +409.3% | +7.8% | 0.461 | -55.3% | +93.3% |
| XLB | class_baseline | drift_soft | +400.4% | +7.7% | 0.460 | -54.5% | +93.3% |
| XLB | bnh | buy_hold | +521.9% | +8.8% | 0.480 | -59.8% | +100.0% |
| XLRE | adaptive_profile | drift_soft | +13.0% | +2.5% | 0.229 | -31.2% | +93.8% |
| XLRE | calib_tilt | calib_tilt | +13.7% | +2.7% | 0.236 | -31.5% | +93.9% |
| XLRE | class_baseline | drift_soft | +13.0% | +2.5% | 0.229 | -31.2% | +93.8% |
| XLRE | bnh | buy_hold | +13.9% | +2.7% | 0.234 | -34.1% | +100.0% |
| BTC | adaptive_profile | balanced | +94.3% | +9.9% | 0.447 | -61.1% | +78.1% |
| BTC | calib_tilt | calib_tilt | +174.2% | +15.3% | 0.699 | -34.8% | +47.8% |
| BTC | class_baseline | crypto_base | +193.3% | +16.4% | 0.714 | -38.7% | +50.7% |
| BTC | bnh | buy_hold | +38.7% | +4.7% | 0.325 | -76.6% | +100.0% |
| ETH | adaptive_profile | falling_knife_defense | -33.6% | -10.9% | -0.232 | -56.8% | +41.8% |
| ETH | calib_tilt | calib_tilt | -37.2% | -12.3% | -0.204 | -58.1% | +45.3% |
| ETH | class_baseline | crypto_base | -38.2% | -12.7% | -0.216 | -59.5% | +46.5% |
| ETH | bnh | buy_hold | -28.7% | -9.1% | 0.121 | -67.5% | +100.0% |
| SOL | adaptive_profile | balanced | -38.8% | -16.8% | -0.083 | -65.8% | +76.9% |
| SOL | calib_tilt | calib_tilt | -16.5% | -6.6% | 0.060 | -50.3% | +51.6% |
| SOL | class_baseline | crypto_base | -11.0% | -4.3% | 0.113 | -46.6% | +51.2% |
| SOL | bnh | buy_hold | -53.6% | -25.1% | -0.116 | -76.3% | +100.0% |

## Verdict table

| symbol | variant | vs class baseline | vs B&H | verdict |
| --- | --- | ---: | ---: | --- |
| BTC | adaptive_profile | CAGR -6.58p / Sharpe -0.267 | CAGR +5.12p / Sharpe +0.122 | baseline 열위 |
| BTC | calib_tilt | CAGR -1.10p / Sharpe -0.014 | CAGR +10.61p / Sharpe +0.375 | baseline 열위 |
| ETH | adaptive_profile | CAGR +1.76p / Sharpe -0.017 | CAGR -1.82p / Sharpe -0.354 | 혼재 |
| ETH | calib_tilt | CAGR +0.37p / Sharpe +0.012 | CAGR -3.20p / Sharpe -0.326 | baseline 대비 개선 |
| SOL | adaptive_profile | CAGR -12.57p / Sharpe -0.197 | CAGR +8.21p / Sharpe +0.033 | baseline 열위 |
| SOL | calib_tilt | CAGR -2.28p / Sharpe -0.053 | CAGR +18.50p / Sharpe +0.177 | baseline 열위 |
| GLD | adaptive_profile | CAGR -0.40p / Sharpe +0.021 | CAGR -0.65p / Sharpe +0.027 | 혼재 |
| GLD | calib_tilt | CAGR +0.15p / Sharpe +0.008 | CAGR -0.09p / Sharpe +0.014 | baseline 대비 개선 |
| QQQ | adaptive_profile | CAGR -2.22p / Sharpe -0.019 | CAGR -3.45p / Sharpe -0.031 | baseline 열위 |
| QQQ | calib_tilt | CAGR -0.18p / Sharpe -0.012 | CAGR -1.41p / Sharpe -0.023 | baseline 열위 |
| SPY | adaptive_profile | CAGR -1.97p / Sharpe -0.006 | CAGR -2.88p / Sharpe -0.010 | baseline 열위 |
| SPY | calib_tilt | CAGR +0.06p / Sharpe +0.004 | CAGR -0.85p / Sharpe -0.000 | baseline 대비 개선 |
| XLB | adaptive_profile | CAGR -2.41p / Sharpe -0.062 | CAGR -3.50p / Sharpe -0.082 | baseline 열위 |
| XLB | calib_tilt | CAGR +0.09p / Sharpe +0.002 | CAGR -1.00p / Sharpe -0.019 | baseline 대비 개선 |
| XLC | adaptive_profile | CAGR -4.77p / Sharpe -0.051 | CAGR -6.26p / Sharpe -0.100 | baseline 열위 |
| XLC | calib_tilt | CAGR +0.00p / Sharpe +0.000 | CAGR -1.49p / Sharpe -0.049 | 혼재 |
| XLE | adaptive_profile | CAGR -0.72p / Sharpe +0.016 | CAGR -1.00p / Sharpe +0.019 | 혼재 |
| XLE | calib_tilt | CAGR -0.07p / Sharpe -0.001 | CAGR -0.35p / Sharpe +0.002 | baseline 열위 |
| XLF | adaptive_profile | CAGR -0.25p / Sharpe +0.043 | CAGR -0.33p / Sharpe +0.052 | 혼재 |
| XLF | calib_tilt | CAGR +0.01p / Sharpe +0.000 | CAGR -0.07p / Sharpe +0.009 | baseline 대비 개선 |
| XLI | adaptive_profile | CAGR -3.76p / Sharpe -0.013 | CAGR -4.50p / Sharpe -0.005 | baseline 열위 |
| XLI | calib_tilt | CAGR -0.04p / Sharpe -0.002 | CAGR -0.79p / Sharpe +0.006 | baseline 열위 |
| XLK | adaptive_profile | CAGR -1.81p / Sharpe +0.000 | CAGR -2.87p / Sharpe -0.004 | 혼재 |
| XLK | calib_tilt | CAGR -0.17p / Sharpe -0.008 | CAGR -1.23p / Sharpe -0.013 | baseline 열위 |
| XLP | adaptive_profile | CAGR -6.11p / Sharpe -0.352 | CAGR -7.12p / Sharpe -0.378 | baseline 열위 |
| XLP | calib_tilt | CAGR -0.01p / Sharpe +0.001 | CAGR -1.01p / Sharpe -0.025 | baseline 대비 개선 |
| XLRE | adaptive_profile | CAGR +0.00p / Sharpe +0.000 | CAGR -0.17p / Sharpe -0.005 | 혼재 |
| XLRE | calib_tilt | CAGR +0.13p / Sharpe +0.007 | CAGR -0.03p / Sharpe +0.002 | baseline 대비 개선 |
| XLU | adaptive_profile | CAGR -5.52p / Sharpe -0.179 | CAGR -6.35p / Sharpe -0.186 | baseline 열위 |
| XLU | calib_tilt | CAGR +0.03p / Sharpe +0.001 | CAGR -0.80p / Sharpe -0.005 | baseline 대비 개선 |
| XLV | adaptive_profile | CAGR +0.00p / Sharpe +0.000 | CAGR -0.84p / Sharpe -0.012 | 혼재 |
| XLV | calib_tilt | CAGR -0.04p / Sharpe -0.001 | CAGR -0.88p / Sharpe -0.014 | baseline 열위 |
| XLY | adaptive_profile | CAGR -1.76p / Sharpe +0.006 | CAGR -2.64p / Sharpe +0.005 | 혼재 |
| XLY | calib_tilt | CAGR -0.05p / Sharpe -0.007 | CAGR -0.93p / Sharpe -0.008 | baseline 열위 |

## Notes

- This is a first-pass initial-calibration design, not a production strategy.
- The profile selector is rule-based and predeclared in the script; it does not grid-search weights.
- ETF baseline is `drift_soft`; crypto baseline is the existing yoon_profile-style `crypto_base`.
- A stronger next test is annual walk-forward reclassification with hysteresis.
