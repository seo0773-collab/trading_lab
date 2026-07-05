# yoon2_final_fx_2 overlay report

- Period: `2007-01-29~2026-07-02` (4084 bars)
- Scope: lower yoon2_final sleeves frozen; only top-level USD/KRW/cash allocation is changed.

## Candidate Results

| candidate | CAGR | MDD | Sharpe | Calmar | USD mean | Cash mean/max | Cash days |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fast_override_only | 16.08% | -17.13% | 1.181 | 0.938 | 51.07% | 0.00%/0.00% | 0 |
| fast_vol | 15.70% | -17.11% | 1.187 | 0.917 | 49.94% | 0.00%/0.00% | 0 |
| final_fx_current | 16.27% | -17.13% | 1.189 | 0.950 | 50.91% | 0.00%/0.00% | 0 |
| fx2_all | 15.44% | -17.11% | 1.176 | 0.902 | 49.12% | 1.66%/35.00% | 194 |
| fx2_all_sleeve | 15.54% | -17.11% | 1.179 | 0.908 | 49.58% | 0.75%/35.00% | 88 |
| gate_fast | 15.81% | -17.13% | 1.169 | 0.923 | 50.20% | 1.66%/35.00% | 194 |
| gate_only | 16.00% | -17.13% | 1.177 | 0.934 | 50.04% | 1.66%/35.00% | 194 |
| gate_sleeve_only | 16.11% | -17.13% | 1.180 | 0.940 | 50.54% | 0.75%/35.00% | 88 |
| gate_vol | 15.57% | -17.11% | 1.182 | 0.910 | 49.02% | 1.66%/35.00% | 194 |
| static_50_50 | 15.11% | -16.94% | 1.165 | 0.892 | 50.00% | 0.00%/0.00% | 0 |
| vol_adjust_only | 15.83% | -17.11% | 1.193 | 0.925 | 49.84% | 0.00%/0.00% | 0 |

## Bootstrap

- `fx2_all` vs `final_fx_current`: P(Sharpe>) `21.0%`, P(CAGR>) `0.1%`, P(MDD shallower) `60.7%`
- `fx2_all` vs `static_50_50`: P(Sharpe>) `71.3%`, P(CAGR>) `83.6%`, P(MDD shallower) `37.5%`
- `gate_sleeve_only` vs `final_fx_current`: P(Sharpe>) `5.0%`, P(CAGR>) `3.6%`, P(MDD shallower) `43.1%`
- `fx2_all_sleeve` vs `final_fx_current`: P(Sharpe>) `22.8%`, P(CAGR>) `0.1%`, P(MDD shallower) `61.0%`
- `fx2_all_sleeve` vs `fx2_all` (bench gate): P(Sharpe>) `62.0%`, P(CAGR>) `76.7%`, P(MDD shallower) `49.1%`

## Walk-Forward

- Folds: `14`
- Avg current: CAGR `21.68%`, MDD `-8.74%`, Sharpe `1.283`
- Avg fx2_all: CAGR `20.01%`, MDD `-8.73%`, Sharpe `1.250`
- fx2_all Sharpe wins vs current: `6/14`
- fx2_all MDD wins vs current: `8/14`
- Avg fx2_all_sleeve: CAGR `20.13%`, MDD `-8.70%`, Sharpe `1.261`
- fx2_all_sleeve Sharpe wins vs current: `5/14`
- fx2_all_sleeve MDD wins vs current: `8/14`
- Avg gate_sleeve_only: CAGR `21.47%`, MDD `-8.74%`, Sharpe `1.271`
- gate_sleeve_only Sharpe wins vs current: `2/14`
- gate_sleeve_only MDD wins vs current: `2/14`

## Parameter Sensitivity

Top candidates with MDD not worse than current:

| candidate | CAGR | MDD | Sharpe | Cash mean |
| --- | ---: | ---: | ---: | ---: |
| vol_blend_0.35 | 15.83% | -17.11% | 1.193 | 0.00% |
| vol_blend_0.5 | 15.63% | -17.10% | 1.193 | 0.00% |
| vol_blend_0.3 | 15.89% | -17.11% | 1.193 | 0.00% |
| vol_blend_0.25 | 15.95% | -17.12% | 1.192 | 0.00% |
| vol_blend_0.2 | 16.02% | -17.12% | 1.192 | 0.00% |
| vol_blend_0.15 | 16.08% | -17.12% | 1.191 | 0.00% |
| vol_blend_0.1 | 16.15% | -17.13% | 1.190 | 0.00% |
| combo_vol_0.25_gate_0.8_dd_-0.15 | 15.84% | -17.12% | 1.189 | 0.64% |

Top candidates by shallowest MDD:

| candidate | CAGR | MDD | Sharpe | Cash mean |
| --- | ---: | ---: | ---: | ---: |
| vol_blend_0.5 | 15.63% | -17.10% | 1.193 | 0.00% |
| vol_blend_0.35 | 15.83% | -17.11% | 1.193 | 0.00% |
| vol_blend_0.3 | 15.89% | -17.11% | 1.193 | 0.00% |
| vol_blend_0.25 | 15.95% | -17.12% | 1.192 | 0.00% |
| combo_vol_0.25_gate_0.8_dd_-0.15 | 15.84% | -17.12% | 1.189 | 0.64% |
| combo_vol_0.25_gate_0.65_dd_-0.15 | 15.76% | -17.12% | 1.185 | 1.12% |
| combo_vol_0.25_gate_0.8_dd_-0.1 | 15.81% | -17.12% | 1.186 | 0.89% |
| combo_vol_0.25_gate_0.8_dd_-0.12 | 15.81% | -17.12% | 1.186 | 0.75% |

### Sleeve-based gate sensitivity (`ctx.navs` top-3 sleeves instead of SPY/KODEX200)

Top candidates with MDD not worse than current:

| candidate | CAGR | MDD | Sharpe | Cash mean |
| --- | ---: | ---: | ---: | ---: |
| sleeve_gate_off_0.5_dd_-0.15_vol_0.3 | 16.27% | -17.13% | 1.189 | 0.18% |
| sleeve_gate_off_0.5_dd_-0.15_vol_0.4 | 16.27% | -17.13% | 1.189 | 0.18% |
| sleeve_gate_off_0.5_dd_-0.15_vol_0.5 | 16.27% | -17.13% | 1.189 | 0.18% |
| sleeve_gate_off_0.5_dd_-0.2_vol_0.3 | 16.27% | -17.13% | 1.189 | 0.01% |
| sleeve_gate_off_0.5_dd_-0.2_vol_0.4 | 16.27% | -17.13% | 1.189 | 0.01% |
| sleeve_gate_off_0.5_dd_-0.2_vol_0.5 | 16.27% | -17.13% | 1.189 | 0.01% |
| sleeve_gate_off_0.5_dd_-0.25_vol_0.3 | 16.27% | -17.13% | 1.189 | 0.00% |
| sleeve_gate_off_0.5_dd_-0.25_vol_0.4 | 16.27% | -17.13% | 1.189 | 0.00% |

Top candidates by shallowest MDD:

| candidate | CAGR | MDD | Sharpe | Cash mean |
| --- | ---: | ---: | ---: | ---: |
| sleeve_gate_off_0.5_dd_-0.15_vol_0.3 | 16.27% | -17.13% | 1.189 | 0.18% |
| sleeve_gate_off_0.5_dd_-0.15_vol_0.4 | 16.27% | -17.13% | 1.189 | 0.18% |
| sleeve_gate_off_0.5_dd_-0.15_vol_0.5 | 16.27% | -17.13% | 1.189 | 0.18% |
| sleeve_gate_off_0.5_dd_-0.2_vol_0.3 | 16.27% | -17.13% | 1.189 | 0.01% |
| sleeve_gate_off_0.5_dd_-0.2_vol_0.4 | 16.27% | -17.13% | 1.189 | 0.01% |
| sleeve_gate_off_0.5_dd_-0.2_vol_0.5 | 16.27% | -17.13% | 1.189 | 0.01% |
| sleeve_gate_off_0.5_dd_-0.25_vol_0.3 | 16.27% | -17.13% | 1.189 | 0.00% |
| sleeve_gate_off_0.5_dd_-0.25_vol_0.4 | 16.27% | -17.13% | 1.189 | 0.00% |

## Drawdown Episodes

| peak | trough | current MDD | fx2(bench) MDD | fx2(sleeve) MDD | gate_sleeve MDD | static MDD | fx2-current | sleeve-current | gate_sleeve cash mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2011-01-27 | 2011-03-15 | -6.26% | -6.32% | -6.32% | -6.26% | -5.49% | -0.06% | -0.06% | 0.00% |
| 2011-05-02 | 2011-08-22 | -16.40% | -16.11% | -16.11% | -16.40% | -15.87% | 0.29% | 0.29% | 1.40% |
| 2014-03-05 | 2014-05-19 | -6.04% | -5.99% | -5.99% | -6.04% | -5.83% | 0.05% | 0.05% | 0.00% |
| 2015-07-21 | 2016-01-20 | -9.65% | -9.53% | -9.53% | -9.65% | -9.29% | 0.12% | 0.12% | 0.29% |
| 2018-01-29 | 2018-12-24 | -10.42% | -10.70% | -10.70% | -10.42% | -10.73% | -0.28% | -0.28% | 0.17% |
| 2020-02-14 | 2020-03-19 | -17.13% | -17.11% | -17.11% | -17.13% | -16.94% | 0.02% | 0.02% | 14.58% |
| 2020-09-02 | 2020-10-30 | -8.75% | -8.98% | -8.98% | -8.75% | -9.09% | -0.23% | -0.23% | 0.00% |
| 2021-01-25 | 2021-01-29 | -6.29% | -6.12% | -6.12% | -6.29% | -5.75% | 0.16% | 0.16% | 0.00% |
| 2022-01-12 | 2022-01-27 | -7.39% | -7.36% | -7.36% | -7.39% | -7.32% | 0.02% | 0.02% | 0.00% |
| 2022-04-20 | 2023-01-03 | -9.53% | -10.57% | -10.08% | -9.59% | -10.17% | -1.04% | -0.56% | 1.68% |
| 2023-07-28 | 2023-11-09 | -11.01% | -11.17% | -11.17% | -11.01% | -11.47% | -0.16% | -0.16% | 4.31% |
| 2024-07-16 | 2024-08-05 | -12.00% | -11.90% | -11.90% | -12.00% | -12.21% | 0.10% | 0.10% | 0.00% |
| 2025-02-19 | 2025-04-08 | -10.32% | -10.20% | -10.20% | -10.32% | -10.14% | 0.12% | 0.12% | 0.00% |
| 2025-11-13 | 2025-11-21 | -6.43% | -5.98% | -5.98% | -6.43% | -5.78% | 0.45% | 0.45% | 0.00% |
| 2026-02-26 | 2026-03-04 | -10.57% | -8.95% | -8.95% | -10.57% | -8.25% | 1.62% | 1.62% | 0.00% |
| 2026-05-14 | 2026-05-20 | -6.51% | -5.66% | -5.66% | -6.51% | -5.29% | 0.85% | 0.85% | 0.00% |
| 2026-06-04 | 2026-06-08 | -6.38% | -5.57% | -5.57% | -6.38% | -5.22% | 0.81% | 0.81% | 0.00% |

## Verdict

fx2_all reduces full-sample MDD versus final_fx_current, but the added cash gate also lowers CAGR/Sharpe. Treat it as a defensive research variant, not a promotion candidate, unless the objective is explicitly lower drawdown over return. Sleeve-based gate (gate_sleeve_only: CAGR 16.11%, MDD -17.13%, Sharpe 1.180; fx2_all_sleeve: CAGR 15.54%, MDD -17.11%, Sharpe 1.179) is compared against the bench-based gate above; see Bootstrap/Drawdown Episodes sections for which signal source actually catches the simultaneous-downturn cases.
