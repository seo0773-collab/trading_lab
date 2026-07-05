# yoon2_final_fx validation

- Period: `2007-01-29~2026-07-02` (4084 bars)
- Scope: top-level USD/KRW allocation only; lower yoon2_final sleeves are unchanged.

## Point estimates

| label | CAGR | MDD | Sharpe | USD range |
| --- | ---: | ---: | ---: | ---: |
| static_50_50 | 15.11% | -16.94% | 1.165 | 50.0-50.0% |
| current_L63_t0.125_s0.10_b40_60 | 16.27% | -17.13% | 1.189 | 40.0-60.0% |
| band_0_100 | 16.35% | -17.13% | 1.184 | 37.5-62.4% |
| lookback_21 | 15.39% | -18.22% | 1.147 | 40.0-60.0% |
| lookback_126 | 15.94% | -18.56% | 1.161 | 40.0-60.0% |
| tilt_0.3 | 16.55% | -17.32% | 1.199 | 40.0-60.0% |
| scale_0.05 | 16.46% | -17.25% | 1.195 | 40.0-60.0% |

## Paired moving-block bootstrap

- Comparison: `current_L63_t0.125_s0.10_b40_60` vs `static_50_50`
- Iterations/block: `5000` / `20` trading days
- P(Sharpe improvement): `84.9%`
- P(CAGR improvement): `99.9%`
- P(MDD shallower): `33.9%`

95% paired CI:

- Excess Sharpe: `-0.026` to `0.083`
- Excess CAGR: `0.39%` to `2.14%`
- MDD difference: `-2.26%` to `1.46%`

## Walk-forward

- Grid: lookback 21/42/63/84/126, tilt 0.05/0.125/0.20/0.30, scale 0.05/0.10/0.20, band 40/60.
- Fold rule: select best train Sharpe over 5 calendar years, test on next calendar year.
- Folds: `14`
- Avg selected: CAGR `21.31%`, MDD `-8.63%`, Sharpe `1.266`
- Avg current: CAGR `21.68%`, MDD `-8.74%`, Sharpe `1.283`
- Avg static: CAGR `19.81%`, MDD `-8.61%`, Sharpe `1.262`
- Current Sharpe wins vs static: `10/14`
- Current MDD wins vs static: `6/14`

Most selected train params:

- `L63_t0.3_s0.05`: 5
- `L126_t0.3_s0.05`: 3
- `L21_t0.3_s0.1`: 2
- `L126_t0.2_s0.05`: 1
- `L21_t0.05_s0.2`: 1
- `L21_t0.2_s0.05`: 1
- `L84_t0.2_s0.1`: 1

## Rolling 3-year stability

- Current CAGR wins: `14/16`
- Current Sharpe wins: `13/16`
- Current MDD wins: `7/16`

## Verdict

The dynamic layer has real return/Sharpe value versus fixed 50/50, but its MDD defense is not robust. The 40/60 band is a pragmatic guardrail rather than the alpha source. The sensitive knobs are lookback, tilt, and scale; aggressive tilt improves in-sample point estimates but is not suitable for promotion without a stronger out-of-sample gate.
