# yoon2_final_fx parameter optimization

- Period: `2007-01-29~2026-07-02` (4084 bars)
- Grid size: `840` parameter sets
- Grid: lookback 21/42/63/84/126/189/252, tilt 0.05/0.10/0.125/0.20/0.30/0.50, scale 0.025/0.05/0.10/0.20, band 45/55 through 0/100.
- Warning: full-sample winners are diagnostic only; use walk-forward and cost sensitivity for promotion decisions.

## Full-sample winners

| objective | label | CAGR | MDD | Sharpe | Calmar |
| --- | --- | --- | --- | --- | --- |
| sharpe | L63_t0.3_s0.025_b30_70 | 19.29% | -21.06% | 1.257 | 0.916 |
| cagr | L63_t0.5_s0.025_b0_100 | 23.86% | -25.71% | 1.212 | 0.928 |
| calmar | L63_t0.5_s0.1_b0_100 | 21.47% | -19.99% | 1.199 | 1.074 |
| sharpe_mdd20 | L63_t0.3_s0.05_b35_65 | 18.17% | -19.81% | 1.239 | 0.917 |
| cagr_mdd20 | L63_t0.5_s0.1_b0_100 | 21.47% | -19.99% | 1.199 | 1.074 |

## Walk-forward objective selection

- Rule: 5 calendar years train, next calendar year test.
- Folds: `14`
- Current avg: CAGR `23.44%`, MDD `-8.86%`, Sharpe `1.313`
- Static avg: CAGR `20.92%`, MDD `-8.77%`, Sharpe `1.279`

| objective | avg CAGR | avg MDD | avg Sharpe | avg Calmar | Sharpe wins | MDD wins |
| --- | --- | --- | --- | --- | --- | --- |
| sharpe | 28.04% | -10.57% | 1.076 | 2.381 | 7/14 | 6/14 |
| cagr | 31.49% | -10.93% | 1.190 | 2.753 | 8/14 | 5/14 |
| calmar | 33.96% | -10.41% | 1.252 | 3.135 | 9/14 | 6/14 |
| sharpe_mdd20 | 28.50% | -10.18% | 1.134 | 2.534 | 7/14 | 6/14 |
| cagr_mdd20 | 35.30% | -10.35% | 1.295 | 3.287 | 9/14 | 5/14 |

## Paired bootstrap

- Iterations/block: `3000` / `20` trading days
- `current_vs_static`: `current` vs `static_50_50`
  P(Sharpe>) `95.1%`, P(CAGR>) `100.0%`, P(MDD shallower) `39.6%`
- `best_sharpe_vs_current`: `L63_t0.3_s0.025_b30_70` vs `current`
  P(Sharpe>) `91.1%`, P(CAGR>) `100.0%`, P(MDD shallower) `28.4%`
- `best_sharpe_mdd20_vs_current`: `L63_t0.3_s0.05_b35_65` vs `current`
  P(Sharpe>) `91.3%`, P(CAGR>) `99.9%`, P(MDD shallower) `32.0%`
- `best_calmar_vs_current`: `L63_t0.5_s0.1_b0_100` vs `current`
  P(Sharpe>) `51.4%`, P(CAGR>) `99.9%`, P(MDD shallower) `11.3%`

## Top-level cost sensitivity

Cost is one-way bps on top-level sleeve turnover at monthly rebalance.

| candidate | cost_bps | CAGR | MDD | Sharpe |
| --- | --- | --- | --- | --- |
| static_50_50 | 0 | 15.49% | -17.89% | 1.169 |
| static_50_50 | 5 | 15.48% | -17.89% | 1.168 |
| static_50_50 | 10 | 15.47% | -17.89% | 1.168 |
| static_50_50 | 25 | 15.45% | -17.89% | 1.166 |
| static_50_50 | 50 | 15.40% | -17.89% | 1.163 |
| current | 0 | 16.90% | -18.42% | 1.209 |
| current | 5 | 16.87% | -18.42% | 1.207 |
| current | 10 | 16.84% | -18.43% | 1.205 |
| current | 25 | 16.75% | -18.44% | 1.200 |
| current | 50 | 16.61% | -18.47% | 1.191 |
| best_sharpe_mdd20 | 0 | 18.17% | -19.81% | 1.239 |
| best_sharpe_mdd20 | 5 | 18.12% | -19.81% | 1.236 |
| best_sharpe_mdd20 | 10 | 18.06% | -19.81% | 1.233 |
| best_sharpe_mdd20 | 25 | 17.90% | -19.83% | 1.223 |
| best_sharpe_mdd20 | 50 | 17.63% | -19.85% | 1.207 |

## Verdict

The optimizer likes wider bands and stronger tanh settings in full sample, but those are not robust enough to replace the current conservative setting. Walk-forward objective selection does not beat the current parameter set cleanly, and MDD defense remains inconsistent. Keep the current final_fx parameters unless a separate out-of-sample gate is added for a more aggressive variant.
