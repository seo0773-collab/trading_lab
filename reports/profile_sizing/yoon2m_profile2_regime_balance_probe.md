# yoon2m Regime-Balanced yoon_profile_2 Probe

- period: 2007-01-29 ~ 2026-07-02
- structure: yoon2m top_k=3, monthly rebalance, USD/KRW 50:50, market filter, safe sleeve
- change: replace per-asset profile score with yoon_profile_2-style 3x3 state weights calibrated to yoon2m regime exposure bands

## Candidates

Rows are `mult_bin` 0..2, columns are `angle_bin` 0..2.

| Candidate | Matrix |
| --- | --- |
| profile2_original | `[[0.85,1.00,1.00],[0.95,1.00,0.90],[0.85,0.90,1.00]]` |
| y2m_regime_map | `[[0.30,0.60,0.90],[0.55,0.90,0.75],[0.30,0.55,0.90]]` |
| y2m_balanced | `[[0.35,0.65,0.90],[0.60,0.90,0.80],[0.45,0.65,0.92]]` |
| y2m_defensive | `[[0.30,0.55,0.80],[0.50,0.85,0.70],[0.30,0.50,0.85]]` |
| y2m_kr_soft | `[[0.30,0.60,0.85],[0.55,0.85,0.70],[0.40,0.55,0.88]]` |

## Results

| Strategy | Total | CAGR | MDD | Sharpe | Vol | Avg US Stock | Avg KR Stock |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| yoon2m baseline | +896.8% | 15.24% | -18.06% | 1.153 | 13.04% | - | - |
| profile2_original | +827.5% | 14.73% | -24.61% | 1.093 | 13.40% | 83.6% | 84.0% |
| y2m_regime_map | +773.4% | 14.31% | -15.43% | 1.149 | 12.30% | 74.8% | 74.6% |
| y2m_balanced | +811.9% | 14.61% | -15.43% | 1.150 | 12.55% | 76.2% | 76.2% |
| y2m_defensive | +755.7% | 14.16% | -13.79% | 1.169 | 11.95% | 71.9% | 71.5% |
| y2m_kr_soft | +758.0% | 14.18% | -14.09% | 1.146 | 12.23% | 73.3% | 73.1% |

## Verdict

Regime-balanced state weights fix the high-exposure problem of `profile2_original`.
The best defensive candidate (`y2m_defensive`) improves MDD and Sharpe versus baseline,
but gives up CAGR and total return. Baseline remains the better growth profile; the
regime-balanced variant is useful only if the objective is shallower drawdown and lower
volatility.
