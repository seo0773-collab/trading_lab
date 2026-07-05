# yoon2m top_k optimization

- Structure fixed: yoon1j_lp USD sleeve + yoon1j_kr KRW sleeve, 50/50 monthly, no SSO.
- Sweep: US top_k 1..11, KR top_k 1..8.
- Metrics are ideal continuous-weight backtests, not integer-share execution.
- Defense score MDD cap: -20.00%

## Key Rows

| row | US k | KR k | CAGR | MDD | Sharpe | total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| benchmark | - | - | 16.48% | -31.35% | 1.071 | 1085.61% |
| current yoon2m | 3 | 3 | 15.24% | -18.06% | 1.153 | 896.77% |
| single-per-sleeve | 1 | 1 | 16.87% | -25.27% | 0.975 | 1151.29% |
| selected defense | 4 | 5 | 14.86% | -15.37% | 1.175 | 844.97% |

## Best Defense Score

| US k | KR k | CAGR | MDD | Sharpe | total | ΔCAGR vs bench | ΔMDD vs bench |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 5 | 14.86% | -15.37% | 1.175 | 844.97% | -1.62% | +15.99% |
| 5 | 5 | 14.76% | -14.71% | 1.171 | 830.93% | -1.73% | +16.65% |
| 3 | 5 | 14.82% | -15.30% | 1.173 | 838.94% | -1.66% | +16.05% |
| 6 | 5 | 14.74% | -14.70% | 1.168 | 828.17% | -1.75% | +16.65% |
| 4 | 6 | 14.38% | -14.98% | 1.166 | 782.76% | -2.10% | +16.38% |
| 4 | 8 | 13.84% | -14.61% | 1.165 | 716.81% | -2.65% | +16.74% |
| 5 | 6 | 14.28% | -14.27% | 1.162 | 769.52% | -2.21% | +17.08% |
| 3 | 8 | 13.79% | -14.55% | 1.164 | 711.61% | -2.69% | +16.81% |
| 5 | 8 | 13.73% | -13.90% | 1.161 | 704.57% | -2.75% | +17.45% |
| 3 | 6 | 14.34% | -14.92% | 1.164 | 776.97% | -2.15% | +16.44% |

## Best Sharpe

| US k | KR k | CAGR | MDD | Sharpe | total | ΔCAGR vs bench | ΔMDD vs bench |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 5 | 14.86% | -15.37% | 1.175 | 844.97% | -1.62% | +15.99% |
| 3 | 5 | 14.82% | -15.30% | 1.173 | 838.94% | -1.66% | +16.05% |
| 5 | 5 | 14.76% | -14.71% | 1.171 | 830.93% | -1.73% | +16.65% |
| 6 | 5 | 14.74% | -14.70% | 1.168 | 828.17% | -1.75% | +16.65% |
| 4 | 6 | 14.38% | -14.98% | 1.166 | 782.76% | -2.10% | +16.38% |
| 4 | 8 | 13.84% | -14.61% | 1.165 | 716.81% | -2.65% | +16.74% |
| 3 | 6 | 14.34% | -14.92% | 1.164 | 776.97% | -2.15% | +16.44% |
| 3 | 8 | 13.79% | -14.55% | 1.164 | 711.61% | -2.69% | +16.81% |
| 5 | 6 | 14.28% | -14.27% | 1.162 | 769.52% | -2.21% | +17.08% |
| 7 | 5 | 14.58% | -14.63% | 1.162 | 807.34% | -1.91% | +16.72% |

## Best CAGR

| US k | KR k | CAGR | MDD | Sharpe | total | ΔCAGR vs bench | ΔMDD vs bench |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 16.87% | -25.27% | 0.975 | 1151.29% | +0.39% | +6.08% |
| 1 | 2 | 16.36% | -21.40% | 1.048 | 1065.13% | -0.13% | +9.95% |
| 2 | 1 | 16.25% | -23.35% | 1.014 | 1047.31% | -0.24% | +8.01% |
| 4 | 1 | 16.13% | -22.36% | 1.039 | 1028.43% | -0.35% | +8.99% |
| 3 | 1 | 16.10% | -22.23% | 1.036 | 1023.10% | -0.39% | +9.12% |
| 1 | 3 | 16.09% | -20.78% | 1.059 | 1022.26% | -0.39% | +10.58% |
| 5 | 1 | 16.03% | -21.75% | 1.036 | 1013.36% | -0.45% | +9.60% |
| 6 | 1 | 16.01% | -21.74% | 1.034 | 1010.06% | -0.47% | +9.62% |
| 7 | 1 | 15.85% | -21.69% | 1.029 | 985.74% | -0.63% | +9.66% |
| 1 | 5 | 15.68% | -17.01% | 1.068 | 960.00% | -0.80% | +14.34% |

## Shallowest MDD

| US k | KR k | CAGR | MDD | Sharpe | total | ΔCAGR vs bench | ΔMDD vs bench |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 11 | 8 | 13.04% | -13.67% | 1.136 | 628.77% | -3.45% | +17.69% |
| 7 | 8 | 13.54% | -13.86% | 1.151 | 683.45% | -2.94% | +17.49% |
| 11 | 7 | 13.23% | -13.87% | 1.132 | 648.53% | -3.26% | +17.49% |
| 5 | 8 | 13.73% | -13.90% | 1.161 | 704.57% | -2.75% | +17.45% |
| 6 | 8 | 13.71% | -13.94% | 1.158 | 701.92% | -2.78% | +17.41% |
| 10 | 8 | 13.13% | -13.96% | 1.137 | 637.94% | -3.36% | +17.39% |
| 7 | 7 | 13.73% | -14.04% | 1.146 | 704.61% | -2.75% | +17.31% |
| 9 | 8 | 13.05% | -14.05% | 1.126 | 629.58% | -3.44% | +17.30% |
| 8 | 8 | 13.27% | -14.05% | 1.137 | 652.95% | -3.22% | +17.30% |
| 5 | 7 | 13.92% | -14.06% | 1.157 | 726.11% | -2.57% | +17.30% |

## Verdict

Current yoon2m 3/3 remains a strong balance point if the objective is low-budget concentration with drawdown defense. The defense-score optimum is 4/5, but it should be treated as a research candidate until bootstrap/walk-forward and integer-share execution are rerun.

## Follow-up Validation

Additional validation files:

- `yoon2m_topk45_validate.json`
- `yoon2m_topk45_integer_2m.json`

### Bootstrap / Walk-forward

| row | CAGR | MDD | Sharpe | P(Sharpe > bench) | P(MDD shallower) | rolling 3y MDD wins |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| current 3/3 | 15.24% | -18.06% | 1.153 | 59.3% | 95.6% | 17/18 |
| candidate 4/5 | 14.86% | -15.37% | 1.175 | 65.8% | 97.6% | 18/18 |

4/5 improves the defensive identity: lower MDD, higher Sharpe, stronger MDD bootstrap and
rolling-window defense. It gives up about 0.38%p CAGR versus 3/3.

### 2M KRW Integer-share Probe

This probe uses conservative frictions and integer-share constraints, so the absolute CAGR is
not directly comparable to the ideal NAV. The relative comparison is useful.

| row | ideal CAGR | integer CAGR | integer MDD | integer Sharpe | ideal corr | invested |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| current 3/3 | 15.24% | 9.57% | -19.87% | 0.774 | 0.905 | 91.0% |
| candidate 4/5 | 14.86% | 9.09% | -20.17% | 0.776 | 0.890 | 90.0% |

The practical low-budget layer weakens 4/5: more holdings means slightly lower investment
rate, lower ideal tracking, and lower CAGR. Therefore 4/5 is a defensive research candidate,
not an immediate replacement for yoon2m 3/3.

Final decision: keep yoon2m at 3/3 for now. Register 4/5 only as a paper/validation variant
if a more defensive yoon2m branch is needed.
