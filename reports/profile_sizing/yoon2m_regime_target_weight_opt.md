# yoon2m Regime Target Weight Optimization

- period: 2007-01-29 ~ 2026-07-02
- candidates: 81-grid around yoon2m observed regime exposure bands
- mapping: rows=`mult_bin` 0..2, cols=`angle_bin` 0..2
- parameterized matrix: `[[D,R,N],[C,N,R],[D,C,N]]`
  - `D`: defense-like cells `(0,0),(2,0)`
  - `C`: caution-like cells `(1,0),(2,1)`
  - `R`: recovery-like cells `(0,1),(1,2)`
  - `N`: normal-like cells `(0,2),(1,1),(2,2)`

## Baseline

| Strategy | Total | CAGR | MDD | Sharpe | Vol | Calmar |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| yoon2m baseline | +896.8% | 15.24% | -18.06% | 1.153 | 13.04% | 0.844 |
| internal 50/50 bench | +1085.6% | 16.48% | -31.35% | 1.071 | 15.36% | 0.526 |

## Best Candidates

| Objective | D | C | R | N | Total | CAGR | MDD | Sharpe | Vol | Calmar |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Balanced / best CAGR with MDD better than baseline | 0.35 | 0.55 | 0.70 | 0.90 | +770.8% | 14.29% | -15.38% | 1.151 | 12.25% | 0.929 |
| Best Sharpe | 0.35 | 0.55 | 0.70 | 0.80 | +699.6% | 13.69% | -14.48% | 1.157 | 11.68% | 0.945 |
| Best Calmar | 0.25 | 0.55 | 0.50 | 0.80 | +678.3% | 13.50% | -13.62% | 1.139 | 11.72% | 0.991 |

## Balanced Candidate Matrix

`D=0.35, C=0.55, R=0.70, N=0.90`

```text
[[0.35, 0.70, 0.90],
 [0.55, 0.90, 0.70],
 [0.35, 0.55, 0.90]]
```

Compared with baseline, this gives up about 0.96%p CAGR but improves MDD by about
2.69%p. Sharpe is effectively tied.

## Verdict

The optimized regime target weights do not beat baseline yoon2m on growth. They do create
credible lower-drawdown variants:

- choose baseline yoon2m for growth and total return;
- choose the balanced candidate if MDD reduction is worth roughly 1%p CAGR drag;
- choose the Calmar candidate only for a more conservative sleeve.

This is an in-sample grid search. Treat the selected weights as research candidates, not
production defaults, until they pass walk-forward and bootstrap checks.
