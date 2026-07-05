# yoon2m profile -> yoon_profile_2 Replacement Probe

- period: 2007-01-29 ~ 2026-07-02
- structure: keep yoon2m top_k=3, monthly rebalance, USD/KRW 50:50, market filter, safe sleeve
- change: replace per-asset `profile-sizing` score with `yoon_profile_2` target weight

## Portfolio

| Strategy | Total | CAGR | MDD | Sharpe | Vol |
| --- | ---: | ---: | ---: | ---: | ---: |
| yoon2m baseline | +896.8% | 15.24% | -18.06% | 1.153 | 13.04% |
| yoon2m profile2 replace | +827.5% | 14.73% | -24.61% | 1.093 | 13.40% |
| yoon2m internal 50/50 bench | +1085.6% | 16.48% | -31.35% | 1.071 | 15.36% |
| SPY B&H KRW | +1157.9% | 16.91% | -30.15% | 0.847 | 21.08% |

## Sleeve Split

| Sleeve | Model | Total | CAGR | MDD | Sharpe |
| --- | --- | ---: | ---: | ---: | ---: |
| US | baseline profile | +1098.2% | 16.56% | -18.16% | 0.909 |
| US | yoon_profile_2 | +1293.1% | 17.65% | -20.55% | 0.957 |
| KR | baseline profile | +573.1% | 12.49% | -41.51% | 0.738 |
| KR | yoon_profile_2 | +391.4% | 10.32% | -61.07% | 0.584 |

## Verdict

Full replacement is worse than baseline. The US sleeve improves modestly, but the KR sleeve
deteriorates sharply. yoon2m's current profile layer is not merely a weak single profile;
combined with the market filter and safe sleeve, it contributes to the low-drawdown portfolio
shape. If this path is continued, test a US-only replacement or market-specific state weights
for KR instead of replacing both sleeves with the same `yoon_profile_2` matrix.
