# yoon2_final_fx_credit_gate 2차 개선 검증

- current(고정 50/50 없는 순수 yoon2_final_fx): CAGR `16.27%`, MDD `-17.13%`, Sharpe `1.189`

## Part 2. 신규 기능 3종 (실제 Yoon1kHandler로 실데이터 검증)

| variant | CAGR | MDD | Sharpe |
| --- | ---: | ---: | ---: |
| v1_baseline (등록됨, min_consec=2) | 16.60% | -16.40% | 1.228 |
| v2_safe_ief | 16.84% | -16.40% | 1.242 |
| v3_credit_and_vix | 16.34% | -16.40% | 1.202 |
| v3b_credit_and_vix_safe_ief | 16.51% | -16.40% | 1.212 |
| v4_continuous_cap1.5 | 16.35% | -16.40% | 1.206 |
| v4_continuous_cap2.0 | 16.35% | -16.40% | 1.205 |
| v4_continuous_cap3.0 | 16.33% | -16.40% | 1.200 |
| v5_continuous_cap2_safe_ief | 16.58% | -16.40% | 1.219 |

### 부트스트랩 (5000회, vs v1_baseline / vs current)

- `v2_safe_ief` vs v1: P(Sharpe>) `79.4%`, P(CAGR>) `82.0%`, P(MDD shallower) `63.1%` | vs current: P(Sharpe>) `88.0%`, P(CAGR>) `79.8%`, P(MDD shallower) `75.0%`
- `v3_credit_and_vix` vs v1: P(Sharpe>) `12.5%`, P(CAGR>) `22.9%`, P(MDD shallower) `35.7%` | vs current: P(Sharpe>) `64.7%`, P(CAGR>) `53.5%`, P(MDD shallower) `62.2%`
- `v3b_credit_and_vix_safe_ief` vs v1: P(Sharpe>) `26.0%`, P(CAGR>) `41.5%`, P(MDD shallower) `37.3%` | vs current: P(Sharpe>) `79.5%`, P(CAGR>) `74.7%`, P(MDD shallower) `68.5%`
- `v4_continuous_cap1.5` vs v1: P(Sharpe>) `12.8%`, P(CAGR>) `18.8%`, P(MDD shallower) `38.2%` | vs current: P(Sharpe>) `71.2%`, P(CAGR>) `54.6%`, P(MDD shallower) `73.5%`
- `v4_continuous_cap2.0` vs v1: P(Sharpe>) `13.7%`, P(CAGR>) `23.0%`, P(MDD shallower) `34.0%` | vs current: P(Sharpe>) `71.4%`, P(CAGR>) `55.6%`, P(MDD shallower) `69.4%`
- `v4_continuous_cap3.0` vs v1: P(Sharpe>) `13.6%`, P(CAGR>) `24.3%`, P(MDD shallower) `31.5%` | vs current: P(Sharpe>) `71.3%`, P(CAGR>) `55.2%`, P(MDD shallower) `70.3%`
- `v5_continuous_cap2_safe_ief` vs v1: P(Sharpe>) `36.1%`, P(CAGR>) `50.4%`, P(MDD shallower) `38.7%` | vs current: P(Sharpe>) `91.2%`, P(CAGR>) `82.9%`, P(MDD shallower) `72.2%`

## Part 1. off_scale 재그리드 (min_consecutive=2 고정, 4x3x3x3=108조합)

- current 대비 Sharpe·MDD 동시 개선 조합: `67/108`

### Sharpe 상위 8개

| label | lookback | sma | threshold | off_scale | CAGR | MDD | Sharpe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L10_S30_T-0.01_O0.3 | 10 | 30 | -0.01 | 0.3 | 17.08% | -14.96% | 1.328 |
| L10_S20_T-0.01_O0.3 | 10 | 20 | -0.01 | 0.3 | 16.74% | -14.96% | 1.312 |
| L10_S30_T-0.01_O0.5 | 10 | 30 | -0.01 | 0.5 | 16.86% | -15.37% | 1.296 |
| L15_S10_T-0.01_O0.3 | 15 | 10 | -0.01 | 0.3 | 16.53% | -12.00% | 1.294 |
| L10_S20_T-0.01_O0.5 | 10 | 20 | -0.01 | 0.5 | 16.62% | -15.37% | 1.286 |
| L10_S10_T-0.01_O0.3 | 10 | 10 | -0.01 | 0.3 | 16.44% | -14.96% | 1.283 |
| L15_S10_T-0.01_O0.5 | 15 | 10 | -0.01 | 0.5 | 16.47% | -12.00% | 1.273 |
| L10_S10_T-0.01_O0.5 | 10 | 10 | -0.01 | 0.5 | 16.40% | -15.37% | 1.265 |
