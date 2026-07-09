# final0706 비중 편차(weight_power) 스터디

- Period: `2007-01-29~2026-07-08` (4087 bars, 원화(KRW) 기준)
- 공식 핸들러(`Yoon1kHandler`/`ProfilePortfolioHandler`)로 계산(재구현 없음). `final0706.json`은 동결 스냅샷이라 수정하지 않고 sub_strategies overrides에 `weight_power`만 즉석 주입한 변형으로 실험.
- `weight_power`: score^p 재정규화 지수(portfolio.py `_target_weights`). 1.0=원본(불변), >1=상위 점수 종목에 더 쏠림(편차 확대).

## 1. 실배분 편차 실측 (top-3 내부, 슬리브 단독)

| sleeve | weight_power | mean HHI | mean HHI(균등가정) | mean 최대종목비중 | mean 보유종목수 |
| --- | ---: | ---: | ---: | ---: | ---: |
| US(yoon1j_lp) | 1.0 | 0.266 | 0.254 | 28.92% | 4.07 |
| KR(yoon1j_kr) | 1.0 | 0.276 | 0.262 | 31.00% | 3.97 |
| US(yoon1j_lp) | 2.0 | 0.289 | 0.254 | 32.06% | 4.07 |
| KR(yoon1j_kr) | 2.0 | 0.309 | 0.262 | 35.45% | 3.97 |
| US(yoon1j_lp) | 4.0 | 0.338 | 0.254 | 36.41% | 4.07 |
| KR(yoon1j_kr) | 4.0 | 0.379 | 0.262 | 41.48% | 3.97 |

(HHI = sum(w_i/exposure)^2, top-3 완전균등이면 1/3≈0.333. 값이 클수록 편중.)

## 2. 전체 파이프라인 성과 비교

### all (2007-01-29~2026-07-08)

| candidate | CAGR | MDD | Sharpe | Total Return |
| --- | ---: | ---: | ---: | ---: |
| weight_power=1.0(원본) | 13.66% | -16.40% | 1.223 | 1104.32% |
| weight_power=2.0 | 13.57% | -16.70% | 1.211 | 1086.17% |
| weight_power=4.0 | 13.40% | -17.04% | 1.189 | 1051.87% |

### test (2022-12-19~2026-07-08)

| candidate | CAGR | MDD | Sharpe | Total Return |
| --- | ---: | ---: | ---: | ---: |
| weight_power=1.0(원본) | 33.61% | -12.00% | 1.951 | 179.77% |
| weight_power=2.0 | 33.50% | -12.00% | 1.943 | 178.96% |
| weight_power=4.0 | 33.29% | -12.00% | 1.929 | 177.45% |

### recent_3y (2023-07-10~2026-07-08)

| candidate | CAGR | MDD | Sharpe | Total Return |
| --- | ---: | ---: | ---: | ---: |
| weight_power=1.0(원본) | 36.88% | -12.00% | 2.009 | 156.05% |
| weight_power=2.0 | 37.05% | -12.00% | 2.015 | 157.04% |
| weight_power=4.0 | 37.27% | -12.00% | 2.022 | 158.25% |

## 3. 페어드 블록 부트스트랩 (vs 원본, iters=8000, block=20)

| 비교 | 기간 | P(Sharpe 우위) | P(CAGR 우위) | P(MDD 더 얕음) |
| --- | --- | ---: | ---: | ---: |
| power2.0_vs_원본 | 2007-01-29~2026-07-08 | 14.0% | 23.4% | 32.9% |
| power4.0_vs_원본 | 2007-01-29~2026-07-08 | 9.1% | 19.1% | 28.3% |
