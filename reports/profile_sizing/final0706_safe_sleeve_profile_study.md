# final0706 안전자산(TLT/GLD) 프로파일 점수화 스터디

- Period: `2007-01-29~2026-07-08` (4087 bars, 원화(KRW) 기준)
- 공식 핸들러로 계산(재구현 없음). `final0706.json`/`yoon1j_lp.json`/`yoon1j_kr.json`은 그대로 두고 sub_strategies overrides에 `safe_sleeve.profile_scored=true`만 즉석 주입.
- 추세 게이트(종가>MA100, ON/OFF)는 원본과 동일하게 유지 — 바뀌는 것은 ON인 TLT·GLD 사이의 배분 비율뿐(균등 → 각자 profile-sizing 점수 비례).

## 1. 슬리브 단독 성과 (US yoon1j_lp / KR yoon1j_kr)

| sleeve | variant | CAGR | MDD | Sharpe | Total Return | 평균 안전자산 비중 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| US(yoon1j_lp) | 원본(균등 배분) | 8.33% | -21.02% | 0.749 | 806.88% | 8.08% |
| KR(yoon1j_kr) | 원본(균등 배분) | 9.88% | -37.55% | 0.706 | 524.13% | 13.54% |
| US(yoon1j_lp) | profile_scored(점수 비례) | 8.41% | -21.02% | 0.755 | 823.40% | 8.09% |
| KR(yoon1j_kr) | profile_scored(점수 비례) | 9.93% | -37.58% | 0.708 | 529.75% | 13.54% |

## 2. 전체 파이프라인(final0706 구조) 성과 비교

### all (2007-01-29~2026-07-08)

| candidate | CAGR | MDD | Sharpe | Total Return |
| --- | ---: | ---: | ---: | ---: |
| 원본(균등 배분) | 13.66% | -16.40% | 1.223 | 1105.24% |
| profile_scored(점수 비례) | 13.74% | -16.15% | 1.227 | 1121.79% |

### test (2022-12-19~2026-07-08)

| candidate | CAGR | MDD | Sharpe | Total Return |
| --- | ---: | ---: | ---: | ---: |
| 원본(균등 배분) | 33.63% | -12.00% | 1.953 | 179.98% |
| profile_scored(점수 비례) | 33.95% | -12.00% | 1.967 | 182.36% |

### recent_3y (2023-07-10~2026-07-08)

| candidate | CAGR | MDD | Sharpe | Total Return |
| --- | ---: | ---: | ---: | ---: |
| 원본(균등 배분) | 36.91% | -12.00% | 2.011 | 156.25% |
| profile_scored(점수 비례) | 37.20% | -12.00% | 2.023 | 157.87% |

## 3. 페어드 블록 부트스트랩 (profile_scored vs 원본, iters=8000, block=20)

| 비교 | 기간 | P(Sharpe 우위) | P(CAGR 우위) | P(MDD 더 얕음) |
| --- | --- | ---: | ---: | ---: |
| profile_scored_vs_원본 | 2007-01-29~2026-07-08 | 82.9% | 88.5% | 60.1% |
