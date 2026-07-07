# final0706: QLD를 US top_k 유니버스에 추가 vs 기존 오버레이 블렌드

- Period: `2007-01-29~2026-07-06` (4085 bars, 원화 기준)
- `final0706_qld_topk` = US 슬리브(yoon1j_lp) universe에 QLD 한 종목만 추가, top_k=3/hold_buffer=2 등 나머지는 원본과 동일 — 공식 ProfilePortfolioHandler 선택 로직(프로파일 컨트래리언 점수) 그대로, KR 슬리브·통화틸트·신용게이트도 원본과 동일.
- `final0706_qld` = 기존 등록 파생(core 63.5%+QLD 36.5% 고정 오버레이, 월간 리밸런싱) 참고용.
- US 슬리브 전체 진입 거래 102건 중 QLD 진입 8건(7.8%).

## all

| candidate | CAGR | MDD | Sharpe | Total Return |
| --- | ---: | ---: | ---: | ---: |
| final0706 (원본) | 16.97% | -16.40% | 1.250 | 1168.65% |
| final0706_qld_topk (US유니버스+QLD, top_k=3 경쟁) | 17.24% | -16.40% | 1.255 | 1217.48% |
| final0706_qld (core63.5%+QLD36.5% 오버레이) | 24.85% | -30.06% | 1.164 | 3553.63% |

## test

| candidate | CAGR | MDD | Sharpe | Total Return |
| --- | ---: | ---: | ---: | ---: |
| final0706 (원본) | 39.37% | -12.00% | 2.053 | 193.36% |
| final0706_qld_topk (US유니버스+QLD, top_k=3 경쟁) | 40.02% | -12.00% | 2.079 | 197.82% |
| final0706_qld (core63.5%+QLD36.5% 오버레이) | 52.38% | -22.32% | 2.017 | 291.77% |

## recent_3y

| candidate | CAGR | MDD | Sharpe | Total Return |
| --- | ---: | ---: | ---: | ---: |
| final0706 (원본) | 43.35% | -12.00% | 2.126 | 168.82% |
| final0706_qld_topk (US유니버스+QLD, top_k=3 경쟁) | 44.14% | -12.00% | 2.155 | 172.91% |
| final0706_qld (core63.5%+QLD36.5% 오버레이) | 49.71% | -22.32% | 1.906 | 202.88% |

## 페어드 부트스트랩 (8000 iters, block=20)

| vs | P(Sharpe 개선) | P(CAGR 개선) | P(MDD 더 얕음) |
| --- | ---: | ---: | ---: |
| qld_topk_vs_원본 | 59.2% | 81.8% | 38.4% |
| qld_overlay_vs_원본 | 33.1% | 99.3% | 0.6% |
| qld_topk_vs_qld_overlay | 68.7% | 0.8% | 99.4% |

## 파일

- JSON: `reports/profile_sizing/final0706_qld_topk_compare.json`
