# yoon_profile_3 멀티에셋 검증 — 크립토 · ETF (MDD/CAGR/Sharpe)

- 2026-07-04. 대상: 크립토 BTC/ETH/SOL (로컬 일봉), ETF SPY/QQQ/GLD (yfinance 일봉 max).
- 변형: yp3(=yoon_profile_3, path_tilt on) vs base(=yoon_profile, tilt off) vs B&H.
- 비용 왕복 20bp. 수치: `yoon_profile_3_multiasset_eval.json`,
  `yoon_profile_3_multiasset_postwarmup.json`.
- **공정성 주의**: phase "all"은 전략이 거래할 수 없는 워밍업(~700봉)을 포함해
  워밍업 중 대세상승이 있었던 자산(ETH 2020-21, SOL 2020-21)은 B&H가 과대평가된다.
  크립토는 워밍업 이후(post-warmup) 창을 기준으로 판정한다.

## 1. 크립토 (post-warmup 공정 창)

| 자산 (창) | 지표 | yp3 | base | B&H |
| --- | --- | --- | --- | --- |
| BTC (2019-07~) | CAGR | **+36.4%** | +34.8% | +20.2% |
| | Sharpe | **1.043** | 0.987 | 0.619 |
| | MDD | **−41.7%** | −44.3% | −76.6% |
| ETH (2021-12~) | CAGR | −6.3% | −5.4% | −13.4% |
| | Sharpe | −0.069 | −0.042 | +0.037 |
| | MDD | −60.8% | −59.5% | −77.6% |
| SOL (2022-07~) | CAGR | +27.9% | **+32.3%** | +8.2% |
| | Sharpe | 0.742 | **0.811** | 0.513 |
| | MDD | −47.5% | −46.6% | −79.3% |

holdout(test) phase: BTC yp3 Sharpe 0.439/MDD −20.2% (B&H 0.160/−53.0%),
SOL yp3 0.042/−36.1% (B&H −0.373/−74.9%), ETH −0.481/−47.2% (B&H 0.012/−67.5%).

## 2. ETF (BTC 튜닝 가중치 그대로 — 이식 테스트)

| 자산 (all) | 지표 | yp3 | base | B&H |
| --- | --- | --- | --- | --- |
| SPY | CAGR / Sharpe / MDD | +1.3% / 0.20 / −23.8% | +2.0% / 0.27 / −21.6% | **+10.8% / 0.65 / −55.2%** |
| QQQ | CAGR / Sharpe / MDD | +5.2% / 0.43 / −27.4% | +5.4% / 0.44 / −27.9% | **+10.8% / 0.52 / −83.0%** |
| GLD | CAGR / Sharpe / MDD | +4.9% / 0.47 / −27.2% | +5.3% / 0.50 / −26.5% | **+10.4% / 0.64 / −45.6%** |

MDD는 절반 이하로 줄지만 CAGR·Sharpe 모두 B&H 대비 크게 열위 — holdout에서도
동일(SPY test Sharpe 0.06 vs 0.86). ETF에서는 MDD 개선이 수익 희생을 정당화하지
못한다.

## 3. 판정

| 자산군 | 판정 | 근거 |
| --- | --- | --- |
| **BTC** | **채택 확정** | 3지표 전부 B&H·base 우위, holdout 포함. 등록 근거 재확인 |
| **SOL** | **base 구조 이식 성립, tilt는 미적용 권고** | base가 CAGR +32.3%/Sharpe 0.811로 B&H 압도(위험조정+절대수익). 단 path_tilt는 SOL에서 −4.4%p 역기여 |
| **ETH** | 판정보류 (방어 우위만) | CAGR·MDD는 B&H보다 낫지만 절대수익 음수·Sharpe 동급 이하. 유효 창이 2021-12 상투 직후 시작 = 베어 단일 레짐 표본 |
| **ETF (SPY/QQQ/GLD)** | **기각** | CAGR·Sharpe 전 phase 열위. yoon_profile_1 주식 이식 기각(2026-07-04)과 일관 — BTC 튜닝 state_weights는 주식·금에 부적합. 주식은 yoon_profile_2 노선(재추정) 유지 |

**path_tilt(M1A1 경로 분리)의 이식성**: BTC에서만 +기여(+1.6%p CAGR, Sharpe
+0.056). ETH −0.9%p, SOL −4.4%p, ETF 소폭 열위 — **BTC 특화 신호**로 본다.
타 자산에 yoon_profile 계열을 쓸 경우 base(tilt off)가 기본 선택.

## 4. 유보 / 한계

- ETH·SOL은 워밍업이 2020-21 강세를 소진해 유효 표본이 짧고(4.5년/4년) 레짐
  편중 — 다음 사이클 데이터 누적 후 재평가.
- SOL의 tilt 역기여는 M1A1 경로 delta가 SOL에서 성립하지 않음을 시사하나,
  자산별 path 스터디는 미실행(사전 등록 검정은 BTC만). 필요 시 SOL로
  yoon_profile_path_study 재실행이 다음 단계.
- ETF 기각은 "BTC 가중치 그대로" 조건 — 자산별 state_weights 재추정 스터디는
  별도 트랙(주식은 yoon_profile_2가 그 노선).
