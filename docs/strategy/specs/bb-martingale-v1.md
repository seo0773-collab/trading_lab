# 전략 명세: bb-martingale-v1

> 모든 절이 채워지기 전에는 백테스트를 시작하지 않습니다. 결정하지 못한 항목은 `미정`으로 표시하고, `미정`이 하나라도 남아 있으면 Gate 0을 시작할 수 없습니다.

## 1. 식별 정보

| 항목 | 값 |
| --- | --- |
| strategy_id | bb-martingale-v1 |
| version | 1 |
| 설명 | BB(20,2.5,0) 신호 + 항상 롱숏 유지 + 마틴게일 물타기 + ADX 추세가속 레짐게이트 |
| 작성일 | 2026-07-09 |
| 상태 | draft (파산 가능 구조라 live 후보 아님, 대시보드 열람/리서치 확인용) |
| config 경로 | configs/strategies/bb_martingale_v1.json |

## 2. 가설과 엣지

- **시장 가설**: 횡보성이 강한(추세강도 ADX 낮은) 종목은 가격이 볼린저밴드 상/하단 사이에서 왕복하는 경향이 있어, 밴드 이탈은 평균회귀로 반전될 확률이 높다.
- **엣지의 원천**: 구조적 제약(마켓메이커/차익거래 자본의 변동성 매도 유인, 유동성 공급자의 재고 관리) — 새로운 정보 없는 순수 변동성 확산은 평균회귀하는 경향.
- **무효화 조건**: holdout에서 파산율(ruin_rate)이 0%가 아니거나, B&H 대비 median Sharpe가 열위이면 이 파라미터 조합은 폐기.
- **선행 연구/참조**:
  - `scripts/bb_crossing_sweep.py` / `reports/bb_crossing_sweep.md` — 무포지션 허용 재상향크로스 방식 BB 파라미터 스윕(length/mult/offset), length=20·mult=2.5·offset=0 채택.
  - `scripts/bb_martingale_meanreversion.py` / `reports/bb_martingale_sweep.md` — 항상 롱/숏 유지(밴드이탈=플립) + 마틴게일 사이징. mult(물타기 배수) 커질수록 파산율 급증 실측(1.5배=0.4%, 2.0배=5.8~6.7%, 3.0배=25~27%).
  - `scripts/bb_martingale_regime_gate.py` / `reports/bb_martingale_regime_sweep.md` — ADX(14) 추세가속 레짐게이트. 파산율은 소폭 개선되나 Sharpe 개선은 불명확(일부 조합은 오히려 악화) — 청산(플립) 신호까지 같이 막는 부작용 때문으로 추정.

## 3. 유니버스와 데이터

| 항목 | 값 |
| --- | --- |
| 대상 심볼 (개발용) | 단일종목 선택형(대시보드 symbol 입력). 리서치는 `reports/bb_crossing_universe.csv`의 30종목(ADX 낮은 미국 상장 종목) 기준 |
| 일반화 자산 세트 | 미실시 — 스크리닝 자체가 "현재 횡보 중인 종목"을 고르므로 임의 자산에 일반화 주장 안 함 |
| asset class 커버리지 | equity/ETF(미국 상장, `.` 없는 티커만 — KR/JP는 비용모델 불일치로 제외) |
| interval | 1d |
| period | max |
| 데이터 소스 | 로컬 캐시 `var/market_data/1d`(있으면) → 없으면 yfinance(`load_cumulative_yfinance`) |
| 최소 bar 수 | ≥ 800(3년) 권장 — BB length(20)+ADX(14) 워밍업 및 통계적 최소 거래수 확보 |
| 데이터 품질 기준 | 중복 인덱스 제거(`normalize_ohlcv`), OHLC 결측 제거, 시간순 정렬 |

## 4. 신호 정의

- **피처**: `close`, BB `basis`(SMA20)/`upper`/`lower`(±2.5SD), `ADX(14)`.
- **방향 규칙**: 무포지션 상태에서 close가 lower 이하 → 롱, upper 이상 → 숏. 보유 중 반대 밴드 터치 시 전량 청산 + 즉시 반대 진입(플립). 항상 롱/숏 중 하나 유지(무포지션 없음).
- **확신도(confidence)**: 별도 확신도 필터 없음(밴드 이탈 자체가 이산 신호). 대신 ADX 레짐게이트가 "추세가속 구간에서는 신규 행위 정지"로 확신도 필터 역할을 대신함.
- **파라미터 목록과 탐색 범위**: BB 파라미터는 `bb_crossing_sweep.py`에서 이미 탐색·고정(재탐색 안 함). 마틴게일/레짐 파라미터는 아래 그리드로 탐색.

| 파라미터 | 기본값 | 탐색 범위 | 비고 |
| --- | --- | --- | --- |
| bb.length | 20 | {10,14,20,30,40,50} (탐색 완료, 고정) | bb_crossing_sweep.py |
| bb.mult | 2.5 | {1.5,2.0,2.5,3.0} (탐색 완료, 고정) | bb_crossing_sweep.py |
| bb.offset | 0 | {0,1,2,3,5} (탐색 완료, 고정) | causal lag만 |
| martingale.add_step | 0.03 | {0.02,0.03,0.05,0.08} | 물타기 트리거 역행폭 |
| martingale.mult | 1.5 | {1.5,2.0,3.0} | 배수 클수록 파산율 급증 |
| martingale.max_levels | 5 | {2,3,4,5} | 손절 없음, 이 값으로만 리스크 캡 |
| martingale.base_lot_fraction | 0.05 | 미탐색(고정) | 최초 랏 = 자본의 5% |
| regime_gate.adx_threshold | 25.0 | 미탐색(고정) | |
| regime_gate.adx_slope_lookback | 10 | 미탐색(고정) | |

## 5. 실행 규칙

| 항목 | 값 |
| --- | --- |
| 체결 방식 | 당봉 종가(익일 시가 지연 미반영 — 단순화) |
| 보유 기간 | 반대 밴드 터치까지 무기한(마틴게일 물타기로 보유 연장 가능) |
| 반대 신호 처리 | 즉시 플립(전량 청산 + 반대 포지션 신규 진입) — 단, 레짐게이트 bad 구간에서는 플립도 정지 |
| long_only | false (항상 롱 또는 숏 유지) |
| 포지션 크기 | 최초 랏 5% + 역행마다 직전 랏의 mult배 추가(최대 max_levels 단계) |
| 손절/익절 | 없음(사용자 지정) — max_levels로만 리스크 상한, 자본<=0이면 파산 처리(equity=0 고정) |

## 6. 비용 모델

| 항목 | 값 |
| --- | --- |
| 수수료 | fee_bps_per_side = 20.103bp (KIS 미국 실조건, `scripts/profile_sizing/config.py:KIS_US_FEE_BPS`) |
| 슬리피지 가정 | 5bp 고정 가산 |
| 비용 민감도 | 미검증 — 편도 비용을 2배로 올린 재검증은 아직 안 함 |

## 7. 리스크 한도

| 항목 | 한도 |
| --- | --- |
| max_drawdown 허용치 | 명시적 상한 없음(손절 없는 구조) — 파산(-100%)까지 가능함을 감안하고 사용 |
| 단일 자산 비중 | 100%(단일 자산), 단 마틴게일 물타기로 명목 노출은 최대 `base_lot_fraction × (mult^(max_levels+1)-1)/(mult-1)` 배까지 커질 수 있음(예: 1.5배/5단계 ≈ 자본의 89%) |
| 거래 빈도 상한 | 없음 |
| kill switch 조건 | 자본(mark-to-market) <= 0 이면 즉시 전략 중단(파산) — 코드상 강제 구현됨 |

## 8. 검증 설계

| 항목 | 값 |
| --- | --- |
| 분할 | 시간순 train 70% / holdout 30% (identification/validation/test 3분할 아님 — 리서치 스크립트 단계의 단순 분할) |
| 파라미터 탐색 허용 구간 | train만(holdout은 확인용) |
| validation 사용 규칙 | 대시보드 phase 위젯은 프로젝트 표준(train_frac 0.6/validation_frac 0.2)으로 별도 재분할 — 리서치 리포트의 train/holdout(0.7/0.3)과 경계가 다름에 유의 |
| holdout(test) 개봉 규칙 | 리서치 단계에서 이미 확인함(개봉 후 재탐색 없음) |
| 컨트롤 실험 | 미실시(placebo/파라미터 섭동/leave-one-asset-out 없음) — 후속 과제 |

## 9. 합격선 (결과 확인 전에 고정)

리서치 단계 사후 확인(사전 고정 아님, 정직하게 기록):

| 지표 | train(≈3.5y) 실측 | holdout(≈1.5y) 실측 |
| --- | --- | --- |
| trades (자산당 중앙값) | 소수(파라미터별 상이, 대체로 한 자릿수) | 소수 |
| median Sharpe | mult=1.5/add_step=3%/max_levels=5: 0.276 | 0.673(게이트 없음) / 0.404(게이트 적용) |
| ruin_rate | mult=1.5: 0.4%, mult=3.0: 25.2% | mult=1.5: 0.4%, mult=3.0: 27.1% |
| B&H 대비 | median Sharpe 우위(파산 0% 조합 한정) | 대체로 우위, 절대수익은 낮음(1~5%대) |
| placebo 대비 | 미실시 | 미실시 |

## 10. 산출물과 등록

- [x] `configs/strategies/bb_martingale_v1.json` 작성
- [x] `src/trading_lab/strategies/registry.py` 등록 (`enabled=True`, `live_eligible=False`)
- [x] 체크리스트 `docs/strategy/checklists/bb-martingale-v1.md` 생성
- [x] 게이트 증거: `reports/bb_crossing_sweep.{md,json}`, `reports/bb_martingale_sweep.{md,json}`, `reports/bb_martingale_regime_sweep.{md,json}`
