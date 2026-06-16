# 전략 명세: fin-sensitivity-v1

> 재무제표 변화 → 주가 반응의 **종목별 민감도**를 rolling Ridge로 학습해 발표
> 이후 20일/60일 예상 수익률을 산출하는 단일 종목 전략(배치 스크리너의 단위).
> 근거 계획: `finance_plan.txt` (특히 §19~§28 trading_lab 통합 보완).
> 미정 항목이 하나라도 남아 있으면 Gate 0(백테스트)을 시작하지 않는다.

## 1. 식별 정보

| 항목 | 값 |
| --- | --- |
| strategy_id | fin-sensitivity-v1 |
| version | 1 |
| 설명 | 재무 팩터 변화에 대한 종목별 주가 민감도 학습 후 예상 수익률 기반 long-only |
| 작성일 | 2026-06-14 |
| 상태 | validation (핸들러 등록·계약 테스트 통과, 실데이터 게이트 진행 전) |
| config 경로 | configs/strategies/fin_sensitivity_v1.json |
| 결과 단위 | 단일 종목 (1 run = 1 종목). 유니버스 병렬은 배치 러너 = §28 / B는 fin-portfolio-v1 |

## 2. 가설과 엣지

- **시장 가설**: 같은 재무 변화라도 종목마다 주가 반응 강도(민감도)가 다르고, 그
  민감도는 시간에 따라 비교적 천천히 변한다. 따라서 한 종목의 과거 "팩터 변화 →
  발표 후 수익률" 관계를 학습하면, 새 발표의 향후 수익률을 부분적으로 예측할 수 있다.
- **엣지의 원천**: 행동 편향(실적 정보의 점진적 반영 = PEAD 계열) + 종목별 이질성을
  무시한 일률적 스크리닝의 비효율. 절대 수준이 아니라 **변화와 반응**을 본다(§3).
- **무효화 조건** (*결과 보기 전 고정*):
  - validation에서 예상 수익률과 실제 수익률의 순위상관(Spearman)이 ≤ 0,
  - 또는 placebo(팩터 셔플) 대비 IC 차이가 부트스트랩 95% CI에서 유의하지 않음,
  - 또는 종목 자기 시계열 표본이 부족해(< train_quarters) 대부분 run이
    `insufficient_train_data`로 떨어짐 → 이 경우 단일 종목(A) 접근을 폐기하고
    횡단면 풀링(fin-portfolio-v1 / B)을 선행한다.
- **선행 연구/참조**: PEAD(post-earnings announcement drift). 기존 DI 전략과 달리
  가격·기술지표가 아니라 **재무 이벤트**를 신호원으로 쓴다.

## 3. 유니버스와 데이터

| 항목 | 값 |
| --- | --- |
| 대상 심볼 (개발용) | 미국 주식 1종목 (예: AAPL). 합성 경로는 SYNTH/RANDOM |
| 유니버스(배치/포트폴리오) | configs로 노출 — 1차 검증 후 §28 배치 러너가 소비 |
| interval | 1d |
| period | max |
| 데이터 소스 | 가격: yfinance / synthetic / csv · 재무: SEC EDGAR companyfacts → parquet · 금리: yfinance ^IRX(13주 T-bill) |
| 재무 데이터 | scripts/fetch_fundamentals.py 가 `var/fundamentals/<SYMBOL>.parquet`로 저장 |
| 거시(금리) 데이터 | 미국 금리(^IRX) 일별 종가를 사이드로드. 합성 경로는 결정적 합성 금리. 발표 즉시 공개라 거래일 종가를 그대로 PIT로 사용 |
| 누적 저장 | Yahoo OHLCV는 백테스트 때 `var/market_data/<interval>/<SYMBOL>.parquet`에 타임스탬프 기준 병합. SEC 재수집은 기존 분기를 보존하고 새 값으로 갱신 |
| 최소 표본 | rolling Ridge train_quarters(기본 16) 이상의 분기 이벤트. 미만이면 학습 보류 |
| 데이터 품질 기준 | 중복 인덱스 0, OHLC 결측 행 제거, 분기재무 as-of 정렬, 결측에도 비중단 |

## 4. 신호 정의

- **피처**(재무 팩터의 **변화량**, 전분기 대비 Δ 후 rolling 표준화):
  매출 성장률, 영업이익, 순이익, ROE, 영업현금흐름, 부채비율, 자본총계.
  밸류에이션(PER/PBR/PSR)은 절대값이 아니라 **과거 평균 대비 z-score**(섹터 상대는
  패널이 있는 B에서 활성). 선택 팩터: EPS 전망(데이터 안정성 낮아 기본 off).
  - **거시 피처(미국 금리 발표)**: `rate_level`(금리 수준의 과거평균 대비 z)과
    `d_rate`(rate_change_lookback≈1분기 금리 변화). 발표 사용가능일 시점 값을 써
    누수 없이 금리 국면에 대한 종목 반응 민감도를 함께 학습한다. `use_rate_feature`로 토글.
- **타깃**: 발표 사용가능일 기준 forward 20일 / 60일 누적수익률(두 모델).
- **민감도 학습**: rolling 12~16분기 Ridge. 각 분기 발표 시 `pred_ret_20d`,
  `pred_ret_60d`와 팩터별 계수(=민감도)를 산출. 윈도우 **안에서만** 표준화/적합
  (전체기간 누수 금지). 표본 부족 시 `insufficient_train_data` 플래그.
- **방향 규칙(진입, long-only, direction=+1)**: 발표 사용가능일 다음 봉에서
  `pred_ret_20d > pred20_min` AND `pred_ret_60d > pred60_min` AND 제외조건 미해당
  AND 시장필터 정상.
- **제외 조건**(§9 plan): 영업이익 적자 지속 / 영업현금흐름 지속 악화 / 자본총계 감소 /
  부채비율 급등 / (매출 감소 ∧ 재고 증가) / 밸류에이션 과거평균 대비 과열 / 결측 과다.

| 파라미터 | 기본값 | 탐색 범위 | 비고 |
| --- | --- | --- | --- |
| train_quarters | 16 | 12–16 | rolling 학습 윈도우(분기) |
| ridge_alpha | 1.0 | 0.1–10 | L2 정규화 강도 |
| pred20_min / pred60_min | 0.0 | 0–0.03 | 진입 임계(예상수익률 하한) |
| availability_lag_q / _a | 45 / 90일 | 고정 | 발표일 부재 시 보수 사용가능일 |

## 5. 실행 규칙

| 항목 | 값 |
| --- | --- |
| 체결 방식 | next_open (발표 사용가능일 다음 봉 시가) |
| 보유 기간 | **시간 한도 없음**. 다음 분기 발표 때 예측·실제값으로 포트폴리오 재구성(rebalance)할 때까지 보유 |
| 청산 사유 | rebalance(다음 발표 재구성) / signal_flip(갱신 예상수익률 음전환) / stop_loss / end_of_data(마지막 포지션) |
| long_only | true (1차) |
| 포지션 크기 | 단일 종목 100%, 중첩 금지 |
| 손절/익절 | 손절: 넓게(stop_loss_pct 기본 0.25). 익절 없음. **보유기간 청산 폐지** — 리밸런싱 중심(§12) |
| 계좌 평가 | equity는 **평가자산(mark-to-market)** — 보유 중 종가로 매일 미실현 손익 반영, 청산 시 실현 확정 |

## 6. 비용 모델

| 항목 | 값 |
| --- | --- |
| 수수료 | fee_bps_per_side (기본 5bp) |
| 슬리피지 가정 | next_open 체결로 갈음 |
| 비용 민감도 | 수수료 2배에서도 validation 순수익 부호 유지 |

## 7. 리스크 한도

| 항목 | 한도 |
| --- | --- |
| max_drawdown 허용치 | -30% 초과 시 재검토 |
| 단일 자산 비중 | 100% (단일 종목 전략) |
| 시장 필터 | SPY/QQQ 장기 MA 아래에서는 신규 진입 제한(§12). 합성·심볼 부재 시 자기 종가 MA 폴백 |
| kill switch | live 차단 상태. 예상수익률 음전환 지속 시 청산은 signal_flip로 처리 |

## 8. 검증 설계

| 항목 | 값 |
| --- | --- |
| 분할 | train(initialize=민감도 학습) / validation / test 시간순. phase로 슬라이스 |
| 파라미터 탐색 허용 구간 | train(initialize)만 |
| validation 사용 규칙 | 합격/불합격 판정만, 탐색 금지 |
| holdout(test) 개봉 | Gate 3 통과 후 1회 |
| 컨트롤 실험 | placebo(팩터 셔플), 파라미터 ±20%, leave-one-quarter, bootstrap CI |
| 누수 검증 | as-of merge 후 미래 재무 0건, rolling 표준화/계수 윈도우 밖 미참조, 타깃>피처 시점 |

## 9. 합격선 (결과 확인 전에 고정)

| 지표 | validation 합격선 | holdout 합격선 |
| --- | --- | --- |
| trades (종목당 최소) | ≥ 8 | ≥ 8 |
| 예측 IC (Spearman, pred vs real) | > 0 | > 0 |
| total_return | > 벤치마크(buy&hold) | > 벤치마크 |
| hit_rate | ≥ 0.45 | |
| sharpe | > 0 | |
| max_drawdown | ≥ -0.30 | |
| placebo 대비 | IC가 placebo 평균보다 유의(95% CI) | |

## 10. 산출물과 등록

- [x] `scripts/finance_sensitivity/` 데이터·모델·신호 모듈 (백테스트 전 구현 완료)
- [x] `configs/strategies/fin_sensitivity_v1.json`
- [x] `docs/strategy/checklists/fin-sensitivity-v1.md`
- [x] `src/trading_lab/strategies/fin_sensitivity.py` 핸들러 (StrategyArtifacts 매핑)
- [x] `registry.py` 등록 (`enabled=True`), `presentation.EXIT_REASON_LABELS`에 rebalance/signal_flip 추가
- [x] 계약 테스트 통과: `tests.test_strategy_contract` + 전체 76개 + 합성 CLI run `succeeded`
- [x] 결과 대시보드 `재무·주가 학습` 탭: 20/60일 IC·오차·방향 적중률,
  예측-실제 산점도, 팩터 민감도, 이벤트별 학습 데이터
- [ ] 실데이터(yfinance + fetch_fundamentals) 게이트 증거를 `var/runs/` 또는 `reports/`에 기록
