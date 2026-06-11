# h=72 PRICE Strategy Generalization Plan

Updated: 2026-06-11 UTC

## Purpose

이 문서는 다음 Codex 세션이 현재 BTC 단일 표본 결과를 일반화 검증하는
작업을 순서대로 구현하고 실행하기 위한 handoff 문서다.

현재 후보는 다음과 같다.

```text
bar interval: 1h
forecast horizon: 72 bars
direction: PRICE
confidence quantile: q=0.85
cycle: EMA 200
cycle slope: EWM span 24, 72-bar linear extrapolation
exit: 72 bars or opposite high-confidence signal
primary cost: 10bp per side
```

현재 BTC 결과는 71건의 거래에 불과하고, 전략 선택에 해당 표본이 사용됐다.
따라서 이 결과를 최종 검정에 다시 사용하지 않는다.

## Non-Negotiable Rules

1. 기본 전략 규칙은 holdout 결과를 보기 전에 고정한다.
2. 자산별 파라미터 튜닝을 금지한다.
3. 신호는 bar `t` 종가까지의 정보로 계산하고 체결은 가장 빠르게 bar
   `t+1` 시가에서 수행한다.
4. rolling 임계값은 현재 bar를 포함하지 않는다.
5. 파라미터 식별과 sigma calibration에는 평가 구간의 미래 데이터를 쓰지
   않는다.
6. validation 결과로 규칙을 한 번 변경하면 새 버전 이름을 부여하고, 기존에
   열어본 test 구간은 다시 holdout으로 간주하지 않는다.
7. 실패 결과와 제외된 자산도 보고서에 남긴다.
8. 기존 사용자 파일과 `_1` 백업 파일을 삭제하거나 되돌리지 않는다.

## Primary Question

다음 조건을 모두 적용해도 h=72 PRICE 신호의 순 비용 차감 수익이 양수인가?

```text
multiple liquid assets
next-bar-open execution
10bp per-side cost
chronological untouched test
fixed parameters
```

## Secondary Questions

- 성과가 암호화폐 한 자산이나 특정 연도에 집중되는가?
- MULT/PRICE 충돌 반전이 다른 자산에서도 유효한가?
- cycle slope 설정을 조금 바꿔도 결과 방향이 유지되는가?
- 실제 결과가 랜덤 및 placebo 전략 분포보다 우수한가?
- 평균 거래 수익과 Sharpe의 불확실성이 어느 정도인가?

## Required Implementation

### 1. Preserve And Test The Baseline

현재 결과를 회귀 기준으로 고정한다.

```text
BTC h=72 PRICE, close execution, 10bp/side:
71 trades, 56.3% hit, +32.1bp/trade, +19.3%, Sharpe 0.65, MDD -23.9%
```

추가할 테스트:

- `build_signals()`의 PRICE 방향 단위 테스트
- rolling quantile이 현재 bar를 제외하는지 테스트
- 비용이 거래당 정확히 왕복 `2 * fee_bps` 적용되는지 테스트
- 반대 신호가 청산에는 적용되지만 edge entry filter와 혼동되지 않는지 테스트
- 기존 BTC close-execution 결과 회귀 테스트

### 2. Add Next-Bar-Open Execution

현재 forecast CSV에는 `open`이 없다. 아래 변경이 먼저 필요하다.

- `flat_chart.compute_features()` 결과에 원본 `open`을 추가한다.
- `run_kalman_pipeline.py`의 forecast CSV에 `open`을 저장한다.
- `conf_filter_backtest.py` 또는 새 재사용 가능 엔진에
  `execution={close,next_open}`을 추가한다.
- `next_open`에서는 bar `t` 신호를 bar `t+1` 시가에 체결한다.
- 반대 신호 청산과 재진입도 bar `t+1` 시가에서 처리한다.
- 보유 기간 72 bars는 실제 체결 bar부터 센다.
- 마지막 미청산 포지션의 강제 청산 규칙을 명시하고 테스트한다.

필수 look-ahead 테스트:

- bar `t+1` 이후 가격을 변경해도 bar `t+1` 진입 결정이 바뀌지 않아야 한다.
- bar `t` 시가를 변경해도 bar `t` 종가 신호가 바뀌지 않아야 한다.
- 신호 시점과 체결 시점이 로그에 별도 컬럼으로 남아야 한다.

거래 로그 필수 컬럼:

```text
asset
signal_time
entry_time
entry_price
exit_signal_time
exit_time
exit_price
direction
exit_reason
gross_return
fee_return
net_return
holding_bars
price_edge
confidence_threshold
mult_price_conflict
split
```

### 3. Add Chronological Splits

각 자산을 시간순으로 다음처럼 분리한다.

```text
identification: first 40%
validation: next 30%
test: final 30%
```

- identification은 Kalman 파라미터와 sigma calibration에만 사용한다.
- validation은 구현 검증과 최대 한 번의 전략 수정에 사용한다.
- test는 최종 실행 전까지 집계 결과를 출력하지 않는 모드를 제공한다.
- rolling feature는 과거 데이터가 필요하므로 split 직전 history를 입력으로
  허용하되, 성과 집계는 해당 split 내부 체결만 포함한다.
- validation에서 진입하고 test에서 청산되는 거래의 귀속 규칙을 명시한다.
  기본값은 entry split에 귀속한다.

CLI 예:

```bash
../.venv/bin/python generalization_runner.py \
  --manifest generalization_assets.csv \
  --config frozen_h72_price.json \
  --phase validation
```

### 4. Freeze Configuration And Asset Manifest

코드에 흩어진 기본값 대신 버전 관리 가능한 설정 파일을 만든다.

`frozen_h72_price.json` 최소 내용:

```json
{
  "version": "h72-price-v1",
  "interval": "1h",
  "horizon": 72,
  "direction": "PRICE",
  "confidence_quantile": 0.85,
  "quantile_window": 2000,
  "cycle_len": 200,
  "fast_window": 120,
  "slow_window": 720,
  "slope_span": 24,
  "slope_mode": "linear",
  "fee_bps_per_side": 10.0,
  "execution": "next_open",
  "exit_on_opposite": true,
  "long_only": false
}
```

자산은 실행 전에 manifest에서 역할을 고정한다. 다운로드 실패나 데이터 부족
때문에 자산을 교체하면 이유와 교체 시점을 기록한다.

권장 discovery/validation 자산:

```text
BTC-USD
ETH-USD
SOL-USD
XRP-USD
SPY
QQQ
GLD
EURUSD=X
```

권장 untouched test 자산:

```text
LTC-USD
LINK-USD
AVAX-USD
IWM
TLT
```

주의:

- yfinance 1h history 제한과 상장 기간을 사전에 검사한다.
- 암호화폐는 24/7, ETF/FX는 세션과 결측 구조가 다르다.
- 72 bars는 모든 자산에서 72 clock-hours가 아니다. 기본 연구 질문은
  `72 bars`로 고정하고 자산군별 의미 차이를 보고한다.
- 미국 ETF의 Sharpe annualization은 약 1,638 bars/year, 24/7 crypto는
  8,760 bars/year를 사용한다.
- 유동성이 낮거나 데이터가 지나치게 짧은 자산은 사전에 정의한 최소 bar 수
  기준으로 제외한다.

### 5. Run Validation Assets Without Per-Asset Tuning

각 자산에서 아래 결과를 저장한다.

```text
trade count
long/short count
hit rate
average gross and net return
median net return
total return
annualized Sharpe
max drawdown
exposure
profit factor
MULT/PRICE agree and conflict counts
conflict-subset PRICE hit and gross return
year/regime breakdown
```

포트폴리오 집계는 두 가지를 모두 보고한다.

- asset-equal: 자산별 성과를 동일 가중
- trade-pooled: 모든 거래를 합산

한 자산이 전체 결과를 지배하는지 `leave-one-asset-out` 결과도 출력한다.

### 6. Add Random And Placebo Controls

최소 세 종류를 구현한다.

1. GBM/null synthetic:
   자산별 평균과 변동성만 맞춘 무예측 수익률 시계열.
2. Moving-block bootstrap:
   수익률의 단기 자기상관과 변동성 군집을 일부 보존한다.
3. Placebo:
   실제 가격 데이터에서 신호 방향 또는 신호 시간을 무작위화하되 거래 수와
   보유 기간 분포를 최대한 유지한다.

OHLC synthetic 생성 시 다음을 지킨다.

- 미래 수익률로 현재 high/low/open을 만들지 않는다.
- 양수 가격을 보장한다.
- seed를 결과 파일에 기록한다.
- validation 단계는 seed 100개, 최종 단계는 가능하면 500개 이상 실행한다.

비교 통계:

```text
actual mean net return percentile in null distribution
actual Sharpe percentile
actual max drawdown percentile
empirical one-sided p-value
```

### 7. Robustness Grid, Not Optimization

아래 grid는 최적 파라미터 선택용이 아니라 주변 안정성 확인용이다.

```text
horizon: 48, 60, 72, 84, 96
confidence q: 0.80, 0.85, 0.90
cycle EMA: 100, 200, 400
slope EWM span: 12, 24, 48, 72
slope mode:
  linear
  damped
  capped horizon return
```

원칙:

- primary 결과는 항상 frozen h72 PRICE 설정이다.
- grid 최고 결과를 primary로 승격하지 않는다.
- heatmap과 양수 셀 비율을 보고한다.
- 한 점에서만 수익이면 불안정으로 판정한다.
- grid를 본 뒤 전략을 변경하면 새 버전과 새 holdout이 필요하다.

### 8. Regime And Concentration Analysis

최소 분석 단위:

- calendar year
- rolling 6-month window
- realized volatility tercile
- cycle slope positive/negative
- long/short
- MULT/PRICE agree/conflict
- top 5 and top 10 winning trades removed

성과가 다음 중 하나에 해당하면 concentration warning을 낸다.

- 단일 자산이 총 순이익의 50% 이상
- 단일 연도가 총 순이익의 50% 이상
- 상위 5개 거래 제거 후 평균 순이익이 0 이하
- conflict 거래만 제거하면 전체 edge가 사라짐

마지막 항목은 자동 실패가 아니라 전략 메커니즘 의존성으로 명시한다.

### 9. Statistical Uncertainty

거래가 겹치고 시계열 의존성이 있으므로 단순 IID t-test만 사용하지 않는다.

- asset 및 시간 block bootstrap
- 평균 net return 95% confidence interval
- median net return 95% confidence interval
- Sharpe 95% confidence interval
- hit rate Wilson interval
- validation grid에는 multiple-testing 경고 또는 보정 결과

최소 표본:

- 자산별 30건 미만은 개별 판정 보류
- 전체 test 거래 200건 미만은 통계 결론을 제한
- 표본 기준 미달은 실패가 아니라 `INCONCLUSIVE`로 판정

## Execution Gates

### Gate 0: Baseline Reproduction

통과 조건:

- 기존 BTC close-execution 수치가 허용 오차 내 재현
- compile, module self-tests, repository smoke tests 통과

실패 시:

- 일반화 작업을 중단하고 회귀 원인을 먼저 수정한다.

### Gate 1: Execution Correctness

통과 조건:

- next-bar-open 단위 테스트 전부 통과
- 거래 로그에서 `entry_time > signal_time`
- 미래 데이터 변경 불변성 테스트 통과

실패 시:

- 성과 수치를 해석하지 않는다.

### Gate 2: Validation Feasibility

통과 조건:

- 최소 3개 암호화폐와 2개 비암호화폐에서 실행 가능
- pooled validation 거래가 100건 이상
- 오류 없이 동일 frozen config 적용

성과는 이 gate의 필수 통과 조건이 아니다.

### Gate 3: Validation Evidence

진행 조건:

- asset-equal 평균 net return > 0
- pooled 평균 net return > 0
- leave-one-asset-out 결과의 과반이 양수
- random/placebo Sharpe 분포의 90 percentile 이상
- next-open 결과가 close-execution 대비 완전히 붕괴하지 않음

미충족 시:

- holdout을 열지 않는다.
- 결과를 `FAILED_VALIDATION`으로 보고하고 원인 분석까지만 수행한다.

### Gate 4: Untouched Test

성공 조건:

- 10bp/side next-open에서 pooled 평균 net return > 0
- asset-equal 평균 net return > 0
- test 자산 과반의 평균 net return > 0
- portfolio Sharpe > 0
- leave-one-asset-out portfolio Sharpe가 과반에서 > 0
- 실제 Sharpe가 placebo 분포의 95 percentile 이상
- top 5 winners 제거 후 pooled 평균 net return > 0
- 심각한 look-ahead 또는 데이터 품질 문제가 없음

실패 조건:

- pooled 또는 asset-equal 평균 net return <= 0
- 성과가 한 자산/연도/소수 거래에만 의존
- next-open과 보수적 비용에서 edge 소멸
- placebo 대비 우위 없음
- 테스트 이후 발견된 look-ahead 또는 split contamination

보류 조건:

- 전체 test 거래가 200건 미만
- 필요한 자산군 데이터가 충분하지 않음
- 신뢰구간이 0을 넓게 포함하지만 점추정은 양수

최종 판정은 다음 셋 중 하나만 사용한다.

```text
PASS_GENERALIZATION
FAILED_GENERALIZATION
INCONCLUSIVE
```

## Expected Files

새 파일 이름은 기존 코드 구조를 확인한 뒤 조정할 수 있지만, 역할은 분리한다.

```text
generalization_runner.py
strategy_execution.py
generalization_assets.csv
frozen_h72_price.json
test_strategy_execution.py
test_generalization_splits.py
../reports/generalization/
```

최종 산출물:

```text
config_snapshot.json
asset_manifest_snapshot.csv
data_quality.csv
forecast_metrics.csv
trades.csv
asset_summary.csv
portfolio_summary.json
regime_summary.csv
robustness_grid.csv
bootstrap_summary.json
placebo_summary.json
GENERALIZATION_REPORT.md
```

모든 결과에는 아래 provenance를 기록한다.

```text
run timestamp UTC
git commit or git diff hash
Python/package versions
data source
download start/end timestamp
config version
random seeds
```

## Recommended Work Order

1. 현재 baseline을 테스트로 고정한다.
2. forecast에 `open`을 추가한다.
3. next-bar-open 체결 엔진과 look-ahead 테스트를 구현한다.
4. chronological split과 split별 집계를 구현한다.
5. frozen config와 asset manifest를 만든다.
6. validation 자산 데이터를 생성하고 데이터 품질을 검사한다.
7. frozen primary validation을 실행한다.
8. random/placebo와 bootstrap을 실행한다.
9. robustness 및 regime 분석을 실행한다.
10. Gate 3을 판정한다.
11. Gate 3 통과 시에만 untouched test를 한 번 실행한다.
12. 최종 보고서와 판정을 작성한다.

## First Session Scope

한 세션에서 모든 자산과 bootstrap을 무리하게 끝내지 않는다. 첫 구현 세션의
완료 조건은 다음과 같다.

- baseline regression test
- forecast `open` column
- next-bar-open execution
- no-look-ahead unit tests
- chronological split
- frozen config and asset manifest skeleton
- BTC와 ETH validation dry run

이 범위가 통과한 뒤 두 번째 세션에서 전체 validation matrix를 실행한다.

## Commands To Run Before Reporting

실제 파일명이 정해지면 명령을 갱신하되 최소 검증은 유지한다.

```bash
../.venv/bin/python -m py_compile \
  run_kalman_pipeline.py conf_filter_backtest.py \
  strategy_execution.py generalization_runner.py

../.venv/bin/python -m unittest -v \
  test_strategy_execution.py test_generalization_splits.py

../.venv/bin/python kalman.py
../.venv/bin/python flat_chart.py
../.venv/bin/python test_vectorbt.py
../.venv/bin/python test_backtrader.py

git diff --check
```

네트워크 smoke test는 데이터 제공자 상태에 따라 별도로 실패 원인을 기록한다.

## Reporting Template

각 세션 종료 시 아래 형식으로 보고한다.

```text
Status:
  gate reached:
  verdict:

Implemented:
  files changed:
  behavior changed:

Tests:
  passed:
  failed:
  not run:

Results:
  execution model:
  assets:
  trades:
  asset-equal avg net:
  pooled avg net:
  Sharpe:
  MDD:
  placebo percentile:

Risks:
  data limitations:
  statistical limitations:
  unresolved issues:

Next exact step:
```

## Prompt For The Next Codex

```text
~/trading_lab/scripts/CODEX_BRINGUP.md와
~/trading_lab/scripts/CODEX_GENERALIZATION_PLAN.md를 전부 읽고,
현재 코드와 git 상태를 확인해줘. 사용자 파일이나 `_1` 파일을 되돌리지 말고
GENERALIZATION_PLAN의 First Session Scope를 순서대로 구현해. 먼저 기존 BTC
close-execution 결과를 회귀 테스트로 고정하고, forecast에 open을 추가한 뒤
bar t 종가 신호를 bar t+1 시가에 체결하는 엔진과 no-look-ahead 테스트를
구현해. chronological split, frozen config, asset manifest skeleton까지 만든
후 BTC와 ETH validation dry run을 실행해. 테스트 결과와 Gate 0/Gate 1 판정을
GENERALIZATION_REPORT 또는 세션 리포트로 남기고 요약해줘.
```
