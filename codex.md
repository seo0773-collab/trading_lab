# DI Kalman M/W 전략 수정 계획

작성일: 2026-06-13

## 진행 상태

2026-06-13 기준:

```text
Phase 0: 완료 - 기준선 및 기존 버킷 모델 보존
Phase 1: 완료 - split 경계 누수, 성과 귀속, P4/P5 통계 분리 수정
Phase 2: 완료 - P1~P4 feature / P5·가격 outcome 데이터셋 구현
Phase 3: 완료 - train-only weighted k-NN 유사도 예측기 구현
Phase 4: 완료 - 유사 이웃 실제 가격 경로 기반 보수적 EV 구현
Phase 5: 완료 - 온라인 P5 후보 경로 재평가 및 의사결정 로그 구현
Phase 6: preflight 완료 - 1h 3개 데이터셋만 백테스트 실행 가능
Phase 7: 미착수
```

Phase 1 검증 결과:

```text
전체 unittest: 51개 통과
합성 4h P4 train 통계: 후보 36, 완결 36
합성 4h P5 train 통계: 후보 36, 완결 10, 미완결/경계 제외 26
split을 넘어간 실제 거래: 0
train transition P1~P5 완결 사례: 74
```

Phase 1 구현 내용:

- baseline trade outcome이 다음 split을 사용하면 학습에서 제외
- transition 학습은 P4와 P5 confirmation이 모두 train인 사례만 사용
- P4 진입 통계와 P5 confirmation 진입 통계를 별도로 적합
- 포지션은 split 마지막 봉에서 `split_boundary`로 강제 청산
- split 마지막 봉의 신호는 다음 split에 진입하지 않도록 차단
- 거래 손익과 bar return이 동일 split에 귀속되도록 통일
- 관련 회귀 테스트 추가

Phase 2~3 검증 결과(합성 4h, 9,000 bars, seed 7):

```text
전체 패턴 사례: 244
train 완결 P1~P5 사례: 154
validation 평가 사례: 38
similarity P5 위치 MAE: 0.8846 mean-leg
train global median MAE: 1.5586 mean-leg
MAE 감소율: 43.24%
q10~q90 coverage: 81.58%
continuation Brier score: 0.2121
평균 effective_n: 30.06
평균 confidence: 0.6324
```

이 수치는 합성 데이터의 연구 기준선일 뿐 전략 수익성을 의미하지 않는다.
실제 가격 경로 기반 기대값과 온라인 상태 모델을 구현했더라도 Phase 6
실데이터 validation을 통과하기 전에는 매매 우위를 주장하지 않는다.

Phase 4~5 검증 결과(합성 4h, 9,000 bars, seed 7):

```text
가격 EV train 사례: 154
가격 EV validation 사례: 38
예측 순수익 MAE: 0.1354
평균 예측 순수익: -0.0523
평균 실현 순수익: -0.0739
q25 하한/표본 기준 진입 가능 사례: 0

온라인 train snapshot: 6,481
온라인 validation snapshot: 1,758
남은 순수익 예측 MAE: 0.1130
온라인 hold 판단: 454
온라인 exit 판단: 1,304
온라인 reverse 판단: 0
```

진입 가능 사례가 0인 것은 구현 오류로 보지 않는다. 현재 합성 데이터와
보수적 q25 하한에서 검증된 양의 기대값이 없다는 뜻이다. 백테스트 성과를
만들 목적으로 `entry_margin`, lower quantile, 최소 표본을 완화하지 않는다.
온라인 판단은 현재 연구 로그만 생성하며 실제 청산/반전 주문에는 연결하지
않았다.

실데이터 preflight 결과:

```text
BTCUSDT 1h: ready, train 3.87년, 완결 P5 1,068
ETHUSDT 1h: ready, train 3.87년, 완결 P5 1,077
SOLUSDT 1h: ready, train 3.50년, 완결 P5 1,029

BTCUSDT 4h: not ready, 완결 P5 279 / 최소 300
ETHUSDT 4h: not ready, 완결 P5 285 / 최소 300
SOLUSDT 4h: not ready, 완결 P5 248 / 최소 300

BTCUSDT 1d: not ready, train 3.87년 / 최소 10년, 완결 P5 36
ETHUSDT 1d: not ready, train 3.87년 / 최소 10년, 완결 P5 43
SOLUSDT 1d: not ready, train 3.50년 / 최소 10년, 완결 P5 29
```

따라서 최초 실데이터 백테스트 대상은 BTC/ETH/SOL 1h로 제한한다. 4h와
1d를 포함하기 위해 최소 표본 기준을 낮추지 않는다.

이 문서는 사용자와 Codex가 논의한 DI Kalman M/W 전략의 목표와 수정
방향을 다음 작업자(Claude 포함)가 바로 이어받을 수 있도록 정리한 실행
계획이다. 저장소 전체 작업 규약은 `CLAUDE.md`가 우선하며, 이 문서는
`di-kalman-mw-v1` 전략의 연구 및 수정 범위만 다룬다.

## 1. 전략 목표

이 전략의 핵심은 **최근 확정 극점 4개(P1~P4)를 이용해 다음 극점 P5의
위치를 과거 5극점 패턴 데이터로 확률 예측하는 것**이다.

```text
학습 사례: 과거에 완성된 P1, P2, P3, P4 -> 실제 P5
현재 입력: 최근 확정된 P1, P2, P3, P4
예측 대상: 아직 확정되지 않은 P5의 위치와 형성 가능성
```

P6를 예측하거나 6극점 구조로 확장하지 않는다.

진입 후 P4에서 시작된 현재 진행 경로가 예상 P5 분포와 가까워지는지
계속 관찰한다. 기존 방향의 유사도와 기대값이 낮아지고 다른 패턴 방향의
유사도와 기대값이 높아지면 포지션 유지, 청산, 반전을 재검토한다.

## 2. 전략 해석

### 2.1 진입 전

1. `+DI Kalman`, `-DI Kalman`에서 인과적으로 확정된 극점을 추출한다.
2. 각 DI 라인의 최근 P1~P4 구조를 정규화된 특징 벡터로 만든다.
3. train 데이터에서 P1~P4가 유사했던 과거 사례를 찾는다.
4. 유사 사례들의 실제 P5 위치를 가중 집계해 조건부 분포를 만든다.
5. 최근 `+DI/-DI` 극점 평균과 실제 가격 경로 결과를 조건에 포함한다.
6. 비용과 불확실성 마진을 차감한 기대값이 기준을 넘으면 다음 봉
   시가에 진입한다.

### 2.2 진입 후

P4 이후 아직 P5로 확정되지 않은 현재 Kalman 경로를 매 봉 갱신한다.
이 값은 확정 극점으로 취급하지 않고, 과거 P4~P5 진행 경로와 비교하는
관측값으로만 사용한다.

```text
EV_keep    = 현재 포지션을 유지할 조건부 기대값
EV_exit    = 지금 청산했을 때의 가치(기본 0, 청산 비용 반영)
EV_reverse = 반대 포지션의 조건부 기대값 - 청산/재진입 비용
```

순간적인 유사도 변화만으로 매매하지 않는다. 최소 유효 표본, 예측
불확실성, 기대값 차이의 안전 마진, 일정 지속 시간(hysteresis)을 함께
요구한다.

## 3. 패턴 데이터 정의

현재 `scripts/di_kalman_mw/extreme_transition.py`의
`TransitionInstance`를 기반으로 확장한다.

### 3.1 입력 특징: P1~P4만 사용

최소 특징 후보:

```text
line                     plus / minus
pattern                  W / M
P1~P4 상대 높이
leg1, leg2, leg3 정규화 진폭
leg1, leg2, leg3 소요 봉 수
retr_ratio
leg3_ratio
p3_vs_p1_norm
right/left width_ratio
P1~P4 전체 span_bars
최근 +DI 극점 4개 평균
최근 -DI 극점 4개 평균
두 평균의 차이와 비율
P4 시점 ATR 및 변동성 regime
```

모든 스케일링 파라미터와 특징 가중치는 train에서만 적합한다.
P5 값, P5 시각, P5 확인 정보는 입력 특징에 절대 포함하지 않는다.

### 3.2 예측 결과

과거 각 사례에는 다음 결과를 저장한다.

```text
P5.value
P5.value - P4.value
(P5.value - P4.value) / mean_leg(P1~P4)
(P5.value - P3.value) / mean_leg(P1~P4)
P5 발생까지의 봉 수
P5 confirmation까지의 봉 수
W일 때 P5 > P3 확률
M일 때 P5 < P3 확률
P4 이후 실제 가격 경로
방향별 MFE / MAE
고정 horizon 수익률
stop / take-profit 선도달 결과
```

P5가 데이터 끝까지 확정되지 않은 사례를 무조건 제거하면 빠르게
확정되는 패턴만 남는 편향이 생긴다. 라이브 미완성 사례와 학습 시계열
끝의 미확정 사례를 구분하고, 시간 예측에는 censoring 처리를 검토한다.

## 4. 유사도와 확률 예측

현재의 tercile 버킷 및 폴백 통계는 기준선으로 유지하되, 주 모델은
연속 거리 기반 가중 이웃 방식으로 교체한다.

초기 구현은 해석 가능한 k-NN/kernel 방식으로 제한한다.

```text
distance_i = weighted_distance(current_features, train_pattern_i)
weight_i   = exp(-(distance_i ** 2) / temperature)
```

가중 이웃으로 다음을 계산한다.

```text
P5 변위의 q10/q25/median/q75/q90
P(continuation)
P5까지 예상 봉 수의 분위수
유효 표본 크기(effective sample size)
최근접 거리
가중치 집중도
예측 신뢰도
```

신뢰도는 유사도 자체와 분리한다. 가까운 사례가 하나뿐인 경우 높은
유사도처럼 보여도 신뢰도는 낮아야 한다.

최소 출력 계약:

```text
prediction_median
prediction_quantiles
p_continuation
effective_n
nearest_distance
confidence
model_fallback
```

## 5. 최근 극점 평균과 기대값

최근 `+DI/-DI` 극점 평균은 기대이익에 임의 배수를 곱하는 용도로 쓰지
않는다. 우선 유사 패턴 검색의 조건 특징으로 사용한다.

현재 `expected_values()`의 다음 방식은 검증 전까지 주 모델에서 제거한다.

```text
평균 이익 * pressure_rr_factor
평균 손실 / pressure_rr_factor
continuation factor를 EV와 TP 양쪽에 중복 반영
```

기대값은 유사 과거 사례의 **실제 가격 경로 결과**로 직접 추정한다.

```text
EV_long  = weighted_mean(realized_long_net_return)
EV_short = weighted_mean(realized_short_net_return)
```

또는 stop/TP 결과를 사용할 경우:

```text
EV = P(TP first) * avg_net_win
     - P(SL first) * avg_net_loss
     - expected_cost
```

신호 조건에는 점 추정치뿐 아니라 보수적 하한을 사용한다.

```text
진입 후보 = EV_lower_bound > entry_margin
```

## 6. 온라인 재평가

P4 확정 후 각 봉에서 현재 Kalman 값과 경과 시간을 이용해 기존 P1~P4
이웃들의 가중치를 갱신한다. 이 과정은 새로운 확정 P5를 만들어내는 것이
아니다.

예상 경로에서 멀어질수록 기존 방향의 가중치는 낮아지고, 현재 경로와
더 잘 맞는 다른 P1~P4 패턴 사례의 가중치는 높아질 수 있다.

재평가 규칙 초안:

```text
유지:
  EV_keep_lower_bound > 0
  기존 방향 confidence >= hold_threshold

청산:
  EV_keep_lower_bound <= exit_threshold
  위 상태가 confirm_bars 이상 지속

반전:
  EV_reverse_lower_bound >
      EV_keep_upper_bound + switch_cost + reversal_margin
  반대 방향 effective_n >= minimum
  위 상태가 confirm_bars 이상 지속
```

하드 ATR stop은 모델 오류와 급변 위험에 대한 최종 안전장치로 유지한다.
모델 기반 청산은 hard stop을 대체하지 않는다.

## 7. 먼저 수정해야 할 검증 결함

다음 결함을 해결하기 전의 validation/test 성능은 신뢰하지 않는다.

### 7.1 split 경계 누수

현재 train 이벤트의 baseline trade와 P5 outcome이 validation 구간까지
진행될 수 있다.

- train 통계에는 결과까지 train 안에서 완결된 사례만 포함한다.
- 또는 split 경계 앞에 최대 보유/확정 기간만큼 purge 구간을 둔다.
- transition 학습 사례는 `p4_conf_idx`뿐 아니라 `p5_conf_idx`도 train
  경계 안에 있어야 한다.

### 7.2 split 성과 귀속 불일치

현재 거래는 signal split, bar return은 달력 split에 귀속될 수 있다.

- 연구 비교용으로 split 경계에서 포지션을 강제 청산하는 방식을 우선한다.
- 거래, 수익률, 지표의 split 귀속 기준을 하나로 통일한다.

### 7.3 P4/P5 통계 혼용

P4 다음 봉 진입 결과로 만든 통계를 P5 확인 진입에 적용하지 않는다.

- 주 전략은 P1~P4에서 P5를 예측하는 P4 진입 모델로 명확히 고정한다.
- P5 confirmation 진입은 별도 비교 기준으로 남기되 별도 outcome
  통계를 사용한다.

### 7.4 표본 충분성

캔들 수 하나만 충족해도 충분하다고 판단하지 않는다.

```text
충분성 = 최소 기간 충족 AND 최소 완결 P5 사례 수 충족
```

방향, 라인, 패턴, regime별 `effective_n`도 별도로 보고한다.

## 8. 구현 단계

### Phase 0: 기준선 동결

- 현재 전체 테스트 42개 통과 상태를 기준선으로 기록한다.
- 현재 버킷 모델 결과를 baseline artifact로 보존한다.
- test 구간은 모델/파라미터 선택에 사용하지 않는다.

완료 기준:

```text
동일 입력/config 재현 가능
기존 h72 전략 계약 유지
현재 결과와 config snapshot 보존
```

### Phase 1: 데이터 경계와 누수 수정

상태: 완료 (2026-06-13)

대상:

```text
scripts/di_kalman_mw/run.py
scripts/di_kalman_mw/stats.py
scripts/di_kalman_mw/extreme_transition.py
scripts/di_kalman_mw/metrics.py
tests/test_di_kalman_mw.py
```

작업:

- split 경계 완결 조건 및 purge 적용
- transition outcome의 `p5_conf_idx` 경계 검사
- split 경계 포지션 처리 규칙 고정
- P4/P5 통계 분리
- 누수 회귀 테스트 추가

### Phase 2: 패턴 데이터셋 확장

상태: 완료 (2026-06-13)

- P1~P4 시간/진폭/pressure/regime 특징 추가
- P5 위치, 시간, 가격 경로 outcome 추가
- 학습 입력과 outcome 컬럼을 명시적으로 분리
- dataset schema/version 저장

예상 신규 모듈:

```text
scripts/di_kalman_mw/pattern_dataset.py
```

기존 `extreme_transition.py`를 무리하게 비대하게 만들지 말고, 극점 전이
기초 자료형은 유지하면서 데이터셋 조립 책임을 분리한다.

### Phase 3: 유사도 예측기

상태: 완료 (2026-06-13)

- train-only scaler와 특징 가중치
- weighted k-NN/kernel predictor
- 거리, effective_n, confidence 계산
- 버킷 모델 대비 MAE/Brier/coverage 비교
- validation에서만 k, temperature, 특징 집합 선택

예상 신규 모듈:

```text
scripts/di_kalman_mw/similarity.py
```

### Phase 4: 가격 기대값 모델

상태: 완료 (2026-06-13)

- 각 유사 사례의 P4 이후 long/short 실제 가격 결과 연결
- 비용 차감 순수익 분포 계산
- EV 점 추정과 보수적 하한 계산
- pressure 임의 배수 제거 후 ablation 비교

비교군:

```text
패턴만
패턴 + 최근 극점 평균
패턴 + 평균 + volatility regime
기존 버킷/pressure 보정식
```

### Phase 5: 온라인 재평가와 청산

상태: 완료 (2026-06-13, 연구 판단 로그까지)

실제 체결 엔진 연결은 Phase 6 백테스트에서 별도 비교군으로 수행한다.

- P4 이후 미확정 경로 snapshot 생성
- 매 봉 유사도, P5 분포, EV 갱신
- hold/exit/reverse 의사결정 기록
- hysteresis, 최소 표본, 비용 마진 적용
- 기존 opposite-pattern exit와 성과 비교

예상 신규 모듈:

```text
scripts/di_kalman_mw/online_state.py
```

### Phase 6: 실데이터 walk-forward 검증

상태: 데이터 수집 및 preflight 완료, 백테스트 실행 대기

- BTC/ETH/SOL의 1h/4h/1d native OHLCV 수집
- anchored 또는 rolling walk-forward
- train에서 적합, validation에서 선택, test는 최종 1회 평가
- 비용 1x/2x, regime, 방향, strong/weak별 결과 보고

현재 `data/raw`가 비어 있으므로 이 단계 전에는 전략 유효성을 주장하지
않는다.

### Phase 7: 대시보드 연결

연구 모델이 독립 테스트를 통과한 뒤에만 공통
`StrategyArtifacts` 핸들러로 연결한다.

추가 표시 후보:

```text
P5 예측 중앙값/구간
p_continuation
effective_n
confidence
현재 EV_long / EV_short
온라인 결정 사유
```

공통 UI에 전략 이름이나 전용 컬럼을 하드코딩하지 않는다.

## 9. 검증 계획

### 9.1 인과성

- P5 값을 변경해도 P1~P4 특징과 진입 시점 예측은 변하지 않아야 한다.
- 시계열을 시점 `t`에서 잘랐을 때 `t` 이전 예측은 전체 데이터 실행과
  동일해야 한다.
- 학습 사례의 P5 확인과 가격 outcome은 train 경계 안에서 끝나야 한다.
- 온라인 업데이트는 해당 봉까지의 값만 사용해야 한다.

### 9.2 예측 품질

- P5 정규화 위치 MAE
- global median 대비 MAE 감소
- quantile coverage 및 calibration
- continuation Brier score
- effective_n 및 거리 구간별 오차
- P5 형성 시간 오차

### 9.3 매매 품질

- net expectancy와 신뢰구간
- profit factor, max drawdown, turnover
- long/short 및 symbol/timeframe별 일관성
- 상위 소수 거래 이익 집중도
- 비용 2배 민감도
- 기존 단순 M/W, DMI crossover, 무신호/항상 보유 기준선 비교

### 9.4 Ablation

각 요소의 실제 추가 가치를 분리한다.

```text
P1~P4 형상만
+ 시간 특징
+ 최근 극점 평균
+ volatility regime
+ 온라인 경로 업데이트
+ 모델 기반 청산
```

## 10. 승인 기준

다음 조건 전에는 실전 가능 또는 검증 완료로 표시하지 않는다.

- split 누수 테스트 통과
- validation에서 global/bucket 기준선보다 P5 예측 개선
- 충분한 완결 P5 사례와 OOS 거래 수 확보
- 주요 symbol/timeframe 다수에서 기대값 부호가 일관됨
- 비용 2배에서도 결과가 완전히 붕괴하지 않음
- test는 모든 모델 선택 후 단 한 번 평가
- 온라인 청산이 단순 hard stop/time stop 대비 OOS 위험조정 성과 개선

## 11. 작업 시 금지 사항

- P6 또는 6극점 예측으로 범위를 확장하지 않는다.
- P5 outcome을 P1~P4 입력 특징에 포함하지 않는다.
- test 결과를 보고 특징, k, temperature, 임계값을 조정하지 않는다.
- DI 평균 비율을 근거 없이 수익/손실에 곱해 EV를 부풀리지 않는다.
- 유사도가 높다는 이유만으로 표본 수와 불확실성을 무시하지 않는다.
- 모델 기반 청산만 믿고 hard risk stop을 제거하지 않는다.
- 전략 연구를 이유로 공통 서비스/UI에 전략별 분기를 추가하지 않는다.

## 12. 다음 작업 순서

다음 작업자는 바로 유사도 모델부터 구현하지 말고 아래 순서를 따른다.

1. split 경계 누수 재현 테스트 작성
2. baseline trade와 P5 outcome의 train 완결 조건 수정
3. split 성과 귀속 통일
4. P4 주 전략과 P5 비교 전략의 통계 분리
5. 패턴 데이터셋 schema 및 특징/outcome 분리
6. 버킷 baseline 평가 고정
7. 연속 유사도 예측기 구현
8. 실제 가격 경로 기반 기대값 구현
9. 온라인 재평가 및 청산 구현
10. 실데이터 walk-forward 평가

이 순서를 바꾸면 누수가 남은 결과를 최적화하거나, P5 예측 정확도와
매매 기대값을 혼동할 가능성이 높다.
