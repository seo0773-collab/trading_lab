# 전략 명세: yoon2 — Kalman MACD 타이밍 (단일종목)

## 1. 아이디어

`macd_raw.txt`(Pine v6 인디케이터)의 **칼만 처리된 라인만** 사용한다. raw
MACD/Signal/Histogram은 쓰지 않는다. 한 종목에 대해 칼만 평활된 모멘텀의
**가속/감속 전환**을 잡아 롱·숏 양방향으로 타이밍한다.

칼만 라인 4종:

| 라인 | 정의 |
| --- | --- |
| `kal_macd` | `kalman_base`에 따라 `kalman(macd_line)`(기본) 또는 `kalman(fast_ema) − kalman(slow_ema)` |
| `kal_signal` | `kalman(EMA(kal_macd, signal_len))` |
| `kal_hist` | `kal_macd − kal_signal` |
| `kal_hist_delta` | `kal_hist − kal_hist[1]` (히스토그램의 1차 변화 = 모멘텀 가속/감속) |

## 2. 신호 규칙 (`entry_trigger`로 트리거 계열 선택)

부호 전환 판정 로직은 공통이고, `entry_trigger`로 *어떤 계열의* 부호 전환을
볼지만 고른다:

- **`delta_turn`(기본)** — `kal_hist_delta`(히스토그램 1차 변화) 부호 전환.
  모멘텀 가속/감속 반전. 가장 선행·가장 시끄러움(검증된 기본 조합의 토대).
- **`cross`** — `kal_hist`(=`kal_macd − kal_signal`)의 0교차 = **라인 크로싱**
  (kal_macd가 kal_signal을 상향/하향 돌파). 덜 선행·덜 시끄러움.

진입 규칙(트리거 계열 `s` 기준, 공통):

- **롱 진입**: `s`가 `confirm_bars`봉 연속 양(+)이고 직전이 ≤0.
- **숏 진입**: `s`가 `confirm_bars`봉 연속 음(−)이고 직전이 ≥0.
- **양방향 stop-and-reverse**: 보유 중 반대 신호가 나면 청산 후 즉시 반대 진입.
- 체결은 `next_open`(신호 봉의 다음 봉 시가). 룩어헤드 없음.
- 노이즈 필터(`min_hist_gap_atr`, `macd_zero_filter`)는 두 트리거에 동일 적용.

### 노이즈 통제 (델타 전환은 가장 선행·가장 시끄러움 → 이게 핵심)

| 파라미터 | 역할 | 기본 |
| --- | --- | --- |
| `kalman_q` / `kalman_r` | 반응성 / 평활도 (1차 노이즈 제거) | 0.01 / 0.10 |
| `confirm_bars` | 같은 부호 델타 연속 N봉일 때만 진입 | 2 |
| `min_hist_gap_atr` | `|kal_hist| ≥ N·ATR`일 때만 진입(0선 잡음 차단) | 0(off) |
| `macd_zero_filter` | `kal_macd` 0선 방향과 정렬된 진입만 허용 | false |
| `atr_stop_mult` | ATR 배수 손절(0=off) | 0 |
| `max_hold_bars` | 최대 보유봉 시간청산(0=off) | 0 |

## 3. 분포 기반 익절 사다리 (`tp_enabled`)

히스토그램 델타 크기 분포를 모아 익절(부분청산) 주문을 분위수에 분산한다.

- **분포 수집**: in-sample(`identification`) 구간에서만 `|kal_hist_delta| / ATR`
  (= 변동성 대비 봉당 모멘텀 가속, 무차원)의 경험분포를 수집한다. 이 분포를
  validation/test에 **고정 적용** → 룩어헤드 없음.
- **ATR 환산**: 분위수 `tp_quantiles`(기본 33/66/90%)에 `tp_atr_scale`(기본 30)을
  곱해 봉당 가속을 다봉 익절 거리(ATR 배수)로 환산. 진입 시 ATR로 가격 목표 산출.
  (히스토그램 델타는 가격 단위라 ATR로 나누면 무차원 비율이 됨.)
- **사다리 청산**: 거리 오름차순 3단계에서 `tp_fractions`(기본 1/3씩)만큼 부분
  익절(`exit_reason="take_profit"`, `take_profit_price` 기록). 잔량은 기존 청산
  (반대 신호/손절/시간/데이터 끝)으로 종료. 트랜치별 보유 분수를 합산해 바별
  익스포저(`position`)에 반영하므로 진입당 `size_frac` 합 = 1.0.
- 같은 봉에서 여러 단계 동시 체결 허용. 손절은 익절보다 먼저 보수적 확인.
- `tp_enabled=false`(기본)면 전량 단일 청산으로, 기존 검증 동작과 정확히 동일.

## 4. 비용·청산

- 비용: `fee_bps_per_side + slippage_bps`를 진입(전량 1회)·각 부분/전량 청산에
  해당 분수만큼 차감(기본 5+5bps).
- 청산 사유: `take_profit`(분포 익절), `opposite`(반대 신호), `stop_loss`(ATR
  손절), `horizon`(시간청산), `end_of_data`(데이터 끝) — 모두
  `presentation.EXIT_REASON_LABELS`에 존재.

## 5. 아티팩트 계약

- `forecast`: OHLC + `atr`, `kal_macd`, `kal_signal`, `kal_hist`,
  `kal_hist_delta` (비OHLC 컬럼은 대시보드 파동 패널에 자동 노출).
- `trades`: 표준 스키마 + `stop_loss_price`, `take_profit_price`,
  `entry_reason`, `size_frac`(부분청산 분수).
- `equity`: 바별 포지션 수익(진입/청산 봉 비용 차감) 누적, 1.0 기준.
- `metrics`: `trades`, `hit_rate`, `total_return`, `sharpe`, `max_drawdown`,
  `profit_factor`, `expectancy`.

## 6. 합격선(게이트) — `checklists/yoon2.md`

핵심 검증: **델타 전환의 선행성이 거래비용을 이기는가.**

**실데이터 결론(`reports/yoon2/compare_all.md`, 5종목·10년·phase=all)**: 델타
전환은 양방향 단독으론 실패(raw both 평균 Sharpe −0.39, 281건 whipsaw). 그러나
**`kal_macd` 0선 추세필터 + long-only**로 가두면 평균 Sharpe 0.83·거래 8~14건으로
역전 — NVDA·MSFT는 Buy & Hold를 위험조정수익·낙폭 모두 상회. 이 조합을
config 기본값(`direction=long`, `macd_zero_filter=true`)으로 채택했다. 양방향은
연구용으로 남긴다.
