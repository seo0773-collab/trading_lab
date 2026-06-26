# 전략 명세: heatmap1 — Volume Profile 신호 (단일종목)

## 1. 아이디어

한 종목의 OHLCV를 **가격×시간 볼륨 프로파일**로 보고, 거래가 집중된 가격대
(POC)와 가치 영역(Value Area)을 기준으로 매매한다. Coinglass 스타일 2D 히트맵은
연구용 아티팩트이고, 대시보드 신호는 그 프로파일에서 뽑은 **1D 레벨**을 쓴다.

데이터 소스 무관: `asset_class='equity'`면 yfinance, `'crypto'`면 ccxt
(`scripts/fetch_ohlcv.py` 재사용). 코어(`scripts/volume_profile.py`)는 정규화
OHLCV만 받는 순수 함수다. yoon3(kalHist 누적 프로파일 게이트) 계보의 다음 단계.

## 2. 레벨 산출 (lookahead 없음)

각 바 t에서 **직전 `lookback`개 바**(또는 `cumulative=true`면 0..t 확장)의
프로파일로부터:

| 레벨 | 정의 |
| --- | --- |
| `poc` | volume 최대 가격 bin 중심 (Point of Control) |
| `vah` / `val` | POC에서 좌우로 더 큰 이웃을 흡수하며 누적 volume이 `va_pct`(기본 0.70)에 도달할 때까지 확장한 가격대의 상·하단 |

볼륨 분배는 균등분배(바 volume을 `[low,high]`가 걸친 bin에 겹침비율로 분배).
t의 레벨은 t 이하 데이터만 사용 → **룩어헤드 없음**. warmup(`lookback`) 전은 NaN.

## 3. 신호 규칙 (`signal_mode`)

- **`va_reversion`(기본)** — 평균회귀. close가 VAL 아래로 갔다가 다시 위로 복귀
  → 롱(POC 회귀 기대). VAH 위에서 복귀 → 숏(long_only면 생략).
- **`va_breakout`** — 추세추종. close가 VAH 상향 돌파 → 롱, VAL 하향 이탈 → 숏.

공통 실행:
- 체결 `next_open`(신호 봉의 다음 봉 시가). 룩어헤드 없음.
- 청산: `poc_target`(POC/추세 목표 도달) · `va_stop`(VA 경계 밖 손절,
  버퍼 = `stop_buffer_frac`×VA폭) · `horizon`(`max_hold_bars`) · `opposite`(반대 신호).
- `long_only`(기본 true) · `min_hold_bars` · 비용(`fee`+`slippage` bps).

## 4. 파라미터

| 키 | 의미 | 기본 |
| --- | --- | --- |
| `asset_class` | equity(yfinance) / crypto(ccxt) | equity |
| `interval` / `period` | TF / 기간 (대시보드 위젯이 주입) | 1d / max |
| `lookback` | 프로파일 윈도우(봉) | 120 |
| `profile_bins` | 가격 bin 수 | 60 |
| `va_pct` | Value Area 비율 | 0.70 |
| `cumulative` | 0..t 누적 프로파일 | false |
| `price_scale` | linear / log | linear |
| `signal_mode` | va_reversion / va_breakout | va_reversion |
| `long_only` | 롱 온리 | true |
| `stop_buffer_frac` | 손절 버퍼(VA폭 배수) | 0.5 |
| `min_hold_bars` / `max_hold_bars` | 최소/최대 보유봉 | 1 / 60 |

## 5. 산출 (StrategyArtifacts)

- `forecast`: close + `poc`/`vah`/`val`(자동 오버레이 노출).
- `trades`: 표준 스키마, `exit_reason` ∈ {poc_target, va_stop, horizon, opposite, end_of_data}.
- `equity`/`metrics`: 표준.
- 2D 히트맵 PNG는 forecast 계약 밖 → 선택적 곁들이 아티팩트(`volume_profile.render`).
