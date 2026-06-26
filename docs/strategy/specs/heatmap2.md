# 전략 명세: heatmap2 — 고볼륨 노드(HVN) 지지/저항 롱숏 (단일종목)

## 1. 아이디어

볼륨 프로파일 히트맵에서 **색이 짙은 가격대 = 거래가 집중된 합의가격(HVN, High
Volume Node)** 을 지지/저항으로 가정한다. 현재가 기준 바로 아래 HVN은 **지지**,
바로 위 HVN은 **저항**. 가격이 지지에서 반등하면 롱, 저항에서 거부되면 숏(양방향).

heatmap1(단일 POC/VA 평균회귀)의 일반화: VA 한 덩어리 대신 프로파일의 **여러 봉우리**를
지지/저항 레벨로 쓴다. yoon3·heatmap1 계보의 다음 단계.

## 2. 레벨 산출 (lookahead 없음)

각 바 t에서 직전 `lookback`개 바(또는 `cumulative=true`면 0..t) 프로파일을 만들고,
`profile_nodes`로 HVN을 추출한다:

| 단계 | 규칙 |
| --- | --- |
| 로컬 피크 | bin이 양 이웃보다 큰 곳 |
| 강도 필터 | 최대 bin의 `node_min_strength`(기본 0.3) 배 이상 |
| 병합 | 서로 `node_min_gap_bins`(기본 3) 이상 떨어진 것만, 강도 상위 `node_top_n`(기본 4)개 |

그 시점 종가(t 이하 데이터) 기준 **아래 가장 가까운 HVN = 지지(`val`)**, **위 가장
가까운 HVN = 저항(`vah`)**, 프로파일 최빈가 = `poc`. heatmap1 시뮬엔진 재사용을 위해
컬럼명을 `poc/vah/val`로 매핑한다(vah=저항, val=지지). warmup·해당 노드 없음은 NaN.

가격축은 `price_scale="log"` 기본(bin이 geomspace → 변동성 큰 구간 저가대 보존).

## 3. 신호 규칙 (`signal_mode`)

heatmap1 엔진을 그대로 상속하므로 모드 의미만 바뀐다:

- **`va_reversion`(기본)** — 지지/저항 반등. close가 지지 아래로 갔다 복귀 → 롱
  (위 저항/POC 회귀 기대). 저항 위에서 복귀 → 숏.
- **`va_breakout`** — close가 저항 상향 돌파 → 롱, 지지 하향 이탈 → 숏.

공통 실행(heatmap1과 동일):
- 체결 `next_open`(신호 봉의 다음 봉 시가). 룩어헤드 없음.
- 청산: `poc_target`(POC/추세 목표) · `va_stop`(S/R 경계 밖 손절, 버퍼 =
  `stop_buffer_frac`×(저항−지지)) · `horizon`(`max_hold_bars`) · `opposite`(반대 신호).
- `long_only`(기본 **false** = 양방향) · `min_hold_bars` · 비용(`fee`+`slippage` bps).

## 4. 파라미터

| 키 | 의미 | 기본 |
| --- | --- | --- |
| `asset_class` | equity(yfinance) / crypto(ccxt) | equity |
| `interval` / `period` | TF / 기간 (대시보드 위젯이 주입) | 1d / max |
| `lookback` | 프로파일 윈도우(봉) | 120 |
| `profile_bins` | 가격 bin 수 | 80 |
| `price_scale` | log / linear | **log** |
| `node_top_n` | HVN 최대 개수 | 4 |
| `node_min_strength` | HVN 최소 강도(최대 대비) | 0.3 |
| `node_min_gap_bins` | HVN 최소 간격(bin) | 3 |
| `signal_mode` | va_reversion / va_breakout | va_reversion |
| `long_only` | 롱 온리 | false |
| `stop_buffer_frac` | 손절 버퍼(S/R폭 배수) | 0.5 |
| `min_hold_bars` / `max_hold_bars` | 최소/최대 보유봉 | 1 / 60 |

## 5. 산출 (StrategyArtifacts)

heatmap1 핸들러 상속 → 계약 동일.
- `forecast`: close + `poc`/`vah`(저항)/`val`(지지) 자동 오버레이.
- `trades`: 표준 스키마, `exit_reason` ∈ {poc_target, va_stop, horizon, opposite, end_of_data},
  `entry_reason`은 "지지 반등/저항 거부/저항 상향돌파/지지 하향이탈"로 표기.
- `equity`/`metrics`: 표준.
- 2D 히트맵(로그스케일)은 extras `heatmap`로 곁들임.
