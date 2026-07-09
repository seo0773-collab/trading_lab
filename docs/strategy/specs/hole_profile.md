# 전략 명세: hole_profile

> **계층적 지그재그(L0→L1→L2) 진행-스윙 프로파일 돌파/지속 모멘텀** 단일종목
> 롱숏 전략(기본 BTC/USDT 일봉). 원본은 `hole_profile.txt`(Pine v6
> "Multiscale Market Structure — Level 0~2 + Time/Price Profiles") — ATR
> 방향전환 기반 원시 지그재그(L0)를 "피벗의 피벗" 재귀 압축으로 L1/L2까지
> 계층화하고, 선택 레벨에서 완결된 스윙(익스트림→익스트림)의 지속시간/가격변동%
> 분포를 프로파일링하는 시각화 지표다. Pine 코드 자체엔 매매 시그널이 없어,
> "hole"(분포 희소구간) 진입을 방향성 가설로 전략화했다.

## 1. 식별 정보

| 항목 | 값 |
| --- | --- |
| strategy_id | hole_profile |
| 분류 | single (단일종목, 기본 BTC/USDT 일봉) |
| 핸들러 | `src/trading_lab/strategies/hole_profile.py` (`HoleProfileHandler`) |
| 지표/상태 | `scripts/profile_sizing/hole_profile_features.py` |
| 파이프라인 | `scripts/profile_sizing/hole_profile.py` |
| config | `configs/strategies/hole_profile.json` |
| 검증 스터디 | `scripts/profile_sizing/hole_profile_study.py` → `reports/profile_sizing/hole_profile_study.json` (+ decile/gate CSV) |
| 원본 | `hole_profile.txt` (Pine v6, 매매 시그널 없는 순수 구조/프로파일 지표) |

## 2. 원본 지표 이식 (`hole_profile_features.py`)

Pine 코드 라인 344~471을 그대로 이식한다(변수명·비교연산자·분기 순서 동일):

1. **L0 원시 지그재그** — ATR 방향전환. `atr = SMA(TR, atr_len) * atr_mult`를
   되돌림 폭으로 삼아 상승/하락 추적 극값(`pend_max`/`pend_min`)이 ATR만큼
   되돌리면 그 극값을 "확정 피벗"(type=1 고점/−1 저점)으로 등록하고 방향을
   뒤집는다(`up_move` 상태기계).
2. **재귀 압축 L0→L1→L2** — `f_next_candidate`(Pine `f_next_candidate`,
   라인 118~239 그대로 포팅): 새 L0 피벗이 등록될 때마다 "피벗의 피벗"인지
   확인해 L1에 0~2개를 캐스케이드로 밀어 넣고, 그 L1 포인트가 다시 L2에
   0~2개를 밀어 넣는다. 동일 타입 연속 방지·동률 처리·"업그레이드" 삽입까지
   Pine과 동일한 분기 순서.
3. **완결 스윙 프로파일** — 각 레벨에서 연속 두 피벗이 새로 등록될 때마다
   `(지속시간_봉, |%변동|)`을 그 레벨의 표본 리스트에 추가(`f_record_profile`
   이식). **Pine에는 없는 확장**: 매 봉마다 "선택 레벨의 마지막 확정 피벗 →
   현재 종가"까지 **진행 중인 스윙**의 `duration_bars`/`pct_move`를 계산하고,
   그 절대값이 지금까지 완결된 스윙들의 분포에서 차지하는 **expanding
   백분위**(`profile_rank`, 무누수 — bisect로 그 시점까지 표본만 사용)를 구한다.

시간 단위는 Pine의 "Bars/Days" 옵션 중 **Bars만** 이식했다(대시보드 관례상
봉 단위 duration이 일관적).

## 3. 방향 가설 — "hole = 돌파/지속" (소진/반전 아님)

핵심 결정: 진행 중인 스윙이 그 레벨의 과거 완결 스윙 분포에서 **상위
percentile(희소구간, "hole")**에 도달했을 때 —

- ❌ 채택 안 함: "이미 드문 크기니 소진/반전" (평균회귀 가설)
- ✅ 채택: "이미 보기 드문 크기로 확장됐다는 것 자체가 이번 스윙이 구조적으로
  다르다(추세 강도가 세다)는 신호" (돌파/지속 모멘텀 가설)

`entry_percentile` 이상이면 진행 방향(`direction` = 마지막 피벗 대비 현재
종가 부호)으로 진입한다.

## 4. 검증 결과 (`hole_profile_study.py`, 2026-07-08, BTC/USDT 일봉 2017-08-17~2026-07-03, 3243봉)

레벨별 direction-signed forward return을 `profile_rank` 10분위/게이트로 나눠
Newey-West(lag=h) t + Bonferroni/BH-FDR(36검정: level×horizon×threshold) 병기.

| 레벨 | 최종 확정 피벗수 | 결과 |
| --- | --- | --- |
| L0 | 934 | decile FDR 유의 0/30, 게이트 전부 무의미(|t|<1.8, 방향도 비일관) |
| **L1** | **287** | decile FDR 유의 0/30이나, **게이트(threshold=0.70) h=5/10/20 전부 FDR 유의**(t=+2.93/+3.15/+3.49) |
| L2 | 79 | decile FDR 유의 1/30(우연 수준), 게이트는 threshold↑일수록 부호 반전·표본 급감(n=13) — 불안정 |

**L1만 채택**(Pine 기본 표시 레벨과 일치 — `showL1=true`, `profileLevel="Level 1"`).
L1 게이트 상세(threshold=0.70, n=317/2770~2755):

| horizon | gate mean | baseline mean | edge | t-stat | FDR |
| --- | --- | --- | --- | --- | --- |
| 5봉 | +2.23% | +0.50% | +1.74%p | +2.93 | 유의 |
| 10봉 | +4.38% | +0.85% | +3.53%p | +3.15 | 유의 |
| 20봉 | +8.44% | +1.73% | +6.70%p | +3.49 | 유의 |

threshold를 0.85/0.90/0.95로 올릴수록 명목 edge는 유지·확대되나(h=20 기준
+3.8~+8.7%p) 표본이 줄며(153→58) FDR 유의성은 사라진다 — **방향은 일관, 통계적
확정은 threshold=0.70 수준까지만**.

### 4.1 분할 안정성 (train_60/val_20/test_20, first_half/second_half)

| 구간 | h=5 edge (t) | h=10 edge (t) | h=20 edge (t) |
| --- | --- | --- | --- |
| first_half (2017-08~2022-01) | +2.04%p (+2.85) | +3.50%p (+3.04) | +6.63%p (+3.32) |
| second_half (2022-01~2026-07) | +0.95%p (+1.04) | +2.37%p (+1.28) | +5.00%p (+1.62) |
| train_60 | +2.00%p (+2.81) | +3.51%p (+3.08) | +6.33%p (+3.21) |
| val_20 | +1.46%p (+0.68) | +3.85%p (+0.76) | +8.76%p (+1.27) |
| test_20 (2024-09~2026-07) | +0.78%p (+1.08) | +1.46%p (+1.10) | +3.21%p (+1.37) |

**5개 구간 × 3 horizon = 15/15 전부 edge 부호가 양(+)** — 방향은 완전히
일관된다. 단 val/test는 표본 축소(n≈58~59)로 |t|<1.65 — **개별 holdout
구간의 통계적 유의성은 미확정**, full-period·train 구간에서만 확정.

## 5. 백테스트 파이프라인 (`hole_profile.py`)

```
state(t) = compute_hole_profile_state(df, atr_len=48, atr_mult=1.0, level="L1")
entry: 무포지션 ∧ profile_rank(t) >= entry_percentile(0.60) ∧ direction(t)≠0
  → 다음봉 시가 진입(direction 그대로, long_only=false면 양방향)
청산 우선순위(봉 순회):
  1. stop_atr — 손절: entry_price ∓ stop_atr_mult(2.0) × entry 시점 ATR (인트라바)
  2. take_profit_atr — 익절: entry_price ± take_profit_atr_mult(2.0) × entry
     시점 ATR (인트라바, 고정 R배수 — Pine 원본엔 없는 확장)
  3. structure_confirmed — 진입 근거였던 피벗이 다음 확정 피벗으로 갱신되면
     (그 레벨 스윙이 공식 종료) 다음봉 시가 청산
  4. percentile_fade — profile_rank가 exit_percentile(0.50) 밑으로 후퇴하면
     다음봉 시가 청산(모멘텀 소진)
  5. horizon — max_hold_bars(60) 시간청산
비용: 왕복 20bp(수수료+슬리피지 각 5bp, heatmap1과 동일 관례)
```

entry/exit_percentile은 §6.1 청산 규칙 스윕 결과로 갱신된 값(최초 등록 시
0.70/0.30 → 스윕 후 0.60/0.50). stop_atr_mult/take_profit_atr_mult는 §6.2
리스크 스윕 결과로 갱신된 값(최초 등록 시 stop=3.0·익절 없음 → 스윕 후
stop=2.0/take_profit=2.0). max_hold_bars(60)는 등록 시점 값을 유지 — 두 스윕
모두에서 유의미한 영향이 없었다.

### 5.1 대시보드 결과창

- **매수/매도 지점**: `build_price_figure`(공통, 전략명 무관)가 트레이드
  계약(`direction`/`entry_time`/`exit_time`)만으로 자동 표시 — 롱 진입(초록
  삼각형)/숏 진입(주황 삼각형)/청산(빨강 x). hole_profile 전용 코드 없음.
- **손절/익절 라인**: `build_price_figure`에 범용 추가(`_trade_level_segments`)
  — 거래마다 `stop_loss_price`/`take_profit_price`가 있으면 진입~청산 구간에만
  수평선을 긋고, 거래 사이는 `None`으로 끊어 **포지션 정리 이후로는 이어지지
  않는다**. §6.2에서 `take_profit_atr_mult`를 도입한 뒤로는 hole_profile도
  두 선을 모두 채운다(도입 전에는 stop_loss_price만 있었음). 다른 전략이 같은
  두 컬럼을 채우면 동일 코드가 자동으로 그린다 — 전략별 분기 없음.
- **두 프로파일**(Pine 원본의 "Time Bin"/"Price % Bin" 테이블 재현): 파이프라인이
  `duration_and_pct_move_profiles()`로 선택 레벨(L1)의 완결 스윙 지속시간(봉)·
  가격변동%(부호 보존) 히스토그램을 `structure.profile_bins`(기본 40, 최초
  등록 시 20에서 세분화) 구간으로 만들어 `extras.duration_profile`/
  `extras.pct_move_profile`로 노출한다. `config.dashboard.panels`에
  `type: "bar"`로 선언돼 있어 결과창 "보조 패널"에 막대그래프로 자동 렌더된다
  (범용 패널 메커니즘 — app.py/presentation.py 추가 수정 없음). **전체 히스토리
  누적**(Pine의 `barstate.islast` 표와 동일하게 phase 윈도로 자르지 않음) — 특정
  phase 구간만 보고 싶다면 별도 표시가 아직 없다.

## 6. 실측 백테스트 결과 (BTC/USDT 일봉, 수수료+슬리피지 왕복 10bp)

### 6.0 최초 등록 config (entry=0.70/exit=0.30/stop=3.0×ATR/hold=60봉) — 폐기, 비교용

| phase | 봉수 | 거래수 | 전략 수익 | Sharpe | MDD | CAGR | B&H 수익 | B&H Sharpe | B&H MDD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all (2017-08~) | 3243 | 31 | +223.1% | 0.527 | −33.4% | 14.12% | +1347.1% | 0.654 | −83.2% |
| validation | 649 | 8 | +0.1% | 0.086 | −27.3% | 0.05% | +257.1% | 1.431 | −26.2% |
| test (holdout) | 649 | 5 | +8.1% | 0.275 | −17.4% | 4.46% | −2.1% | 0.160 | −53.0% |

거래 8~31건의 소표본이라 청산 규칙(3× ATR 손절/0.30 percentile_fade/60봉
시간청산)이 §4의 신호 edge를 제대로 살리는지 의문이 있었다 — §6.1에서 스윕.

### 6.1 청산 규칙 파라미터 스윕 (`hole_profile_exit_sweep.py`, 2026-07-08)

`entry_percentile × exit_percentile × stop_atr_mult × max_hold_bars` 4축
그리드(4×6×7×6=1008조합, `compute_hole_profile_state`는 1회만 계산해 재사용).
**과최적화 방지**: all-period Sharpe로 바로 고르지 않고 **train_60 구간
Sharpe**(train 거래수 ≥15 필터)로 후보를 추리고 validation/test는 검증에만
사용했다.

**핵심 발견 — stop_atr_mult·max_hold_bars는 사실상 죽은 파라미터였다.**
최초 config(entry=0.70) 거래로그를 보면 31건 중 28건이 `structure_confirmed`
(진입 근거 피벗이 갱신되는 구조적 청산)로 종료되고 `percentile_fade`/`horizon`은
**단 한 번도 발동하지 않았다** — stop_atr_mult∈{1.5~6.0, 사실상 무한대},
max_hold_bars∈{0(비활성)~120} 전 조합에서 train/val/test/all 성과가 거의
동일했다(구조적 확정이 그 전에 항상 선점). 실제로 유의미한 두 축은
`entry_percentile`과 `exit_percentile`이었다:

| entry | exit | train Sharpe(n) | val Sharpe(n) | test Sharpe(n) | all Sharpe | all MDD |
| --- | --- | --- | --- | --- | --- | --- |
| 0.70(기존) | 0.30(기존) | 0.695(18) | 0.086(8) | 0.275(5) | 0.527 | −33.4% |
| 0.70 | −(비활성)~0.5 | 0.632(19) | ≈0.00(8) | 0.263(5) | ≈0.47 | −38% |
| **0.60(채택)** | **0.50(채택)** | **0.707(24)** | **0.785(10)** | **0.232(8)** | **0.640** | **−40.7%** |
| 0.80 | any | 0.043(17) | 0.056(5) | −0.485(3) | ≈−0.01 | −45.4% |
| 0.90 | any | −0.114(16) | n=1 | n=1 | ≈−0.06 | −51.9% |

- entry_percentile=0.70(기존)은 **validation Sharpe가 exit_percentile 값과
  무관하게 거의 0**(−0.05~0.09) — 구조적으로 취약한 지점이었다.
- entry_percentile=0.60으로 낮추면 거래수가 늘고(18→24 train, 8→10 val)
  validation Sharpe가 0.09→0.79로 크게 개선, train/all도 개선. §4의 원시
  신호 재검증(threshold=0.60, n=457)도 h=5/10/20 전부 NW t=+2.9~+3.2로
  독립적으로 유의 — entry=0.60 채택이 백테스트 우연이 아니라 신호 자체가
  뒷받침하는 결정임을 확인했다.
- entry_percentile≥0.80은 거래수가 급감(train n≤17)하며 성과가 완전히
  무너진다 — 진입 문턱을 더 좁히는 방향은 기각.
- exit_percentile은 entry=0.60 기준 0.1~0.5 전 구간에서 성과가 비슷했으나
  **0.50이 train/val Sharpe와 all MDD(−40.7% vs 다른 값들의 −51~−52%)
  전부에서 가장 우수**해 채택했다.

**§6.1 시점 config (entry=0.60/exit=0.50/stop=3.0×ATR/익절 없음/hold=60봉)**:

| phase | 봉수 | 거래수 | 전략 수익 | Sharpe | MDD | CAGR | B&H 수익 | B&H Sharpe | B&H MDD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all (2017-08~) | 3243 | 42 | +410.4% | 0.640 | −40.7% | 20.16% | +1347.1% | 0.654 | −83.2% |
| train (60%) | ~1946 | 24 | +235.6% | 0.707 | −40.7% | 25.55% | — | — | — |
| validation | 649 | 10 | +42.5% | 0.785 | −22.2% | 22.10% | +257.1% | 1.431 | −26.2% |
| test (holdout) | 649 | 8 | +7.1% | 0.232 | −20.3% | 3.95% | −2.1% | 0.160 | −53.0% |

**정직한 결론(§6.1 시점)**: 청산 규칙 스윕으로 all-period Sharpe 0.527→0.640
(거의 B&H의 0.654에 근접), CAGR 14.1%→20.2%, validation Sharpe 0.086→0.785로
개선했다. 대가는 all MDD 악화(−33.4%→−40.7%, 다만 B&H의 −83.2%보다는 여전히
훨씬 얕음)와 test Sharpe 소폭 하락(0.275→0.232, 절대수익은 +8.1%→+7.1%로
거의 동일)이다. **개선의 실체는 "청산 규칙 튜닝"이 아니라 "진입 문턱을
0.70→0.60으로 낮춰 거래수를 늘린 것"** — 이 시점까지는 stop_atr_mult/
max_hold_bars가 이 백테스트 구간에서 사실상 영향이 없었다(§6.2에서 뒤집힘).

### 6.2 리스크 파라미터(stop/take_profit) 스윕 (`hole_profile_risk_sweep.py`, 2026-07-08)

사용자 피드백("익절/손절을 더 보수적으로") 반영. §6.1까지는 `take_profit_atr_mult`
자체가 없어 이익 실현이 전적으로 구조 확정(structure_confirmed)에 의존했다 —
실거래 예시(§ "예시" 대화)에서 $58,353 고점까지 갔다가 $46,277로 되돌린 뒤에야
청산된 사례처럼, 큰 수익이 상당 부분 반납된 뒤 청산되는 경우가 있었다. 이번
스윕은 entry=0.60/exit=0.50/hold=60을 고정하고 `stop_atr_mult`(0.5~3.0)ᅟ×
`take_profit_atr_mult`(0=비활성~5.0) 64조합을 스윕했다(train Sharpe 기준 선택 +
val/test 확인, 과최적화 방지 방법론은 §6.1과 동일).

| stop×ATR | take_profit×ATR | train Sharpe(n) | val Sharpe(n) | test Sharpe(n) | all Sharpe(n) | all MDD |
| --- | --- | --- | --- | --- | --- | --- |
| 3.0(§6.1) | 없음 | 0.707(24) | 0.785(10) | 0.232(8) | 0.640(42) | −40.7% |
| **2.0(채택)** | **2.0(채택)** | **0.873(52)** | **1.066(19)** | **0.274(13)** | **0.805(84)** | **−41.8%** |
| 3.0 | 2.0 | 0.872(48) | 0.715(19) | 0.311(13) | 0.753(80) | −41.8% |
| 2.5 | 2.0 | 0.838(50) | 1.066(19) | 0.311(13) | 0.786(82) | −41.8% |
| 3.0 | 2.5 | 0.861(44) | 0.675(18) | −0.330(13) | 0.656(75) | −38.5% |

- stop=2.0/take_profit=2.0이 **train/validation/test/all 4구간 전부에서
  동시에 개선**됐다 — train Sharpe만 좋아지고 나머지가 나빠지는 전형적인
  과최적화 패턴이 아니라는 뜻. 거래수도 42→84로 거의 2배(더 짧은 보유기간
  + 더 이른 확정 청산 덕에 회전율이 늘어남).
- 청산 사유 분포가 건전해졌다: `take_profit_atr` 36건(43%, 평균 +8.6%),
  `structure_confirmed` 31건(37%, 평균 −1.1% — 큰 승자 다수가 이제
  take_profit에서 먼저 잡힘), `stop_atr` 11건(13%, 평균 −9.3%),
  `percentile_fade` 6건(7%, 평균 −6.2%). §6.1 시점엔 `stop_atr`/`take_profit_atr`
  가 각각 0건에 가까웠던 것과 대조적 — 이제 손절/익절이 실제로 "발동하는"
  파라미터가 됐다.
- stop을 3.0보다 더 조여도(1.5·1.0·0.75·0.5) train Sharpe는 계속 낮아졌다
  (손절이 너무 taut해 정상 변동에도 잘리는 whipsaw) — 2.0이 하한에 가까운
  실질적 최적점.
- take_profit을 1.0~1.5로 더 타이트하게 잡으면 거래수는 늘지만(최대 126건)
  train Sharpe는 오히려 하락(0.66~0.75) — 너무 빨리 먹고 나가 큰 스윙의
  upside를 놓친다. 2.0~2.5가 상한에 가까운 균형점.

**최종 채택 config (entry=0.60/exit=0.50/stop=2.0×ATR/take_profit=2.0×ATR/hold=60봉)**:

| phase | 거래수 | 전략 수익 | Sharpe | MDD | CAGR |
| --- | --- | --- | --- | --- | --- |
| all (2017-08~) | 84 | (미집계, all Sharpe/CAGR만 §6.2 표 참조) | 0.805 | −41.8% | 24.9% |
| train (60%) | 52 | — | 0.873 | — | — |
| validation | 19 | — | 1.066 | — | — |
| test (holdout) | 13 | — | 0.274 | — | — |

vs §6.1: all Sharpe 0.640→0.805, CAGR 20.2%→24.9%, MDD −40.7%→−41.8%(거의
동일), 거래수 42→84. **"보수적으로"의 실제 효과**: 손절을 3.0→2.0으로 조여
꼬리위험을 줄이고, 없던 익절(2.0×ATR)을 신설해 큰 수익이 구조 확정을 기다리며
반납되는 경우를 줄였다 — 그 결과 위험조정 성과가 전 구간에서 함께 개선됐다.

## 7. 주의 / 무효화 조건

- **표본 단일 자산·단일 레짐**: BTC 강세 드리프트(2017~2026)가 baseline
  자체를 끌어올린 상태에서의 edge다. 하락장/횡보장 별도 표본, 타 자산(주식
  지수 등)에서 방향이 유지되는지 미검증 — 등록 시점 기준 **BTC 일봉 단일
  검증**.
- val/test 개별 구간은 |t|<1.65로 통계적 미확정(§4.1) — "방향성 tilt"이지
  "확정 알파"가 아니다. entry_percentile=0.60 채택은 원시신호 재검증(§6.1,
  n=457, NW t=+2.9~+3.2)으로 뒷받침되나, 이 역시 전체 구간(all) 기준이고
  val/test 개별 구간 재확인은 아니다.
- L0/L2는 명시적으로 신호에 부적합함을 확인했다(§4) — 다른 레벨로 재시도
  시 반드시 스터디 재실행.
- **청산 규칙 스윕(§6.1)과 리스크 파라미터 스윕(§6.2) 모두 train Sharpe 기준
  선택 + val/test 확인이라는 절차를 거쳤지만 여전히 같은 단일 자산·단일
  히스토리 안에서의 재사용이다** — 진짜 out-of-sample 검증(신규 데이터,
  타 자산)이 아니면 "그럴듯한 재적합" 위험은 남는다. §6.2는 4구간 동시 개선이라
  §6.1보다 과최적화 신호가 약하지만, 파라미터 2개(stop/take_profit)를 같은
  히스토리에서 추가로 탐색한 만큼 자유도가 늘어난 것도 사실이다.
- ATR 방향전환 파라미터(`atr_len=48`, `atr_mult=1.0`)는 Pine 원본 기본값을
  그대로 썼다 — 스윕/재튜닝 안 함(플랜 고정값). 재튜닝 시 스터디 재실행 필수.
- **무효화**: 신규 데이터 누적 후 스터디 재실행에서 L1 게이트 FDR 유의성이
  사라지거나, 페이퍼/타자산 검증에서 방향 일관성이 깨지면 폐기.
- `live_eligible=False` — 페이퍼 검증·타 자산 이식 검증 전 승격 금지.

## 8. 이력

- 2026-07-08: `hole_profile.txt`(Pine) 리뷰 → 방향 가설(돌파/지속 vs
  소진/반전) 및 검증 자산(BTC 일봉) 사용자 확인 → L0/L1/L2 이식 + 프로파일
  백분위 스터디 → L1·threshold=0.70만 채택 → 전략 등록.
- 2026-07-08: 청산 규칙 파라미터 스윕(§6.1) — stop_atr_mult/max_hold_bars는
  구조적 청산이 선점해 사실상 무의미함을 확인, entry_percentile 0.70→0.60·
  exit_percentile 0.30→0.50으로 갱신(train Sharpe 기준 선택 + val/test 확인,
  원시신호 재검증으로 뒷받침). config/문서 갱신(이 문서).
- 2026-07-08: 대시보드 결과창 보강(§5.1) — 손절/익절선을 `build_price_figure`에
  범용 추가(거래별로 끊어 그림), 원본 Pine의 두 프로파일(Time Bin/Price % Bin)을
  `extras`+`dashboard.panels`(bar)로 노출. 매수/매도 마커는 기존 공통 로직이
  이미 지원 확인.
- 2026-07-08: 사용자 피드백("프로파일 bin 더 촘촘히, 손절/익절 보수적으로")
  반영 — `structure.profile_bins` 20→40. `take_profit_atr_mult` 신설(§6.2
  리스크 스윕) + `stop_atr_mult` 3.0→2.0으로 조여 4구간(train/val/test/all)
  동시 개선(all Sharpe 0.640→0.805, CAGR 20.2%→24.9%, 거래수 42→84). 새
  exit_reason(`stop_atr`/`take_profit_atr`/`structure_confirmed`/
  `percentile_fade`)을 `EXIT_REASON_LABELS`에 뒤늦게 등록(누락 발견·수정).
