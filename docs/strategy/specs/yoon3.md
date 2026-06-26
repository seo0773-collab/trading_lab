# 전략 명세: yoon3 (칼만 히스토그램 누적프로파일 모멘텀 게이트)

> [[profile-portfolio-v1]] 엔진([[yoon1b]] 수익 우선형)에 **칼만 히스토그램
> 누적프로파일 모멘텀 게이트**를 곱한 변형. macd_raw.txt(Pine v6)의 kalHist를
> 값의 누적분포 백분위로 자기적응 정규화하고, [g_min,1.0] 게이트로 yoon1b
> 종목 점수에 곱한다(블렌드: 저가권 × 모멘텀). 핸들러는 `ProfilePortfolioHandler`
> 공유(전용 코드 없음, config + 게이트 로직만). 게이트 모듈은
> `scripts/profile_sizing/momentum.py`.

## 1. 식별 정보

| 항목 | 값 |
| --- | --- |
| strategy_id | yoon3 |
| 베이스 | yoon1b(megacap30, monthly·gain1.25·SPY200MA필터) |
| config | configs/strategies/yoon3.json |
| 작성일 | 2026-06-21 |
| 상태 | validation (실데이터 검증 진행) |
| 결과 단위 | 1 run = 1 포트폴리오(가상 심볼 PORTFOLIO) |

## 2. 핵심 아이디어 — 저가권 × 모멘텀 블렌드

yoon1b는 저가권(컨트래리언) 점수로 종목을 고르는 방어형 사이저다. 단일종목
yoon2(칼만 MACD 타이밍)의 교훈은 "이진 진입/청산은 whipsaw·best-day-miss로
자멸, 연속 노출이 답"이었다. yoon3는 그 둘을 **곱셈 게이트**로 합친다.

각 종목에 대해:

```
base_score  = yoon1b final_target_weight          (가격 cycle_multiple 프로파일 → 버킷 → 캡)
kalHist      = kalMacd − kalSignal                 (macd_raw.txt 이중 칼만 평활 모멘텀)
z            = clip(kalHist / rolling_std, ±z_clip)
mom_pct      = kalHist 누적프로파일 내 z의 하위 백분위 (0~1, 무누수)
g_mom        = g_min + (1 − g_min)·mom_pct         (momentum 방향)
yoon3_score  = base_score · g_mom
```

이후 top-K 선택·노출=mean(top-K 점수)·regime_cap·market_filter는 yoon1b 그대로.

**왜 원시 칼만 추세신호 이식(`trend_overlay.signal=kalman`)의 실패와 다른가**:
그건 `floor=1.0` 포화로 무효였다. yoon3는 히스토그램을 *자기 분포 백분위*로
정규화(종목·국면 스케일 차 제거)하고 `floor`가 아니라 **곱셈 게이트**로 넣어
포화되지 않는다.

## 3. 무누수·불변식

- 백분위는 시작~현재 봉까지의 값만 누적(과거·현재). 엔진에서 점수가 한 번 더
  shift(1)되므로 체결은 전봉 신호 기준.
- warmup/결측 구간 게이트 = 1.0(불변).
- `mom_gate.enabled=false`(다른 전략 기본) 시 점수·동작 완전 불변 — additive.

## 4. 파라미터 (config `mom_gate`)

| 키 | 기본 | 의미 |
| --- | --- | --- |
| g_min | 0.5 | 게이트 바닥(모멘텀 최저에서도 노출 절반 유지 → best-day-miss 차단) |
| fast/slow/signal_len | 12/26/9 | MACD 길이(macd_raw.txt) |
| kalman_q / kalman_r | 0.01 / 0.10 | 칼만 반응성/평활도 |
| kalman_base | MACD Line | "MACD Line" \| "Fast/Slow EMA" |
| norm_window | 252 | rolling std 정규화 창 |
| bin_count | 120 | 누적프로파일 bin 수 |
| rolling_window | 0 | 0=누적(expanding), >0=최근 N봉 롤링 분포 |
| z_clip | 4.0 | 비닝 범위 [-z_clip, z_clip] |
| direction | momentum | momentum(백분위↑→게이트↑) \| contrarian |

## 5. 검증 결과 (megacap30, 주 벤치마크 SPY)

근거 리포트: `reports/profile_sizing/yoon3_momentum_gate.md`
(`scripts/yoon3_momentum_gate_compare.py`). 규율 = **val Sharpe ≥ yoon1b**.

| phase | 변형 | 노출 | CAGR | Sharpe | MDD |
| --- | --- | ---: | ---: | ---: | ---: |
| val | yoon1b | 77% | +11.9% | 0.932 | -19.5% |
| val | 게이트 g0.5 | 65% | +10.2% | **0.957** | -15.1% |
| val | contrarian g0.5 | 66% | +8.9% | 0.816(기각) | -18.6% |
| test | yoon1b | 91% | +19.2% | **1.335** | -22.9% |
| test | 게이트 g0.5 | 80% | +15.8% | 1.307 | **-18.6%** |
| all | yoon1b | 85% | +15.7% | 1.143 | -32.0% |
| all | 게이트 g0.5 | 70% | +12.6% | 1.136 | **-25.4%** |

**판정**: 모멘텀 게이트는 val Sharpe를 소폭 올리고(g0.5 +0.025, g0.3 +0.046)
**MDD를 전 phase에서 개선**하지만, 결정적 holdout(test)에서 Sharpe는 base보다
소폭 낮다(1.307 < 1.335). 즉 yoon3는 yoon1b를 위험조정으로 **이기지 못하며**,
수익을 내주고 낙폭을 줄이는 **방어 다이얼**(yoon1c 성격)이다. **방향(momentum
vs contrarian)은 확정적**: contrarian은 val Sharpe −0.115로 기각 — 블렌드는
"저가권 × *상승* 모멘텀"이 옳다. g_min↓일수록 더 방어적(노출↓·MDD↓·수익↓).
config 기본 g_min=0.5는 중간값.

## 6. 한계·주의

- 칼만 신호 이식은 본 세션에서 다회 실패 이력 → 회의적으로 검증. base(yoon1b)를
  못 넘으면 등록만 유지하고 운영 채택하지 않는다.
- 메가캡 강세장의 절대수익은 여전히 집중 리스크 없이 못 얻는다(세션 반복 교훈).
  yoon3의 기대 가치는 "회복 진입 타이밍/노출 공백 보강"이지 알파 생성이 아니다.
