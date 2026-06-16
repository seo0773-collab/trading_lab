# 전략 명세: profile-portfolio-v1

> 여러 종목을 병렬 평가해 **상위 K개 상승 종목을 러프하게 추종**하되, 개별 종목의
> profile-sizing 방어 로직(regime cap·DEFENSE)이 합산되어 시장 전반 하락 시 현금
> 비중이 자동으로 오르는 **다종목 포트폴리오** 전략. 근거: profile_plan.txt + 단일
> 종목 [[profile-sizing-v1]] 확장.

## 1. 식별 정보

| 항목 | 값 |
| --- | --- |
| strategy_id | profile-portfolio-v1 |
| 설명 | 상위 K개 상승종목 추종 + 개별 방어 합산(현금화) 포트폴리오 |
| 작성일 | 2026-06-16 |
| 상태 | validation (핸들러 등록·계약/단위 테스트 통과, 30종목 실데이터 확인) |
| config | configs/strategies/profile_portfolio_v1.json |
| 결과 단위 | **1 run = 1 포트폴리오**(가상 심볼 PORTFOLIO). 종목별 OHLCV를 MultiIndex wide로 모아 단일 NAV equity로 환산 |

## 2. 알고리즘

1. 유니버스 각 종목 i의 profile-sizing `final_target_weight`(percentile 저가권↑ +
   추세 floor + regime cap)를 **점수 sᵢ** 로 산출.
2. 매 리밸런스(기본 월간) 점수 sᵢ>0 종목 중 **상위 K개**(기본 10) 선택.
3. **전체 주식 노출 = mean(top-K 점수)** — 모두 강세면 노출↑(추종), 모두 약세(DEFENSE)면
   노출↓(방어). 개별 방어가 합산되어 포트폴리오 현금 비중을 자동 결정.
4. 그 노출을 점수 비례로 top-K에 배분, 나머지 현금. 레버리지·공매도 없음.
5. 리밸런스 사이에는 보유분이 가격에 따라 drift, 다음 리밸런스에서 목표로 리셋.

**상승 추종 강도 레버**: 개별 `trend_overlay.floor`(기본 0.9)가 상승추세 종목 점수를
끌어올려 노출을 높인다. floor를 낮추면 더 방어적(노출↓·MDD↓·수익↓).

## 3. 데이터·실행

| 항목 | 값 |
| --- | --- |
| 유니버스 | config `universe`(기본 섹터분산 대형주 30) |
| interval/period | 1d / max |
| 리밸런스 | `rebalance_freq` ∈ monthly(기본)/weekly/daily |
| top_k | 기본 10 (튜너블) |
| 비용 | 리밸런스 회전율 × (fee+slippage) |
| 무누수 | 종목별 cumulative profile은 과거·현재만 사용. 점수는 당일 정보로 산출, 다음 봉부터 평가 |
| 벤치마크 | 같은 유니버스 **equal-weight buy & hold**(분산 B&H) |

## 4. 합격선 (결과 확인 전 고정)

| 지표 | 합격선 |
| --- | --- |
| MDD | equal-weight B&H보다 얕을 것 |
| Sharpe | equal-weight B&H 대비 −0.1 이내 |
| CAGR | (참고) B&H 근접 — 방어로 일부 열위 허용 |

## 5. 30종목 실데이터 결과 (phase=all)

파라미터 스윕(`reports/profile_sizing/portfolio_sweep.md`, top_k×리밸런스×floor)을
**validation에서 선정 → test(holdout)에서 확인**(과적합 점검)한 결과, 검증 1위는
top_k=20·monthly·floor=1.0이며 holdout에서도 견고했다(val Sharpe−B&H +0.029 →
test −0.013). 이를 기본 config로 채택.

| 구성 | 평균 노출 | CAGR | MDD | Sharpe |
| --- | ---: | ---: | ---: | ---: |
| top_k=10, floor=0(방어형) | 53% | 9.4% | -34.4% | 0.926 |
| top_k=10, floor=0.9 | 82% | 16.0% | -40.7% | 1.041 |
| **top_k=20, floor=1.0(기본·스윕 최선)** | **79%** | **14.1%** | **-32.5%** | **1.072** |
| equal-weight B&H | 100% | 18.0% | -53.5% | 1.045 |

기본 config는 **Sharpe로 B&H를 추월(1.072 > 1.045)** 하면서 MDD는 21%p 얕다(-32.5%
vs -53.5%). 분산(20종목)이 노출 79%에서도 낙폭을 억제. CAGR은 B&H 대비 -3.9%p로
방어 비용. 합격선(MDD↓·Sharpe ≥ B&H−0.1) 충족.

핵심 패턴: (1) 분산↑(top_k=20)일수록 위험조정↑, (2) 상승 추종↑(floor=1.0)일수록 좋음,
(3) floor=0(방어형)·집중(top_k 5)은 하위권.

### 시장 레짐 필터 (SPY 200MA, off_scale 0.5) — 기본 활성

시장 지수가 장기 MA 아래면 전체 목표 노출을 절반으로 줄여 전면 약세장을 회피한다.

| 구성 | 노출 | CAGR | MDD | Sharpe | (test) MDD | (test) Sharpe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 필터 OFF | 79% | 14.1% | -32.5% | 1.072 | -28.5% | 1.190 |
| **필터 ON(기본)** | 76% | 13.9% | **-30.1%** | **1.132** | **-20.9%** | **1.230** |
| equal-weight B&H | 100% | 18.0% | -53.5% | 1.045 | -33.8% | 1.204 |

필터는 **수익을 거의 깎지 않으면서(CAGR −0.2%p) MDD를 더 줄이고 Sharpe를 올린다.**
holdout에서 효과가 특히 커서(MDD -28.5%→-20.9%) Sharpe가 B&H(1.204)도 추월(1.230).

### 생존편향 점검 (reports/profile_sizing/survivorship.md)

기본 30종목은 살아남은 대형 우량주라 결과가 낙관적일 수 있다. 닷컴·금융위기·원자재·
리테일에서 크게 부진했던 종목을 섞은 확장 유니버스(55종목 로드)로 재평가:

| 유니버스 | 전략 Sharpe | B&H Sharpe | 전략−B&H |
| --- | ---: | ---: | ---: |
| 우량주 30 | 1.132 | 1.045 | +0.087 |
| 확장 55(부진 포함) | 1.122 | 0.983 | **+0.138** |

부진 종목을 섞자 **B&H는 악화(1.045→0.983, 부진 종목을 떠안음)하지만 전략은 유지
(1.132→1.122, top-K가 부진 종목을 회피)** → 전략의 우위가 오히려 확대. 전략의 가치가
"우량주 선별 덕"이 아니라 **전략 자체**임을 시사. 단 상장폐지로 데이터가 사라진 종목
(예: WBA)은 빠져 완전한 survivorship-free는 아님(데이터 한계). 전략 MDD는 확장에서
-30%→-36%로 다소 악화하나 B&H(-53%) 대비 여전히 크게 얕다.

### 재점검 정정 (2026-06-16)

코드 재검토에서 두 가지를 바로잡았다.

1. **체결 타이밍 무누수**: 포트폴리오가 리밸런스일 종가로 점수·시장레짐을 보고 같은
   종가에 체결하던 미세 미래참조를 제거(신호는 전봉 기준, 당봉 종가 체결로 1봉 지연).
   영향은 미미했다(Sharpe 1.132→1.133) — 결과가 룩어헤드 덕이 아님을 확인.
2. **벤치마크 정정**: 기존 `buy_hold`는 사실 **매일 균등 재조정 지수**였다. 사용자가
   뜻한 "사서 묻어두기"에 맞춰 **진짜 buy & hold**(각 종목 첫 상장일 1/N 투입 후 보유,
   리밸런싱 없음)로 교체. `benchmark_ew`로 매일재조정 지수도 함께 노출.

정정 후(기본 config, 진짜 buy & hold 기준):

| | 전략 (all) | B&H (all) | 전략 (test) | B&H (test) |
| --- | ---: | ---: | ---: | ---: |
| CAGR | 13.9% | 12.3% | 17.3% | 14.8% |
| MDD | -29.0% | -53.2% | -21.5% | -27.4% |
| Sharpe | **1.133** | 0.821 | **1.301** | 0.992 |

진짜 묻어두기 기준으로는 전략이 **수익(CAGR)도 앞서고, Sharpe는 크게 앞서며(+0.31),
MDD는 절반 수준**이다. 이전에 "수익은 B&H에 못 미친다"던 결론은 벤치마크가 매일
재조정 지수(분산 효과로 수익·Sharpe가 높음)였기 때문이며, 정정 후엔 전 지표 우위.

## 6. 산출물과 등록

- [x] `scripts/profile_sizing/portfolio.py`(compute_universe·rebalance_dates·simulate_portfolio·benchmark)
- [x] `src/trading_lab/strategies/profile_portfolio.py` 핸들러(MultiIndex wide → NAV equity)
- [x] `configs/strategies/profile_portfolio_v1.json` + registry 등록
- [x] 계약 테스트 자동 포함 + 전체 테스트 통과
- [ ] top_k·리밸런스·floor 스윕 + holdout(test) 개봉으로 live 판정
