# 전략 명세: yoon1g (회복 레버리지 슬리브)

> [[profile-portfolio-v1]] 엔진(섹터 방어형 [[yoon1f]])에 **회복 게이팅 레버리지
> 슬리브**를 더한 변형. "깊은 저가권 + 회복 확인"에서만 2x 섹터 ETF를 섞어 상방을
> 보강하되, 위기 바닥의 false 반등을 시장필터로 차단한다. 핸들러는
> `ProfilePortfolioHandler` 공유(전용 코드 없음, config + 게이팅 로직만).

## 1. 식별 정보

| 항목 | 값 |
| --- | --- |
| strategy_id | yoon1g |
| 베이스 | yoon1f(섹터 ETF 11종 + TLT·GLD, monthly·gain1.25·SPY200MA필터) |
| config | configs/strategies/yoon1g.json |
| 작성일 | 2026-06-21 |
| 상태 | validation (holdout·full-cycle 2007~ 실데이터 검증, 118 테스트 통과) |
| 결과 단위 | 1 run = 1 포트폴리오(가상 심볼 PORTFOLIO) |

## 2. 핵심 아이디어 — 항상 레버리지(decay 함정)가 아니라 선택적 레버리지

레버리지는 마진이든 일일리셋 ETF든 시장이 비용을 받아간다(마진=이자, 상품=decay+
증폭낙폭). 세션 내 5회 반복 확인. 그래서 **상시 레버리지가 아니라, 컨트래리언
엔진이 "깊은 저가권 + 회복 확인"을 가리키는 좁은 구간에서만** 2x를 태운다.

레버리지 슬리브(2x 섹터 ETF) 점수는 **두 조건 동시 만족** 시에만 살리고, 그 외엔
0으로 죽인다(`compute_universe` 게이팅, additive·기본 off):

1. **자기 국면이 RECOVERY** — 깊은 저가권에서 회복 램프 진입(바닥 칼잡기·횡보 decay 회피).
2. **시장필터 ON (SPY > 200MA)** — 시장 추세까지 정상일 때만. RECOVERY는 자기 국면
   기준이라 2008류 위기 바닥의 false 반등에 물릴 수 있어, 시장 추세로 이를 차단.

## 3. 유니버스·파라미터

| 항목 | 값 |
| --- | --- |
| 1x 분산 | XLK·XLF·XLV·XLE·XLI·XLP·XLY·XLU·XLB·XLRE + TLT·GLD (12) |
| 2x 슬리브 | ROM·UYG·RXL·DIG·UXI·UGE·UCC·UPW·UYM·URE (10, 1x 섹터 1:1 대응) |
| top_k | 12 |
| leverage_sleeve | `{enabled, symbols=2x, regimes=["RECOVERY"], require_market_on=true}` |
| 나머지 | yoon1f 상속(gain 1.25·SPY 200MA 필터·SMA 추세 오버레이) |

## 4. 검증 결과 (2007-02~, 주 벤치마크 SPY)

### phase=test (holdout, OOS)

| 변형 | CAGR | Sharpe | MDD |
| --- | ---: | ---: | ---: |
| 1x (12, =yoon1f) | +8.4% | 0.972 | -8.7% |
| MIX RECOVERY 게이팅 | +9.7% | 1.019 | -9.1% |
| **yoon1g (RECOVERY+시장ON)** | **+10.0%** | **1.065** | -9.1% |
| SPY | +18.2% | 1.106 | -18.8% |

### phase=all (2008 포함)

| 변형 | CAGR | Sharpe | MDD |
| --- | ---: | ---: | ---: |
| 1x (12, =yoon1f) | +7.7% | 0.812 | -17.6% |
| MIX RECOVERY 게이팅 | +8.1% | 0.774 | -22.3% |
| **yoon1g (RECOVERY+시장ON)** | +8.8% | **0.851** | **-17.6%** |
| SPY | +10.9% | 0.624 | -55.2% |

**시장필터 보강 효과**: RECOVERY 단독은 full-cycle서 1x 열위(Sharpe 0.77·MDD-22.3%)
였으나, 시장ON 동시 요구로 **Sharpe 0.851 > 1x 0.812·MDD -17.6%(=1x 동일)** 로
2008 취약성을 완치하면서 holdout 이득은 오히려 ↑(Sharpe 1.019→1.065). 연도별 2x
점유율에서 위기 바닥·약세장 헛반등만 제거(2008 11%→0%, 2022 5%→0%), 회복기
(09·16·20)는 유지.

## 5. 한계·주의

- 강세장 절대수익은 여전히 SPY에 미달(집중 리스크 없이 강세장 수익 없음 — 세션 반복 교훈).
  yoon1g의 가치는 **위험조정·방어**이며 레버리지는 회복 램프 상방만 좁게 보강.
- 2x ETF 데이터 inception(~2007)에 검증 기간이 묶임. 더 긴 사이클 표본은 부재.
- 슬리브는 시장필터(`market_filter.enabled`)가 켜져 있어야 require_market_on이 작동.

근거 리포트: reports/profile_sizing/yoon1g_recovery_leverage.md,
reports/profile_sizing/recovery_leverage_sleeve.md,
reports/profile_sizing/leveraged_etf_universe.md.
