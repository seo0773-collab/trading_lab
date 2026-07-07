# 전략 명세: final0706_qld

> `final0706` 방어형 코어에 QLD(2x Nasdaq100) 레버리지 슬리브를 상위 계층에서
> 혼합한 파생 전략. 목표는 MDD를 SPY/QQQ 원화환산 buy&hold와 비슷한 수준(-30%)까지만
> 허용하고, 그 범위 안에서 CAGR을 최대화하는 것.

## 1. 식별 정보

| 항목 | 값 |
| --- | --- |
| strategy_id | final0706_qld |
| 코어 | [[final0706]] 63.5% |
| 레버리지 슬리브 | QLD(2x Nasdaq100) 36.5% |
| 리밸런싱 | 월간 |
| 기준 통화 | KRW, USD 슬리브(QLD)는 USD/KRW 환산 |
| 핸들러 | `Yoon1kLeverageMixHandler`(`src/trading_lab/strategies/yoon1k_leverage_mix.py`) — 기존 `yoon1k_levmix`/`yoon1a`/`yoon2a`와 동일 core+overlay 블렌드 패턴 재사용, 신규 코드 없음 |
| config | `configs/strategies/final0706_qld.json` |

## 2. 배경

final0706은 CAGR 16.84%/MDD -16.40%/Sharpe 1.242로 위험조정 성과는 우수하지만
절대 CAGR은 SPY(16.91%)와 비슷하고 QQQ(23.85%)·QLD(34.88%) buy&hold에는 못 미친다
([[final0706-strategy]] 참조). "MDD를 좀 더 허용하면 CAGR을 얼마나 끌어올릴 수
있는가"라는 질문에 답하기 위해, 기존에 검증된 core+QLD 오버레이 블렌드 패턴
(`yoon1k_levmix`: yoon2 60%+QLD 40%)을 final0706에 그대로 적용했다.

## 3. 스윕 방법 및 결과

`scripts/profile_sizing/final0706_qld_mix_sweep.py`: raw 데이터를 1회만 로드하고
(심볼 구성이 비중에 무관하므로) QLD 비중을 0%→100%까지 11개 그리드로 스윕,
공식 `Yoon1kLeverageMixHandler.build_artifacts`로 각 지점의 CAGR/MDD/Sharpe를 실측.

| QLD 비중 | CAGR | MDD | Sharpe |
| --- | ---: | ---: | ---: |
| 0%(final0706 원본) | 16.84% | -16.40% | 1.242 |
| 30% | 23.42% | -26.20% | 1.204 |
| **36.5%(채택)** | **24.72%** | **-30.04%** | **1.160** |
| 40% | 25.42% | -32.08% | 1.137 |
| 100%(QLD 단독) | 28.56% | -70.94% | 0.782 |

MDD -30% 상한과의 교차점을 그리드 30%/40% 사이에서 선형보간(36.5%) 후 그 지점을
다시 실제로 백테스트해 재검증했다(보간 추정 CAGR 24.71% vs 실측 24.72% — 거의 정확히
일치, 근방에서 CAGR-MDD 관계가 국소적으로 선형에 가깝다).

**최종 채택**: core_weight=0.635, QLD weight=0.365.

## 4. 벤치마크 대비 부트스트랩 (8000회, block=20, 페어드 무빙블록)

| vs | P(Sharpe 개선) | P(CAGR 개선) | P(MDD 더 얕음) |
| --- | ---: | ---: | ---: |
| SPY 원화 buy&hold | 99.7% | 99.8% | 41.6% |
| QQQ 원화 buy&hold | 96.9% | 75.6% | 50.5% |
| QLD 원화 buy&hold(단독) | 99.7% | 4.3% | 100.0% |
| final0706(원본) | 32.0% | 99.3% | 0.5%(당연히 MDD는 커짐) |

**해석**: SPY·QQQ와는 비슷한 MDD 수준에서 CAGR·Sharpe 모두 확실히 우위. QLD
단독과는 Sharpe·MDD는 크게 이기지만 절대 CAGR은 이기지 못한다(QLD 단독의 34.88%가
여전히 더 높음) — 이 전략의 정체성은 "QLD 단독보다 낫다"가 아니라 **"SPY/QQQ와
동일한 낙폭 수준에서 그 둘보다 CAGR·Sharpe가 우위인 지점을 찾은 것"**이다.

## 5. 알려진 한계

- final0706 자체가 아직 페이퍼 전진검증 전(`live_eligible=false`)이므로 이 파생도
  같은 게이트를 그대로 상속한다.
- QLD 40% 이상 구간은 MDD가 -32%를 넘어가며 Sharpe가 빠르게 하락(100%: Sharpe
  0.782)한다 — QLD 비중이 과도해지면 레버리지 ETF 고유의 리스크(변동성 끌림·환전
  비용·세금)가 부각될 수 있음. 이 스크립트는 세금·QLD 자체 운용보수·환전 스프레드를
  별도 반영하지 않는다([[final0706-strategy]]·[[yoon1k_leverage_mix]] 계열과 동일 한계).
- 36.5%라는 비중은 그리드 스윕 + 선형보간으로 구한 근사치이며, 별도의 워크포워드나
  롤링 안정성 검증은 아직 수행하지 않았다(1차 등록 — 후속 검증 대상).
- QLD는 2006년 상장이라 그 이전 표본은 없다(final0706 자체 표본 시작 2007-01-29와
  거의 일치해 실질적 제약은 적음).

## 6. 승격 정책

`final0706`이 승격(페이퍼 3개월 + KIS 실수수료 확정)되기 전까지는
`live_eligible=false` 유지. 승격 이후에도 이 파생은 워크포워드/롤링 안정성 검증을
별도로 거친 뒤 검토한다.
