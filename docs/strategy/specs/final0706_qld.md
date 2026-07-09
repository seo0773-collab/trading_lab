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

> **[정정 2026-07-07]** 이 문서의 CAGR류 절대 수치는 봉 수 기준 연환산 버그 +
> .KS 데이터 갭(2008년 0봉)으로 과대표기됐다(진단: `final0706_data_gap_diagnosis.md`).
> 달력 기준·권위 구간(2011-01~) 정정치: **final0706_qld CAGR 22.51%/MDD -30.06%/
> Sharpe 1.132** (vs final0706 원본 14.39%/-16.40%/1.176). 스윕의 비중-MDD 관계와
> 부트스트랩 상대 비교는 같은 인덱스 기반이라 유효하다.

## 2. 배경

final0706은 CAGR 16.84%/MDD -16.40%/Sharpe 1.242(원공표, 위 정정 참조)로 위험조정 성과는 우수하지만
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

**2026-07-09**: 이 config는 `core_strategy` 참조 구조라 자체 `safe_sleeve` 설정이
없고 런타임에 `configs/strategies/final0706.json`을 그대로 읽는다. final0706
베이스에 승격된 `safe_sleeve.profile_scored=true`([[final0706-strategy]] 참조)를
이 파일 수정 없이 자동 상속한다.

## 7. QLD 오버레이의 위험게이트 추종 (`gate_follow`, 2026-07-09)

**문제 지적(사용자)**: `final0706_safe_scored`(→ base 승격)는 안전자산 슬리브가
추세 게이트 ON 시 방어 자산으로 전환되는데, `final0706_qld`의 QLD 36.5%
오버레이는 core의 신용스프레드 위험게이트와 무관하게 **항상 고정 비중**이었다
— core가 노출 0.5배로 줄이고 IEF로 대피해도 QLD 슬리브는 그대로 남아있는
구조적 공백이었다.

**변경**: `Yoon1kLeverageMixHandler`(`src/trading_lab/strategies/yoon1k_leverage_mix.py`)에
overlay별 `gate_follow: true` 옵션을 추가했다. core가 이미 계산해 자기
`forecast["target_w___CASH__"]`에 노출하는 신호(게이트 발동 시 `1-exposure`,
final0706 기준 발동 시 0.5)를 그대로 재사용해 QLD 목표비중도 동일 비율로
줄이고, 줄인 만큼을 core와 같은 안전자산(`risk_gate.safe_asset`, 여기선
IEF)으로 돌린다. 게이트 로직 자체는 재구현하지 않았다(신호만 재사용) — core가
없거나 신호가 없는 다른 leverage_mix 전략(`yoon1k_levmix`/`yoon1a`/`yoon2a`)은
`gate_follow`를 켜지 않으면 기존 고정비중 동작이 그대로 보존된다.

config: `leverage_mix.overlays[0].gate_follow = true` (`configs/strategies/final0706_qld.json`).
선택 파라미터: `gate_follow_frac`(기본 1.0, 게이트 강도 대비 축소 비율),
`gate_safe_asset`(기본은 core의 `risk_gate.safe_asset`을 자동 상속).

**실측** (`scripts/profile_sizing/final0706_qld_gate_follow_validate.py`,
2007-01-29~2026-07, phase="all", 달력 기준 CAGR):

| 구성 | CAGR | MDD | Sharpe |
| --- | ---: | ---: | ---: |
| 고정비중(변경 전) | 20.09% | -30.02% | 1.152 |
| gate_follow=true(변경 후) | **20.50%** | **-28.42%** | **1.198** |

세 지표 모두 개선(트레이드오프 없음). 페어드 부트스트랩(8000회, block=20):
P(Sharpe 개선) 87.4% · P(CAGR 개선) 72.1% · P(MDD 더 얕음) 76.7% — 방향이
일관돼 `gate_follow=true`를 기본값으로 채택했다.

> 위 20.09%/20.50%는 이 표를 만든 실측 스크립트와 동일한 phase="all"(연환산
> 버그 수정 후 달력 기준) 수치이며, 문서 상단 "권위 정정치"(2011-01~ 구간)
> CAGR 22.51%와는 표본 구간이 달라 직접 비교 불가 — gate_follow 전/후는
> **동일 구간·동일 코드 경로**로 비교했으므로 상대 비교는 유효하다.

대시보드에서는 `sleeve_weights` 패널에 `__CASH__` 레그가 자동으로 잡혀
QLD 게이트 대피 국면이 면적으로 보인다(`leg___CASH__` 라벨: "QLD 게이트
대피(IEF)").
