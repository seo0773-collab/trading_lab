# 전략 명세: {STRATEGY_ID}

> 모든 절이 채워지기 전에는 백테스트를 시작하지 않습니다. 결정하지 못한 항목은 `미정`으로 표시하고, `미정`이 하나라도 남아 있으면 Gate 0을 시작할 수 없습니다.

## 1. 식별 정보

| 항목 | 값 |
| --- | --- |
| strategy_id | {STRATEGY_ID} |
| version | 1 |
| 설명 | {DESCRIPTION} |
| 작성일 | {DATE} |
| 상태 | draft → validation → holdout → live-candidate 중 하나 |
| config 경로 | configs/strategies/{CONFIG_FILE} |

## 2. 가설과 엣지

- **시장 가설**: 어떤 비효율/구조를 이용하는가? (한 문단, 지표 이름이 아니라 시장 현상으로 서술)
- **엣지의 원천**: 왜 이 엣지가 아직 남아 있는가? (위험 프리미엄 / 행동 편향 / 구조적 제약 등)
- **무효화 조건**: 어떤 결과가 나오면 이 가설을 폐기하는가? *결과를 보기 전에 작성.*
  - 예: validation에서 avg_net_bps ≤ 0, 또는 placebo 신호와 성과 차이 없음
- **선행 연구/참조**: 관련 문서, 기존 전략과의 차이점

## 3. 유니버스와 데이터

| 항목 | 값 |
| --- | --- |
| 대상 심볼 (개발용) | 예: BTC-USD |
| 일반화 자산 세트 | configs/assets/generalization.csv 또는 별도 목록 |
| asset class 커버리지 | 예: crypto 4 / etf 3 / fx 1 |
| interval | 예: 1h |
| period | 예: 720d |
| 데이터 소스 | yfinance / csv / synthetic |
| 최소 bar 수 | 예: forecast 기준 ≥ 4,000 (validation 구간 거래 수 확보 근거 포함) |
| 데이터 품질 기준 | 중복 인덱스 0, OHLC 결측 행 제거, 시간순 정렬 |

## 4. 신호 정의

- **피처**: 사용하는 입력 (예: m_fast, m_slow, mult_close, q_scale …)
- **방향 규칙**: long/short 신호가 정해지는 정확한 조건
- **확신도(confidence)**: 정의와 임계값 방식 (예: rolling quantile 0.85, window 2000, 반드시 signal bar 이전 데이터만 사용)
- **파라미터 목록과 탐색 범위**: 각 파라미터의 기본값, 탐색 범위, 탐색을 identification 구간에서만 수행한다는 명시

| 파라미터 | 기본값 | 탐색 범위 | 비고 |
| --- | --- | --- | --- |
| horizon | 72 | 고정/24–168 | |

## 5. 실행 규칙

| 항목 | 값 |
| --- | --- |
| 체결 방식 | next_open (신호 다음 bar 시가) / close |
| 보유 기간 | horizon bar 후 청산 / 조건 청산 |
| 반대 신호 처리 | exit_on_opposite true/false |
| long_only | true/false |
| 포지션 크기 | 예: 100% 단일 포지션, 중첩 금지 |
| 손절/익절 | 사용 여부와 규칙 (미사용이면 명시) |

## 6. 비용 모델

| 항목 | 값 |
| --- | --- |
| 수수료 | fee_bps_per_side (예: 10bp) |
| 슬리피지 가정 | 예: next_open 체결로 갈음 / 추가 N bp |
| 비용 민감도 | 수수료 2배에서도 avg_net_bps > 0 이어야 하는가? |

## 7. 리스크 한도

| 항목 | 한도 |
| --- | --- |
| max_drawdown 허용치 | 예: -25% 초과 시 전략 재검토 |
| 단일 자산 비중 | 예: 100% (단일 자산 전략) |
| 거래 빈도 상한 | 예: 자산당 연 N회 이하/이상 |
| kill switch 조건 | live 가정 시 중단 조건 (현재 live 차단 상태여도 기록) |

## 8. 검증 설계

| 항목 | 값 |
| --- | --- |
| 분할 | identification {IDENT_FRAC} / validation {VALID_FRAC} / test 나머지 (시간순) |
| 파라미터 탐색 허용 구간 | identification만 |
| validation 사용 규칙 | 합격/불합격 판정만, 탐색 금지 |
| holdout(test) 개봉 규칙 | Gate 3 통과 후 1회, 개봉 후 수정 시 새 버전 |
| 컨트롤 실험 | random/placebo 신호, 파라미터 섭동(±20%), leave-one-asset-out, bootstrap CI |

## 9. 합격선 (결과 확인 전에 고정)

`summarize_execution` 지표 기준. 빈칸 없이 작성합니다.

| 지표 | validation 합격선 | holdout 합격선 |
| --- | --- | --- |
| trades (자산당 최소) | 예: ≥ 10 | ≥ 10 |
| avg_net_bps | 예: > 0 | > 0 |
| hit_rate | 예: ≥ 0.45 | |
| sharpe | 예: > 0 | |
| max_drawdown | 예: ≥ -0.30 | |
| 자산군 간 일관성 | 예: asset class 과반에서 avg_net_bps > 0 | |
| placebo 대비 | 예: placebo 평균보다 유의하게 우수 (bootstrap 95% CI) | |

## 10. 산출물과 등록

- [ ] `configs/strategies/{CONFIG_FILE}` 작성 (필수 키: `scripts/new_strategy.py`의 REQUIRED_CONFIG_KEYS 참조)
- [ ] `src/trading_lab/strategies/registry.py` 등록 (`enabled=True`, `live_eligible=False`)
- [ ] 체크리스트 `docs/strategy/checklists/{STRATEGY_ID}.md` 생성
- [ ] 게이트 증거를 `var/runs/` run_id 또는 `reports/` 경로로 기록
