# 전략 수립 환경

새 전략은 "명세 작성 → config 생성 → registry 등록 → 게이트 통과" 순서로만 플랫폼에 진입합니다. holdout test 구간과 live 거래는 게이트를 통과하기 전까지 잠겨 있습니다.

## 워크플로

```bash
# 1. 스캐폴딩: config + 명세 + 체크리스트 생성
python scripts/new_strategy.py my-strategy-v1 --description "전략 한 줄 설명"

# 2. 명세 작성: docs/strategy/specs/my-strategy-v1.md 의 모든 항목 채우기
#    (STRATEGY_SPEC_TEMPLATE.md 기준, "미정" 항목이 남아 있으면 백테스트 금지)

# 3. config 조정: configs/strategies/my_strategy_v1.json

# 4. registry 등록: src/trading_lab/strategies/registry.py 에
#    스캐폴딩 출력에 표시된 StrategyDefinition 스니펫 추가
#    (enabled=True, live_eligible=False 로 시작)

# 5. 게이트 진행: docs/strategy/checklists/my-strategy-v1.md 를 위에서부터 통과
trading-lab backtest --symbol BTC-USD --phase validation
```

## 파이프라인 블럭 다이어그램 (공통 vs 커스텀)

범례: `═` 박스 = **공통 인프라**(전략을 추가해도 손대지 않는 구간), `─` 박스 = **전략별 커스텀**(전략마다 갈아 끼우는 구간).

```
                        ┌─ 전략별 커스텀 (전략마다 작성) ─────────────────┐
                        │                                                │
                        │  docs/strategy/specs/<id>.md        ← 명세     │
                        │  docs/strategy/checklists/<id>.md   ← 게이트   │
                        │  configs/strategies/<id>.json       ← 파라미터 │
                        │  registry.py 항목 1개               ← 등록     │
                        └───────────────┬────────────────────────────────┘
                                        │ strategy_id로 조회
                                        ▼
╔═══════════════════════════════════════════════════════════════════════╗
║ 진입점 (공통)                                                          ║
║   ./run_dashboard.sh → ui/app.py "새 백테스트"   trading-lab backtest  ║
║   (전략 드롭다운 = list_strategies())            (CLI, --phase)        ║
╚═══════════════════════════════╤═══════════════════════════════════════╝
                                ▼
╔═══════════════════════════════════════════════════════════════════════╗
║ BacktestService.run()  (service.py)                          (공통)    ║
║   registry 조회 → config JSON 로드 → phase 검증(test 잠금)             ║
╚═══════════════════════════════╤═══════════════════════════════════════╝
                                ▼
╔═══════════════════════════════════════════════════════════════════════╗
║ 데이터 로딩  load_market_data()                              (공통)    ║
║   yfinance / csv / synthetic → OHLCV 정제·정렬                         ║
╚═══════════════════════════════╤═══════════════════════════════════════╝
                                ▼
┌───────────────────────────────────────────────────────────────────────┐
│ 신호 생성  (전략별 커스텀 — 현재 유일한 코드 분기 지점)                │
│                                                                       │
│   현재: h72 Kalman 파이프라인으로 고정 (research_adapter.py)          │
│     compute_features → identify_params → run_filter                   │
│     → calibrate_sigma → build_signals                                 │
│                                                                       │
│   산출 계약(이것만 지키면 어떤 로직이든 교체 가능):                   │
│     direction(+1/-1), confidence, expected_edge 시리즈                │
│                                                                       │
│   ※ 신호 로직이 다른 전략 추가 시 여기에 디스패치 1회 작업 필요       │
└───────────────────────────────┬───────────────────────────────────────┘
                                ▼
╔═══════════════════════════════════════════════════════════════════════╗
║ 분할 + 실행 엔진  (strategy_execution.py)                    (공통)    ║
║   chronological_splits: identification / validation / test            ║
║   run_execution: rolling 임계값(shift 1) → next_open 체결              ║
║                  수수료 차감 → horizon/반대신호 청산 → 비중복 포지션   ║
║   ← Gate 1 테스트가 검증하는 구간 (전략 무관 재사용)                   ║
╚═══════════════════════════════╤═══════════════════════════════════════╝
                                ▼
╔═══════════════════════════════════════════════════════════════════════╗
║ 지표 + 아티팩트  (공통)                                                ║
║   summarize_execution: trades, hit_rate, avg_net_bps, sharpe, mdd …    ║
║   var/runs/{run_id}/: config·예측·거래·equity·리포트·차트              ║
║   SQLite 인덱스 (var/trading_lab.sqlite3)                              ║
╚═══════════════════════════════╤═══════════════════════════════════════╝
                                ▼
╔═══════════════════════════════════════════════════════════════════════╗
║ 소비 (공통)                                                            ║
║   대시보드 결과/비교/시스템 화면        게이트 체크리스트 증거(run_id) ║
╚═══════════════════════════════════════════════════════════════════════╝
```

- 위쪽 커스텀 블럭(명세·config·registry)은 코드가 아니라 선언입니다. `scripts/new_strategy.py`가 생성하며, 여기까지만 작성하면 공통 구간 전체가 그대로 동작합니다.
- 가운데 신호 생성 블럭이 유일하게 전략 로직이 들어가는 코드 구간입니다. 경계 계약은 "direction/confidence 시리즈를 내놓는 것"이고, 그 아래 실행 엔진부터는 다시 공통입니다. config 파라미터만 다른 전략은 이 블럭도 수정 없이 재사용합니다.
- Gate 1(실행 정확성)은 공통 실행 엔진을 검증하므로 한 번 통과하면 전략 무관하게 유효하고, Gate 2~4(성과·일반화·holdout)는 전략마다 다시 수행합니다.

## 전략을 구성하는 실제 파일

전략은 `strategy_id` 문자열 하나로 타겟되며, 해석 순서는 다음과 같습니다.

```
strategy_id ("h72-price-v1")
  → src/trading_lab/strategies/registry.py 의 _STRATEGIES 딕셔너리
    → StrategyDefinition.config_path
      → configs/strategies/h72_price_v1.json  (파라미터)
  → src/trading_lab/research_adapter.py        (scripts/ 를 sys.path에 추가)
    → 신호 생성 코드 (아래 표)
```

활성 전략 `h72-price-v1` 기준 실제 파일:

| 역할 | 파일 | 구분 |
| --- | --- | --- |
| 타겟팅(조회) | `src/trading_lab/strategies/registry.py` | 커스텀 (항목 1개 추가) |
| 파라미터 | `configs/strategies/h72_price_v1.json` | 커스텀 |
| 피처 계산 | `scripts/flat_chart.py` (`compute_features`) | 커스텀 (현 신호 로직) |
| 예측 필터 | `scripts/run_kalman_pipeline.py` (`identify_params`, `run_filter`, `calibrate_sigma`) + `scripts/kalman.py` | 커스텀 (현 신호 로직) |
| 신호 규칙 | `scripts/conf_filter_backtest.py` (`build_signals`) | 커스텀 (현 신호 로직) |
| 어댑터(연결) | `src/trading_lab/research_adapter.py` | 공통 (신호 로직 교체 시 디스패치 지점) |
| 실행 엔진 | `scripts/strategy_execution.py` (`run_execution`, `chronological_splits`, `summarize_execution`) | 공통 |
| 명세/게이트 | `docs/strategy/specs/`, `docs/strategy/checklists/` | 커스텀 |

`strategies/cycle_reversion.py`는 레거시 연구 코드로, 전체 구간 profile에 look-ahead bias가 있어 registry에 등록되어 있지 않습니다 (`docs/research/README.md` 참조).

## 파일 배치

- `DASHBOARD_GUIDE.md`: 대시보드(run_dashboard.sh) 연동 데이터 계약 — forecast/trades/equity 컬럼 규칙, 보조지표 자동 노출, config `dashboard` 블록
- `STRATEGY_SPEC_TEMPLATE.md`: 전략 명세에 반드시 들어가야 하는 요소(기준) 정의
- `STRATEGY_TEST_CHECKLIST.md`: 마스터 테스트 리스트 (Gate 0~5)
- `specs/<strategy-id>.md`: 전략별 명세 (스캐폴딩이 생성)
- `checklists/<strategy-id>.md`: 전략별 체크리스트 사본 (스캐폴딩이 생성, 진행 상황 기록용)

## 원칙

- 명세의 **무효화 조건**(어떤 결과가 나오면 전략을 폐기하는가)을 백테스트 전에 먼저 적습니다.
- holdout test 구간은 전략당 1회만 개봉하고, 개봉 후 파라미터 수정은 새 버전(`-v2`)으로 취급합니다.
- 모든 합격선 수치는 명세에 먼저 기록하고, 결과를 본 뒤 합격선을 낮추지 않습니다.
- Gate 통과 증거는 `var/runs/{run_id}/` 또는 `reports/` 의 실제 산출물 경로로 체크리스트에 남깁니다.
