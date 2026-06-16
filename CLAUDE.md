# Trading Lab — 작업 규약 (필독)

이 파일은 모든 AI 세션에 자동 로드됩니다. **전략을 추가하거나 수정하기 전에 반드시 읽고 따르세요.**
이 프로젝트의 백테스트는 `./run_dashboard.sh` 한 줄로 동작하는 **공통 파이프라인**을 중심으로 설계되어 있습니다. 새 전략을 추가하는 사람(또는 AI)이 이 구조를 모르고 손대면 대시보드 전체가 깨집니다. 아래 계약을 지키면 `app.py`/`presentation.py`/`service.py`를 **한 줄도 고치지 않고** 새 전략이 동작합니다.

## 0. 절대 규칙 (TL;DR)

- 전략은 **3개의 선언 + 1개의 핸들러**로만 추가한다: ① registry 항목, ② config JSON, ③ 명세/체크리스트(docs), ④ `StrategyHandler` 구현.
- 아래 "공통 인프라" 파일은 **전략 추가를 이유로 수정하지 않는다.** 특정 전략 이름·컬럼·파라미터를 이 파일들에 하드코딩하면 안 된다.
- 작업을 끝내기 전에 **반드시** 전체 테스트를 통과시킨다 (4절). 특히 `tests/test_strategy_contract.py`는 파이프라인 계약을 강제하는 회귀 테스트다.

## 1. 실행 체인 (이 구조를 훼손하지 말 것)

```
./run_dashboard.sh
  → python -m trading_lab ui            (src/trading_lab/cli.py)
  → streamlit run src/trading_lab/ui/app.py
       ├─ BacktestService.run()         (service.py — 공통: 데이터→핸들러→아티팩트→DB)
       │    └─ get_handler(strategy_id).load_data / build_artifacts   ← 전략별 코드 (유일한 분기점)
       └─ render_run_result()           (app.py — 공통: 아티팩트만 읽어 시각화)
```

CLI도 같은 서비스를 호출한다: `PYTHONPATH=src python -m trading_lab backtest --strategy <id> --symbol SYNTH --chart-type random --synthetic --phase validation`.

### 공통 인프라 — 전략 추가 시 수정 금지

| 파일 | 역할 |
| --- | --- |
| `run_dashboard.sh`, `src/trading_lab/cli.py` | 진입점 |
| `src/trading_lab/service.py` (`BacktestService`) | 데이터 로딩 → 핸들러 호출 → 아티팩트 기록 → run_name/DB |
| `src/trading_lab/artifacts.py` | `var/runs/<run_name>/` 파일 기록 |
| `src/trading_lab/ui/app.py`, `ui/presentation.py` | 결과/비교/시스템 화면 (아티팩트만 소비) |
| `src/trading_lab/strategies/base.py` | `StrategyHandler` 프로토콜 + `StrategyArtifacts` 계약 |

화면에서 부족한 표현이 있으면 전략 전용 페이지·차트 함수를 추가하지 말고, `presentation.py`의 **범용** 함수를 확장한다(특정 전략에 결합 금지).

### 전략별 커스텀 — 새 전략은 여기만 작성

| 항목 | 위치 |
| --- | --- |
| 등록 1줄 | `src/trading_lab/strategies/registry.py` 의 `_STRATEGIES` |
| 파라미터 | `configs/strategies/<id>.json` |
| 핸들러 구현 | `src/trading_lab/strategies/<id>.py` (`StrategyHandler` 구현체) |
| 명세 / 게이트 | `docs/strategy/specs/<id>.md`, `docs/strategy/checklists/<id>.md` |

스캐폴딩: `python scripts/new_strategy.py <id>-v1 --description "..."`.

## 2. 핸들러 계약 (`StrategyHandler` → `StrategyArtifacts`)

새 전략은 `src/trading_lab/strategies/base.py`의 `StrategyHandler`를 구현하고 registry의 `handler_factory`("module:Callable")로 연결한다. 두 메서드를 제공한다.

- `load_data(symbol, config, *, csv_path=None, synthetic=False) -> DataFrame`
  - `config["interval"]`(TF), `config["period"]`(기간)을 사용해 OHLCV를 로드한다. **대시보드의 TF/기간 위젯은 이 두 config 키로 흘러들어온다** — 별도 인자를 만들지 말 것.
  - `synthetic=True`면 재현 가능한 합성 데이터를 반환해야 한다 (계약 테스트가 이 경로를 사용).
- `build_artifacts(raw, config, *, symbol, phase, bars_per_year) -> StrategyArtifacts`

반환하는 `StrategyArtifacts`가 지켜야 할 스키마(전체는 `docs/strategy/DASHBOARD_GUIDE.md`):

- `forecast`: DatetimeIndex 프레임, 최소 `close` 포함. OHLC 외 숫자 컬럼은 보조지표로 **자동 노출**된다(전략별 차트 코드 작성 금지 — 컬럼만 추가).
- `trades`: `direction`(±1), `entry_time`, `entry_price`, `exit_time`, `exit_price`, `net_return`(소수), `exit_reason`. 선택: `stop_loss_price`, `take_profit_price`, `entry_reason`. 새 `exit_reason` 라벨은 `presentation.py`의 `EXIT_REASON_LABELS`에 추가.
- `equity`: 1.0 기준 정규화 누적 성장 시리즈 (금액 환산은 대시보드 책임).
- `metrics`: 최소 `trades`, `hit_rate`, `total_return`, `sharpe`, `max_drawdown`.

## 3. 결과 저장 네이밍 (변경 금지)

`BacktestService`가 실행마다 `var/runs/<run_name>/`에 아티팩트를 남긴다. `run_name` 규칙(`service.py:_run_name`):

```
{몇번째}_{전략이름}_{타입}_{세부타입}_{YYDDHH}
  타입 = 주식 | 크립토 | 합성     (CHART_TYPE_LABELS)
  세부타입 = 종목명(정제됨)        예: 12_di-kalman-mw-v1_크립토_BTC_261214
```

이 형식은 `tests/test_platform.py`와 `tests/test_strategy_contract.py`가 강제한다. 바꾸려면 두 테스트를 함께 갱신할 것.

## 4. 마무리 전 검증 (필수)

```bash
PYTHONPATH=src python -m unittest discover tests -q
```

- `tests/test_strategy_contract.py` — 등록된 **모든** 전략을 합성 데이터로 파이프라인에 통과시켜 위 계약(아티팩트·run_name·StrategyArtifacts 스키마)을 검증한다. 전략을 추가하면 이 테스트가 자동으로 새 전략을 포함한다.
- `tests/test_platform.py` — 서비스/스토리지/네이밍 회귀.
- `tests/test_dashboard_presentation.py` — 표시 계약 회귀.

빠른 단건 확인: 위 1절의 CLI `backtest --synthetic` 후 `status=succeeded`와 forecast/trades/equity/metrics 아티팩트 존재 확인.

## 5. 참고 문서

- `docs/strategy/DASHBOARD_GUIDE.md` — 대시보드 데이터 계약 (forecast/trades/equity 컬럼, 보조지표 자동 노출, config `dashboard` 블록). **권위 있는 최신 문서.**
- `docs/strategy/README.md` — 전략 등록 워크플로와 게이트(Gate 0~5). 단, 신호 생성을 `research_adapter.py` 단일 디스패치로 설명한 부분은 **구식**이다. 현재 확장 지점은 위 2절의 `StrategyHandler`다.
