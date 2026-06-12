# Trading Lab

로컬 연구, 표준 백테스트, 실행 이력, 리포트, 대시보드를 하나의 플랫폼으로 관리합니다. 실제 주문은 브로커 중립 인터페이스만 제공하며 기본 어댑터가 모든 live 주문을 차단합니다.

## 플랫폼 실행

```bash
source .venv/bin/activate
python -m pip install -e .
trading-lab init
./run_dashboard.sh
```

CLI 예시:

```bash
trading-lab backtest --symbol BTC-USD
trading-lab backtest --symbol SYNTH --synthetic --phase all
trading-lab runs
```

각 실행은 `var/runs/{run_id}/`에 설정, 예측, 거래, equity, 지표, Markdown 리포트와 HTML 차트를 남기며 SQLite 인덱스는 `var/trading_lab.sqlite3`에 저장됩니다. 활성 전략은 `h72-price-v1` 하나이고 holdout test와 live 거래는 잠겨 있습니다.

## 전략 수립 환경

새 전략은 `docs/strategy/`의 명세 기준과 Gate 0~5 테스트 체크리스트를 따라 추가합니다.

```bash
python scripts/new_strategy.py my-strategy-v1 --description "전략 한 줄 설명"
```

config(`configs/strategies/`), 명세(`docs/strategy/specs/`), 게이트 체크리스트(`docs/strategy/checklists/`)가 생성됩니다. 워크플로, 파이프라인 블럭 다이어그램(공통 vs 커스텀 구간), 전략을 구성하는 실제 파일 목록은 `docs/strategy/README.md`를 참조하세요.

## Legacy 연구 화면

TradingView PineScript `평면차트 12`를 Python으로 옮기기 위한 로컬 연구 환경입니다. 지표 계산은 대시보드와 백테스트에서 같은 `indicators` API를 사용합니다.

## 환경 준비

Python 3.12 가상환경이 프로젝트의 `.venv`에 있다고 가정합니다.

```bash
cd ~/trading_lab
source .venv/bin/activate
python -m pip install -r requirements-core.txt
python -m pip install -r requirements-backtest.txt
```

API 키는 필요하지 않습니다. 향후 키가 필요해도 `.env.example`만 템플릿으로 사용하고 실제 `.env`는 버전 관리에서 제외하세요.

## 실행

```bash
source ./activate.sh
./run_dashboard.sh
```

직접 실행할 수도 있습니다.

```bash
source .venv/bin/activate
python -m trading_lab ui
```

## 검증

프로젝트 루트에서 실행합니다.

```bash
python scripts/test_data.py
python scripts/test_vectorbt.py
python scripts/test_backtrader.py
python -m compileall src indicators strategies dashboard scripts tests
```

`test_data.py`는 Yahoo Finance 네트워크 접근이 필요한 smoke test입니다. vectorbt와 Backtrader 테스트는 재현성을 위해 합성 OHLCV를 사용합니다.

## 구조

- `indicators/kalman.py`: 1D 및 constant-velocity Kalman filter
- `indicators/cycle_base.py`: SMA/Kalman base cycle
- `indicators/cycle_multiple.py`: OHLC cycle multiple
- `indicators/volume_profile.py`: 0~5x profile, POC, weighted percentile
- `indicators/gaussian_profile.py`: Gaussian fit, expected profile, gap/deficit
- `indicators/flat_chart.py`: 대시보드/전략 공통 계산 파이프라인
- `strategies/cycle_reversion.py`: vectorbt 목표 비중 기반 예제 전략
- `src/trading_lab/ui/app.py`: 활성 백테스트 대시보드
- `src/trading_lab/ui/presentation.py`: 결과 차트 및 거래 리포트 표시 로직
- `dashboard/app.py`: 레거시 Wave Viewer 연구 화면
- `scripts/`: 데이터와 백테스트 smoke test

## 전략 예제

`run_cycle_reversion_backtest()`는 cycle multiple close가 lower percentile 아래에 있다가 회복할 때 100% 목표 비중으로 진입합니다. POC에서 기본 50% 목표 비중으로 줄이고 upper percentile에서 전량 청산합니다. `fees`, `slippage`, `init_cash`, `partial_target`을 인자로 받습니다.

현재 profile과 Gaussian 계산은 전체 입력 구간을 사용합니다. 실전 성능 평가 전에는 look-ahead bias를 피하기 위해 rolling/expanding profile로 바꾸고 PineScript 원본의 정확한 수식 및 초기화 규칙과 대조해야 합니다.

## Dashboard

```bash
./run_dashboard.sh
```

활성 대시보드는 다음 네 화면으로 구성됩니다.

- `새 백테스트`: 심볼, 평가 구간, 초기 계좌 금액을 지정하고 실행
- `결과`: 가격/예측 및 진입·청산 파형, 보조지표 파형, 계좌 금액/낙폭 파형, 거래별 상세 리포트
- `비교`: 여러 실행의 성과 지표와 누적 수익률 파형 비교
- `시스템`: 실행 상태, 저장 경로, live 거래 비활성 상태 확인

거래 리포트에는 거래 번호, 방향, 진입/청산 시각과 가격, 손절/익절가, 결과 손익률, 거래 후 계좌 금액, 진입 근거가 표시됩니다. 현재 `h72-price-v1`은 고정 손절/익절 주문을 사용하지 않으므로 해당 가격은 `미사용`으로 표시됩니다.
