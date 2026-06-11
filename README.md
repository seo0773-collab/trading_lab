# Trading Lab

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
streamlit run dashboard/app.py
```

## 검증

프로젝트 루트에서 실행합니다.

```bash
python scripts/test_data.py
python scripts/test_vectorbt.py
python scripts/test_backtrader.py
python -m compileall indicators strategies scanner dashboard scripts
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
- `dashboard/app.py`: yfinance + Plotly + Streamlit 연구 화면
- `scripts/`: 데이터와 백테스트 smoke test

## 전략 예제

`run_cycle_reversion_backtest()`는 cycle multiple close가 lower percentile 아래에 있다가 회복할 때 100% 목표 비중으로 진입합니다. POC에서 기본 50% 목표 비중으로 줄이고 upper percentile에서 전량 청산합니다. `fees`, `slippage`, `init_cash`, `partial_target`을 인자로 받습니다.

현재 profile과 Gaussian 계산은 전체 입력 구간을 사용합니다. 실전 성능 평가 전에는 look-ahead bias를 피하기 위해 rolling/expanding profile로 바꾸고 PineScript 원본의 정확한 수식 및 초기화 규칙과 대조해야 합니다.

## Wave Viewer

```bash
./run_dashboard.sh
```

브라우저의 `Wave Viewer` 탭에서 다음 정보를 같은 시간축으로 확인할 수 있습니다.

- 가격 캔들 및 base cycle
- 매수(초록 삼각형), 부분 청산(노란 다이아몬드), 전량 청산(빨간 삼각형)
- cycle multiple과 lower/POC/Gaussian mean/upper 기준선
- 목표 포지션 비중
- 백테스트 자산곡선과 drawdown

`Volume Profile` 탭은 observed/Gaussian expected profile을, `Signals & Trades` 탭은 전략 이벤트·실제 주문·trade leg를 표시합니다. 차트는 드래그 및 스크롤 확대, 더블클릭 초기화, `3M/6M/1Y/3Y/All` 범위 버튼을 지원합니다.
