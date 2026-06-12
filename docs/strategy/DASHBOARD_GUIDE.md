# 대시보드 연동 가이드 (전략 작성자용)

`./run_dashboard.sh`는 전략을 모릅니다. 실행 체인은 다음과 같고, 대시보드는 **실행이 남긴 아티팩트와 config만 읽어서** 화면을 구성합니다.

```
./run_dashboard.sh
  → python -m trading_lab ui            (cli.py)
  → streamlit run src/trading_lab/ui/app.py
       ├─ BacktestService.run()         (service.py — 실행·아티팩트 생성)
       └─ render_run_result()           (app.py — 아티팩트 소비·시각화)
```

새 전략이 이 문서의 **데이터 계약**만 지키면 `app.py` / `presentation.py`를 한 줄도 수정하지 않고 가격 차트, 보조지표 파형, 계좌 곡선, 트레이드 리포트, 실행 비교가 전부 동작합니다. 전략 등록 절차 자체는 [README.md](README.md)의 워크플로를 따르세요.

## 1. 대시보드가 소비하는 아티팩트 계약

`BacktestService.run()`이 `var/runs/{run_id}/`에 기록하는 아티팩트 중 대시보드가 직접 읽는 것은 세 가지이며, 하나라도 없으면 결과 화면이 오류로 종료됩니다.

### forecast (필수)

DatetimeIndex를 가진 프레임. 컬럼 규칙:

| 컬럼 | 필수 | 용도 |
| --- | --- | --- |
| `open`, `close` | 필수 | 가격 차트, 체결 가정, 파생 지표 계산 |
| `price_mid_{h}`, `price_lo_{h}`, `price_hi_{h}` | 선택 | `h == config.horizon`일 때 가격 차트에 예측 중앙값/구간 오버레이 |
| 그 외 모든 숫자 컬럼 | 선택 | **보조지표 선택지로 자동 노출** (아래 2절) |

### trades (필수)

| 컬럼 | 필수 | 비고 |
| --- | --- | --- |
| `entry_time`, `entry_price`, `exit_time`, `exit_price` | 필수 | 가격 차트 마커 + 트레이드 리포트 |
| `direction` | 필수 | `+1` 롱 / `-1` 숏 |
| `net_return` | 필수 | 소수 단위 수익률 (0.05 = 5%) |
| `exit_reason` | 필수 | 한글 라벨은 `presentation.py`의 `EXIT_REASON_LABELS`로 변환. 새 청산 사유를 쓰면 그 딕셔너리에 라벨을 추가 |
| `stop_loss_price`, `take_profit_price` | 선택 | 없거나 전부 NaN이면 "미사용"으로 표시 |
| `price_edge`, `confidence_threshold`, `mult_price_conflict` | 선택 | 진입 근거 문구 생성에 사용, 없으면 "n/a" |

### equity (필수)

`equity` 컬럼 하나를 가진 프레임. **초기값 1.0 기준 정규화** 곡선이어야 하며, 대시보드가 초기 자본을 곱해 계좌 금액·낙폭·실행 비교를 그립니다.

### metrics (필수 키)

`summarize_execution`이 기본으로 채우는 키 중 대시보드가 직접 읽는 것: `trades`, `hit_rate`, `total_return`, `sharpe`, `max_drawdown`. (`initial_capital`, `final_account_value` 등은 service가 자동 추가)

## 2. 보조지표 파형 — 동작 규칙

결과 화면의 "보조지표 파형" 섹션은 다음 규칙으로 동작합니다. **전략별 차트 코드를 새로 작성하지 마세요.**

1. **자동 노출**: forecast의 OHLC(`open/high/low/close`) 제외 모든 숫자 컬럼이 멀티셀렉트 선택지가 됩니다. 즉, 사용자에게 보여주고 싶은 지표는 **forecast 프레임에 컬럼으로 추가**하면 끝입니다.
2. **파생 지표**: `price_mid_{horizon}` 컬럼과 config의 `horizon > 0`이 있으면 `expected_edge_pct`(예상 변동폭 %)와 `entry_threshold_pct`(rolling 분위수 진입 임계값 %)가 자동 생성됩니다 (`confidence_quantile`, `quantile_window` 사용).
3. **패널 배치**: 선택된 지표는 스케일(중앙값의 자릿수)이 비슷한 것끼리 같은 패널에 겹쳐 그려지고, 스케일이 다르면 자동으로 별도 패널로 분리됩니다. 파생 % 지표 쌍은 항상 전용 패널입니다.

구현 위치: `presentation.py`의 `indicator_series()` / `build_waveform_figure()`.

### config의 dashboard 블록 (표시 힌트)

전략 config JSON에 선택적으로 `dashboard` 블록을 선언해 기본 화면을 제어합니다. 없으면 선택지 앞 4개가 기본 선택됩니다.

```json
"dashboard": {
  "default_indicators": ["m_filt", "m_fast", "expected_edge_pct", "entry_threshold_pct"],
  "indicator_labels": {
    "m_filt": "Filtered",
    "m_fast": "Fast"
  }
}
```

- `default_indicators`: 결과 화면 진입 시 기본 선택될 지표(컬럼명 또는 파생 지표 키). 존재하지 않는 이름은 무시됩니다.
- `indicator_labels`: 컬럼명 → 표시 이름 매핑. 선언하지 않은 컬럼은 컬럼명 그대로 표시됩니다.

## 3. 대시보드가 읽는 config 키

| 키 | 기본값 | 사용처 |
| --- | --- | --- |
| `horizon` | 72 (app.py) / 0 (service.py) | 예측 오버레이 정렬, 파생 지표, 진입 근거 문구 |
| `confidence_quantile` | 0.85 | 파생 임계값 계산 |
| `quantile_window` | 2000 | 파생 임계값 rolling 윈도 |
| `execution` | `next_open` | 진입 근거 문구 ("다음 시가" / "신호 종가") |
| `dashboard.*` | 없음 | 2절 표시 힌트 |

위 키가 의미 없는 전략이면 생략해도 됩니다 — 전부 기본값으로 동작하고, `price_mid_*`가 없으면 관련 오버레이·파생 지표는 조용히 빠집니다.

## 4. 해야 할 일 / 하지 말아야 할 일

**해야 할 일**

- 보여주고 싶은 내부 상태(필터 값, 레짐, 신뢰도 등)는 forecast 프레임의 숫자 컬럼으로 내보내고, `dashboard.default_indicators`에 핵심 지표만 선언
- 새 `exit_reason` 값을 도입하면 `presentation.py`의 `EXIT_REASON_LABELS`에 한글 라벨 추가
- equity는 반드시 1.0 정규화 — 금액 환산은 대시보드 책임

**하지 말아야 할 일**

- `app.py` / `presentation.py`에 특정 전략의 컬럼명·전략 이름·파라미터를 하드코딩하지 않기 (이 결합을 제거한 것이 현재 구조의 목적)
- 전략 전용 Streamlit 페이지나 차트 함수를 추가하지 않기 — 계약(forecast 컬럼 + dashboard 블록)으로 해결이 안 되는 경우에만 `presentation.py`의 범용 함수를 확장
- config 키 이름 재활용 주의: `horizon`, `confidence_quantile`, `quantile_window`, `execution`은 대시보드가 위 표의 의미로 해석하므로 다른 의미로 쓰지 않기

## 5. 빠른 검증

UI를 띄우기 전에 CLI로 계약 충족 여부를 확인할 수 있습니다.

```bash
PYTHONPATH=src python -m trading_lab backtest --strategy <id> --symbol SYNTH \
    --chart-type random --synthetic --phase validation
# status=succeeded + forecast/trades/equity/metrics 아티팩트 확인 후
./run_dashboard.sh   # "결과" 메뉴에서 해당 실행 선택
```

`tests/test_dashboard_presentation.py`가 표시 계약(트레이드 리포트 필드, 지표 추출, 패널 분리)을 회귀 테스트하므로 전략 추가 후 함께 실행하세요.
