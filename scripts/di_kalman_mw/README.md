# DI Kalman M/W 전략 (plan.txt 구현)

`+DI / -DI`를 Kalman으로 평활한 뒤 극점 4개의 M/W 배열과 압력(pressure) 우위로
Long/Short 신호를 만들고, train/validation/test 백테스트를 수행하는 독립 리서치
패키지다. plan.txt §16에 따라 플랫폼(`research_adapter`)과 연결하지 않는다.

## 모듈 구성

| 파일 | 역할 | plan.txt |
|---|---|---|
| `config.py` | 전체 파라미터, 조합 A~D, train 충분성 기준 | §4, §17 |
| `dmi.py` | Wilder DMI/ATR + DI Kalman 평활 | §5A.4 |
| `extremes.py` | reversal threshold 기반 극점 추출 (idx/confirmation_idx 분리) | §5A.1 |
| `events.py` | M/W 분류, setup 이벤트, pressure feature | §1, §5, §5A.2, §6 |
| `stats.py` | 2-Pass EV 통계 (Pass1 가상 트레이드 → 버킷 집계) | §7, §7A |
| `signals.py` | Variant A(P4)/B(P5) 신호 + 필터 | §8 |
| `backtest.py` | 체결 규칙(§12A) 포함 백테스트 엔진 | §12, §12A |
| `splits.py` | chronological split + train 충분성 검사 | §4 |
| `metrics.py` | 성능 지표, pressure_aligned 분해 | §13, §14 |
| `extreme_transition.py` | P1~P4 조건부 P5 위치/continuation 기준선 | `codex.md` |
| `pattern_dataset.py` | P1~P4 입력과 P5/가격 outcome 분리 데이터셋 | `codex.md` Phase 2 |
| `similarity.py` | train-only weighted k-NN P5 유사도 예측 | `codex.md` Phase 3 |
| `expectation.py` | 유사 사례 실제 가격 경로 기반 순수익 EV | `codex.md` Phase 4 |
| `online_state.py` | P4 이후 P5 후보 경로 온라인 재평가 | `codex.md` Phase 5 |
| `preflight.py` | 실데이터 기간/완결 P5 표본 준비 검사 | `codex.md` Phase 6 |
| `run.py` | 파이프라인 실행기 + 산출물 저장 | §13, §15 |
| `viz.py` | 대시보드용 plotly 차트 빌더 | - |
| `../fetch_ohlcv.py` | Phase 0 ccxt 데이터 수집 | §3A |

## 사용법

```bash
# Phase 0: 데이터 수집 (네트워크 필요)
python scripts/fetch_ohlcv.py

# 합성 데이터로 파이프라인 검증
python scripts/di_kalman_mw/run.py --synthetic --combo A

# 백테스트 전 데이터 준비 검사(성과 계산 없음)
python scripts/di_kalman_mw/preflight.py

# 실데이터 백테스트 (조합 B = P5 확인 진입 + 고정 2R)
python scripts/di_kalman_mw/run.py --data data/raw/BTCUSDT_4h.parquet \
    --symbol BTCUSDT --timeframe 4h --combo B
```

산출물은 `reports/di_kalman_mw/{symbol}_{timeframe}_{features|events|signals|trades|equity}.csv`,
`_metrics.json`, `_train_stats.json`, `_summary.csv` (§13).

추가 연구 산출물:

```text
{symbol}_{timeframe}_pattern_dataset.csv
{symbol}_{timeframe}_similarity_metrics.json
{symbol}_{timeframe}_price_expectations.csv
{symbol}_{timeframe}_price_expectation_metrics.json
{symbol}_{timeframe}_online_decisions.csv
{symbol}_{timeframe}_online_metrics.json
```

`similarity_ev=on`이면 기존 pressure EV 대신 유사 사례 순수익 q25 하한을
진입 필터로 사용한다. 기본값은 검증 전 동작 보존을 위해 `off`다.
`online_revaluation=on`은 hold/exit/reverse 연구 로그를 생성하지만 실제
주문을 체결하지 않는다. 실제 온라인 청산/반전 비교는 Phase 6에서 수행한다.

현재 실데이터 preflight에서는 BTC/ETH/SOL 1h만 최소 train 기간과 완결
P5 표본 기준을 모두 통과했다. 4h와 1d는 `reports/di_kalman_mw/preflight.json`
의 `ready=false`이며 최초 백테스트 대상에서 제외한다.

## 대시보드

`./run_dashboard.sh` 실행 후 "새 백테스트"에서 전략 **`di-kalman-mw-v1`** 을
선택하면 공통 플로우(yfinance 데이터 → 백테스트 → "결과" 탭)로 실행된다.
"전략 파라미터" 패널에서 조합 A~D, reversal/Kalman/비용을 조절할 수 있고
(`configs/strategies/di_kalman_mw_v1.json`의 `tunables` 스키마 기반), 결과 화면에
+DI/-DI Kalman 파형과 train/val/test 구간별 성과가 표시된다.

전략 핸들러(`src/trading_lab/strategies/di_kalman_mw.py`)가 이 리서치 파이프라인을
호출해 공통 `BacktestService`/registry에 연결한다. CLI로도 동일하게 실행 가능:
`trading-lab backtest --strategy di-kalman-mw-v1 --symbol BTC-USD --chart-type crypto`.
별도 데이터(parquet/csv)나 민감도 스윕은 아래 `run.py` 직접 실행을 그대로 쓰면 된다.

## Phase 4 robustness 예시

```bash
for sym in BTCUSDT ETHUSDT SOLUSDT; do
  for tf in 1h 4h 1d; do
    python scripts/di_kalman_mw/run.py --data data/raw/${sym}_${tf}.parquet \
        --symbol $sym --timeframe $tf --combo A
  done
done

# Kalman / reversal / 비용 민감도
python scripts/di_kalman_mw/run.py --synthetic --kalman-q 0.005
python scripts/di_kalman_mw/run.py --synthetic --reversal-mult 1.5
python scripts/di_kalman_mw/run.py --synthetic --cost-mult 2.0
```

## 주의 (plan §16)

- train 구간 지표는 in-sample이다 (EV 통계가 train에서 추정됨, §7A).
- 같은 봉에서 stop/TP 동시 터치 시 stop 우선 체결을 가정한다 (§12A).
- trailing stop은 close에서 레벨 갱신 후 다음 봉부터 intrabar 적용된다.
- DI invalidation exit(§9 Stop Type 3)은 보조 청산 조건으로만 권장되어
  초기 버전에서는 구현 범위에서 제외했다 (§12 엔진 요구사항에는 없음).

테스트: `python -m unittest tests.test_di_kalman_mw -v`
