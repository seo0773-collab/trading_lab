#!/usr/bin/env python
"""유니버스 배치 IC 측정 + placebo 순열검정 (finance_plan.txt §28).

단일 종목 IC가 작은 표본(5종목)에서 평균 약(弱)양수였는데, 그것이 진짜 엣지인지
방법론이 만든 허상인지 판정한다. 각 종목의 예측 IC(예측 vs 실제 초과수익)를 모아
분포를 보고, **placebo**(재무 피처를 무작위로 섞어 피처→타깃 관계를 끊은 것)와
순열검정으로 비교한다. real 평균 IC가 placebo 분포보다 유의하게 높아야 엣지가 있다.

공통 대시보드 파이프라인이 아니라 연구 오케스트레이션이므로, finance_sensitivity
모듈을 직접 호출한다(거래/equity 불요 — 예측 vs 실제만 필요). 종목별 이벤트 테이블은
1회만 빌드하고, placebo는 그 테이블의 피처만 셔플해 rolling_predict를 재실행한다.

Usage:
    python scripts/finance_sensitivity/batch.py --symbols MSFT,AAPL,... --placebo 200
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from finance_sensitivity.config import FinSensitivityConfig, config_from_dict
from finance_sensitivity.dataset import build_event_table
from finance_sensitivity.fundamentals import feature_columns
from finance_sensitivity.model import rolling_predict

DEFAULT_UNIVERSE = [
    # 섹터 분산 대형/중형주.
    "MSFT", "AAPL", "NVDA", "GOOGL", "AMZN", "META", "AVGO", "TSLA",
    "JPM", "BAC", "WFC", "GS",
    "JNJ", "PFE", "MRK", "UNH",
    "KO", "PEP", "PG", "WMT", "MCD", "COST",
    "CAT", "DE", "HON", "GE",
    "XOM", "CVX", "NEE",
    "DIS", "NKE", "INTC", "CSCO", "ORCL",
]


def _ic(table: pd.DataFrame, horizon: int) -> tuple[float, int]:
    """예측 vs 실제 초과수익의 Spearman IC와 표본 수."""
    pred, real = f"pred_ret_{horizon}d", f"ret_{horizon}d"
    if pred not in table or real not in table:
        return (float("nan"), 0)
    valid = table[[pred, real]].dropna()
    if len(valid) < 10:
        return (float("nan"), len(valid))
    return (float(valid[pred].corr(valid[real], method="spearman")), len(valid))


def _load_market(cfg: FinSensitivityConfig) -> pd.Series | None:
    try:
        from run_kalman_pipeline import load_yfinance
        return load_yfinance(cfg.market_filter.symbol, cfg.interval, cfg.period)["close"]
    except Exception:
        return None


def build_symbol_table(
    symbol: str, cfg: FinSensitivityConfig, market_close: pd.Series | None,
) -> pd.DataFrame | None:
    """종목의 이벤트 테이블(예측 전)을 1회 빌드. 가격/재무 없으면 None."""
    from run_kalman_pipeline import load_yfinance
    fund_path = ROOT / "var" / "fundamentals" / f"{symbol.upper()}.parquet"
    if not fund_path.exists():
        return None
    try:
        raw = load_yfinance(symbol, cfg.interval, cfg.period)
    except Exception:
        return None
    fundamentals = pd.read_parquet(fund_path)
    return build_event_table(raw, fundamentals, cfg, market_close=market_close)


def placebo_ic(
    table: pd.DataFrame, cfg: FinSensitivityConfig, horizon: int, rng: np.random.Generator,
) -> float:
    """피처 행을 한 번 셔플(피처↔타깃 관계 파괴) 후 IC를 측정 — placebo 1회."""
    feats = [c for c in feature_columns(cfg) if c in table.columns]
    shuffled = table.copy()
    order = rng.permutation(len(shuffled))
    shuffled[feats] = shuffled[feats].to_numpy()[order]
    out = rolling_predict(shuffled, cfg)["table"]
    return _ic(out, horizon)[0]


def _progress(phase: str, done: int, total: int, note: str = "") -> None:
    pct = int(round(100 * done / total)) if total else 100
    print(f"PROGRESS {phase} {pct}% ({done}/{total}) {note}".rstrip(), flush=True)


def run_batch(
    symbols: list[str], cfg: FinSensitivityConfig, *,
    horizon: int = 60, n_placebo: int = 200, seed: int = 7,
    verbose: bool = True,
) -> dict:
    print("PROGRESS start 0% (시장지수 SPY 로드 중)", flush=True)
    market_close = _load_market(cfg)
    rng = np.random.default_rng(seed)

    tables: dict[str, pd.DataFrame] = {}
    real_rows = []
    n_sym = len(symbols)
    step = max(1, n_sym // 10)  # 약 10회만 알림(과다 방지)
    for i, symbol in enumerate(symbols, 1):
        table = build_symbol_table(symbol, cfg, market_close)
        if table is None or table.empty:
            real_rows.append({"symbol": symbol, "ic": float("nan"), "n": 0,
                              "status": "no_data"})
        else:
            predicted = rolling_predict(table, cfg)["table"]
            ic, n = _ic(predicted, horizon)
            tables[symbol] = table
            real_rows.append({"symbol": symbol, "ic": ic, "n": n, "status": "ok"})
            time.sleep(0.05)
        if verbose and (i % step == 0 or i == n_sym):
            _progress("load+real", i, n_sym, symbol)

    real = pd.DataFrame(real_rows)
    usable = [r["symbol"] for r in real_rows if r["status"] == "ok" and not np.isnan(r["ic"])]
    real_agg = float(real.loc[real["symbol"].isin(usable), "ic"].mean()) if usable else float("nan")
    frac_pos = float((real.loc[real["symbol"].isin(usable), "ic"] > 0).mean()) if usable else float("nan")

    # placebo: 매 반복마다 모든 종목 피처를 셔플→IC, 종목 평균 = placebo 집계 1개.
    placebo_aggs = []
    pstep = max(1, n_placebo // 10)
    for k in range(1, n_placebo + 1):
        ics = [placebo_ic(tables[s], cfg, horizon, rng) for s in usable]
        ics = [v for v in ics if not np.isnan(v)]
        if ics:
            placebo_aggs.append(float(np.mean(ics)))
        if verbose and (k % pstep == 0 or k == n_placebo):
            _progress("placebo", k, n_placebo)
    placebo_aggs = np.array(placebo_aggs)

    # 순열검정 p값: placebo 집계가 real 집계 이상일 빈도.
    if len(placebo_aggs) and not np.isnan(real_agg):
        p_value = float((np.sum(placebo_aggs >= real_agg) + 1) / (len(placebo_aggs) + 1))
        placebo_mean = float(placebo_aggs.mean())
        placebo_std = float(placebo_aggs.std(ddof=0))
    else:
        p_value = placebo_mean = placebo_std = float("nan")

    verdict = _verdict(real_agg, frac_pos, p_value)
    return {
        "horizon": horizon,
        "n_symbols_usable": len(usable),
        "real_mean_ic": real_agg,
        "real_frac_positive": frac_pos,
        "placebo_mean_ic": placebo_mean,
        "placebo_std": placebo_std,
        "p_value": p_value,
        "n_placebo": int(len(placebo_aggs)),
        "verdict": verdict,
        "per_symbol": real.to_dict(orient="records"),
    }


def _verdict(real_agg: float, frac_pos: float, p_value: float) -> str:
    if np.isnan(real_agg) or np.isnan(p_value):
        return "표본 부족 — 판정 불가"
    if real_agg <= 0:
        return "기각: real 평균 IC ≤ 0 (명세 §2 무효화 조건)"
    if p_value < 0.05:
        return f"엣지 시사: real IC가 placebo 대비 유의 (p={p_value:.3f}) — A 발전 가치"
    if p_value < 0.20:
        return f"약한/모호: placebo 대비 경계 (p={p_value:.3f}) — B 횡단면 풀링 검토"
    return f"엣지 불충분: placebo와 차이 없음 (p={p_value:.3f}) — A 단일종목 가설 폐기 검토"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", default=",".join(DEFAULT_UNIVERSE))
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--placebo", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "configs" / "strategies" / "fin_sensitivity_v1.json",
    )
    parser.add_argument(
        "--outdir", type=Path, default=ROOT / "reports" / "finance_sensitivity",
    )
    args = parser.parse_args(argv)

    cfg = config_from_dict(json.loads(args.config.read_text(encoding="utf-8")))
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    result = run_batch(
        symbols, cfg, horizon=args.horizon, n_placebo=args.placebo, seed=args.seed,
    )

    print(f"\n=== 배치 IC{args.horizon} 판정 (n={result['n_symbols_usable']}종목) ===")
    print(f"real 평균 IC = {result['real_mean_ic']:.4f}  "
          f"(양수 비율 {result['real_frac_positive']:.0%})")
    print(f"placebo 평균 IC = {result['placebo_mean_ic']:.4f} "
          f"± {result['placebo_std']:.4f}  (n_placebo={result['n_placebo']})")
    print(f"순열검정 p값 = {result['p_value']:.4f}")
    print(f"판정: {result['verdict']}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    out = args.outdir / f"batch_ic{args.horizon}.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n리포트: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
