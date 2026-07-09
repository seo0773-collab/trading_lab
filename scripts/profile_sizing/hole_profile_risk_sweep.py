"""hole_profile 리스크 파라미터(stop_atr_mult × take_profit_atr_mult) 스윕.

hole_profile_exit_sweep.py(2026-07-08)에서 entry_percentile=0.60/exit_percentile
=0.50를 채택한 뒤, 사용자가 "익절/손절을 더 보수적으로" 요청했다. 기존 스윕은
stop_atr_mult(1.5~6.0, 무제한)만 봤고 take_profit_atr_mult는 아예 없었다
(구조/percentile 청산에만 의존) — 이번엔 그 두 축만 고정 entry/exit 위에서
촘촘히 스윕해 손절을 더 조이고 실제 익절(고정 R배수) 도입 여지를 확인한다.

방법론은 hole_profile_exit_sweep.py와 동일: state는 1회만 계산, train_60
Sharpe로 후보를 추리고 validation/test는 검증에만 쓴다(과최적화 방지).

Usage:
    PYTHONPATH=scripts .venv/bin/python scripts/profile_sizing/hole_profile_risk_sweep.py
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import pandas as pd

from profile_sizing.config import config_from_dict
from profile_sizing.engine import performance
from profile_sizing.hole_profile import HoleProfileConfig, simulate_hole_profile
from profile_sizing.hole_profile_features import compute_hole_profile_state
from profile_sizing.run import rebased_equity, slice_window

ROOT = Path(__file__).resolve().parents[2]

ENTRY_PCTL = 0.60
EXIT_PCTL = 0.50
MAX_HOLD = 60

STOP_ATR = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0)
TAKE_PROFIT_ATR = (0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)   # 0.0 = 비활성

MIN_TRAIN_TRADES = 15
COST_BPS_PER_SIDE = 5.0
SLIPPAGE_BPS = 5.0


def _phase_perf(bar_return: pd.Series, window, interval: str) -> dict:
    equity = rebased_equity(bar_return, window)
    ret = bar_return.reindex(window).fillna(0.0)
    return performance(equity, ret, interval)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        default=ROOT / "data" / "raw" / "BTCUSDT_1d.parquet")
    parser.add_argument("--outdir", type=Path,
                        default=ROOT / "reports" / "profile_sizing")
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args(argv)

    raw = pd.read_parquet(args.data)
    df = raw.dropna(subset=["open", "high", "low", "close"]).sort_index()
    index = pd.DatetimeIndex(pd.to_datetime(df.index))
    if index.tz is not None:
        index = index.tz_localize(None)
    df = df.copy()
    df.index = index
    print(f"BTC {df.index[0].date()}~{df.index[-1].date()} ({len(df)}봉)")
    print(f"고정: entry_percentile={ENTRY_PCTL}, exit_percentile={EXIT_PCTL}, "
          f"max_hold_bars={MAX_HOLD}")

    base_cfg = config_from_dict({"interval": "1d", "train_frac": 0.6, "validation_frac": 0.2})
    windows = {p: slice_window(index, p, base_cfg) for p in ("all", "validation", "test")}
    windows["train"] = index[: int(len(index) * 0.6)]

    state = compute_hole_profile_state(
        df, atr_len=48, atr_mult=1.0, level="L1", min_profile_samples=30,
    )

    combos = list(itertools.product(STOP_ATR, TAKE_PROFIT_ATR))
    print(f"grid size = {len(combos)}")

    rows = []
    for stop_mult, tp_mult in combos:
        cfg = HoleProfileConfig(
            entry_percentile=ENTRY_PCTL, exit_percentile=EXIT_PCTL,
            stop_atr_mult=stop_mult, take_profit_atr_mult=tp_mult,
            max_hold_bars=MAX_HOLD,
            fee_bps_per_side=COST_BPS_PER_SIDE, slippage_bps=SLIPPAGE_BPS,
        )
        trades, position, bar_return = simulate_hole_profile(df, state, cfg)
        if trades.empty:
            continue
        entries = pd.DatetimeIndex(pd.to_datetime(trades["entry_time"]))
        row = {"stop_atr_mult": stop_mult, "take_profit_atr_mult": tp_mult}
        ok = True
        for phase, window in windows.items():
            n_trades = int(((entries >= window[0]) & (entries <= window[-1])).sum())
            perf = _phase_perf(bar_return, window, "1d")
            row[f"{phase}_trades"] = n_trades
            row[f"{phase}_sharpe"] = perf["sharpe"]
            row[f"{phase}_return"] = perf["total_return"]
            row[f"{phase}_mdd"] = perf["max_drawdown"]
            row[f"{phase}_cagr"] = perf["cagr"]
            if phase == "train" and n_trades < MIN_TRAIN_TRADES:
                ok = False
        reason_counts = trades["exit_reason"].value_counts().to_dict()
        row["exit_reasons"] = reason_counts
        row["stop_hit_rate"] = reason_counts.get("stop_atr", 0) / len(trades)
        row["take_profit_hit_rate"] = reason_counts.get("take_profit_atr", 0) / len(trades)
        if ok:
            rows.append(row)

    table = pd.DataFrame(rows)
    table = table[table["train_sharpe"].notna()]
    table = table.sort_values("train_sharpe", ascending=False).reset_index(drop=True)

    args.outdir.mkdir(parents=True, exist_ok=True)
    table.drop(columns=["exit_reasons"]).to_csv(
        args.outdir / "hole_profile_risk_sweep.csv", index=False
    )

    baseline = table[(table["stop_atr_mult"] == 3.0) & (table["take_profit_atr_mult"] == 0.0)]
    print("\n=== 기존(sweep 전) 리스크 설정: stop=3.0/take_profit=비활성 ===")
    if len(baseline):
        b = baseline.iloc[0]
        print(f"  train sharpe={b.train_sharpe:.3f} n={int(b.train_trades)} | "
              f"val sharpe={b.validation_sharpe:.3f} n={int(b.validation_trades)} | "
              f"test sharpe={b.test_sharpe:.3f} n={int(b.test_trades)} | "
              f"all sharpe={b.all_sharpe:.3f} n={int(b.all_trades)} mdd={b.all_mdd*100:.1f}% "
              f"stop_hit={b.stop_hit_rate*100:.0f}% tp_hit={b.take_profit_hit_rate*100:.0f}%")

    print(f"\n=== train Sharpe 상위 {args.top_n} ===")
    cols = ["stop_atr_mult", "take_profit_atr_mult", "train_sharpe", "train_trades",
            "validation_sharpe", "validation_trades", "test_sharpe", "test_trades",
            "all_sharpe", "all_trades", "all_mdd", "all_cagr",
            "stop_hit_rate", "take_profit_hit_rate"]
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(table[cols].head(args.top_n).to_string(index=False))

    payload = {
        "data": str(args.data),
        "fixed": {"entry_percentile": ENTRY_PCTL, "exit_percentile": EXIT_PCTL,
                  "max_hold_bars": MAX_HOLD},
        "grid": {"stop_atr_mult": list(STOP_ATR), "take_profit_atr_mult": list(TAKE_PROFIT_ATR)},
        "min_train_trades": MIN_TRAIN_TRADES,
        "n_eligible_combos": int(len(table)),
        "baseline": baseline.drop(columns=["exit_reasons"]).to_dict(orient="records")
        if len(baseline) else None,
        "top_by_train_sharpe": table.drop(columns=["exit_reasons"]).head(30).to_dict(orient="records"),
    }
    path = args.outdir / "hole_profile_risk_sweep.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nJSON: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
