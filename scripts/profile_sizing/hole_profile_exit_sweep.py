"""hole_profile 청산 규칙 파라미터 스윕: stop_atr_mult × exit_percentile × max_hold_bars × entry_percentile.

hole_profile_study.py는 **진입신호**(profile_rank 게이트)만 검증했다. 실제
`simulate_hole_profile`의 청산 규칙(손절/percentile_fade/시간청산) 파라미터는
설계 당시 임의값이었고, 기본 config(entry=0.70/exit=0.30/stop=3.0/hold=60)의
실거래 로그를 보면 31건 중 28건이 ``structure_confirmed``(진입 근거 피벗이
갱신돼 자동 종료)로 빠져 percentile_fade·horizon은 **한 번도 발동하지 않았다**
— 즉 그 두 파라미터가 사실상 죽은 설정이었다. 이 스크립트는 네 파라미터를
같이 스윕해 실제로 개선 여지가 있는지 확인한다.

방법론(과최적화 방지): compute_hole_profile_state는 파라미터와 무관하므로
1회만 계산하고 시뮬레이션만 반복한다. **train_60 구간의 Sharpe로 후보를
고르고 validation/test/all은 검증용으로만 본다** — all-period Sharpe로 바로
고르면 test 구간에 대한 암묵적 과최적화가 된다.

Usage:
    PYTHONPATH=scripts .venv/bin/python scripts/profile_sizing/hole_profile_exit_sweep.py
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from profile_sizing.config import config_from_dict
from profile_sizing.engine import performance
from profile_sizing.hole_profile import HoleProfileConfig, simulate_hole_profile
from profile_sizing.hole_profile_features import compute_hole_profile_state
from profile_sizing.run import rebased_equity, slice_window

ROOT = Path(__file__).resolve().parents[2]

ENTRY_PCTL = (0.60, 0.70, 0.80, 0.90)
EXIT_PCTL = (-1.0, 0.10, 0.20, 0.30, 0.40, 0.50)   # -1.0 = percentile_fade 비활성
STOP_ATR = (1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 1.0e6)     # 1e6 ~= 손절 비활성
MAX_HOLD = (0, 20, 40, 60, 90, 120)                 # 0 = 시간청산 비활성

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

    base_cfg = config_from_dict({"interval": "1d", "train_frac": 0.6, "validation_frac": 0.2})
    windows = {p: slice_window(index, p, base_cfg) for p in ("all", "validation", "test")}
    windows["train"] = index[: int(len(index) * 0.6)]

    state = compute_hole_profile_state(
        df, atr_len=48, atr_mult=1.0, level="L1", min_profile_samples=30,
    )

    combos = list(itertools.product(ENTRY_PCTL, EXIT_PCTL, STOP_ATR, MAX_HOLD))
    print(f"grid size = {len(combos)}")

    rows = []
    for entry_p, exit_p, stop_mult, max_hold in combos:
        cfg = HoleProfileConfig(
            entry_percentile=entry_p, exit_percentile=exit_p,
            stop_atr_mult=stop_mult, max_hold_bars=max_hold,
            fee_bps_per_side=COST_BPS_PER_SIDE, slippage_bps=SLIPPAGE_BPS,
        )
        trades, position, bar_return = simulate_hole_profile(df, state, cfg)
        if trades.empty:
            continue
        entries = pd.DatetimeIndex(pd.to_datetime(trades["entry_time"]))
        row = {
            "entry_percentile": entry_p, "exit_percentile": exit_p,
            "stop_atr_mult": stop_mult, "max_hold_bars": max_hold,
        }
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
        row["exit_reason_counts"] = trades["exit_reason"].value_counts().to_dict()
        if ok:
            rows.append(row)

    table = pd.DataFrame(rows)
    table = table[table["train_sharpe"].notna()]
    table = table.sort_values("train_sharpe", ascending=False).reset_index(drop=True)

    args.outdir.mkdir(parents=True, exist_ok=True)
    table.drop(columns=["exit_reason_counts"]).to_csv(
        args.outdir / "hole_profile_exit_sweep.csv", index=False
    )

    baseline = table[
        (table["entry_percentile"] == 0.70) & (table["exit_percentile"] == 0.30)
        & (table["stop_atr_mult"] == 3.0) & (table["max_hold_bars"] == 60)
    ]
    print("\n=== 기존 config baseline (entry=0.70/exit=0.30/stop=3.0/hold=60) ===")
    if len(baseline):
        b = baseline.iloc[0]
        print(f"  train sharpe={b.train_sharpe:.3f} n={int(b.train_trades)} | "
              f"val sharpe={b.validation_sharpe:.3f} n={int(b.validation_trades)} | "
              f"test sharpe={b.test_sharpe:.3f} n={int(b.test_trades)} | "
              f"all sharpe={b.all_sharpe:.3f} n={int(b.all_trades)} mdd={b.all_mdd*100:.1f}%")
    else:
        print("  (baseline이 min_train_trades 필터에서 제외됨)")

    print(f"\n=== train Sharpe 상위 {args.top_n} (train n>={MIN_TRAIN_TRADES}) ===")
    cols = ["entry_percentile", "exit_percentile", "stop_atr_mult", "max_hold_bars",
            "train_sharpe", "train_trades", "validation_sharpe", "validation_trades",
            "test_sharpe", "test_trades", "all_sharpe", "all_trades", "all_mdd", "all_cagr"]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table[cols].head(args.top_n).to_string(index=False))

    payload = {
        "data": str(args.data),
        "grid": {
            "entry_percentile": list(ENTRY_PCTL), "exit_percentile": list(EXIT_PCTL),
            "stop_atr_mult": list(STOP_ATR), "max_hold_bars": list(MAX_HOLD),
        },
        "min_train_trades": MIN_TRAIN_TRADES,
        "n_eligible_combos": int(len(table)),
        "baseline": baseline.drop(columns=["exit_reason_counts"]).to_dict(orient="records")
        if len(baseline) else None,
        "top_by_train_sharpe": table.drop(columns=["exit_reason_counts"]).head(50).to_dict(orient="records"),
    }
    path = args.outdir / "hole_profile_exit_sweep.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nJSON: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
