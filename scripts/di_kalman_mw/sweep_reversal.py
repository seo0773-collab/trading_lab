#!/usr/bin/env python
"""Reversal-parameter sensitivity sweep (overfit-spike vs plateau check).

Runs the full pipeline in-memory (outdir=None, no file writes) across a grid
of reversal_mult x reversal_std_window on the same synthetic data used for
the saved SYNTH_4h reports, and prints train (in-sample) vs test (OOS) PF /
return / trade count so a sharp corner spike is easy to spot.
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from di_kalman_mw.config import ExtremeConfig, combo_config  # noqa: E402
from di_kalman_mw.run import make_synthetic_ohlcv, run_pipeline  # noqa: E402

MULTS = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
WINDOWS = [5, 10, 20, 35, 50, 75]
COMBO = "A"


def fmt(x, nd=2):
    return "  n/a" if x is None else f"{x:>5.{nd}f}"


def main() -> int:
    df = make_synthetic_ohlcv(n=9000, seed=7, timeframe="4h")
    base = combo_config(COMBO)
    print(f"combo={COMBO} variant={base.signal.entry_variant} "
          f"bars={len(df)}  (synthetic seed=7)\n")

    for window in WINDOWS:
        print(f"=== reversal_std_window = {window} ===")
        print(f"{'mult':>5} | {'events':>6} | "
              f"{'tr_tr':>5} {'tr_pf':>6} {'tr_ret':>7} | "
              f"{'te_tr':>5} {'te_pf':>6} {'te_ret':>7}")
        for mult in MULTS:
            cfg = dataclasses.replace(
                base,
                extremes=ExtremeConfig(
                    reversal_mult=mult, reversal_std_window=window
                ),
            )
            m, _ = run_pipeline(df, "SYNTH", "4h", cfg, outdir=None, combo=COMBO)
            tr = m["train_metrics"]
            te = m["test_metrics"]
            print(
                f"{mult:>5.2f} | {m['n_events']:>6} | "
                f"{tr['num_trades']:>5} {fmt(tr['profit_factor'])} "
                f"{fmt(tr['total_return'], 3)} | "
                f"{te['num_trades']:>5} {fmt(te['profit_factor'])} "
                f"{fmt(te['total_return'], 3)}"
            )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
