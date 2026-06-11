"""Deterministic smoke test for the flat-chart vectorbt strategy."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd


def sample_ohlcv(rows: int = 600) -> pd.DataFrame:
    index = pd.date_range("2022-01-01", periods=rows, freq="D")
    phase = np.linspace(0.0, 16.0 * np.pi, rows)
    trend = np.linspace(100.0, 130.0, rows)
    close = trend * (1.0 + 0.16 * np.sin(phase))
    open_ = close * (1.0 + 0.004 * np.sin(phase + 0.5))
    high = np.maximum(open_, close) * 1.012
    low = np.minimum(open_, close) * 0.988
    volume = 1_000_000 * (1.2 + 0.2 * np.cos(phase))
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=index,
    )


def main() -> int:
    try:
        from strategies.cycle_reversion import run_cycle_reversion_backtest
    except Exception as exc:
        print(f"ERROR: strategy import failed: {exc!r}", file=sys.stderr)
        return 1

    try:
        portfolio, calculated, profile, summary, targets = run_cycle_reversion_backtest(
            sample_ohlcv(), length=80, bins=160, fees=0.001, slippage=0.001
        )
    except Exception as exc:
        print(f"ERROR: vectorbt strategy failed: {exc!r}", file=sys.stderr)
        return 1

    required = {"base_cycle", "cm_open", "cm_high", "cm_low", "cm_close"}
    missing = required.difference(calculated.columns)
    if missing or profile["value"].sum() <= 0 or not np.isfinite(summary["poc"]):
        print(f"ERROR: invalid flat-chart output; missing={sorted(missing)}", file=sys.stderr)
        return 1

    print(
        "OK: vectorbt flat-chart strategy",
        f"orders={targets.notna().sum()}",
        f"end_value={float(portfolio.value().iloc[-1]):.2f}",
        f"poc={summary['poc']:.4f}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
