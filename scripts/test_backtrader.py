"""Deterministic Backtrader engine smoke test."""

import sys

import numpy as np
import pandas as pd


def main() -> int:
    try:
        import backtrader as bt
    except ImportError as exc:
        print(f"ERROR: backtrader import failed: {exc}", file=sys.stderr)
        print("Install with: pip install -r requirements-backtest.txt", file=sys.stderr)
        return 1

    class SmaCross(bt.Strategy):
        def __init__(self):
            fast = bt.indicators.SimpleMovingAverage(self.data.close, period=10)
            slow = bt.indicators.SimpleMovingAverage(self.data.close, period=30)
            self.cross = bt.indicators.CrossOver(fast, slow)

        def next(self):
            if not self.position and self.cross > 0:
                self.buy()
            elif self.position and self.cross < 0:
                self.close()

    index = pd.date_range("2023-01-01", periods=180, freq="D")
    close = 100.0 + np.linspace(0.0, 10.0, len(index)) + 8.0 * np.sin(
        np.linspace(0.0, 10.0 * np.pi, len(index))
    )
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1000.0,
        },
        index=index,
    )

    try:
        engine = bt.Cerebro(stdstats=False)
        engine.adddata(bt.feeds.PandasData(dataname=frame))
        engine.addstrategy(SmaCross)
        engine.broker.setcash(10_000.0)
        engine.broker.setcommission(commission=0.001)
        engine.run()
    except Exception as exc:
        print(f"ERROR: Backtrader smoke test failed: {exc!r}", file=sys.stderr)
        return 1

    print(f"OK: backtrader {bt.__version__}; final_value={engine.broker.getvalue():.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
