from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from conf_filter_backtest import (
    BTConfig,
    build_signals,
    rolling_conf_threshold,
    run_backtest,
)
from flat_chart import FlatChartConfig, compute_features
from strategy_execution import ExecutionConfig, run_execution


ROOT = Path(__file__).resolve().parents[1]


def frame(open_: list[float], close: list[float]) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=len(close), freq="1h")
    return pd.DataFrame({"open": open_, "close": close}, index=index)


class SignalTests(unittest.TestCase):
    def test_feature_output_preserves_open(self):
        index = pd.date_range("2025-01-01", periods=8, freq="1h")
        source = pd.DataFrame({
            "open": np.arange(100.0, 108.0),
            "high": np.arange(101.0, 109.0),
            "low": np.arange(99.0, 107.0),
            "close": np.arange(100.5, 108.5),
            "volume": np.ones(8),
        }, index=index)
        features = compute_features(
            source,
            FlatChartConfig(cycle_len=2, fast_window=2, slow_window=3),
        )
        pd.testing.assert_series_equal(
            features["open"], source["open"], check_names=False
        )

    def test_price_direction_uses_forecast_price(self):
        df = pd.DataFrame({
            "close": [100.0, 100.0],
            "mult_close": [1.0, 1.0],
            "m_slow": [1.0, 1.0],
            "pup_72": [0.9, 0.1],
            "mhat_72": [0.9, 1.1],
            "price_mid_72": [110.0, 90.0],
        })
        signals, _ = build_signals(df, 72)
        self.assertEqual(signals["mult_dir"].tolist(), [-1.0, 1.0])
        self.assertEqual(signals["price_dir"].tolist(), [1.0, -1.0])

    def test_rolling_threshold_excludes_current_bar(self):
        conf = pd.Series([1.0, 2.0, 100.0])
        threshold = rolling_conf_threshold(conf, 0.5, 2)
        self.assertEqual(threshold.iloc[2], 1.5)


class ExecutionTests(unittest.TestCase):
    def config(self, **kwargs) -> ExecutionConfig:
        values = {
            "horizon": 2,
            "fee_bps": 10.0,
            "conf_quantile": 0.5,
            "quantile_window": 2,
            "execution": "next_open",
        }
        values.update(kwargs)
        return ExecutionConfig(**values)

    def test_next_open_entry_and_round_trip_fee(self):
        df = frame([100, 101, 103, 105], [100, 102, 104, 106])
        direction = pd.Series([0, 1, 1, 1], index=df.index)
        confidence = pd.Series([0.1, 1.0, 1.0, 1.0], index=df.index)
        result = run_execution(
            df, direction, confidence, self.config(),
            asset="TEST",
        )
        trade = result.trades.iloc[0]
        self.assertEqual(trade["signal_time"], df.index[1])
        self.assertEqual(trade["entry_time"], df.index[2])
        self.assertGreater(trade["entry_time"], trade["signal_time"])
        self.assertAlmostEqual(trade["fee_return"], 0.002)
        self.assertAlmostEqual(
            trade["net_return"], trade["gross_return"] - 0.002
        )
        self.assertAlmostEqual(
            result.equity.iloc[-1], 1.0 + trade["net_return"]
        )

    def test_future_prices_do_not_change_entry_decision(self):
        base = frame(
            [100, 101, 102, 103, 104],
            [100, 101, 102, 103, 104],
        )
        changed = base.copy()
        changed.loc[changed.index[3]:, ["open", "close"]] *= 10
        direction = pd.Series([0, 1, 1, 1, 1], index=base.index)
        confidence = pd.Series([0.1, 1, 1, 1, 1], index=base.index)
        first = run_execution(
            base, direction, confidence, self.config(horizon=3), asset="TEST"
        ).trades.iloc[0]
        second = run_execution(
            changed, direction, confidence, self.config(horizon=3), asset="TEST"
        ).trades.iloc[0]
        self.assertEqual(first["signal_time"], second["signal_time"])
        self.assertEqual(first["entry_time"], second["entry_time"])
        self.assertEqual(first["entry_price"], second["entry_price"])

    def test_signal_bar_open_does_not_change_close_signal(self):
        base = frame([100, 101, 102, 103], [100, 101, 102, 103])
        changed = base.copy()
        changed.loc[changed.index[1], "open"] = 999
        direction = pd.Series([0, 1, 1, 1], index=base.index)
        confidence = pd.Series([0.1, 1, 1, 1], index=base.index)
        first = run_execution(
            base, direction, confidence, self.config(), asset="TEST"
        ).trades.iloc[0]
        second = run_execution(
            changed, direction, confidence, self.config(), asset="TEST"
        ).trades.iloc[0]
        self.assertEqual(first["signal_time"], second["signal_time"])
        self.assertEqual(first["entry_time"], second["entry_time"])
        self.assertEqual(first["entry_price"], second["entry_price"])

    def test_opposite_exit_ignores_edge_filter_for_exit(self):
        df = frame(
            [100, 100, 100, 100, 100, 100],
            [100, 100, 100, 100, 100, 100],
        )
        direction = pd.Series([0, 1, 1, -1, -1, -1], index=df.index)
        confidence = pd.Series([0.1, 1, 1, 1, 1, 1], index=df.index)
        edge = pd.Series([0, 1, 1, 0, 0, 0], index=df.index)
        result = run_execution(
            df, direction, confidence,
            self.config(horizon=10, edge_mult=1.0),
            asset="TEST", expected_edge=edge,
        )
        self.assertEqual(len(result.trades), 1)
        self.assertEqual(result.trades.iloc[0]["exit_reason"], "opposite")

    def test_entry_filter_does_not_block_opposite_exit(self):
        df = frame(
            [100, 100, 100, 100, 100, 100],
            [100, 100, 100, 100, 100, 100],
        )
        direction = pd.Series([0, 1, 1, -1, -1, -1], index=df.index)
        confidence = pd.Series([0.1, 1, 1, 1, 1, 1], index=df.index)
        allowed = pd.Series([True, True, True, False, False, False], index=df.index)
        result = run_execution(
            df, direction, confidence,
            self.config(horizon=10),
            asset="TEST", entry_allowed=allowed,
        )
        self.assertEqual(len(result.trades), 1)
        self.assertEqual(result.trades.iloc[0]["exit_reason"], "opposite")

    def test_short_specific_horizon(self):
        df = frame(
            [100, 100, 100, 100, 100, 100],
            [100, 100, 100, 100, 100, 100],
        )
        direction = pd.Series([0, -1, -1, -1, -1, -1], index=df.index)
        confidence = pd.Series([0.1, 1, 1, 1, 1, 1], index=df.index)
        result = run_execution(
            df, direction, confidence,
            self.config(horizon=4, short_horizon=2, exit_on_opposite=False),
            asset="TEST",
        )
        self.assertEqual(result.trades.iloc[0]["holding_bars"], 2)
        self.assertEqual(result.trades.iloc[0]["exit_reason"], "horizon")

    def test_short_position_size_scales_return_and_fee(self):
        df = frame([100, 100, 100, 90], [100, 100, 100, 90])
        direction = pd.Series([0, -1, -1, -1], index=df.index)
        confidence = pd.Series([0.1, 1, 1, 1], index=df.index)
        result = run_execution(
            df, direction, confidence,
            self.config(
                horizon=2, short_size=0.25, exit_on_opposite=False,
            ),
            asset="TEST",
        )
        trade = result.trades.iloc[0]
        self.assertEqual(trade["position_size"], 0.25)
        self.assertAlmostEqual(trade["fee_return"], 0.0005)
        self.assertAlmostEqual(
            trade["gross_return"], 0.25 * (1.0 - 90.0 / 100.0)
        )


class BaselineRegressionTest(unittest.TestCase):
    def test_btc_h72_close_baseline(self):
        path = ROOT / "reports" / "BTCUSD_V1_forecast.csv"
        if not path.exists():
            self.skipTest("ignored BTC forecast artifact is unavailable")
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        required = [
            "close", "mult_close", "m_slow", "pup_72",
            "mhat_72", "price_mid_72",
        ]
        df = df.dropna(subset=required)
        signals, _ = build_signals(df, 72)
        result = run_backtest(
            df,
            signals["price_dir"],
            signals["price_conf"],
            BTConfig(horizon=72, fee_bps=10, conf_quantile=0.85),
            "PRICE",
            signals["price_edge"],
        )
        self.assertEqual(result.n_trades, 71)
        self.assertAlmostEqual(result.hit_rate, 0.5633802817, places=6)
        self.assertAlmostEqual(result.avg_net_bps, 32.1, delta=0.1)
        self.assertAlmostEqual(result.total_return, 0.193, delta=0.001)
        self.assertAlmostEqual(result.sharpe, 0.65, delta=0.01)
        self.assertAlmostEqual(result.max_dd, -0.239, delta=0.001)


if __name__ == "__main__":
    unittest.main()
