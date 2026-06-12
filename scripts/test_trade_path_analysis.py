from __future__ import annotations

import unittest

import pandas as pd

from trade_path_analysis import (
    summarize_asset_paths,
    summarize_paths,
    trade_pnl_paths,
)


class TradePathTests(unittest.TestCase):
    def test_realized_pnl_is_carried_after_exit(self):
        index = pd.date_range("2025-01-01", periods=5, freq="1h")
        forecast = pd.DataFrame({
            "open": [100, 101, 102, 103, 104],
            "close": [101, 102, 103, 104, 105],
        }, index=index)
        trades = pd.DataFrame([{
            "asset": "TEST",
            "entry_time": index[1],
            "entry_price": 101.0,
            "exit_time": index[3],
            "holding_bars": 2,
            "direction": 1,
            "net_return": 103.0 / 101.0 - 1.0 - 0.002,
            "exit_reason": "opposite",
        }])
        paths = trade_pnl_paths(
            forecast, trades, fee_bps=10, horizon=4,
        )
        realized = paths[paths["hour"] >= 3]
        self.assertTrue((realized["state"] == "realized").all())
        self.assertEqual(realized["net_return"].nunique(), 1)

    def test_horizon_checkpoint_equals_realized_exit(self):
        index = pd.date_range("2025-01-01", periods=4, freq="1h")
        forecast = pd.DataFrame({
            "open": [100, 101, 102, 103],
            "close": [101, 102, 103, 104],
        }, index=index)
        net = 103.0 / 101.0 - 1.0 - 0.002
        trades = pd.DataFrame([{
            "asset": "TEST",
            "entry_time": index[1],
            "entry_price": 101.0,
            "exit_time": index[3],
            "holding_bars": 2,
            "direction": 1,
            "net_return": net,
            "exit_reason": "horizon",
        }])
        paths = trade_pnl_paths(forecast, trades, fee_bps=10, horizon=2)
        self.assertAlmostEqual(paths.iloc[-1]["net_return"], net)
        self.assertEqual(paths.iloc[-1]["state"], "realized")

    def test_summary_keeps_full_cohort(self):
        paths = pd.DataFrame({
            "asset": ["A"] * 4,
            "entry_time": pd.to_datetime(["2025-01-01", "2025-01-02"] * 2),
            "direction": [1, -1, 1, -1],
            "hour": [1, 1, 4, 4],
            "net_return": [0.01, -0.01, 0.02, -0.02],
        })
        summary = summarize_paths(paths)
        all_rows = summary[summary["side"] == "ALL"]
        self.assertEqual(all_rows["trades"].tolist(), [2, 2])

    def test_asset_summary_separates_long_and_short(self):
        paths = pd.DataFrame({
            "asset": ["A", "A", "A", "A"],
            "direction": [1, -1, 1, -1],
            "hour": [4, 4, 24, 24],
            "net_return": [0.01, -0.01, 0.02, -0.02],
        })
        summary = summarize_asset_paths(paths)
        at_four = summary[summary["hour"] == 4]
        self.assertEqual(set(at_four["side"]), {"LONG", "SHORT"})


if __name__ == "__main__":
    unittest.main()
