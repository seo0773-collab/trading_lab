from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from trading_lab.service import BacktestRequest, BacktestService
from trading_lab.ui.presentation import (
    build_account_figure,
    build_indicator_figure,
    build_price_figure,
    build_trade_overview,
    build_trade_report,
)


class TradeReportTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.date_range("2026-01-01", periods=5, freq="1h", tz="UTC")
        self.equity = pd.Series(
            [1.0, 1.0, 1.05, 1.05, 1.10], index=self.index, name="equity"
        )
        self.trades = pd.DataFrame([{
            "entry_time": self.index[1],
            "entry_price": 100.0,
            "exit_time": self.index[2],
            "exit_price": 105.2,
            "direction": 1,
            "exit_reason": "horizon",
            "net_return": 0.05,
            "price_edge": 0.08,
            "confidence_threshold": 0.04,
            "mult_price_conflict": False,
        }])

    def test_report_has_requested_trade_fields(self):
        report = build_trade_report(
            self.trades,
            self.equity,
            initial_capital=10_000.0,
            horizon=72,
            execution="next_open",
        )
        row = report.iloc[0]
        self.assertEqual(row["trade_number"], 1)
        self.assertEqual(row["side"], "롱")
        self.assertTrue(np.isnan(row["stop_loss_price"]))
        self.assertTrue(np.isnan(row["take_profit_price"]))
        self.assertAlmostEqual(row["net_return_pct"], 5.0)
        self.assertAlmostEqual(row["account_value_after"], 10_500.0)
        self.assertIn("72봉 예상 상승", row["entry_reason"])
        self.assertIn("다음 시가 체결", row["entry_reason"])

    def test_trade_overview_splits_long_and_short_metrics(self):
        trades = pd.DataFrame([
            {"direction": 1, "net_return": 0.05, "exit_time": self.index[2]},
            {"direction": 1, "net_return": -0.01, "exit_time": pd.NaT},
            {"direction": -1, "net_return": 0.02, "exit_time": self.index[3]},
        ])
        overview = build_trade_overview(trades)
        self.assertEqual(overview["long_trades"], 2)
        self.assertEqual(overview["short_trades"], 1)
        self.assertAlmostEqual(overview["long_avg_return"], 0.02)
        self.assertAlmostEqual(overview["short_avg_return"], 0.02)
        self.assertAlmostEqual(overview["long_close_rate"], 0.5)
        self.assertAlmostEqual(overview["short_close_rate"], 1.0)

    def test_figures_include_price_indicators_and_account(self):
        forecast = pd.DataFrame({
            "close": [100, 101, 102, 103, 104],
            "mult_close": [1.0, 1.1, 1.0, 0.9, 1.0],
            "m_fast": [1.0] * 5,
            "m_filt": [1.0] * 5,
            "m_slow": [1.0] * 5,
            "price_mid_2": [102, 103, 104, 105, 106],
            "price_lo_2": [100, 101, 102, 103, 104],
            "price_hi_2": [104, 105, 106, 107, 108],
        }, index=self.index)
        price = build_price_figure(
            forecast, self.trades, symbol="TEST", horizon=2
        )
        indicators = build_indicator_figure(
            forecast,
            horizon=2,
            confidence_quantile=0.5,
            quantile_window=2,
        )
        account = build_account_figure(
            self.equity, initial_capital=10_000.0, symbol="TEST"
        )
        self.assertGreaterEqual(len(price.data), 5)
        self.assertEqual(len(indicators.data), 6)
        self.assertEqual(len(account.data), 2)
        self.assertEqual(float(account.data[0].y[-1]), 11_000.0)


class BacktestRequestTests(unittest.TestCase):
    def test_initial_capital_must_be_positive(self):
        request = BacktestRequest(
            strategy_id="h72-price-v1",
            symbol="TEST",
            initial_capital=0,
        )
        with self.assertRaisesRegex(ValueError, "initial_capital"):
            BacktestService._validate_request(request, True)


if __name__ == "__main__":
    unittest.main()
