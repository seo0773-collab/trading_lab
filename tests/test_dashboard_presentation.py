from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from trading_lab.service import BacktestRequest, BacktestService
from trading_lab.ui.presentation import (
    available_extra_kinds,
    build_account_figure,
    build_bar_figure,
    build_price_indicator_figure,
    build_price_figure,
    build_scatter_figure,
    build_trade_overview,
    build_trade_report,
    build_waveform_figure,
    indicator_series,
    resolve_extra_panels,
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
        account = build_account_figure(
            self.equity,
            initial_capital=10_000.0,
            symbol="TEST",
            benchmark_price=forecast["close"],
        )
        self.assertGreaterEqual(len(price.data), 5)
        self.assertEqual(len(account.data), 3)
        self.assertEqual(float(account.data[0].y[-1]), 11_000.0)
        self.assertEqual(float(account.data[1].y[-1]), 10_400.0)
        self.assertEqual(account.data[0].name, "전략 계좌 (+10.00%)")
        self.assertEqual(account.data[1].name, "Buy & Hold (+4.00%)")

        series = indicator_series(
            forecast, horizon=2, confidence_quantile=0.5, quantile_window=2
        )
        self.assertNotIn("close", series)
        for name in (
            "mult_close", "m_fast", "m_filt", "m_slow",
            "expected_edge_pct", "entry_threshold_pct",
        ):
            self.assertIn(name, series)

        selected = ("mult_close", "m_fast", "expected_edge_pct")
        waveform = build_waveform_figure(
            {name: series[name] for name in selected},
            labels={"mult_close": "Cycle multiple"},
        )
        self.assertEqual(len(waveform.data), 3)
        names = {trace.name for trace in waveform.data}
        self.assertIn("Cycle multiple", names)
        self.assertIn("예상 변동폭 %", names)
        # 파생 % 지표는 원본 컬럼과 다른 패널(축)에 배치됩니다.
        column_axes = {
            trace.yaxis for trace in waveform.data
            if trace.name in ("Cycle multiple", "m_fast")
        }
        derived_axes = {
            trace.yaxis for trace in waveform.data
            if trace.name == "예상 변동폭 %"
        }
        self.assertEqual(len(column_axes), 1)
        self.assertTrue(column_axes.isdisjoint(derived_axes))

        combined = build_price_indicator_figure(
            forecast,
            self.trades,
            symbol="TEST",
            horizon=2,
            series_map={name: series[name] for name in selected},
            labels={"mult_close": "Cycle multiple"},
        )
        price_axes = {
            trace.yaxis for trace in combined.data
            if trace.name in ("종가", "롱 진입", "청산")
        }
        indicator_axes = {
            trace.yaxis for trace in combined.data
            if trace.name in ("Cycle multiple", "m_fast", "예상 변동폭 %")
        }
        self.assertEqual(price_axes, {"y"})
        self.assertNotIn("y", indicator_axes)
        self.assertEqual(combined.layout.xaxis.matches, "x3")
        self.assertEqual(combined.layout.xaxis2.matches, "x3")


class ExtraPanelTests(unittest.TestCase):
    """전략별 보조 패널 선언/렌더의 공통 인프라 계약 (config add/delete 구조)."""

    def _run(self, kinds):
        return {"artifacts": [{"kind": k, "path": f"/tmp/{k}.json"} for k in kinds]}

    def test_available_excludes_core_artifacts(self):
        run = self._run([
            "forecast", "trades", "equity", "metrics", "manifest",
            "pred_vs_real", "sensitivity_table",
        ])
        self.assertEqual(
            available_extra_kinds(run), ["pred_vs_real", "sensitivity_table"]
        )

    def test_declared_panel_overrides_and_undeclared_defaults_to_table(self):
        dashboard = {"panels": [
            {"kind": "pred_vs_real", "type": "scatter", "label": "예측 vs 실제",
             "x": "pred_ret_60d", "y": "ret_60d", "default": True},
        ]}
        panels = resolve_extra_panels(dashboard, ["pred_vs_real", "mystery"])
        by_kind = {p["kind"]: p for p in panels}
        self.assertEqual(by_kind["pred_vs_real"]["type"], "scatter")
        self.assertEqual(by_kind["pred_vs_real"]["label"], "예측 vs 실제")
        # 선언 없는 extras 도 표로 자동 노출(코드 수정 불필요).
        self.assertEqual(by_kind["mystery"]["type"], "table")
        self.assertEqual(by_kind["mystery"]["label"], "mystery")

    def test_panel_figures_build(self):
        frame = pd.DataFrame({
            "pred_ret_60d": [0.01, -0.02, 0.03],
            "ret_60d": [0.02, -0.01, 0.04],
            "factor": ["a", "b", "c"],
            "sensitivity_mean": [0.5, -0.3, 0.1],
        })
        scatter = build_scatter_figure(
            frame, "pred_ret_60d", "ret_60d", label="예측 vs 실제"
        )
        bar = build_bar_figure(
            frame, "factor", "sensitivity_mean", label="민감도"
        )
        self.assertGreaterEqual(len(scatter.data), 1)
        self.assertEqual(len(bar.data), 1)


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
