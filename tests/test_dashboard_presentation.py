from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from trading_lab.service import BacktestRequest, BacktestService
from trading_lab.ui.presentation import (
    available_extra_kinds,
    build_account_figure,
    build_bar_figure,
    build_heatmap_figure,
    build_price_indicator_figure,
    build_price_figure,
    build_scatter_figure,
    build_stacked_area_figure,
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

    def test_portfolio_trade_report_preserves_symbol(self):
        trades = self.trades.assign(symbol=["AAPL"], entry_reason=["AAPL top20 편입"])
        report = build_trade_report(
            trades,
            self.equity,
            initial_capital=10_000.0,
            horizon=72,
            execution="next_open",
        )

        self.assertIn("symbol", report.columns)
        self.assertEqual(report.iloc[0]["symbol"], "AAPL")
        self.assertEqual(report.iloc[0]["entry_reason"], "AAPL top20 편입")

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

    def test_portfolio_forecast_uses_nav_axis_and_no_synthetic_buy_hold(self):
        forecast = pd.DataFrame({
            "close": [10_000, 10_200, 10_100, 10_500, 10_800],
            "stock_exposure": [0.8, 0.7, 0.7, 0.9, 0.9],
            "cash_ratio": [0.2, 0.3, 0.3, 0.1, 0.1],
            "n_holdings": [20, 20, 20, 20, 20],
        }, index=self.index)
        trades = pd.DataFrame([{
            "entry_time": self.index[1],
            "entry_price": 150.0,
            "exit_time": self.index[3],
            "exit_price": 180.0,
            "direction": 1,
            "exit_reason": "rebalance_out",
            "net_return": 0.18,
            "symbol": "AAPL",
        }])

        price = build_price_figure(
            forecast, trades, symbol="PORT", horizon=0
        )
        self.assertEqual(price.data[0].name, "포트폴리오 NAV")
        self.assertEqual(price.layout.yaxis.title.text, "NAV")
        self.assertEqual(float(price.data[1].y[0]), 10_200.0)
        self.assertIn("진입가", price.data[1].hovertemplate)
        self.assertEqual(price.data[1].customdata[0][1], "AAPL")

        account = build_account_figure(
            self.equity,
            initial_capital=10_000.0,
            symbol="PORT",
            benchmark_price=None,
        )
        self.assertEqual(len(account.data), 2)
        self.assertEqual(account.layout.title.text, "PORT 전략 계좌")

        benchmark = pd.Series(
            [1.0, 1.01, 1.02, 1.03, 1.04],
            index=self.index,
            name="benchmark",
        )
        account_with_benchmark = build_account_figure(
            self.equity,
            initial_capital=10_000.0,
            symbol="PORT",
            benchmark_equity=benchmark,
        )
        self.assertEqual(len(account_with_benchmark.data), 3)
        self.assertEqual(account_with_benchmark.data[1].name, "Buy & Hold (+4.00%)")
        self.assertEqual(
            account_with_benchmark.layout.title.text,
            "PORT 전략 vs Buy & Hold",
        )

    def test_stacked_area_figure_shows_portfolio_wave(self):
        frame = pd.DataFrame({
            "time": pd.date_range("2026-01-01", periods=3, freq="1D"),
            "cash": [1.0, 0.4, 0.2],
            "AAPL": [0.0, 0.35, 0.5],
            "MSFT": [0.0, 0.25, 0.3],
        })

        figure = build_stacked_area_figure(
            frame, x="time", label="포트폴리오 웨이브"
        )

        self.assertEqual(
            [trace.name for trace in figure.data], ["현금", "AAPL", "MSFT"]
        )
        self.assertTrue(all(trace.stackgroup == "portfolio" for trace in figure.data))
        self.assertEqual(figure.layout.yaxis.title.text, "비중")
        self.assertEqual(tuple(figure.layout.yaxis.range), (0, 1))


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

    def test_heatmap_panel_figure_builds(self):
        # wide 프레임(time + 가격 bin 컬럼) → heatmap trace (z = price x time)
        frame = pd.DataFrame({
            "time": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "10.00": [1.0, 2.0, 3.0],
            "11.00": [0.0, 5.0, 1.0],
        })
        fig = build_heatmap_figure(frame, label="볼륨 프로파일 히트맵")
        self.assertEqual(len(fig.data), 1)
        self.assertEqual(fig.data[0].type, "heatmap")
        self.assertEqual(np.asarray(fig.data[0].z).shape, (2, 3))

    def test_price_figure_heatmap_overlay_is_bottom_layer(self):
        # 청산 히트맵 trace를 넘기면 가격선보다 먼저(맨 아래) 그려져야 한다.
        from trading_lab.ui.presentation import build_heatmap_trace

        frame = pd.DataFrame({
            "time": ["2020-01-01", "2020-01-02", "2020-01-03"],
            "10.00": [1.0, 2.0, 3.0],
            "11.00": [0.0, 5.0, 1.0],
        })
        overlay = build_heatmap_trace(frame, overlay=True, showscale=False)
        self.assertIsNotNone(overlay)
        index = pd.date_range("2020-01-01", periods=3, freq="D")
        forecast = pd.DataFrame({"close": [10.0, 10.5, 11.0]}, index=index)
        fig = build_price_figure(
            forecast, pd.DataFrame(), symbol="TEST", horizon=0,
            heatmap_overlay=overlay,
        )
        self.assertEqual(fig.data[0].type, "heatmap")
        self.assertEqual(fig.data[1].type, "scatter")

    def test_heatmap_trace_returns_none_for_empty(self):
        from trading_lab.ui.presentation import build_heatmap_trace

        self.assertIsNone(build_heatmap_trace(pd.DataFrame()))
        self.assertIsNone(
            build_heatmap_trace(pd.DataFrame({"time": ["2020-01-01"]}))
        )


class BacktestRequestTests(unittest.TestCase):
    def test_initial_capital_must_be_positive(self):
        request = BacktestRequest(
            strategy_id="h72-price-v1",
            symbol="TEST",
            initial_capital=0,
        )
        with self.assertRaisesRegex(ValueError, "initial_capital"):
            BacktestService._validate_request(request, True)

    def test_mixed_chart_type_is_allowed_for_portfolios(self):
        request = BacktestRequest(
            strategy_id="yoon1",
            symbol="PORTFOLIO",
            chart_type="mixed",
        )
        BacktestService._validate_request(request, True)


class DashboardContractTests(unittest.TestCase):
    """전략 추가 시 UI 공통 인프라를 수정하지 않아도 되게 하는 계약 회귀."""

    def test_declared_panel_types_are_registered(self):
        """등록된 모든 전략 config의 dashboard.panels[].type은 PANEL_RENDERERS에
        있어야 한다(오타·미등록 타입을 조기에 잡는다). 'table'은 명시 폴백."""
        from trading_lab.strategies import list_strategies
        from trading_lab.ui.config import strategy_config_dict
        from trading_lab.ui.result_sections import PANEL_RENDERERS

        allowed = set(PANEL_RENDERERS) | {"table"}
        for definition in list_strategies():
            config = strategy_config_dict(definition.strategy_id)
            panels = (config.get("dashboard") or {}).get("panels") or []
            for panel in panels:
                ptype = panel.get("type")
                if ptype is not None:
                    self.assertIn(
                        ptype, allowed,
                        f"{definition.strategy_id}: 미등록 panel type {ptype!r} "
                        "— presentation.py 빌더 + PANEL_RENDERERS에 등록하세요",
                    )

    def test_common_ui_modules_have_no_strategy_id_hardcoding(self):
        """UI 공통 인프라(presentation·result_sections)에 특정 전략 id를
        하드코딩하면 안 된다 — 전략 분기는 config/registry 메타로만 한다."""
        from pathlib import Path

        from trading_lab.strategies import list_strategies
        import trading_lab.ui.presentation as presentation
        import trading_lab.ui.result_sections as result_sections

        ids = [d.strategy_id for d in list_strategies()]
        for module in (presentation, result_sections):
            text = Path(module.__file__).read_text(encoding="utf-8")
            for strategy_id in ids:
                self.assertNotIn(
                    strategy_id, text,
                    f"{module.__name__} 에 전략 id {strategy_id!r} 하드코딩 금지",
                )


if __name__ == "__main__":
    unittest.main()
