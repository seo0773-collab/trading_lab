"""fin-sensitivity-v1 데이터/모델/신호 회귀 (finance_plan.txt §13·§21·§22·§25).

백테스트 전 단계의 핵심 계약을 강제한다:
- as-of 결합 후 미래 재무가 보이지 않음(누수 0),
- rolling 예측이 미래 이벤트를 잘라도 불변(인과/윈도우 표준화 누수 0),
- 합성 데이터의 결정성과 주입 인과(예측 IC > 0).
공통 파이프라인 계약(StrategyArtifacts) 검증은 핸들러 등록 후 test_strategy_contract가 담당.
"""
from __future__ import annotations

import dataclasses
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for _p in (str(ROOT), str(SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from finance_sensitivity.availability import (  # noqa: E402
    AVAILABLE_DATE, PERIOD_END, asof_join, with_available_date,
)
from finance_sensitivity.config import FinSensitivityConfig  # noqa: E402
from finance_sensitivity.dataset import build_event_table  # noqa: E402
from finance_sensitivity.fundamentals import feature_columns  # noqa: E402
from finance_sensitivity.model import rolling_predict  # noqa: E402
from finance_sensitivity.signals import TRADE_COLUMNS, build_trades  # noqa: E402
from finance_sensitivity.synthetic import (  # noqa: E402
    make_synthetic_fundamentals, make_synthetic_ohlcv,
)
from trading_lab.strategies.fin_sensitivity import (  # noqa: E402
    FinSensitivityHandler,
)


def _setup(n_bars: int = 2600, seed: int = 7):
    cfg = FinSensitivityConfig(seed=seed, synthetic_bars=n_bars)
    ohlcv = make_synthetic_ohlcv(n_bars, seed, cfg=cfg)
    funds = make_synthetic_fundamentals(
        pd.DatetimeIndex(ohlcv.index), cfg
    )
    return cfg, ohlcv, funds


class SyntheticTests(unittest.TestCase):
    def test_ohlcv_deterministic(self):
        a = make_synthetic_ohlcv(1500, 7)
        b = make_synthetic_ohlcv(1500, 7)
        pd.testing.assert_frame_equal(a, b)

    def test_fundamentals_deterministic(self):
        cfg, ohlcv, _ = _setup()
        idx = pd.DatetimeIndex(ohlcv.index)
        pd.testing.assert_frame_equal(
            make_synthetic_fundamentals(idx, cfg),
            make_synthetic_fundamentals(idx, cfg),
        )

    def test_ohlc_well_formed(self):
        _, ohlcv, _ = _setup(1500)
        self.assertTrue((ohlcv["high"] >= ohlcv["low"]).all())
        self.assertTrue((ohlcv["high"] >= ohlcv["close"]).all())
        self.assertTrue((ohlcv["low"] <= ohlcv["close"]).all())


class AvailabilityLeakTests(unittest.TestCase):
    def test_available_after_period_end(self):
        cfg, _, funds = _setup()
        avail = with_available_date(funds, cfg)
        # 사용가능일은 항상 기준일보다 뒤(보수 45/90일).
        self.assertTrue((avail[AVAILABLE_DATE] > avail[PERIOD_END]).all())

    def test_asof_no_future_leak(self):
        cfg, ohlcv, funds = _setup()
        joined = asof_join(
            ohlcv.index, funds, ["operating_income"], cfg
        )
        # 각 거래일에 붙은 available_date 는 그 날 이하(미래 발표 미참조).
        days = pd.DatetimeIndex(joined.index)
        seen = pd.to_datetime(joined[AVAILABLE_DATE])
        valid = seen.notna().to_numpy()
        self.assertTrue((seen[valid] <= days[valid]).all())

    def test_event_targets_after_features(self):
        cfg, ohlcv, funds = _setup()
        table = build_event_table(ohlcv, funds, cfg)
        realized = table.dropna(subset=["ret20_time"])
        # 타깃 실현 시점은 진입(=사용가능일 이후) 보다 뒤.
        self.assertTrue(
            (pd.to_datetime(realized["ret20_time"])
             > pd.to_datetime(realized["entry_time"])).all()
        )
        self.assertTrue(
            (pd.to_datetime(realized["entry_time"])
             > pd.to_datetime(realized[AVAILABLE_DATE])).all()
        )


class RollingModelTests(unittest.TestCase):
    def test_learning_summary_reports_prediction_quality(self):
        table = pd.DataFrame({
            "pred_ret_20d": [0.01, -0.02, 0.03],
            "ret_20d": [0.02, -0.01, 0.04],
            "pred_ret_60d": [0.04, -0.01, 0.02],
            "ret_60d": [0.03, -0.03, 0.01],
        })

        summary = FinSensitivityHandler._learning_summary(table)

        self.assertEqual(list(summary["horizon_days"]), [20, 60])
        self.assertEqual(list(summary["samples"]), [3, 3])
        self.assertTrue((summary["direction_accuracy"] == 1.0).all())
        self.assertTrue(summary["spearman_ic"].notna().all())

    def test_prediction_is_causal_under_truncation(self):
        """미래 이벤트를 잘라도 과거 이벤트의 예측이 불변 = 누수 없음."""
        cfg, ohlcv, funds = _setup()
        table = build_event_table(ohlcv, funds, cfg)
        full = rolling_predict(table, cfg)["table"]

        k = len(table) - 6
        truncated = rolling_predict(
            table.iloc[:k].reset_index(drop=True), cfg
        )["table"]

        # 충분히 과거(예측이 산출된) 이벤트에서 두 결과가 동일해야 한다.
        compare = min(k, len(truncated))
        a = full["pred_ret_20d"].iloc[:compare].to_numpy()
        b = truncated["pred_ret_20d"].iloc[:compare].to_numpy()
        both = ~np.isnan(a) & ~np.isnan(b)
        self.assertGreater(both.sum(), 0, "비교할 예측이 없음")
        np.testing.assert_allclose(a[both], b[both], rtol=1e-9, atol=1e-12)

    def test_injected_signal_is_learned(self):
        """주입 인과 → 예측과 실제 forward 수익률의 순위상관이 양수."""
        cfg, ohlcv, funds = _setup(2600)
        table = build_event_table(ohlcv, funds, cfg)
        out = rolling_predict(table, cfg)["table"]
        valid = out.dropna(subset=["pred_ret_20d", "ret_20d"])
        self.assertGreaterEqual(len(valid), 10)
        ic = valid["pred_ret_20d"].corr(valid["ret_20d"], method="spearman")
        self.assertGreater(ic, 0.0, f"예측 IC가 0 이하: {ic:.3f}")


class TradeRuleTests(unittest.TestCase):
    def test_trades_schema_and_no_overlap(self):
        cfg, ohlcv, funds = _setup()
        table = rolling_predict(
            build_event_table(ohlcv, funds, cfg), cfg
        )["table"]
        trades = build_trades(table, ohlcv, cfg)
        for col in TRADE_COLUMNS:
            self.assertIn(col, trades.columns)
        if not trades.empty:
            # 단일 종목 100% — 포지션 중첩 금지.
            entries = pd.to_datetime(trades["entry_time"]).to_numpy()
            exits = pd.to_datetime(trades["exit_time"]).to_numpy()
            self.assertTrue((entries[1:] >= exits[:-1]).all())
            reasons = set(trades["exit_reason"])
            self.assertTrue(reasons.issubset(
                {"rebalance", "signal_flip", "stop_loss", "end_of_data"}
            ))
            # 보유기간(max_hold) 청산은 더 이상 발생하지 않는다.
            self.assertNotIn("horizon", reasons)


class RateFeatureTests(unittest.TestCase):
    def test_rate_features_present_and_causal(self):
        """금리 피처가 이벤트 테이블·피처 셋에 들어가고, 주입 인과를 학습한다."""
        from finance_sensitivity.macro import make_synthetic_rates

        cfg, ohlcv, funds = _setup(2600)
        rates = make_synthetic_rates(pd.DatetimeIndex(ohlcv.index), cfg)
        table = build_event_table(ohlcv, funds, cfg, rates=rates)
        for col in ("rate_level", "d_rate"):
            self.assertIn(col, table.columns)
            self.assertIn(col, feature_columns(cfg))

        out = rolling_predict(table, cfg)["table"]
        # 금리 민감도 계수가 산출된다(피처가 모델에 실제로 들어감).
        self.assertIn("sens_d_rate", out.columns)
        self.assertTrue(out["sens_d_rate"].notna().any())

    def test_rate_feature_off_drops_columns(self):
        cfg = dataclasses.replace(
            FinSensitivityConfig(seed=7, synthetic_bars=1500),
            use_rate_feature=False,
        )
        self.assertNotIn("rate_level", feature_columns(cfg))


class MarkToMarketEquityTests(unittest.TestCase):
    def test_equity_marks_to_market_during_holding(self):
        """평가자산: 보유 구간에서 종가에 따라 매일 값이 변한다(step 아님)."""
        idx = pd.bdate_range("2020-01-01", periods=20)
        close = pd.Series(np.linspace(100.0, 120.0, 20), index=idx)
        trades = pd.DataFrame({
            "entry_time": [idx[2]], "entry_price": [100.0],
            "exit_time": [idx[10]], "exit_price": [110.0],
            "net_return": [0.10],
        })
        equity = FinSensitivityHandler._equity_series(trades, idx, close)
        self.assertEqual(len(equity), len(idx))
        held = equity.loc[idx[3]:idx[9]]
        # 보유 중 미실현 평가가 매일 달라진다(고정 step이 아님).
        self.assertGreater(held.nunique(), 1)
        # 청산 후에는 실현 자본(1.10)으로 평탄.
        self.assertAlmostEqual(float(equity.iloc[-1]), 1.10, places=6)
        self.assertTrue((equity > 0).all())


class BatchPlaceboTests(unittest.TestCase):
    def test_placebo_destroys_injected_signal(self):
        """합성: 진짜 IC는 양수, placebo(피처 셔플)는 0 근처여야 한다(§28 판정 근거)."""
        from finance_sensitivity.batch import _ic, placebo_ic

        cfg, ohlcv, funds = _setup(2600)
        cfg = dataclasses.replace(cfg, feature_set="qoq")  # 합성 인과는 QoQ에 주입
        table = build_event_table(ohlcv, funds, cfg)
        real = rolling_predict(table, cfg)["table"]
        real_ic, n = _ic(real, 20)
        self.assertGreaterEqual(n, 10)
        self.assertGreater(real_ic, 0.0)

        rng = np.random.default_rng(0)
        placebo = [placebo_ic(table, cfg, 20, rng) for _ in range(15)]
        placebo = [v for v in placebo if not np.isnan(v)]
        self.assertTrue(placebo)
        self.assertLess(abs(float(np.mean(placebo))), real_ic)


if __name__ == "__main__":
    unittest.main()
