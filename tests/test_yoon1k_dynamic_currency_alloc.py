from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from trading_lab.strategies.yoon1k import Yoon1kHandler


class Yoon1kDynamicCurrencyAllocationTests(unittest.TestCase):
    def _raw(self, idx: pd.DatetimeIndex) -> pd.DataFrame:
        spy = pd.DataFrame(
            {"close": np.linspace(100.0, 160.0, len(idx))}, index=idx)
        kodex = pd.DataFrame(
            {"close": np.linspace(100.0, 103.0, len(idx))}, index=idx)
        fx = pd.DataFrame(
            {"close": np.linspace(1000.0, 1200.0, len(idx))}, index=idx)
        return pd.concat({
            "__BENCH__": pd.concat({"SPY": spy, "069500.KS": kodex}, axis=1),
            "__FX__": pd.concat({"KRW=X": fx}, axis=1),
        }, axis=1)

    def _config(self) -> dict:
        return {
            "base_currency": "KRW",
            "fx_symbol": "KRW=X",
            "benchmark_symbols": {"USD": "SPY", "KRW": "069500.KS"},
            "currency_allocation": {
                "kind": "relative_momentum_fx_tilt",
                "lookback": 20,
                "fx_lookback": 20,
                "signal_lag": 1,
                "neutral": 0.5,
                "tilt": 0.15,
                "scale": 0.10,
                "fx_weight": 0.5,
                "band": [0.35, 0.65],
            },
            "sub_strategies": [
                {"config": "us", "currency": "USD", "weight": 0.5},
                {"config": "kr", "currency": "KRW", "weight": 0.5},
            ],
        }

    def test_relative_momentum_fx_tilt_favors_usd_within_band(self) -> None:
        idx = pd.date_range("2024-01-01", periods=80, freq="B")
        raw = self._raw(idx)
        static = pd.DataFrame(0.5, index=idx, columns=["us", "kr"])

        weights = Yoon1kHandler._target_weight_frame(
            self._config(), raw, idx, [0.5, 0.5])

        self.assertGreater(float(weights["us"].iloc[-1]), 0.5)
        self.assertLessEqual(float(weights["us"].max()), 0.65)
        self.assertGreaterEqual(float(weights["us"].min()), 0.35)
        pd.testing.assert_series_equal(
            weights["us"].iloc[:21], static["us"].iloc[:21], check_names=False)

    def test_missing_benchmark_falls_back_to_static_weights(self) -> None:
        idx = pd.date_range("2024-01-01", periods=80, freq="B")
        raw = self._raw(idx).drop(columns=("__BENCH__", "SPY", "close"))

        weights = Yoon1kHandler._target_weight_frame(
            self._config(), raw, idx, [0.5, 0.5])

        self.assertTrue((weights["us"] == 0.5).all())
        self.assertTrue((weights["kr"] == 0.5).all())


if __name__ == "__main__":
    unittest.main()
