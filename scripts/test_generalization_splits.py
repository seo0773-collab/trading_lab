from __future__ import annotations

import unittest

import pandas as pd

from strategy_execution import ExecutionConfig, chronological_splits, run_execution


class ChronologicalSplitTests(unittest.TestCase):
    def test_default_split_counts_and_order(self):
        index = pd.date_range("2025-01-01", periods=100, freq="1h")
        split = chronological_splits(index)
        self.assertEqual((split == "identification").sum(), 40)
        self.assertEqual((split == "validation").sum(), 30)
        self.assertEqual((split == "test").sum(), 30)
        self.assertTrue((split.iloc[:40] == "identification").all())
        self.assertTrue((split.iloc[40:70] == "validation").all())
        self.assertTrue((split.iloc[70:] == "test").all())

    def test_invalid_fractions_fail(self):
        index = pd.RangeIndex(10)
        with self.assertRaises(ValueError):
            chronological_splits(index, identification_frac=0.8, validation_frac=0.3)

    def test_entries_are_limited_to_requested_split(self):
        index = pd.date_range("2025-01-01", periods=20, freq="1h")
        prices = pd.DataFrame(
            {"open": range(100, 120), "close": range(100, 120)},
            index=index,
        )
        direction = pd.Series(1.0, index=index)
        confidence = pd.Series(1.0, index=index)
        split = chronological_splits(index)
        result = run_execution(
            prices,
            direction,
            confidence,
            ExecutionConfig(
                horizon=2, conf_quantile=0.5,
                quantile_window=2, execution="next_open",
            ),
            asset="TEST",
            split=split,
            entry_split="validation",
        )
        self.assertFalse(result.trades.empty)
        self.assertTrue((result.trades["split"] == "validation").all())


if __name__ == "__main__":
    unittest.main()
