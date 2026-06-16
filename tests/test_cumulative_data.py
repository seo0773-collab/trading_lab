from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import fetch_fundamentals
from trading_lab import market_data


def _ohlcv(start: str, closes: list[float]) -> pd.DataFrame:
    index = pd.date_range(start, periods=len(closes), freq="D")
    return pd.DataFrame(
        {
            "Open": closes,
            "High": [value + 1 for value in closes],
            "Low": [value - 1 for value in closes],
            "Close": closes,
            "Volume": [100] * len(closes),
        },
        index=index,
    )


class CumulativeMarketDataTests(unittest.TestCase):
    def test_accumulates_and_refreshes_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            responses = iter([
                _ohlcv("2024-01-01", [10.0, 11.0]),
                _ohlcv("2024-01-02", [12.0, 13.0]),
            ])

            def download(*args, **kwargs):
                return next(responses)

            with patch.object(market_data, "var_dir", return_value=root):
                market_data.load_cumulative_yfinance(
                    "AAPL", "1d", "max", downloader=download
                )
                result = market_data.load_cumulative_yfinance(
                    "AAPL", "1d", "max", downloader=download
                )
                path = market_data.market_data_path("AAPL", "1d")

            self.assertEqual(list(result["close"]), [10.0, 12.0, 13.0])
            self.assertTrue(path.exists())

    def test_uses_cache_when_refresh_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(market_data, "var_dir", return_value=root):
                expected = market_data.load_cumulative_yfinance(
                    "MSFT", "1d", "max", downloader=lambda *a, **k: _ohlcv(
                        "2024-01-01", [20.0, 21.0]
                    ),
                )

                def fail(*args, **kwargs):
                    raise OSError("offline")

                actual = market_data.load_cumulative_yfinance(
                    "MSFT", "1d", "max", downloader=fail
                )

            pd.testing.assert_frame_equal(actual, expected)


class CumulativeFundamentalTests(unittest.TestCase):
    def test_preserves_history_and_updates_period(self) -> None:
        old = pd.DataFrame({
            "period_end": ["2023-03-31", "2023-06-30"],
            "report_type": ["quarter", "quarter"],
            "announce_date": ["2023-05-01", "2023-08-01"],
            "revenue": [100.0, 110.0],
        })
        new = pd.DataFrame({
            "period_end": ["2023-06-30", "2023-09-30"],
            "report_type": ["quarter", "quarter"],
            "announce_date": ["2023-08-02", "2023-11-01"],
            "revenue": [111.0, 120.0],
            "operating_income": [None, 12.0],
        })
        old["operating_income"] = [10.0, 11.0]

        result = fetch_fundamentals.merge_fundamentals(old, new)

        self.assertEqual(list(result["revenue"]), [100.0, 111.0, 120.0])
        self.assertEqual(list(result["operating_income"]), [10.0, 11.0, 12.0])
        self.assertTrue(result["period_end"].is_monotonic_increasing)


if __name__ == "__main__":
    unittest.main()
