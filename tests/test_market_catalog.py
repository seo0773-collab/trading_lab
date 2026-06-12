from __future__ import annotations

import unittest
from unittest.mock import patch

from trading_lab.market_catalog import filter_market_options, load_market_options


class MarketCatalogTests(unittest.TestCase):
    def test_options_are_sorted_by_live_market_cap(self):
        with patch(
            "trading_lab.market_catalog._load_market_caps",
            return_value={"ETH-USD": 3.0, "BTC-USD": 2.0, "XRP-USD": 1.0},
        ):
            options = load_market_options("crypto")
        self.assertEqual([item.symbol for item in options[:3]], [
            "ETH-USD", "BTC-USD", "XRP-USD",
        ])

    def test_search_preserves_market_cap_order(self):
        with patch(
            "trading_lab.market_catalog._load_market_caps",
            return_value={"SPY": 3.0, "AAPL": 2.0},
        ):
            options = load_market_options("stock")
        filtered = filter_market_options(options, "p")
        self.assertEqual([item.symbol for item in filtered[:2]], ["SPY", "AAPL"])
        self.assertEqual(filter_market_options(options, "nvidia")[0].symbol, "NVDA")


if __name__ == "__main__":
    unittest.main()
