from __future__ import annotations

import unittest

from trading_lab.ui.backtest_controls import enabled_strategy_ids


class BacktestControlsTests(unittest.TestCase):
    def test_multi_portfolio_mode_lists_yoon_variants(self) -> None:
        strategy_ids = enabled_strategy_ids(portfolio_only=True)

        for strategy_id in ("yoon1", "yoon1b", "yoon1c", "yoon1d"):
            self.assertIn(strategy_id, strategy_ids)

    def test_single_portfolio_mode_excludes_multi_portfolio_strategies(self) -> None:
        strategy_ids = enabled_strategy_ids(portfolio_only=False)

        for strategy_id in ("yoon1", "yoon1b", "yoon1c", "yoon1d"):
            self.assertNotIn(strategy_id, strategy_ids)


if __name__ == "__main__":
    unittest.main()
