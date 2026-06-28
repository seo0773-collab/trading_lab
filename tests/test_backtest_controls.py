from __future__ import annotations

import unittest

from trading_lab.strategies.registry import (
    CATEGORY_HIERARCHICAL,
    CATEGORY_PORTFOLIO,
    CATEGORY_SINGLE,
)
from trading_lab.ui.backtest_controls import enabled_strategy_ids, is_portfolio_strategy
from trading_lab.strategies import get_strategy, list_strategies


class BacktestControlsTests(unittest.TestCase):
    def test_multi_portfolio_mode_lists_yoon_variants(self) -> None:
        strategy_ids = enabled_strategy_ids(portfolio_only=True)

        for strategy_id in ("yoon1", "yoon1b", "yoon1c", "yoon1d"):
            self.assertIn(strategy_id, strategy_ids)

    def test_single_portfolio_mode_excludes_multi_portfolio_strategies(self) -> None:
        strategy_ids = enabled_strategy_ids(portfolio_only=False)

        for strategy_id in ("yoon1", "yoon1b", "yoon1c", "yoon1d", "yoon1k"):
            self.assertNotIn(strategy_id, strategy_ids)

    def test_category_filter_groups_by_declared_category(self) -> None:
        # 분류는 핸들러 추측이 아니라 registry의 category 선언으로 결정된다.
        portfolio = enabled_strategy_ids(category=CATEGORY_PORTFOLIO)
        hierarchical = enabled_strategy_ids(category=CATEGORY_HIERARCHICAL)
        single = enabled_strategy_ids(category=CATEGORY_SINGLE)

        self.assertIn("yoon1b", portfolio)
        self.assertIn("yoon1k", hierarchical)
        self.assertNotIn("yoon1k", portfolio)
        self.assertNotIn("yoon1b", single)
        self.assertNotIn("yoon1k", single)

    def test_hierarchical_is_not_single_but_counts_as_portfolio_group(self) -> None:
        # 계층 포트폴리오는 단일자산이 아니므로 '포트폴리오 아님' 필터에서 빠진다.
        self.assertTrue(is_portfolio_strategy(get_strategy("yoon1k")))

    def test_every_strategy_has_valid_category(self) -> None:
        from trading_lab.strategies.registry import STRATEGY_CATEGORIES

        for definition in list_strategies():
            self.assertIn(definition.category, STRATEGY_CATEGORIES, definition.strategy_id)


if __name__ == "__main__":
    unittest.main()
