"""rebalance_plan(브로커 무관 주문 계산) 단위 테스트."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from profile_sizing.live.rebalance import rebalance_plan  # noqa: E402


class TestRebalancePlan(unittest.TestCase):
    def test_all_cash_buys_to_target(self):
        plan = rebalance_plan({"AAPL": 0.5, "MSFT": 0.5}, {}, {"AAPL": 100.0, "MSFT": 200.0},
                              cash=1000.0)
        self.assertEqual(plan["account_value"], 1000.0)
        by = {o["symbol"]: o for o in plan["orders"]}
        self.assertEqual(by["AAPL"]["side"], "BUY")
        self.assertEqual(by["AAPL"]["qty"], 5)   # 500/100
        self.assertEqual(by["MSFT"]["qty"], 2)   # 500/200

    def test_sell_overweight_buy_underweight(self):
        # 계좌 = 현금200 + AAPL 8*100=800 = 1000. 목표 AAPL 0.5(=500) → 3주 매도.
        plan = rebalance_plan({"AAPL": 0.5, "MSFT": 0.5}, {"AAPL": 8.0},
                              {"AAPL": 100.0, "MSFT": 200.0}, cash=200.0)
        by = {o["symbol"]: o for o in plan["orders"]}
        self.assertEqual(by["AAPL"]["side"], "SELL")
        self.assertEqual(by["AAPL"]["qty"], 3)
        self.assertEqual(by["MSFT"]["side"], "BUY")
        # 매도가 먼저 정렬
        self.assertEqual(plan["orders"][0]["side"], "SELL")

    def test_min_trade_value_skips_small(self):
        # AAPL 약간 미달이지만 변화액 < min_trade_value면 스킵.
        plan = rebalance_plan({"AAPL": 1.0}, {"AAPL": 9.0}, {"AAPL": 100.0},
                              cash=100.0, min_trade_value=200.0)
        self.assertEqual(plan["orders"], [])

    def test_sell_capped_at_holdings(self):
        # 목표 0 → 보유 전량 매도, 보유 이상으로 팔지 않음.
        plan = rebalance_plan({}, {"TSLA": 3.0}, {"TSLA": 50.0}, cash=0.0)
        o = plan["orders"][0]
        self.assertEqual(o["side"], "SELL")
        self.assertEqual(o["qty"], 3)

    def test_integer_shares_default(self):
        # 1000 * 0.5 / 300 = 1.67 → 1주(정수 절사).
        plan = rebalance_plan({"NVDA": 0.5}, {}, {"NVDA": 300.0}, cash=1000.0)
        self.assertEqual(plan["orders"][0]["qty"], 1)

    def test_missing_price_skipped(self):
        plan = rebalance_plan({"AAPL": 1.0}, {}, {}, cash=1000.0)
        self.assertIn("AAPL", plan["skipped_no_price"])
        self.assertEqual(plan["orders"], [])


if __name__ == "__main__":
    unittest.main()
