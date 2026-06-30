"""rebalance_plan(브로커 무관 주문 계산) 단위 테스트."""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from profile_sizing.live.execution_layer import integer_execution_plan  # noqa: E402
from profile_sizing.live.rebalance import rebalance_plan  # noqa: E402
from profile_sizing.live.yoon1k_kis_rebalance import plan_currency_allocation  # noqa: E402
from profile_sizing.paper_trade import buy_hold_snapshot  # noqa: E402


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

    def test_integer_execution_layer_reports_expected_weights(self):
        plan = integer_execution_plan(
            {"A": 0.5, "B": 0.5},
            {},
            {"A": 1_500_000.0, "B": 100_000.0},
            cash=2_000_000.0,
            min_trade_value=0.0,
        )
        by = {o["symbol"]: o for o in plan["orders"]}
        self.assertEqual(by["A"]["qty"], 1.0)
        self.assertGreaterEqual(by["B"]["qty"], 1.0)
        self.assertIn("target_fit", plan)
        self.assertGreater(plan["target_fit"], 0.0)
        rows = {r["symbol"]: r for r in plan["execution_rows"]}
        self.assertIn("CASH", rows)
        self.assertAlmostEqual(rows["A"]["expected_w"], 0.75)

    def test_integer_execution_layer_keeps_strategy_targets_separate(self):
        targets = {"A": 0.6}
        plan = integer_execution_plan(targets, {}, {"A": 700.0}, cash=1000.0)
        self.assertEqual(targets, {"A": 0.6})
        self.assertIn("tracking_error_abs", plan)


class TestYoon2aLiveSetup(unittest.TestCase):
    """yoon2a 모의 전진검증 평탄화 계약."""

    def test_sub_strategies_match_leverage_mix(self):
        # live 평탄화(sub_strategies)가 백테스트 leverage_mix 와 일관돼야 한다.
        cfg = json.loads((ROOT / "configs" / "strategies" / "yoon2a.json")
                         .read_text(encoding="utf-8"))
        mix = cfg["leverage_mix"]
        core_w = float(mix["core_weight"])
        # yoon2 코어는 내부 yoon1j_lp/yoon1j_kr 50:50.
        expected = {"yoon1j_lp": core_w * 0.5, "yoon1j_kr": core_w * 0.5}
        for ov in mix["overlays"]:
            expected["sso_bh" if ov["symbol"] == "SSO" else ov["symbol"]] = float(ov["weight"])
        got = {s["config"]: float(s["weight"]) for s in cfg["sub_strategies"]}
        self.assertEqual(set(got), set(expected))
        for sid, w in expected.items():
            self.assertAlmostEqual(got[sid], w, places=6)
        self.assertAlmostEqual(sum(got.values()), 1.0, places=6)

    def test_buy_hold_snapshot_targets(self):
        snap = buy_hold_snapshot("sso_bh", "sso", "2026-06-26")
        self.assertEqual(snap["targets"], {"SSO": 1.0})
        self.assertEqual(snap["data_through"], "2026-06-26")
        self.assertEqual(snap["n_holdings"], 1)
        self.assertEqual(snap["exposure"], 1.0)

    def test_currency_allocation_three_sleeves(self):
        fx = 1300.0
        sleeves = {
            "yoon1j_lp": {"currency": "USD", "account_value": 100.0, "holdings_value": 100.0},
            "yoon1j_kr": {"currency": "KRW", "account_value": 130_000.0, "holdings_value": 130_000.0},
            "sso_bh": {"currency": "USD", "account_value": 100.0, "holdings_value": 100.0},
        }
        target_w = {"yoon1j_lp": 0.325, "yoon1j_kr": 0.325, "sso_bh": 0.35}
        # 현재 각 1/3, 목표 0.325/0.325/0.35 → 최대 드리프트 ~0.0167.
        alloc = plan_currency_allocation(sleeves, target_w, fx, band=0.01)
        self.assertAlmostEqual(sum(alloc["cur_w"].values()), 1.0, places=6)
        self.assertTrue(alloc["rebalance"])  # 드리프트 0.0167 > 밴드 0.01
        # 밴드 안(0.05)이면 환전 생략.
        self.assertFalse(plan_currency_allocation(sleeves, target_w, fx, band=0.05)["rebalance"])
        # 배정자본은 목표비중 × 총자본(통화 환산) 이어야 한다.
        tot = alloc["total_krw"]
        self.assertAlmostEqual(alloc["budgets"]["sso_bh"] + sleeves["sso_bh"]["holdings_value"],
                               tot * 0.35 / fx, places=4)


if __name__ == "__main__":
    unittest.main()
