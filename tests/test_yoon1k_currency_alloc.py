"""plan_currency_allocation(yoon1k 통화 50/50 ±밴드 배분) 단위 테스트.

A3 결론 코드화: 드리프트 ≤밴드면 환전 생략, 밴드 밖이면 50/50 복원.
배정현금 = 배정자본 − 보유평가(음수 가능) → rebalance_plan 에 cash 로 주면
account=배정자본 이 되어 슬리브가 50% 목표로 사이징된다.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from profile_sizing.live.yoon1k_kis_rebalance import (  # noqa: E402
    plan_currency_allocation, rebalance_plan)

FX = 1000.0  # 원/달러 (계산 단순화)


class TestCurrencyAllocation(unittest.TestCase):
    def _sleeves(self, kr_val, us_val, kr_hold=0.0, us_hold=0.0):
        return {
            "kr": {"currency": "KRW", "account_value": kr_val, "holdings_value": kr_hold},
            "us": {"currency": "USD", "account_value": us_val, "holdings_value": us_hold},
        }

    def test_within_band_no_rebalance(self):
        # KR 5,200,000원 + US $4,800(=4,800,000원) → 52/48%, 드리프트 2%p ≤ 5%p
        s = self._sleeves(5_200_000, 4_800)
        a = plan_currency_allocation(s, {"kr": 0.5, "us": 0.5}, FX, band=0.05)
        self.assertFalse(a["rebalance"])
        self.assertEqual(a["budgets"], {})           # 환전 없음
        self.assertAlmostEqual(a["total_krw"], 10_000_000)
        self.assertAlmostEqual(a["cur_w"]["us"], 0.48, places=4)

    def test_outside_band_rebalances_to_5050(self):
        # KR 7,000,000 + US $3,000(=3,000,000) → 70/30%, 드리프트 20%p > 5%p
        s = self._sleeves(7_000_000, 3_000)
        a = plan_currency_allocation(s, {"kr": 0.5, "us": 0.5}, FX, band=0.05)
        self.assertTrue(a["rebalance"])
        # 총 1,000만 → 각 500만 목표. KR 700만→500만(200만 유출), US 300만→500만(200만 유입)
        self.assertAlmostEqual(a["transfers"]["kr"], -2_000_000)
        self.assertAlmostEqual(a["transfers"]["us"], +2_000_000)
        # 배정현금: 보유 0 이므로 배정자본 전액. US 는 $5,000(=500만÷1000)
        self.assertAlmostEqual(a["budgets"]["kr"], 5_000_000)
        self.assertAlmostEqual(a["budgets"]["us"], 5_000)

    def test_budget_minus_holdings_targets_alloc_capital(self):
        # 보유가 있을 때: 배정현금 = 배정자본 − 보유평가. rebalance_plan account=배정자본 확인.
        # US 슬리브: $3,000 (보유 $2,000 + 현금 $1,000), KR 700만 → US 50% 목표 = $5,000 자본
        s = self._sleeves(7_000_000, 3_000, us_hold=2_000)
        a = plan_currency_allocation(s, {"kr": 0.5, "us": 0.5}, FX, band=0.05)
        budget_us = a["budgets"]["us"]           # 5,000 − 2,000 = 3,000
        self.assertAlmostEqual(budget_us, 3_000)
        # 이 배정현금을 cash 로 rebalance_plan 에 주면 account = 배정자본 $5,000
        plan = rebalance_plan({"VOO": 0.5}, {"VOO": 20.0}, {"VOO": 100.0}, budget_us)
        self.assertAlmostEqual(plan["account_value"], 5_000.0)  # 3,000 cash + 2,000 보유

    def test_total_krw_override(self):
        s = self._sleeves(5_000_000, 5_000)
        a = plan_currency_allocation(s, {"kr": 0.5, "us": 0.5}, FX,
                                     total_krw=20_000_000, band=0.05)
        self.assertAlmostEqual(a["total_krw"], 20_000_000)
        # 현재 10M 평가인데 총자본 20M 강제 → 각 1000만 배정(추가 납입 가정)
        self.assertTrue(a["rebalance"])
        self.assertAlmostEqual(a["budgets"]["kr"], 10_000_000)


if __name__ == "__main__":
    unittest.main()
