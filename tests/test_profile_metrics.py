"""profile_sizing.metrics 단위 테스트 — 달력 기준 연환산 (개선계획 A1).

핵심 불변식: 인덱스에 결측 구간(예: final0706의 2007-02~2009-04 KR 데이터 갭)이
있어도 CAGR은 달력 경과 시간 기준으로 계산돼야 한다. 봉 수/252 방식은 갭만큼
연수를 줄여 CAGR을 부풀린다(실측: 16.84% vs 달력 13.96%).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "src"), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from profile_sizing.engine import performance  # noqa: E402
from profile_sizing.metrics import coverage_label, equity_metrics, years_elapsed  # noqa: E402


def _bdays(start: str, end: str) -> pd.DatetimeIndex:
    return pd.bdate_range(start, end)


class YearsElapsedTest(unittest.TestCase):
    def test_calendar_basis_with_gap(self):
        # 2년 갭이 있어도 경과 연수는 달력 기준 ~4년이어야 한다.
        idx = _bdays("2007-01-01", "2007-02-01").append(_bdays("2009-04-01", "2011-01-01"))
        yrs = years_elapsed(idx)
        self.assertAlmostEqual(yrs, 4.0, delta=0.1)
        # 봉 수 기준(len/252)이었다면 ~1.9"년"으로 절반 이하가 된다.
        self.assertGreater(yrs, len(idx) / 252.0 * 1.5)

    def test_non_datetime_falls_back_to_bars(self):
        self.assertAlmostEqual(years_elapsed(pd.RangeIndex(504)), 2.0)


class EquityMetricsTest(unittest.TestCase):
    def test_gap_year_cagr_not_inflated(self):
        # 총수익 +50%를 달력 5년에 걸쳐 달성. 중간 2년이 통째로 결측이어도
        # CAGR ≈ 1.5**(1/5)-1 ≈ 8.45%여야 한다(봉 수 기준이면 ~14.5%로 부풀려짐).
        idx = _bdays("2010-01-01", "2011-06-30").append(_bdays("2013-07-01", "2015-01-01"))
        eq = pd.Series([1.0 + 0.5 * i / (len(idx) - 1) for i in range(len(idx))], index=idx)
        m = equity_metrics(eq)
        self.assertAlmostEqual(m["cagr"], 1.5 ** (1 / 5) - 1, delta=0.005)
        self.assertAlmostEqual(m["years"], 5.0, delta=0.05)

    def test_contiguous_daily_close_to_bars_basis(self):
        # 연속 일봉에서는 달력 기준과 봉 수 기준이 거의 같아야 한다(회귀 보호).
        idx = _bdays("2018-01-02", "2021-01-02")
        eq = pd.Series([1.0 * 1.0003 ** i for i in range(len(idx))], index=idx)
        m = equity_metrics(eq)
        bars_cagr = eq.iloc[-1] ** (252.0 / len(eq)) - 1.0
        self.assertAlmostEqual(m["cagr"], bars_cagr, delta=0.01)

    def test_unnormalized_start_ok(self):
        idx = _bdays("2020-01-01", "2022-01-01")
        eq = pd.Series([100.0 * 1.0004 ** i for i in range(len(idx))], index=idx)
        m = equity_metrics(eq)
        self.assertAlmostEqual(m["total_return"], eq.iloc[-1] / eq.iloc[0] - 1.0, places=9)
        self.assertGreater(m["cagr"], 0.0)

    def test_engine_performance_uses_calendar(self):
        # engine.performance도 같은 규칙을 따라야 한다(갭 인덱스에서 CAGR 미부풀림).
        idx = _bdays("2010-01-01", "2011-06-30").append(_bdays("2013-07-01", "2015-01-01"))
        eq = pd.Series([1.0 + 0.5 * i / (len(idx) - 1) for i in range(len(idx))], index=idx)
        ret = eq.pct_change().fillna(0.0)
        perf = performance(eq, ret, "1d")
        self.assertAlmostEqual(perf["cagr"], 1.5 ** (1 / 5) - 1, delta=0.005)


class CoverageLabelTest(unittest.TestCase):
    def test_gap_is_visible_in_label(self):
        idx = _bdays("2007-01-29", "2007-02-08").append(_bdays("2009-04-09", "2010-01-04"))
        label = coverage_label(idx)
        self.assertIn("2007-01-29~2010-01-04", label)
        self.assertIn("결측", label)
        self.assertIn("2009-04-09", label)

    def test_no_gap_no_noise(self):
        label = coverage_label(_bdays("2020-01-01", "2021-01-01"))
        self.assertNotIn("결측", label)


if __name__ == "__main__":
    unittest.main()
