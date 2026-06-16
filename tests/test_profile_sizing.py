"""profile-sizing-v1 단위 테스트 (profile_plan.txt §15).

연구 모듈(지표·프로파일·국면·사이징·엔진)의 핵심 불변식을 검증한다. 파이프라인
계약(아티팩트·run_name·StrategyArtifacts)은 test_strategy_contract.py가 자동 포함하므로
여기서는 전략 고유 로직(무누수·rolling/cumulative·방어장 매수금지·B&H 비교)에 집중한다.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "src"), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dataclasses import replace  # noqa: E402

from profile_sizing.config import (  # noqa: E402
    ProfileSizingConfig, TrendOverlay, config_from_dict,
)
from profile_sizing.engine import buy_hold_equity, portfolio_returns  # noqa: E402
from profile_sizing.indicators import compute_cycle, moving_average  # noqa: E402
from profile_sizing.profile import compute_profile  # noqa: E402
from profile_sizing.regime import classify  # noqa: E402
from profile_sizing.run import run_pipeline  # noqa: E402
from profile_sizing.sizing import (  # noqa: E402
    base_target_weight, build_weights, trend_boost,
)
from profile_sizing.synthetic import make_synthetic_ohlcv  # noqa: E402


class IndicatorTests(unittest.TestCase):
    def test_sma_matches_rolling_mean(self) -> None:
        s = pd.Series(np.arange(1, 21, dtype=float))
        sma = moving_average(s, 5, "SMA")
        self.assertTrue(np.isnan(sma.iloc[3]))  # length 미만 NaN
        self.assertAlmostEqual(sma.iloc[4], 3.0)  # mean(1..5)

    def test_cycle_multiple_and_invalid_base(self) -> None:
        raw = make_synthetic_ohlcv(400, 1)
        cycle = compute_cycle(raw, ProfileSizingConfig().base_cycle)
        # warmup(length-1)구간 base_cycle NaN → cm_close NaN.
        self.assertTrue(cycle["base_cycle"].iloc[:199].isna().all())
        valid = cycle.dropna(subset=["cm_close"])
        # 유효 구간에서 cm_close = close / base_cycle.
        np.testing.assert_allclose(
            valid["cm_close"].to_numpy(),
            (valid["close"] / valid["base_cycle"]).to_numpy(),
            rtol=1e-9,
        )


class ProfileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = ProfileSizingConfig()
        self.raw = make_synthetic_ohlcv(1500, 3)
        self.cycle = compute_cycle(self.raw, self.cfg.base_cycle)
        self.profile = compute_profile(self.cycle, self.raw, self.cfg.profile)

    def test_percentile_bounded_0_1(self) -> None:
        pct = self.profile["cumulative_percentile"].dropna()
        self.assertTrue(((pct >= 0.0) & (pct <= 1.0)).all())

    def test_mid_50_within_profile_range(self) -> None:
        mid = self.profile["cumulative_mid_50"].dropna()
        self.assertTrue(((mid >= self.cfg.profile.min_mult)
                         & (mid <= self.cfg.profile.max_mult)).all())

    def test_lower_percentile_below_mid_50(self) -> None:
        # 하위 20% 위치는 50% 위치보다 항상 같거나 낮아야 한다(단조 누적).
        merged = self.profile.dropna(
            subset=["cumulative_lower_percentile", "cumulative_mid_50"]
        )
        self.assertTrue(
            (merged["cumulative_lower_percentile"]
             <= merged["cumulative_mid_50"] + 1e-9).all()
        )

    def test_no_lookahead_cumulative_profile(self) -> None:
        # 미래 봉을 잘라도 과거 구간의 누적 percentile은 변하지 않아야 한다(무누수).
        cut = 1000
        cyc2 = compute_cycle(self.raw.iloc[:cut], self.cfg.base_cycle)
        prof2 = compute_profile(cyc2, self.raw.iloc[:cut], self.cfg.profile)
        a = self.profile["cumulative_percentile"].iloc[:cut].to_numpy()
        b = prof2["cumulative_percentile"].to_numpy()
        both = ~(np.isnan(a) | np.isnan(b))
        np.testing.assert_allclose(a[both], b[both], rtol=1e-9, atol=1e-9)


class SizingTests(unittest.TestCase):
    def test_bucket_weight_decreasing_in_percentile(self) -> None:
        cfg = ProfileSizingConfig()
        pct = np.array([0.05, 0.25, 0.5, 0.7, 0.9])
        w = base_target_weight(pct, cfg)
        np.testing.assert_array_equal(w, np.array([0.80, 0.60, 0.50, 0.30, 0.10]))

    def test_invalid_percentile_zero_weight(self) -> None:
        w = base_target_weight(np.array([np.nan, 0.1]), ProfileSizingConfig())
        self.assertEqual(w[0], 0.0)

    def test_defense_blocks_new_buy(self) -> None:
        cfg = ProfileSizingConfig()
        # 저가권(낮은 percentile) → 높은 base weight 인데 DEFENSE면 증액 금지.
        pct = np.full(5, 0.05)        # base target 0.80
        regime = np.array(["DEFENSE"] * 5, dtype=object)
        rec = np.zeros(5, dtype=int)
        out = build_weights(pct, regime, rec, cfg)
        # 시작 비중 0에서 방어장 매수 금지 → 계속 0.
        self.assertTrue((out["actual_weight"].to_numpy() == 0.0).all())

    def test_defense_caps_existing_position(self) -> None:
        cfg = ProfileSizingConfig()
        # NORMAL에서 비중을 키운 뒤 DEFENSE 전환 시 cap(0.3)까지 축소되어야 한다.
        pct = np.concatenate([np.full(10, 0.05), np.full(20, 0.05)])
        regime = np.array(["NORMAL"] * 10 + ["DEFENSE"] * 20, dtype=object)
        rec = np.zeros(30, dtype=int)
        out = build_weights(pct, regime, rec, cfg)
        self.assertGreater(out["actual_weight"].iloc[9], 0.3)
        self.assertLessEqual(out["actual_weight"].iloc[-1], 0.3 + 1e-9)

    def test_rebalance_step_limited(self) -> None:
        cfg = ProfileSizingConfig()
        pct = np.full(3, 0.05)  # target 0.80
        regime = np.array(["NORMAL"] * 3, dtype=object)
        out = build_weights(pct, regime, np.zeros(3, dtype=int), cfg)
        # 한 봉 최대 변화 0.20.
        self.assertAlmostEqual(out["actual_weight"].iloc[0], 0.20)
        self.assertAlmostEqual(out["actual_weight"].iloc[1], 0.40)


class TrendOverlayTests(unittest.TestCase):
    def test_disabled_by_default_no_boost(self) -> None:
        cfg = ProfileSizingConfig()  # trend_overlay.enabled=False
        regime = np.array(["NORMAL"] * 4, dtype=object)
        ts = np.array([0.0, 0.2, 0.5, 1.0])
        np.testing.assert_array_equal(trend_boost(regime, ts, cfg), np.zeros(4))

    def test_boost_only_in_applied_regimes(self) -> None:
        cfg = replace(ProfileSizingConfig(),
                      trend_overlay=TrendOverlay(enabled=True, boost_gain=0.8,
                                                 max_boost=0.4))
        regime = np.array(["NORMAL", "CAUTION", "DEFENSE", "RECOVERY"], dtype=object)
        ts = np.full(4, 0.25)  # boost = 0.2
        boost = trend_boost(regime, ts, cfg)
        self.assertAlmostEqual(boost[0], 0.2)   # NORMAL 적용
        self.assertEqual(boost[1], 0.0)         # CAUTION 미적용(방어 보존)
        self.assertEqual(boost[2], 0.0)         # DEFENSE 미적용
        self.assertAlmostEqual(boost[3], 0.2)   # RECOVERY 적용

    def test_boost_capped(self) -> None:
        cfg = replace(ProfileSizingConfig(),
                      trend_overlay=TrendOverlay(enabled=True, boost_gain=1.0,
                                                 max_boost=0.4))
        regime = np.array(["NORMAL"], dtype=object)
        self.assertAlmostEqual(trend_boost(regime, np.array([2.0]), cfg)[0], 0.4)

    def test_overlay_raises_exposure(self) -> None:
        raw = make_synthetic_ohlcv(2500, 7)
        base_cfg = config_from_dict({})
        trend_cfg = config_from_dict(
            {"trend_overlay": {"enabled": True, "boost_gain": 0.8, "max_boost": 0.4}}
        )
        base_w = run_pipeline(raw, base_cfg)["forecast"]["actual_weight"].mean()
        trend_w = run_pipeline(raw, trend_cfg)["forecast"]["actual_weight"].mean()
        self.assertGreater(trend_w, base_w)

    def test_floor_lifts_uptrend_weight_above_boost(self) -> None:
        # floor는 상승추세 봉의 비중을 boost-only보다 더 끌어올려야 한다.
        raw = make_synthetic_ohlcv(2500, 7)
        boost_cfg = config_from_dict(
            {"trend_overlay": {"enabled": True, "boost_gain": 0.8, "max_boost": 0.4}}
        )
        floor_cfg = config_from_dict(
            {"trend_overlay": {"enabled": True, "boost_gain": 0.8,
                               "max_boost": 0.4, "floor": 0.9}}
        )
        boost_w = run_pipeline(raw, boost_cfg)["forecast"]["actual_weight"].mean()
        floor_w = run_pipeline(raw, floor_cfg)["forecast"]["actual_weight"].mean()
        self.assertGreater(floor_w, boost_w)

    def test_floor_not_applied_in_defense(self) -> None:
        # floor가 있어도 DEFENSE에선 적용되지 않아 방어가 보존된다.
        cfg = config_from_dict(
            {"trend_overlay": {"enabled": True, "floor": 0.9,
                               "apply_regimes": ["NORMAL", "RECOVERY"]}}
        )
        regime = np.array(["DEFENSE"] * 5, dtype=object)
        ts = np.full(5, 0.3)  # 상승추세 강도 양수지만 DEFENSE라 미적용
        out = build_weights(np.full(5, 0.9), regime, np.zeros(5, dtype=int), cfg,
                            trend_strength=ts)
        self.assertTrue((out["actual_weight"].to_numpy() == 0.0).all())


class AccountSimTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = config_from_dict({})
        self.raw = make_synthetic_ohlcv(2000, 7)
        self.out = run_pipeline(self.raw, self.cfg)
        self.close = pd.Series(self.raw["close"].to_numpy(), index=self.raw.index)
        self.weight = self.out["forecast"]["actual_weight"]

    def test_reinvest_matches_engine_equity(self) -> None:
        from profile_sizing.account import simulate_account
        acct = simulate_account(self.close, self.weight, self.cfg,
                                initial_capital=10_000.0, reinvest=True)
        # 복리 계좌 최종가치 ≈ 정규화 equity × 초기자본 (비용 타이밍 차이 1% 이내).
        engine_final = float(self.out["equity"].iloc[-1]) * 10_000.0
        acct_final = float(acct["account_value"].iloc[-1])
        self.assertAlmostEqual(acct_final / engine_final, 1.0, delta=0.01)

    def test_reinvest_beats_non_reinvest_when_profitable(self) -> None:
        from profile_sizing.account import simulate_account
        on = simulate_account(self.close, self.weight, self.cfg, reinvest=True)
        off = simulate_account(self.close, self.weight, self.cfg, reinvest=False)
        # 합성 시계열이 순상승이면 복리가 고정원금보다 최종가치가 크거나 같다.
        if float(on["account_value"].iloc[-1]) > 10_000.0:
            self.assertGreaterEqual(
                float(on["account_value"].iloc[-1]),
                float(off["account_value"].iloc[-1]) - 1e-6,
            )

    def test_no_leverage_and_nonneg(self) -> None:
        from profile_sizing.account import simulate_account
        acct = simulate_account(self.close, self.weight, self.cfg)
        self.assertTrue((acct["invested_ratio"] <= 1.0 + 1e-9).all())
        self.assertTrue((acct["account_value"] > 0).all())


class PortfolioTests(unittest.TestCase):
    def setUp(self) -> None:
        from profile_sizing.portfolio import compute_universe
        self.cfg = config_from_dict({})
        panels = {f"SYN{i}": make_synthetic_ohlcv(1500, 7 + i) for i in range(4)}
        self.scores, self.prices = compute_universe(panels, self.cfg)

    def test_exposure_bounded(self) -> None:
        from profile_sizing.portfolio import simulate_portfolio
        sim = simulate_portfolio(self.scores, self.prices, self.cfg,
                                 top_k=2, rebal_freq="monthly")
        exp = sim["forecast"]["stock_exposure"]
        self.assertTrue(((exp >= -1e-9) & (exp <= 1.0 + 1e-9)).all())
        self.assertGreater(float(sim["nav"].iloc[-1]), 0.0)

    def test_market_filter_reduces_exposure_in_downtrend(self) -> None:
        from profile_sizing.portfolio import simulate_portfolio
        # 단조 하락 시장 → 항상 MA 아래 → 필터가 노출을 축소해야 한다.
        down = pd.Series(np.linspace(100.0, 50.0, len(self.prices)),
                         index=self.prices.index)
        off = simulate_portfolio(self.scores, self.prices, self.cfg,
                                 top_k=2, rebal_freq="monthly")
        on = simulate_portfolio(self.scores, self.prices, self.cfg,
                                top_k=2, rebal_freq="monthly",
                                market_close=down, market_ma_len=50,
                                market_off_scale=0.5)
        self.assertLess(float(on["forecast"]["stock_exposure"].mean()),
                        float(off["forecast"]["stock_exposure"].mean()))


class EngineTests(unittest.TestCase):
    def test_buy_hold_equals_price_growth(self) -> None:
        raw = make_synthetic_ohlcv(500, 5)
        close = pd.Series(raw["close"].to_numpy(), index=raw.index)
        bnh = buy_hold_equity(close)
        self.assertAlmostEqual(
            float(bnh.iloc[-1]), float(close.iloc[-1] / close.iloc[0]), places=6
        )

    def test_zero_weight_flat_equity(self) -> None:
        raw = make_synthetic_ohlcv(300, 5)
        close = pd.Series(raw["close"].to_numpy(), index=raw.index)
        weight = pd.Series(0.0, index=raw.index)
        port_ret = portfolio_returns(close, weight, ProfileSizingConfig())
        self.assertTrue(np.allclose(port_ret.to_numpy(), 0.0))

    def test_pipeline_outputs_and_bounds(self) -> None:
        cfg = config_from_dict({})
        raw = make_synthetic_ohlcv(2000, 7)
        out = run_pipeline(raw, cfg)
        w = out["forecast"]["actual_weight"]
        self.assertTrue(((w >= 0.0) & (w <= 1.0)).all())
        self.assertEqual(len(out["equity"]), len(raw))
        self.assertGreater(float(out["equity"].iloc[-1]), 0.0)


if __name__ == "__main__":
    unittest.main()
