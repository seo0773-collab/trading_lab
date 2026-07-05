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
from profile_sizing.cycle_state import (  # noqa: E402
    expanding_percentile_rank, zigzag_moves,
)
from profile_sizing.engine import buy_hold_equity, portfolio_returns  # noqa: E402
from profile_sizing.indicators import compute_cycle, moving_average  # noqa: E402
from profile_sizing.profile import (  # noqa: E402
    compute_profile, contribution_matrix, rolling_profile_matrix,
)
from profile_sizing.regime import classify  # noqa: E402
from profile_sizing.rolling_expectancy import (  # noqa: E402
    compute_rolling_profile_features, expectancy_weights,
    run_rolling_expectancy_pipeline, walk_forward_expectancy,
)
from profile_sizing.run import rebased_equity, run_pipeline, slice_window  # noqa: E402
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


class RollingExpectancyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = config_from_dict({
            "base_cycle": {"length": 20},
            "profile": {
                "min_mult": 0.0,
                "max_mult": 3.0,
                "bin_count": 60,
                "rolling_window": 10,
                "weight_mode": "time",
                "accumulation_mode": "range_uniform",
            },
            "smoothing": {"sigma_bins": 1.5},
            "expectancy": {
                "primary_horizon": 5,
                "score_bins": [-1.0, -0.5, 0.0, 0.5, 1.0],
                "min_samples": 5,
                "train_window": 80,
                "refit_every": 7,
                "cost_bps_roundtrip": 10.0,
            },
            "sizing": {
                "expectancy_scale_bps": 25.0,
                "min_positive_expectancy_bps": 1.0,
                "rebalance_threshold": 0.0,
                "max_trade_weight_per_bar": 0.5,
            },
        })
        self.raw = make_synthetic_ohlcv(320, 11)
        self.cycle = compute_cycle(self.raw, self.cfg.base_cycle)

    def test_rolling_profile_matrix_matches_direct_window_sum(self) -> None:
        contrib, _ = contribution_matrix(self.cycle, self.raw, self.cfg.profile)
        roll = rolling_profile_matrix(contrib, self.cfg.profile.rolling_window)
        t = 120
        direct = contrib[t - self.cfg.profile.rolling_window + 1:t + 1].sum(axis=0)
        np.testing.assert_allclose(roll[t], direct, rtol=1e-12, atol=1e-12)

    def test_smooth_distribution_candle_location_features_are_bounded(self) -> None:
        features, prob = compute_rolling_profile_features(
            self.cycle, self.raw, self.cfg
        )
        row_sum = prob.sum(axis=1)
        valid_rows = row_sum > 0.0
        np.testing.assert_allclose(row_sum[valid_rows], 1.0, rtol=1e-9, atol=1e-9)
        for col in (
            "open_percentile", "high_percentile", "low_percentile",
            "close_percentile", "rolling_percentile", "candle_mid_percentile",
            "body_mid_percentile", "close_in_candle_percentile",
        ):
            vals = features[col].dropna()
            self.assertTrue(((vals >= 0.0) & (vals <= 1.0)).all(), col)
        score = features["profile_score"].dropna()
        self.assertTrue(((score >= -1.0) & (score <= 1.0)).all())

    def test_walk_forward_expectancy_unchanged_by_future_tail(self) -> None:
        features, _ = compute_rolling_profile_features(self.cycle, self.raw, self.cfg)
        exp, _ = walk_forward_expectancy(features, self.raw["close"], self.cfg)

        cut = 240
        raw2 = self.raw.iloc[:cut]
        cycle2 = compute_cycle(raw2, self.cfg.base_cycle)
        features2, _ = compute_rolling_profile_features(cycle2, raw2, self.cfg)
        exp2, _ = walk_forward_expectancy(features2, raw2["close"], self.cfg)

        a = exp["expectancy_bps"].iloc[:cut].to_numpy()
        b = exp2["expectancy_bps"].to_numpy()
        both = ~(np.isnan(a) | np.isnan(b))
        np.testing.assert_allclose(a[both], b[both], rtol=1e-9, atol=1e-9)

    def test_expectancy_weights_are_long_only_and_step_limited(self) -> None:
        exp = pd.Series([np.nan, -10.0, 10.0, 60.0], dtype=float)
        weights = expectancy_weights(exp, self.cfg)
        self.assertTrue(((weights["actual_weight"] >= 0.0)
                         & (weights["actual_weight"] <= 1.0)).all())
        self.assertLessEqual(
            float(weights["actual_weight"].diff().abs().max()),
            self.cfg.expectancy_sizing.max_trade_weight_per_bar + 1e-9,
        )

    def test_pipeline_returns_forecast_equity_and_tables(self) -> None:
        out = run_rolling_expectancy_pipeline(self.raw, self.cfg)
        self.assertEqual(len(out["forecast"]), len(self.raw))
        self.assertEqual(len(out["equity"]), len(self.raw))
        self.assertIn("close_percentile", out["forecast"])
        self.assertIn("expectancy_bps", out["forecast"])
        self.assertIn("actual_weight", out["forecast"])
        self.assertGreater(float(out["equity"].iloc[-1]), 0.0)
        self.assertFalse(out["expectancy_tables"].empty)


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

    def test_phase_rebased_equity_starts_at_one(self) -> None:
        for phase in ("validation", "test", "all"):
            window = slice_window(self.raw.index, phase, self.cfg)
            equity = rebased_equity(self.out["port_ret"], window)
            self.assertAlmostEqual(float(equity.iloc[0]), 1.0)


class CycleStateFieldTests(unittest.TestCase):
    """2D 상태 필드 재료: 연속 percentile rank(인과) + zigzag 큰 파동 라벨."""

    def test_expanding_rank_is_causal_and_bounded(self) -> None:
        rng = np.random.default_rng(3)
        idx = pd.date_range("2020-01-01", periods=400, freq="D")
        values = pd.Series(rng.normal(size=400).cumsum(), index=idx)
        rank_full = expanding_percentile_rank(values, 50)
        rank_head = expanding_percentile_rank(values.iloc[:200], 50)
        self.assertTrue(rank_full.iloc[:49].isna().all())
        finite = rank_full.dropna()
        self.assertTrue(((finite > 0.0) & (finite <= 1.0)).all())
        # 무누수: 미래 데이터를 붙여도 과거 시점의 rank는 변하지 않는다.
        pd.testing.assert_series_equal(rank_full.iloc[:200], rank_head)

    def test_zigzag_labels_three_legs_with_signs(self) -> None:
        # 100→200(+100%) → 120(−40%) → 180(+50%), threshold 30%
        path = np.concatenate([
            np.linspace(100.0, 200.0, 40),
            np.linspace(200.0, 120.0, 30),
            np.linspace(120.0, 180.0, 30),
        ])
        idx = pd.date_range("2021-01-01", periods=len(path), freq="D")
        moves = zigzag_moves(pd.Series(path, index=idx), threshold=0.3)
        mid = moves["move_id"].to_numpy()
        mdir = moves["move_dir"].to_numpy()
        self.assertEqual(sorted(set(mid[mid >= 0])), [0, 1, 2])
        for leg, want_dir in ((0, 1), (1, -1), (2, 1)):
            bars = np.flatnonzero(mid == leg)
            self.assertTrue((mdir[bars] == want_dir).all())
            # leg 구간은 연속이고 진폭은 임계 이상
            self.assertTrue((np.diff(bars) == 1).all())
            amp = path[bars[-1]] / path[bars[0]] - 1.0
            self.assertGreaterEqual(abs(amp), 0.3)
        # 인접 leg가 공유하는 pivot 봉은 새 leg에 귀속 → 라벨은 서로 겹치지 않는다.
        self.assertEqual(mid[39], 1)
        self.assertEqual(mid[69], 2)

    def test_zigzag_small_oscillation_has_no_labels(self) -> None:
        t = np.arange(200, dtype=float)
        path = 100.0 + 5.0 * np.sin(t / 7.0)   # ±5% 진동 — 30% 임계 미달
        moves = zigzag_moves(pd.Series(path), threshold=0.3)
        self.assertTrue((moves["move_dir"].to_numpy() == 0).all())
        self.assertTrue((moves["move_id"].to_numpy() == -1).all())

    def test_path_tilt_default_off_keeps_targets(self) -> None:
        from profile_sizing.yoon_profile import (
            apply_path_tilt, state_target_weights, yoon_profile_config,
        )
        state = pd.DataFrame({
            "mult_bin": [1, 1, 2, 0], "angle_bin": [1, 1, 1, 0],
            "mult_pct": [0.5, 0.4, 0.9, 0.1],
        })
        ycfg = yoon_profile_config({})
        target = state_target_weights(state, ycfg)
        tilted, _ = apply_path_tilt(target, state, ycfg)
        pd.testing.assert_series_equal(tilted, target)

    def test_path_tilt_overrides_by_cell_and_velocity_sign(self) -> None:
        from profile_sizing.yoon_profile import (
            apply_path_tilt, state_target_weights, yoon_profile_config,
        )
        n = 30
        state = pd.DataFrame({
            "mult_bin": [1] * n, "angle_bin": [1] * n,
            # 앞 절반 상승(vel>=0), 뒤 절반 하강(vel<0)
            "mult_pct": list(np.linspace(0.2, 0.8, 15))
            + list(np.linspace(0.8, 0.2, 15)),
        })
        ycfg = yoon_profile_config({"path_tilt": {
            "enabled": True, "vel_window": 3,
            "overrides": [
                {"mult_bin": 1, "angle_bin": 1, "arrival": "up", "weight": 1.0},
                {"mult_bin": 1, "angle_bin": 1, "arrival": "down", "weight": 0.5},
            ],
        }})
        target = state_target_weights(state, ycfg)
        tilted, vel = apply_path_tilt(target, state, ycfg)
        # vel NaN 워밍업(앞 3봉)은 base 0.8 유지
        self.assertTrue((tilted.iloc[:3] == 0.8).all())
        self.assertTrue((tilted[vel >= 0].dropna() == 1.0).all())
        self.assertTrue((tilted[vel < 0] == 0.5).all())

    def test_state_field_points_carry_threshold_grid_labels(self) -> None:
        # 대시보드 임계값 노브용 사전 계산 라벨(move_dir_<pct>) 계약
        from trading_lab.strategies.yoon_profile import YoonProfileHandler

        path = np.concatenate([
            np.linspace(100.0, 200.0, 40),
            np.linspace(200.0, 120.0, 30),
        ])
        idx = pd.date_range("2022-01-01", periods=len(path), freq="D")
        forecast = pd.DataFrame({
            "close": path,
            "mult_pct": np.linspace(0.05, 0.95, len(path)),
            "angle_pct": np.linspace(0.05, 0.95, len(path)),
            "mult_bin": 1,
            "angle_bin": 1,
            "target_weight": 0.5,
        }, index=idx)
        points = YoonProfileHandler._state_field_points(
            forecast, {"state_field": {"move_threshold": 0.3}}
        )
        for pct in (10, 15, 20, 25, 30, 40, 50):
            self.assertIn(f"move_dir_{pct}", points.columns)
            self.assertIn(f"move_id_{pct}", points.columns)
        # 평문 라벨 = 기본 임계값(30%) 그리드 라벨
        self.assertTrue(
            (points["move_dir"] == points["move_dir_30"]).all()
        )
        # 낮은 임계값일수록 라벨 봉이 줄지 않는다(단조 포함 관계는 아님 — 최소한
        # 30% leg(+100%/−40%)는 모든 임계값에서 잡힌다)
        self.assertGreater(int((points["move_dir_10"] != 0).sum()), 0)
        self.assertGreater(int((points["move_dir_50"] != 0).sum()), 0)
        # 좌표는 bin 셀(1) 안으로 클립
        self.assertTrue(((points["x"] > 0.5) & (points["x"] < 1.5)).all())
        self.assertTrue(((points["y"] > 0.5) & (points["y"] < 1.5)).all())


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

    def test_short_hedge_adds_spy_short_in_market_downtrend(self) -> None:
        from profile_sizing.portfolio import simulate_portfolio
        down = pd.Series(np.linspace(100.0, 50.0, len(self.prices)),
                         index=self.prices.index)
        sim = simulate_portfolio(
            self.scores, self.prices, self.cfg,
            top_k=2, rebal_freq="monthly",
            market_close=down, market_ma_len=50, market_off_scale=0.5,
            short_hedge_close=down, short_hedge_symbol="SPY",
            short_hedge_ratio=0.5, short_hedge_max=0.25,
        )

        forecast = sim["forecast"]
        self.assertGreater(float(forecast["short_exposure"].max()), 0.0)
        self.assertLess(
            float(forecast["net_exposure"].mean()),
            float(forecast["stock_exposure"].mean()),
        )
        self.assertLessEqual(
            float(sim["short_target_weights"]["SPY_SHORT"].max()), 0.25 + 1e-9
        )
        self.assertIn(-1, set(sim["trades"]["direction"].astype(int)))

    def test_short_hedge_trailing_take_profit_closes_short(self) -> None:
        from profile_sizing.portfolio import simulate_portfolio
        idx = pd.date_range("2026-01-01", periods=12, freq="1D")
        scores = pd.DataFrame({
            "A": [0.6] * len(idx),
            "B": [0.5] * len(idx),
        }, index=idx)
        prices = pd.DataFrame({
            "A": [100.0] * len(idx),
            "B": [100.0] * len(idx),
        }, index=idx)
        hedge = pd.Series(
            [100.0, 99.0, 95.0, 90.0, 85.0, 88.0,
             92.0, 94.0, 93.0, 92.0, 91.0, 90.0],
            index=idx,
        )

        sim = simulate_portfolio(
            scores, prices, self.cfg,
            top_k=2, rebal_freq="daily",
            market_close=hedge, market_ma_len=2, market_off_scale=0.5,
            short_hedge_close=hedge, short_hedge_symbol="SPY",
            short_hedge_ratio=0.5, short_hedge_max=0.25,
            short_hedge_trailing_pct=0.05,
        )

        short_trades = sim["trades"][sim["trades"]["direction"] == -1]
        self.assertIn("trailing_take_profit", set(short_trades["exit_reason"]))
        trailing_exit = short_trades[
            short_trades["exit_reason"] == "trailing_take_profit"
        ].iloc[0]
        self.assertGreater(
            float(trailing_exit["net_return"]),
            0.0,
        )

    def test_rsi_filter_reduces_exposure_when_rsi_ma_below_threshold(self) -> None:
        from profile_sizing.portfolio import simulate_portfolio
        # 단조 하락 RSI 기준 시리즈 → warmup 이후 RSI MA가 50 미만 → 노출 축소.
        down = pd.Series(np.linspace(100.0, 50.0, len(self.prices)),
                         index=self.prices.index)
        off = simulate_portfolio(self.scores, self.prices, self.cfg,
                                 top_k=2, rebal_freq="monthly")
        on = simulate_portfolio(self.scores, self.prices, self.cfg,
                                top_k=2, rebal_freq="monthly",
                                rsi_close=down, rsi_len=14, rsi_ma_len=14,
                                rsi_threshold=50.0, rsi_below_scale=0.25)
        self.assertLess(float(on["forecast"]["stock_exposure"].mean()),
                        float(off["forecast"]["stock_exposure"].mean()))
        self.assertIn("rsi_ma", on["forecast"].columns)


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
