"""Internal validation for the DI Kalman M/W strategy (plan.txt 3, 16).

Covers: reproducibility, no-lookahead in extreme extraction and event
generation, entry strictly after the signal bar, chronological splits,
and the same-bar stop/TP tie-break rule.
"""
from __future__ import annotations

import dataclasses
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for p in (str(ROOT), str(SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from di_kalman_mw import run as runner  # noqa: E402
from di_kalman_mw.backtest import run_backtest  # noqa: E402
from di_kalman_mw.config import (  # noqa: E402
    CostConfig, ExitConfig, ExtremeConfig, OnlineConfig, SignalConfig,
    SimilarityEvConfig, combo_config,
)
from di_kalman_mw.dmi import dmi, kalman_di  # noqa: E402
from di_kalman_mw.events import build_events  # noqa: E402
from di_kalman_mw.expectation import (  # noqa: E402
    PriceExpectationModel, evaluate_price_expectation,
)
from di_kalman_mw.extreme_transition import (  # noqa: E402
    build_transition_stats,
    completed_instances_for_split,
    enumerate_instances,
    evaluate_transition,
    lookup_transition,
)
from di_kalman_mw.extremes import Extreme, extract_extremes  # noqa: E402
from di_kalman_mw.pattern_dataset import (  # noqa: E402
    NUMERIC_FEATURE_COLUMNS,
    OUTCOME_COLUMNS,
    build_pattern_frame,
    completed_split_frame,
)
from di_kalman_mw.online_state import (  # noqa: E402
    OnlinePrediction, OnlineStateModel, build_online_snapshots,
    decide_position, evaluate_online_state,
)
from di_kalman_mw.similarity import (  # noqa: E402
    PatternSimilarityModel, SimilarityConfig, evaluate_similarity,
)
from di_kalman_mw.splits import split_labels  # noqa: E402
from di_kalman_mw.stats import build_train_stats  # noqa: E402


def research_config():
    """Combo A with filters relaxed so synthetic data produces trades."""
    cfg = combo_config("A")
    return dataclasses.replace(
        cfg,
        signal=SignalConfig(
            entry_variant="p4", pressure_score_min=0.0,
            require_positive_ev=False,
        ),
    )


class PipelineFixture:
    df = None
    plus_kalman = None
    minus_kalman = None


_SIMILARITY_FIXTURE = None


def similarity_fixture():
    global _SIMILARITY_FIXTURE
    if _SIMILARITY_FIXTURE is not None:
        return _SIMILARITY_FIXTURE
    df = runner.make_synthetic_ohlcv(6000, seed=11)
    cfg = combo_config("A")
    plus_di, minus_di = dmi(df, cfg.indicators.di_len)
    plus_k = kalman_di(
        plus_di, cfg.indicators.kalman_q, cfg.indicators.kalman_r
    )
    minus_k = kalman_di(
        minus_di, cfg.indicators.kalman_q, cfg.indicators.kalman_r
    )
    plus_ext = extract_extremes(plus_k, cfg.extremes)
    minus_ext = extract_extremes(minus_k, cfg.extremes)
    instances = (
        enumerate_instances(plus_ext, "plus")
        + enumerate_instances(minus_ext, "minus")
    )
    labels = split_labels(len(df), cfg.split)
    frame = build_pattern_frame(
        df,
        pd.Series(2.0, index=df.index),
        instances,
        plus_ext,
        minus_ext,
        labels,
    )
    _SIMILARITY_FIXTURE = {
        "df": df,
        "plus_k": plus_k,
        "minus_k": minus_k,
        "labels": labels,
        "frame": frame,
    }
    return _SIMILARITY_FIXTURE


def setUpModule():
    PipelineFixture.df = runner.make_synthetic_ohlcv(4000, seed=3)
    plus_di, minus_di = dmi(PipelineFixture.df, 14)
    PipelineFixture.plus_kalman = kalman_di(plus_di, 0.01, 1.0)
    PipelineFixture.minus_kalman = kalman_di(minus_di, 0.01, 1.0)


class ExtremeTests(unittest.TestCase):
    def test_alternation_and_confirmation_order(self):
        ext = extract_extremes(PipelineFixture.plus_kalman, ExtremeConfig())
        self.assertGreater(len(ext), 10)
        for e in ext:
            self.assertGreater(e.confirmation_idx, e.idx)
        for a, b in zip(ext, ext[1:]):
            self.assertNotEqual(a.kind, b.kind)
            self.assertLess(a.idx, b.idx)
            self.assertLess(a.confirmation_idx, b.confirmation_idx)

    def test_no_lookahead(self):
        cfg = ExtremeConfig()
        full = extract_extremes(PipelineFixture.plus_kalman, cfg)
        k = 2500
        truncated = extract_extremes(
            PipelineFixture.plus_kalman.iloc[:k], cfg
        )
        expected = [e for e in full if e.confirmation_idx < k]
        self.assertEqual(truncated, expected)


class EventTests(unittest.TestCase):
    def test_events_do_not_use_future_data(self):
        cfg = research_config()
        plus_full = extract_extremes(PipelineFixture.plus_kalman, cfg.extremes)
        minus_full = extract_extremes(PipelineFixture.minus_kalman, cfg.extremes)
        events_full = build_events(plus_full, minus_full, cfg.patterns)
        self.assertGreater(len(events_full), 0)

        k = 2500
        plus_cut = extract_extremes(
            PipelineFixture.plus_kalman.iloc[:k], cfg.extremes
        )
        minus_cut = extract_extremes(
            PipelineFixture.minus_kalman.iloc[:k], cfg.extremes
        )
        events_cut = build_events(plus_cut, minus_cut, cfg.patterns)
        expected = [
            (ev.event_idx, ev.direction, ev.plus_j, ev.minus_j,
             round(ev.pressure_score, 12))
            for ev in events_full if ev.event_idx < k
        ]
        got = [
            (ev.event_idx, ev.direction, ev.plus_j, ev.minus_j,
             round(ev.pressure_score, 12))
            for ev in events_cut
        ]
        self.assertEqual(got, expected)


class PipelineTests(unittest.TestCase):
    def test_entry_strictly_after_signal_bar(self):
        cfg = research_config()
        _, artifacts = runner.run_pipeline(
            PipelineFixture.df, "SYNTH", "4h", cfg, outdir=None
        )
        trades = artifacts["trades"]
        self.assertGreater(len(trades), 0)
        self.assertTrue((trades["entry_time"] > trades["signal_time"]).all())

    def test_reproducible(self):
        cfg = research_config()
        _, a1 = runner.run_pipeline(
            PipelineFixture.df, "SYNTH", "4h", cfg, outdir=None
        )
        _, a2 = runner.run_pipeline(
            PipelineFixture.df, "SYNTH", "4h", cfg, outdir=None
        )
        pd.testing.assert_frame_equal(a1["trades"], a2["trades"])
        pd.testing.assert_frame_equal(a1["equity"], a2["equity"])

    def test_outputs_written(self):
        cfg = research_config()
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            runner.run_pipeline(
                PipelineFixture.df, "SYNTH", "4h", cfg, outdir=outdir
            )
            for suffix in (
                "features.csv", "events.csv", "signals.csv", "trades.csv",
                "equity.csv", "metrics.json", "train_stats.json",
                "summary.csv", "pattern_dataset.csv",
                "similarity_metrics.json", "price_expectations.csv",
                "price_expectation_metrics.json", "online_decisions.csv",
                "online_metrics.json",
            ):
                self.assertTrue(
                    (outdir / f"SYNTH_4h_{suffix}").exists(), suffix
                )

    def test_pipeline_keeps_separate_p4_and_p5_training_stats(self):
        metrics, _ = runner.run_pipeline(
            PipelineFixture.df, "SYNTH", "4h", research_config(), outdir=None
        )
        self.assertEqual(
            set(metrics["train_stats_by_variant"]), {"p4", "p5"}
        )
        self.assertEqual(metrics["train_stats"]["entry_variant"], "p4")
        self.assertEqual(
            metrics["train_stats_by_variant"]["p5"]["entry_variant"], "p5"
        )


class VizTests(unittest.TestCase):
    def test_dashboard_figures_build(self):
        from di_kalman_mw.viz import (
            build_di_figure, build_equity_figure, build_price_figure,
            split_metrics_table,
        )

        cfg = research_config()
        metrics, artifacts = runner.run_pipeline(
            PipelineFixture.df, "SYNTH", "4h", cfg, outdir=None
        )
        price_fig = build_price_figure(
            PipelineFixture.df, artifacts["trades"], artifacts["labels"]
        )
        di_fig = build_di_figure(
            PipelineFixture.df, artifacts["plus_kalman"],
            artifacts["minus_kalman"], artifacts["plus_extremes"],
            artifacts["minus_extremes"], artifacts["events"],
        )
        equity_fig = build_equity_figure(artifacts["equity"])
        self.assertGreaterEqual(len(price_fig.data), 1)
        self.assertGreaterEqual(len(di_fig.data), 4)
        self.assertEqual(len(equity_fig.data), 1)
        table = split_metrics_table(metrics)
        self.assertEqual(len(table), 3)


class SplitTests(unittest.TestCase):
    def test_chronological_split(self):
        cfg = combo_config("A").split
        labels = split_labels(1000, cfg)
        self.assertEqual(list(labels[:600]), ["train"] * 600)
        self.assertEqual(list(labels[600:800]), ["validation"] * 200)
        self.assertEqual(list(labels[800:]), ["test"] * 200)

    def test_train_stats_exclude_outcome_crossing_split_boundary(self):
        index = pd.date_range(
            "2024-01-01", periods=8, freq="4h", tz="UTC"
        )
        df = pd.DataFrame(
            {
                "open": [100.0] * 8,
                "high": [100.5] * 8,
                "low": [99.5] * 8,
                "close": [100.0] * 8,
            },
            index=index,
        )
        atr_series = pd.Series(1.0, index=index)
        event = SimpleNamespace(
            event_idx=3, direction="long", pressure_aligned=True
        )
        stats = build_train_stats(
            df,
            atr_series,
            [event],
            combo_config("A").stats,
            outcome_end_exclusive=6,
            entry_variant="p4",
        )
        self.assertEqual(stats["n_candidates"], 1)
        self.assertEqual(stats["n_simulated"], 0)
        self.assertEqual(stats["n_incomplete"], 1)

    def test_transition_training_requires_p5_confirmation_in_same_split(self):
        instances = enumerate_instances(
            TransitionTests._w_then_p5(12.0), "plus"
        )
        labels = np.array(
            ["train"] * 9 + ["validation"], dtype=object
        )
        selected = completed_instances_for_split(
            instances, labels, "train"
        )
        self.assertEqual(selected, [])


class EngineTests(unittest.TestCase):
    def _engine_run(self, bars, exits):
        """Run the engine with one fabricated long signal at bar 2."""
        index = pd.date_range(
            "2024-01-01", periods=len(bars), freq="4h", tz="UTC"
        )
        df = pd.DataFrame(bars, columns=["open", "high", "low", "close"],
                          index=index)
        atr_series = pd.Series(1.0, index=index)
        event = SimpleNamespace(
            pressure_rr_factor=1.0, pressure_score=0.6, pressure_aligned=True
        )
        signal = SimpleNamespace(
            signal="long", signal_idx=2, event=event,
            raw_expected_value=0.0, pressure_adjusted_expected_value=0.0,
        )
        costs = CostConfig(fee_rate=0.0, slippage_rate=0.0)
        labels = np.array(["train"] * len(df), dtype=object)
        trades, _ = run_backtest(
            df, atr_series, [signal], [], exits, costs, "both", labels, "p4"
        )
        return trades

    def test_same_bar_stop_takes_priority_over_tp(self):
        # entry at bar 3 open=100, stop=98 (ATR 1 * mult 2), TP=102 (1R).
        # bar 4 touches both: low 97 / high 103 -> stop must fill first.
        bars = [
            [100, 100.5, 99.5, 100],
            [100, 100.5, 99.5, 100],
            [100, 100.5, 99.5, 100],  # signal bar
            [100, 100.5, 99.5, 100],  # entry bar
            [100, 103.0, 97.0, 100],  # both stop and TP touched
            [100, 100.5, 99.5, 100],
        ]
        exits = ExitConfig(
            stop_type="atr", atr_stop_mult=2.0, tp_type="fixed_r",
            rr_target=1.0, opposite_exit=False, max_hold_bars=48,
        )
        trades = self._engine_run(bars, exits)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades.iloc[0]["exit_reason"], "stop")
        self.assertAlmostEqual(trades.iloc[0]["exit_price"], 98.0)

    def test_gap_through_stop_fills_at_open(self):
        bars = [
            [100, 100.5, 99.5, 100],
            [100, 100.5, 99.5, 100],
            [100, 100.5, 99.5, 100],  # signal bar
            [100, 100.5, 99.5, 100],  # entry bar (stop = 98)
            [95, 96.0, 94.0, 95],     # gap open below stop
        ]
        exits = ExitConfig(
            stop_type="atr", atr_stop_mult=2.0, tp_type="none",
            opposite_exit=False, max_hold_bars=48,
        )
        trades = self._engine_run(bars, exits)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades.iloc[0]["exit_reason"], "stop")
        self.assertAlmostEqual(trades.iloc[0]["exit_price"], 95.0)

    def test_time_stop_exits_next_open(self):
        bars = [[100, 100.5, 99.5, 100]] * 10
        exits = ExitConfig(
            stop_type="atr", atr_stop_mult=10.0, tp_type="none",
            opposite_exit=False, max_hold_bars=2,
        )
        trades = self._engine_run(bars, exits)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades.iloc[0]["exit_reason"], "time_stop")
        # entry bar 3, time stop evaluated at close of bar 5, filled at 6.
        self.assertEqual(trades.iloc[0]["bars_held"], 3)

    def test_position_is_closed_before_next_split(self):
        bars = [[100, 100.5, 99.5, 100]] * 8
        index = pd.date_range(
            "2024-01-01", periods=len(bars), freq="4h", tz="UTC"
        )
        df = pd.DataFrame(
            bars, columns=["open", "high", "low", "close"], index=index
        )
        atr_series = pd.Series(1.0, index=index)
        event = SimpleNamespace(
            pressure_rr_factor=1.0, pressure_score=0.6,
            pressure_aligned=True,
        )
        signal = SimpleNamespace(
            signal="long", signal_idx=2, event=event,
            raw_expected_value=0.0, pressure_adjusted_expected_value=0.0,
        )
        labels = np.array(
            ["train"] * 5 + ["validation"] * 3, dtype=object
        )
        trades, equity = run_backtest(
            df, atr_series, [signal], [],
            ExitConfig(
                atr_stop_mult=10.0, tp_type="none",
                opposite_exit=False, max_hold_bars=48,
            ),
            CostConfig(fee_rate=0.0, slippage_rate=0.0),
            "both", labels, "p4",
        )
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades.iloc[0]["exit_reason"], "split_boundary")
        self.assertEqual(trades.iloc[0]["exit_time"], index[4])
        self.assertTrue((equity.loc[index[5]:, "bar_return"] == 0.0).all())


class TransitionTests(unittest.TestCase):
    """다음 극점(P5) 값 변위 확률구조 (extreme_transition)."""

    @staticmethod
    def _w_then_p5(p5_value: float) -> list[Extreme]:
        # W 패턴(P1..P4) = (L,H,L,H), P3>P1, 그 다음 교대 극점 P5는 L.
        return [
            Extreme(0, "L", 10.0, 1),
            Extreme(2, "H", 14.0, 3),
            Extreme(4, "L", 11.0, 5),
            Extreme(6, "H", 15.0, 7),
            Extreme(8, "L", p5_value, 9),
        ]

    def test_classifies_and_normalizes_by_mean_leg(self):
        inst = enumerate_instances(self._w_then_p5(12.0), "plus")
        self.assertEqual(len(inst), 1)
        x = inst[0]
        self.assertEqual(x.pattern, "W")
        # mean_leg = (|14-10|+|11-14|+|15-11|)/3 = 11/3
        self.assertAlmostEqual(x.mean_leg, 11.0 / 3.0, places=6)
        # dv = 12 - 15 = -3 ; dv_norm = -3 / (11/3)
        self.assertAlmostEqual(x.dv, -3.0, places=6)
        self.assertAlmostEqual(x.dv_norm, -3.0 / (11.0 / 3.0), places=6)

    def test_continuation_definition(self):
        # W: P5 > P3(=11) 이면 higher-low → continuation True.
        self.assertTrue(enumerate_instances(self._w_then_p5(12.0), "plus")[0].continuation)
        self.assertFalse(enumerate_instances(self._w_then_p5(9.0), "plus")[0].continuation)

    def test_m_pattern_displacement_is_positive(self):
        m = [
            Extreme(0, "H", 15.0, 1),
            Extreme(2, "L", 11.0, 3),
            Extreme(4, "H", 14.0, 5),  # P3 < P1 → M
            Extreme(6, "L", 10.0, 7),
            Extreme(8, "H", 13.0, 9),  # P5 = H, dv = +3
        ]
        x = enumerate_instances(m, "minus")[0]
        self.assertEqual(x.pattern, "M")
        self.assertGreater(x.dv_norm, 0.0)
        self.assertTrue(x.continuation)  # P5(13) < P3(14) → lower-high

    def test_causality_decision_before_outcome(self):
        # 모든 has_p5 인스턴스는 P4 확정이 P5 확정보다 엄격히 앞선다.
        for x in enumerate_instances(self._w_then_p5(12.0), "plus"):
            if x.has_p5:
                self.assertLess(x.p4_conf_idx, x.p5_conf_idx)

    def test_features_independent_of_p5(self):
        # P5 값이 달라도 버킷팅 특징(P1..P4)은 동일해야 한다 (look-ahead 차단).
        a = enumerate_instances(self._w_then_p5(12.0), "plus")[0]
        b = enumerate_instances(self._w_then_p5(7.0), "plus")[0]
        self.assertEqual(a.features, b.features)
        self.assertNotEqual(a.dv_norm, b.dv_norm)

    def test_lookup_falls_back_when_undersampled(self):
        # 표본이 적으면 정밀 버킷·패턴을 건너뛰고 전역으로 폴백, 더 적으면 None.
        train = enumerate_instances(self._w_then_p5(12.0), "plus")
        stats = build_transition_stats(train, min_bucket=30, min_global=10)
        self.assertIsNone(lookup_transition(stats, "W", train[0].features))
        stats_loose = build_transition_stats(train, min_bucket=1, min_global=1)
        hit = lookup_transition(stats_loose, "W", train[0].features)
        self.assertIsNotNone(hit)

    def test_build_and_evaluate_on_synthetic(self):
        df = runner.make_synthetic_ohlcv(6000, seed=5)
        cfg = combo_config("A")
        plus_di, minus_di = dmi(df, cfg.indicators.di_len)
        plus_k = kalman_di(plus_di, cfg.indicators.kalman_q, cfg.indicators.kalman_r)
        minus_k = kalman_di(minus_di, cfg.indicators.kalman_q, cfg.indicators.kalman_r)
        plus_ext = extract_extremes(plus_k, cfg.extremes)
        minus_ext = extract_extremes(minus_k, cfg.extremes)
        labels = split_labels(len(df), cfg.split)
        instances = (
            enumerate_instances(plus_ext, "plus")
            + enumerate_instances(minus_ext, "minus")
        )
        train = [x for x in instances if labels[x.p4_conf_idx] == "train"]
        val = [x for x in instances if labels[x.p4_conf_idx] == "validation"]
        self.assertGreater(len(train), 0)
        stats = build_transition_stats(train, min_bucket=10)
        self.assertIsNotNone(stats["global"])
        self.assertEqual(stats["n_train"], len(train))
        evaluation = evaluate_transition(stats, val)
        self.assertEqual(
            evaluation["n_evaluated"],
            sum(1 for x in val if x.has_p5 and np.isfinite(x.dv_norm)),
        )


class PatternDatasetTests(unittest.TestCase):
    @staticmethod
    def _frame_for_p5(p5_value: float) -> pd.DataFrame:
        extremes = TransitionTests._w_then_p5(p5_value)
        instances = enumerate_instances(extremes, "plus")
        index = pd.date_range(
            "2024-01-01", periods=12, freq="4h", tz="UTC"
        )
        close = np.linspace(100.0, 111.0, 12)
        df = pd.DataFrame(
            {
                "open": close,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
            },
            index=index,
        )
        labels = np.array(["train"] * len(df), dtype=object)
        return build_pattern_frame(
            df,
            pd.Series(2.0, index=index),
            instances,
            extremes,
            [],
            labels,
        )

    def test_p5_change_does_not_change_p1_p4_features(self):
        a = self._frame_for_p5(12.0).iloc[0]
        b = self._frame_for_p5(7.0).iloc[0]
        pd.testing.assert_series_equal(
            a.loc[list(NUMERIC_FEATURE_COLUMNS)],
            b.loc[list(NUMERIC_FEATURE_COLUMNS)],
            check_names=False,
        )
        self.assertNotEqual(a["p5_dv_norm"], b["p5_dv_norm"])

    def test_feature_and_outcome_contracts_are_disjoint(self):
        self.assertTrue(
            set(NUMERIC_FEATURE_COLUMNS).isdisjoint(OUTCOME_COLUMNS)
        )

    def test_fixed_horizon_outcome_does_not_cross_split(self):
        index = pd.date_range(
            "2024-01-01", periods=60, freq="4h", tz="UTC"
        )
        close = np.linspace(100.0, 159.0, 60)
        df = pd.DataFrame(
            {
                "open": close,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
            },
            index=index,
        )
        extremes = TransitionTests._w_then_p5(12.0)
        all_train = np.array(["train"] * 60, dtype=object)
        all_train_frame = build_pattern_frame(
            df,
            pd.Series(2.0, index=index),
            enumerate_instances(extremes, "plus"),
            extremes,
            [],
            all_train,
        )
        self.assertTrue(
            pd.notna(all_train_frame.iloc[0]["directional_return_20"])
        )
        labels = np.array(
            ["train"] * 20 + ["validation"] * 40, dtype=object
        )
        boundary_frame = build_pattern_frame(
            df,
            pd.Series(2.0, index=index),
            enumerate_instances(extremes, "plus"),
            extremes,
            [],
            labels,
        )
        self.assertTrue(
            pd.isna(boundary_frame.iloc[0]["directional_return_20"])
        )


class SimilarityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture = similarity_fixture()
        cls.frame = fixture["frame"]

    def test_fit_and_predict_exposes_confidence(self):
        train = completed_split_frame(self.frame, "train")
        validation = completed_split_frame(self.frame, "validation")
        model = PatternSimilarityModel(
            SimilarityConfig(neighbors=20, min_neighbors=5)
        ).fit(train)
        prediction = model.predict_one(validation.iloc[0])
        for key in (
            "prediction_median", "q10", "q90", "p_continuation",
            "effective_n", "nearest_distance", "confidence",
            "model_fallback",
        ):
            self.assertIn(key, prediction)
        self.assertGreaterEqual(prediction["confidence"], 0.0)
        self.assertLessEqual(prediction["confidence"], 1.0)

    def test_evaluation_uses_only_complete_validation_rows(self):
        train = completed_split_frame(self.frame, "train")
        validation = completed_split_frame(self.frame, "validation")
        model = PatternSimilarityModel(
            SimilarityConfig(neighbors=20, min_neighbors=5)
        ).fit(train)
        evaluation = evaluate_similarity(model, validation)
        self.assertEqual(evaluation["n_evaluated"], len(validation))


class ExpectationTests(unittest.TestCase):
    def test_price_expectation_uses_net_returns_and_lower_bound(self):
        frame = similarity_fixture()["frame"]
        train = completed_split_frame(frame, "train")
        validation = completed_split_frame(
            frame, "validation"
        )
        model = PriceExpectationModel(
            SimilarityEvConfig(
                neighbors=20, min_neighbors=5, entry_margin=-1.0
            )
        ).fit(train)
        prediction = model.predict_one(
            validation.iloc[0],
            CostConfig(fee_rate=0.001, slippage_rate=0.001),
        )
        self.assertIn("expected_net_return", prediction)
        self.assertIn("ev_lower_bound", prediction)
        self.assertLessEqual(
            prediction["net_q25"], prediction["net_q75"]
        )
        evaluation = evaluate_price_expectation(
            model, validation, CostConfig()
        )
        self.assertEqual(evaluation["n_evaluated"], len(validation))

    def test_train_prediction_excludes_its_own_outcome(self):
        frame = similarity_fixture()["frame"]
        train = completed_split_frame(frame, "train")
        model = PriceExpectationModel(
            SimilarityEvConfig(neighbors=20, min_neighbors=5)
        ).fit(train)
        row = train.iloc[0]
        neighbors, _, _ = model._model.neighbor_sample(row)
        self.assertNotIn(
            int(row["instance_id"]),
            set(neighbors["instance_id"].astype(int)),
        )


class OnlineStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture = similarity_fixture()
        cls.snapshots = build_online_snapshots(
            fixture["df"],
            fixture["plus_k"],
            fixture["minus_k"],
            fixture["frame"],
            fixture["labels"],
        )

    def test_snapshots_and_outcomes_stay_in_one_split(self):
        self.assertGreater(len(self.snapshots), 0)
        self.assertFalse(self.snapshots["remaining_directional_return"].isna().any())

    def test_online_model_evaluates_validation_without_refitting(self):
        train = self.snapshots[self.snapshots["split"] == "train"]
        validation = self.snapshots[
            self.snapshots["split"] == "validation"
        ]
        model = OnlineStateModel(
            OnlineConfig(neighbors=20, min_neighbors=5)
        ).fit(train)
        evaluation, decisions = evaluate_online_state(
            model, validation, CostConfig()
        )
        self.assertEqual(evaluation["n_evaluated"], len(decisions))
        self.assertGreater(len(decisions), 0)
        self.assertTrue(
            set(decisions["decision"]).issubset({"hold", "exit", "reverse"})
        )

    def test_decision_requires_persistence_and_margin(self):
        config = OnlineConfig(
            min_neighbors=10, min_confidence=0.2,
            confirm_bars=2, reversal_margin=0.01,
        )
        keep = OnlinePrediction(
            expected_net_return=-0.01, lower_bound=-0.02,
            upper_bound=0.0, effective_n=20, confidence=0.8,
            nearest_distance=0.1,
        )
        reverse = OnlinePrediction(
            expected_net_return=0.05, lower_bound=0.04,
            upper_bound=0.06, effective_n=20, confidence=0.8,
            nearest_distance=0.1,
        )
        self.assertEqual(
            decide_position(keep, config, adverse_bars=1), "hold"
        )
        self.assertEqual(
            decide_position(
                keep, config, reverse=reverse,
                switch_cost=0.005, adverse_bars=2,
            ),
            "reverse",
        )


if __name__ == "__main__":
    unittest.main()
