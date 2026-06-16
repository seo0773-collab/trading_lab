#!/usr/bin/env python
"""DI Kalman M/W strategy backtest runner (plan.txt).

Standalone research script (plan 16: not wired into research_adapter).

Usage:
    python scripts/di_kalman_mw/run.py --synthetic --combo A
    python scripts/di_kalman_mw/run.py --data data/raw/BTCUSDT_4h.parquet \
        --symbol BTCUSDT --timeframe 4h --combo B

Phase 4 sensitivity examples (plan 15):
    python scripts/di_kalman_mw/run.py --synthetic --combo A --cost-mult 2.0
    for m in 0.5 1.0 1.5; do \
        python scripts/di_kalman_mw/run.py --synthetic --reversal-mult $m; done
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from di_kalman_mw.backtest import run_backtest  # noqa: E402
from di_kalman_mw.config import (  # noqa: E402
    StrategyConfig, bars_per_year, combo_config,
)
from di_kalman_mw.dmi import atr, dmi, kalman_di  # noqa: E402
from di_kalman_mw.events import build_events  # noqa: E402
from di_kalman_mw.expectation import (  # noqa: E402
    PriceExpectationModel, evaluate_price_expectation, expectation_lookup,
    predict_price_frame,
)
from di_kalman_mw.extreme_transition import (  # noqa: E402
    build_transition_stats, completed_instances_for_split,
    enumerate_instances,
)
from di_kalman_mw.extremes import extract_extremes  # noqa: E402
from di_kalman_mw.metrics import (  # noqa: E402
    compute_metrics, pressure_alignment_breakdown, sanitize, split_metrics,
)
from di_kalman_mw.pattern_dataset import (  # noqa: E402
    build_pattern_frame, completed_split_frame,
)
from di_kalman_mw.online_state import (  # noqa: E402
    OnlineStateModel, build_online_snapshots, evaluate_online_state,
)
from di_kalman_mw.signals import (  # noqa: E402
    generate_signals, p5_confirmation_index,
)
from di_kalman_mw.similarity import (  # noqa: E402
    PatternSimilarityModel, evaluate_similarity,
)
from di_kalman_mw.splits import split_labels, train_sufficiency  # noqa: E402
from di_kalman_mw.stats import (  # noqa: E402
    build_train_stats, expected_values, lookup_stats,
)

SUMMARY_METRIC_KEYS = (
    "num_trades", "total_return", "profit_factor", "win_rate",
    "expectancy", "max_drawdown", "sharpe", "avg_bars_held",
)


def _pandas_freq(timeframe: str) -> str:
    tf = timeframe.lower()
    if tf.endswith("m"):
        return tf[:-1] + "min"
    if tf.endswith("d"):
        return tf[:-1] + "D"
    if tf.endswith("w"):
        return tf[:-1] + "W"
    return tf


def make_synthetic_ohlcv(
    n: int = 9000, seed: int = 7, timeframe: str = "4h",
    start: str = "2020-01-01",
) -> pd.DataFrame:
    """Regime-cycling synthetic OHLCV for pipeline validation (plan 3)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    drift = (
        0.0035 * np.sin(2.0 * np.pi * t / 240.0)
        + 0.0008 * np.sin(2.0 * np.pi * t / 1100.0)
    )
    ret = drift + rng.normal(0.0, 0.012, n)
    close = 100.0 * np.exp(np.cumsum(ret))
    open_ = np.empty(n)
    open_[0] = 100.0
    open_[1:] = close[:-1]
    span = np.abs(rng.normal(0.0, 0.004, n))
    high = np.maximum(open_, close) * (1.0 + span)
    low = np.minimum(open_, close) * (1.0 - span)
    index = pd.date_range(
        start, periods=n, freq=_pandas_freq(timeframe), tz="UTC"
    )
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.uniform(10.0, 1000.0, n),
        },
        index=index,
    )


def load_data(path: Path) -> pd.DataFrame:
    if path.suffix in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    df.columns = [str(c).lower() for c in df.columns]
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")
    else:
        df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    missing = {"open", "high", "low", "close"} - set(df.columns)
    if missing:
        raise ValueError(f"missing OHLC columns: {sorted(missing)}")
    return df


def events_features_frame(df, events, stats, cfg, symbol, timeframe, labels):
    """plan 5 feature table, one row per pattern event."""
    rows = []
    for ev in events:
        st = lookup_stats(stats, ev.direction, ev.pressure_aligned, cfg.stats)
        raw = adjusted = float("nan")
        if st is not None:
            raw, adjusted = expected_values(st, ev.pressure_rr_factor, cfg.costs)
        row = {
            "symbol": symbol,
            "timeframe": timeframe,
            "event_time": df.index[ev.event_idx],
            "event_idx": ev.event_idx,
            "tier": ev.tier,
            "plus_pattern_type": ev.plus_pattern,
            "minus_pattern_type": ev.minus_pattern,
            "plus_shape": ev.plus_shape,
            "minus_shape": ev.minus_shape,
            "setup_shape": ev.setup_shape,
            "setup_width_ratio": ev.setup_width_ratio,
        }
        for k in range(4):
            row[f"plus_p{k + 1}_value"] = ev.plus_p[k].value
        for k in range(4):
            row[f"minus_p{k + 1}_value"] = ev.minus_p[k].value
        row.update({
            "plus_extreme_mean_4": ev.plus_extreme_mean_4,
            "minus_extreme_mean_4": ev.minus_extreme_mean_4,
            "di_pressure_spread": ev.di_pressure_spread,
            "long_pressure_score": ev.long_pressure_score,
            "short_pressure_score": ev.short_pressure_score,
            "long_rr_factor": ev.long_rr_factor,
            "short_rr_factor": ev.short_rr_factor,
            "predicted_direction": ev.direction,
            "pressure_aligned": ev.pressure_aligned,
            "pressure_score": ev.pressure_score,
            "pressure_rr_factor": ev.pressure_rr_factor,
            "raw_expected_value": raw,
            "pressure_adjusted_expected_value": adjusted,
            "split": labels[ev.event_idx],
        })
        rows.append(row)
    return pd.DataFrame(rows)


def events_frame(df, events, labels):
    rows = []
    for ev in events:
        row = {
            "event_time": df.index[ev.event_idx],
            "event_idx": ev.event_idx,
            "direction": ev.direction,
            "tier": ev.tier,
            "plus_pattern": ev.plus_pattern,
            "minus_pattern": ev.minus_pattern,
            "setup_shape": ev.setup_shape,
            "split": labels[ev.event_idx],
        }
        for side, points in (("plus", ev.plus_p), ("minus", ev.minus_p)):
            for k, ext in enumerate(points):
                row[f"{side}_p{k + 1}_idx"] = ext.idx
                row[f"{side}_p{k + 1}_conf_idx"] = ext.confirmation_idx
                row[f"{side}_p{k + 1}_value"] = ext.value
        rows.append(row)
    return pd.DataFrame(rows)


def signals_frame(df, signals_by_variant, symbol, timeframe):
    rows = []
    for sigs in signals_by_variant.values():
        for s in sigs:
            rows.append({
                "timestamp": df.index[s.signal_idx],
                "bar_index": s.signal_idx,
                "symbol": symbol,
                "timeframe": timeframe,
                "entry_variant": s.entry_variant,
                "predicted_direction": s.direction,
                "signal": s.signal,
                "tier": s.event.tier,
                "setup_shape": s.event.setup_shape,
                "confidence": s.confidence,
                "raw_expected_value": s.raw_expected_value,
                "pressure_adjusted_expected_value": s.pressure_adjusted_expected_value,
                "pressure_score": s.event.pressure_score,
                "pressure_rr_factor": s.event.pressure_rr_factor,
                "continuation_score": s.continuation_score,
                "transition_bucket": s.transition_bucket,
                "similarity_expected_return": s.similarity_expected_return,
                "similarity_ev_lower_bound": s.similarity_ev_lower_bound,
                "similarity_effective_n": s.similarity_effective_n,
                "similarity_confidence": s.similarity_confidence,
                "similarity_fallback": s.similarity_fallback,
                "entry_price_candidate": s.entry_price_candidate,
                "stop_price_candidate": s.stop_price_candidate,
                "take_profit_candidate": s.take_profit_candidate,
                "filter_reason": s.filter_reason,
                "stats_bucket": s.stats_bucket,
            })
    return pd.DataFrame(rows)


def run_pipeline(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    cfg: StrategyConfig,
    outdir: Path | None = None,
    combo: str = "",
) -> tuple[dict, dict]:
    """Full Phase 1-3 pipeline. Returns (metrics, artifacts)."""
    ind = cfg.indicators
    plus_di, minus_di = dmi(df, ind.di_len)
    atr_series = atr(df, ind.atr_len)
    plus_kalman = kalman_di(plus_di, ind.kalman_q, ind.kalman_r)
    minus_kalman = kalman_di(minus_di, ind.kalman_q, ind.kalman_r)
    plus_ext = extract_extremes(plus_kalman, cfg.extremes)
    minus_ext = extract_extremes(minus_kalman, cfg.extremes)
    events = build_events(plus_ext, minus_ext, cfg.patterns)

    labels = split_labels(len(df), cfg.split)
    train_end = int(np.flatnonzero(labels != "train")[0])
    train_events = [ev for ev in events if labels[ev.event_idx] == "train"]
    suff = train_sufficiency(
        timeframe, df.index[labels == "train"], len(train_events)
    )
    stats_by_variant = {
        "p4": build_train_stats(
            df,
            atr_series,
            train_events,
            cfg.stats,
            outcome_end_exclusive=train_end,
            entry_variant="p4",
        ),
        "p5": build_train_stats(
            df,
            atr_series,
            train_events,
            cfg.stats,
            decision_index=lambda ev: p5_confirmation_index(
                ev, plus_ext, minus_ext
            ),
            outcome_end_exclusive=train_end,
            entry_variant="p5",
        ),
    }

    # plan 6/7: next-extreme continuation stats, fit on train instances only,
    # used to tilt EV and the pressure_rr take-profit at signal time.
    band = cfg.patterns.parallel_band
    trans_instances = (
        enumerate_instances(plus_ext, "plus", cfg.patterns.strict, band)
        + enumerate_instances(minus_ext, "minus", cfg.patterns.strict, band)
    )
    trans_train = completed_instances_for_split(
        trans_instances, labels, "train"
    )
    transition_stats = build_transition_stats(
        trans_train, min_bucket=cfg.signal.transition_min_bucket
    )
    pattern_frame = build_pattern_frame(
        df,
        atr_series,
        trans_instances,
        plus_ext,
        minus_ext,
        labels,
    )
    similarity_evaluation = {
        "n_train": 0,
        "n_evaluated": 0,
        "status": "insufficient_data",
    }
    similarity_train = pd.DataFrame()
    similarity_validation = pd.DataFrame()
    try:
        similarity_train = completed_split_frame(pattern_frame, "train")
        similarity_validation = completed_split_frame(
            pattern_frame, "validation"
        )
        similarity_model = PatternSimilarityModel().fit(similarity_train)
        similarity_evaluation = {
            "n_train": len(similarity_train),
            "status": "evaluated",
            **evaluate_similarity(similarity_model, similarity_validation),
        }
    except ValueError:
        pass
    price_evaluation = {
        "n_train": 0,
        "n_evaluated": 0,
        "status": "insufficient_data",
    }
    price_predictions = pd.DataFrame()
    try:
        price_model = PriceExpectationModel(cfg.similarity_ev).fit(
            similarity_train
        )
        price_evaluation = {
            "n_train": len(similarity_train),
            "status": "evaluated",
            **evaluate_price_expectation(
                price_model, similarity_validation, cfg.costs
            ),
        }
        price_predictions = predict_price_frame(
            price_model,
            pattern_frame,
            cfg.costs,
        )
    except ValueError:
        pass
    online_snapshots = pd.DataFrame()
    online_evaluation = {
        "n_train": 0,
        "n_evaluated": 0,
        "status": "disabled",
    }
    online_decisions = pd.DataFrame()
    if cfg.online.enabled:
        online_snapshots = build_online_snapshots(
            df,
            plus_kalman,
            minus_kalman,
            pattern_frame,
            labels,
        )
        try:
            online_train = online_snapshots[
                online_snapshots["split"] == "train"
            ]
            online_validation = online_snapshots[
                online_snapshots["split"] == "validation"
            ]
            online_model = OnlineStateModel(cfg.online).fit(online_train)
            evaluation, online_decisions = evaluate_online_state(
                online_model, online_validation, cfg.costs
            )
            online_evaluation = {
                "n_train": len(online_train),
                "status": "evaluated",
                **evaluation,
            }
        except (KeyError, ValueError):
            online_evaluation["status"] = "insufficient_data"
    similarity_expectations = expectation_lookup(price_predictions)

    bpy = bars_per_year(timeframe)
    runs: dict[str, dict] = {}
    for variant in ("p4", "p5"):
        sigs = generate_signals(
            df, atr_series, events, plus_ext, minus_ext, variant,
            cfg.signal, cfg.exits, cfg.stats, stats_by_variant[variant],
            cfg.costs,
            transition_stats,
            similarity_expectations,
            cfg.similarity_ev,
        )
        trades, equity = run_backtest(
            df, atr_series, sigs, events, cfg.exits, cfg.costs,
            cfg.direction_mode, labels, variant,
        )
        runs[variant] = {
            "signals": sigs,
            "trades": trades,
            "equity": equity,
            "metrics": compute_metrics(trades, equity["bar_return"], bpy),
        }

    primary = cfg.signal.entry_variant
    stats = stats_by_variant[primary]
    trades = runs[primary]["trades"]
    equity = runs[primary]["equity"]
    by_split = split_metrics(trades, equity, bpy)

    metrics: dict = {
        "symbol": symbol,
        "timeframe": timeframe,
        "combo": combo,
        "entry_variant": primary,
        "insufficient_train_data": suff["insufficient_train_data"],
        "train_sufficiency": suff,
        **runs[primary]["metrics"],
        "p4_variant_return": runs["p4"]["metrics"]["total_return"],
        "p5_variant_return": runs["p5"]["metrics"]["total_return"],
        "p4_variant_num_trades": runs["p4"]["metrics"]["num_trades"],
        "p5_variant_num_trades": runs["p5"]["metrics"]["num_trades"],
        "train_metrics": by_split["train"],
        "validation_metrics": by_split["validation"],
        "test_metrics": by_split["test"],
        "pressure_alignment": pressure_alignment_breakdown(trades),
        "n_events": len(events),
        "train_stats": stats,
        "train_stats_by_variant": stats_by_variant,
        "similarity_evaluation": similarity_evaluation,
        "price_expectation_evaluation": price_evaluation,
        "online_evaluation": online_evaluation,
        "config": dataclasses.asdict(cfg),
        # plan 7A: train-split metrics are in-sample because train stats
        # are estimated from the same window.
        "train_metrics_in_sample": True,
    }
    val_m = by_split["validation"]
    test_m = by_split["test"]
    metrics["acceptance"] = {
        "validation_profit_factor_gt_1_2": (val_m["profit_factor"] or 0) > 1.2,
        "validation_num_trades_ge_50": val_m["num_trades"] >= 50,
        "test_profit_factor_gt_1_0": (test_m["profit_factor"] or 0) > 1.0,
        "test_expectancy_gt_0": (test_m["expectancy"] or 0) > 0,
    }

    artifacts = {
        "events": events,
        "signals": {v: runs[v]["signals"] for v in runs},
        "trades": trades,
        "equity": equity,
        "stats": stats,
        "transition_stats": transition_stats,
        "pattern_dataset": pattern_frame,
        "similarity_evaluation": similarity_evaluation,
        "price_expectation_evaluation": price_evaluation,
        "price_expectations": price_predictions,
        "online_snapshots": online_snapshots,
        "online_evaluation": online_evaluation,
        "online_decisions": online_decisions,
        "labels": labels,
        "plus_extremes": plus_ext,
        "minus_extremes": minus_ext,
        "plus_kalman": plus_kalman,
        "minus_kalman": minus_kalman,
        "atr": atr_series,
    }

    if outdir is not None:
        outdir.mkdir(parents=True, exist_ok=True)
        prefix = outdir / f"{symbol}_{timeframe}"
        events_features_frame(
            df, events, stats, cfg, symbol, timeframe, labels
        ).to_csv(f"{prefix}_features.csv", index=False)
        events_frame(df, events, labels).to_csv(
            f"{prefix}_events.csv", index=False
        )
        signals_frame(df, artifacts["signals"], symbol, timeframe).to_csv(
            f"{prefix}_signals.csv", index=False
        )
        trades.to_csv(f"{prefix}_trades.csv", index=False)
        equity.to_csv(f"{prefix}_equity.csv", index_label="timestamp")
        with open(f"{prefix}_metrics.json", "w", encoding="utf-8") as fh:
            json.dump(sanitize(metrics), fh, indent=2, default=str)
        with open(f"{prefix}_train_stats.json", "w", encoding="utf-8") as fh:
            json.dump(sanitize(stats), fh, indent=2)
        pattern_frame.to_csv(
            f"{prefix}_pattern_dataset.csv", index=False
        )
        with open(
            f"{prefix}_similarity_metrics.json", "w", encoding="utf-8"
        ) as fh:
            json.dump(sanitize(similarity_evaluation), fh, indent=2)
        price_predictions.to_csv(
            f"{prefix}_price_expectations.csv", index=False
        )
        with open(
            f"{prefix}_price_expectation_metrics.json",
            "w",
            encoding="utf-8",
        ) as fh:
            json.dump(sanitize(price_evaluation), fh, indent=2)
        online_decisions.to_csv(
            f"{prefix}_online_decisions.csv", index=False
        )
        with open(
            f"{prefix}_online_metrics.json", "w", encoding="utf-8"
        ) as fh:
            json.dump(sanitize(online_evaluation), fh, indent=2)
        summary_rows = []
        for split_name in ("train", "validation", "test"):
            m = by_split[split_name]
            summary_rows.append({
                "symbol": symbol,
                "timeframe": timeframe,
                "combo": combo,
                "entry_variant": primary,
                "split": split_name,
                **{k: m.get(k) for k in SUMMARY_METRIC_KEYS},
                "insufficient_train_data": suff["insufficient_train_data"],
            })
        pd.DataFrame(summary_rows).to_csv(f"{prefix}_summary.csv", index=False)

    return metrics, artifacts


def apply_overrides(cfg: StrategyConfig, args) -> StrategyConfig:
    ind_over = {
        k: v
        for k, v in {
            "kalman_q": args.kalman_q,
            "kalman_r": args.kalman_r,
        }.items()
        if v is not None
    }
    ext_over = {
        k: v
        for k, v in {
            "reversal_mult": args.reversal_mult,
            "reversal_std_window": args.reversal_window,
        }.items()
        if v is not None
    }
    cfg = dataclasses.replace(
        cfg,
        indicators=dataclasses.replace(cfg.indicators, **ind_over),
        extremes=dataclasses.replace(cfg.extremes, **ext_over),
    )
    if args.entry_variant is not None:
        cfg = dataclasses.replace(
            cfg, signal=dataclasses.replace(cfg.signal, entry_variant=args.entry_variant)
        )
    if args.cost_mult is not None:
        cfg = dataclasses.replace(
            cfg, costs=dataclasses.replace(cfg.costs, cost_mult=args.cost_mult)
        )
    if args.direction is not None:
        cfg = dataclasses.replace(cfg, direction_mode=args.direction)
    return cfg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="OHLCV parquet/csv path")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--synthetic-bars", type=int, default=9000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--timeframe", default="4h")
    parser.add_argument("--combo", default="A", help="plan 17 combo A|B|C|D")
    parser.add_argument("--entry-variant", choices=["p4", "p5"], default=None)
    parser.add_argument("--direction", choices=["long", "short", "both"], default=None)
    parser.add_argument("--kalman-q", type=float, default=None)
    parser.add_argument("--kalman-r", type=float, default=None)
    parser.add_argument("--reversal-mult", type=float, default=None)
    parser.add_argument("--reversal-window", type=int, default=None)
    parser.add_argument("--cost-mult", type=float, default=None)
    parser.add_argument(
        "--outdir", type=Path,
        default=Path(__file__).resolve().parents[2] / "reports" / "di_kalman_mw",
    )
    args = parser.parse_args(argv)

    if args.synthetic:
        df = make_synthetic_ohlcv(
            args.synthetic_bars, args.seed, args.timeframe
        )
        symbol = args.symbol if args.data else "SYNTH"
    elif args.data:
        df = load_data(args.data)
        symbol = args.symbol
    else:
        parser.error("--data or --synthetic is required")
        return 2
    if args.synthetic and not args.data:
        symbol = "SYNTH" if args.symbol == "BTCUSDT" else args.symbol

    cfg = apply_overrides(combo_config(args.combo), args)
    metrics, _ = run_pipeline(
        df, symbol, args.timeframe, cfg, args.outdir, combo=args.combo.upper()
    )

    print(
        f"[di_kalman_mw] {symbol} {args.timeframe} combo={args.combo.upper()} "
        f"variant={metrics['entry_variant']} bars={len(df)} "
        f"events={metrics['n_events']}"
    )
    if metrics["insufficient_train_data"]:
        print("  WARNING: insufficient_train_data (plan 4)")
    for split_name in ("train", "validation", "test"):
        m = metrics[f"{split_name}_metrics"]
        note = " (in-sample)" if split_name == "train" else ""
        print(
            f"  {split_name:>10}{note}: trades={m['num_trades']} "
            f"return={m['total_return']!r} pf={m['profit_factor']!r} "
            f"expectancy={m['expectancy']!r} mdd={m['max_drawdown']!r}"
        )
    print(
        f"  p4_variant_return={metrics['p4_variant_return']!r} "
        f"({metrics['p4_variant_num_trades']} trades) / "
        f"p5_variant_return={metrics['p5_variant_return']!r} "
        f"({metrics['p5_variant_num_trades']} trades)"
    )
    print(f"  acceptance={metrics['acceptance']}")
    print(f"  outputs -> {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
