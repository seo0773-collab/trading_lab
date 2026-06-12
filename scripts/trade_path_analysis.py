#!/usr/bin/env python3
"""Analyze mark-to-market PnL paths after entry and conservative short rules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from conf_filter_backtest import build_signals
from strategy_execution import (
    ExecutionConfig,
    chronological_splits,
    rolling_conf_threshold,
    run_execution,
    summarize_execution,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_DIR = ROOT / "reports" / "generalization"
CHECKPOINTS = (1, 4, 12, 24, 36, 48, 60, 72)


def trade_pnl_paths(
    forecast: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    fee_bps: float,
    horizon: int,
) -> pd.DataFrame:
    """Return per-trade net PnL at each post-entry hour.

    Before exit, PnL is marked at the relevant hourly close with round-trip
    costs deducted. After exit, realized net PnL is carried forward so every
    checkpoint retains the full entry cohort.
    """
    if trades.empty:
        return pd.DataFrame(columns=[
            "asset", "entry_time", "direction", "hour", "net_return",
            "state", "exit_reason",
        ])
    fee = fee_bps / 1e4
    locations = {timestamp: i for i, timestamp in enumerate(forecast.index)}
    rows = []
    for trade in trades.itertuples(index=False):
        entry_time = pd.Timestamp(trade.entry_time)
        exit_time = pd.Timestamp(trade.exit_time)
        entry_i = locations[entry_time]
        exit_i = locations[exit_time]
        holding_bars = int(exit_i - entry_i)
        for hour in range(1, horizon + 1):
            close_i = entry_i + hour - 1
            checkpoint_time = (
                forecast.index[close_i] if close_i < len(forecast) else None
            )
            if close_i >= len(forecast) or hour >= holding_bars:
                net_return = float(trade.net_return)
                state = "realized"
            else:
                mark = float(forecast["close"].iloc[close_i])
                size = float(getattr(trade, "position_size", 1.0))
                gross = size * int(trade.direction) * (
                    mark / float(trade.entry_price) - 1.0
                )
                net_return = gross - size * 2 * fee
                state = "active"
            rows.append({
                "asset": trade.asset,
                "entry_time": entry_time,
                "direction": int(trade.direction),
                "hour": hour,
                "checkpoint_time": checkpoint_time,
                "net_return": net_return,
                "state": state,
                "exit_reason": trade.exit_reason,
            })
    return pd.DataFrame(rows)


def summarize_paths(paths: pd.DataFrame) -> pd.DataFrame:
    records = []
    groups = [
        ("ALL", paths),
        ("LONG", paths[paths["direction"] > 0]),
        ("SHORT", paths[paths["direction"] < 0]),
    ]
    for side, frame in groups:
        for hour in CHECKPOINTS:
            values = frame.loc[frame["hour"] == hour, "net_return"]
            if values.empty:
                continue
            records.append({
                "side": side,
                "hour": hour,
                "trades": int(len(values)),
                "avg_pnl_pct": float(values.mean() * 100),
                "median_pnl_pct": float(values.median() * 100),
                "positive_pct": float((values > 0).mean() * 100),
                "p25_pct": float(values.quantile(0.25) * 100),
                "p75_pct": float(values.quantile(0.75) * 100),
            })
    return pd.DataFrame(records)


def excursion_summary(paths: pd.DataFrame) -> pd.DataFrame:
    per_trade = paths.groupby(
        ["asset", "entry_time", "direction"], as_index=False
    ).agg(
        best_pnl_pct=("net_return", lambda x: float(x.max() * 100)),
        worst_pnl_pct=("net_return", lambda x: float(x.min() * 100)),
        final_pnl_pct=("net_return", lambda x: float(x.iloc[-1] * 100)),
        best_hour=("net_return", lambda x: int(x.reset_index(drop=True).idxmax() + 1)),
        worst_hour=("net_return", lambda x: int(x.reset_index(drop=True).idxmin() + 1)),
    )
    records = []
    for side, frame in (
        ("ALL", per_trade),
        ("LONG", per_trade[per_trade["direction"] > 0]),
        ("SHORT", per_trade[per_trade["direction"] < 0]),
    ):
        records.append({
            "side": side,
            "trades": int(len(frame)),
            "avg_best_pnl_pct": float(frame["best_pnl_pct"].mean()),
            "avg_worst_pnl_pct": float(frame["worst_pnl_pct"].mean()),
            "median_best_hour": float(frame["best_hour"].median()),
            "median_worst_hour": float(frame["worst_hour"].median()),
            "gave_back_profit_pct": float(
                ((frame["best_pnl_pct"] > 0) & (frame["final_pnl_pct"] <= 0)).mean() * 100
            ),
        })
    return pd.DataFrame(records)


def summarize_asset_paths(paths: pd.DataFrame) -> pd.DataFrame:
    records = []
    for asset in sorted(paths["asset"].unique()):
        asset_paths = paths[paths["asset"] == asset]
        for side, frame in (
            ("LONG", asset_paths[asset_paths["direction"] > 0]),
            ("SHORT", asset_paths[asset_paths["direction"] < 0]),
        ):
            for hour in (4, 24, 48, 72):
                values = frame.loc[frame["hour"] == hour, "net_return"]
                records.append({
                    "asset": asset,
                    "side": side,
                    "hour": hour,
                    "trades": int(len(values)),
                    "avg_pnl_pct": float(values.mean() * 100),
                    "median_pnl_pct": float(values.median() * 100),
                    "positive_pct": float((values > 0).mean() * 100),
                })
    return pd.DataFrame(records)


def conservative_masks(
    signals: dict[str, pd.Series], config: dict,
) -> dict[str, pd.Series]:
    direction = signals["price_dir"]
    confidence = signals["price_conf"]
    long_mask = direction > 0
    agree = signals["mult_dir"] == direction
    q90 = rolling_conf_threshold(
        confidence, 0.90, config["quantile_window"]
    )
    q95 = rolling_conf_threshold(
        confidence, 0.95, config["quantile_window"]
    )
    return {
        "BASELINE": pd.Series(True, index=direction.index),
        "LONG_ONLY_ENTRY": long_mask,
        "SHORT_Q90": long_mask | ((direction < 0) & (confidence >= q90)),
        "SHORT_Q90_AGREE": (
            long_mask | ((direction < 0) & agree & (confidence >= q90))
        ),
        "SHORT_Q95_AGREE": (
            long_mask | ((direction < 0) & agree & (confidence >= q95))
        ),
    }


def run_candidate(
    forecast: pd.DataFrame, asset: str, bars_per_year: int,
    config: dict, name: str, entry_allowed: pd.Series,
    short_horizon: int | None = None,
    short_size: float = 1.0,
) -> tuple[pd.DataFrame, dict]:
    signals, _ = build_signals(forecast, config["horizon"])
    split = chronological_splits(
        forecast.index,
        config["identification_frac"],
        config["validation_frac"],
    )
    result = run_execution(
        forecast,
        signals["price_dir"],
        signals["price_conf"],
        ExecutionConfig(
            horizon=config["horizon"],
            short_horizon=short_horizon,
            fee_bps=config["fee_bps_per_side"],
            conf_quantile=config["confidence_quantile"],
            quantile_window=config["quantile_window"],
            execution=config["execution"],
            exit_on_opposite=config["exit_on_opposite"],
            short_size=short_size,
        ),
        asset=asset,
        expected_edge=signals["price_edge"],
        mult_direction=signals["mult_dir"],
        entry_allowed=entry_allowed,
        split=split,
        entry_split="validation",
    )
    summary = summarize_execution(result, bars_per_year)
    shorts = result.trades[result.trades["direction"] < 0]
    summary.update({
        "candidate": name,
        "asset": asset,
        "short_avg_net_bps": (
            float(shorts["net_return"].mean() * 1e4) if len(shorts) else np.nan
        ),
        "short_hit_pct": (
            float((shorts["net_return"] > 0).mean() * 100) if len(shorts) else np.nan
        ),
    })
    result.trades["candidate"] = name
    return result.trades, summary


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No data."
    display = frame.copy()
    for column in display.columns:
        display[column] = display[column].map(
            lambda value: "" if pd.isna(value) else str(value).replace("|", "\\|")
        )
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join("---" for _ in display.columns) + " |"
    rows = [
        "| " + " | ".join(row) + " |"
        for row in display.astype(str).itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--config", default="frozen_h72_price.json")
    args = parser.parse_args()

    report_dir = Path(args.report_dir)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parent / config_path
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = pd.read_csv(report_dir / "asset_manifest_snapshot.csv")
    baseline_trades = pd.read_csv(
        report_dir / "trades.csv",
        parse_dates=["signal_time", "entry_time", "exit_signal_time", "exit_time"],
    )

    all_paths = []
    candidate_trades = []
    candidate_summaries = []
    for row in manifest.itertuples(index=False):
        forecast = pd.read_csv(
            report_dir / f"{row.name}_forecast.csv",
            index_col=0, parse_dates=True,
        )
        asset_trades = baseline_trades[baseline_trades["asset"] == row.symbol]
        all_paths.append(trade_pnl_paths(
            forecast, asset_trades,
            fee_bps=config["fee_bps_per_side"],
            horizon=config["horizon"],
        ))
        signals, _ = build_signals(forecast, config["horizon"])
        masks = conservative_masks(signals, config)
        candidates = [
            ("BASELINE", masks["BASELINE"], None),
            ("LONG_ONLY_ENTRY", masks["LONG_ONLY_ENTRY"], None),
            ("SHORT_24H", masks["BASELINE"], 24),
            ("SHORT_Q90", masks["SHORT_Q90"], None),
            ("SHORT_Q90_AGREE", masks["SHORT_Q90_AGREE"], None),
            ("SHORT_Q90_AGREE_24H", masks["SHORT_Q90_AGREE"], 24),
            ("SHORT_Q90_AGREE_24H_25PCT", masks["SHORT_Q90_AGREE"], 24),
            ("SHORT_Q95_AGREE", masks["SHORT_Q95_AGREE"], None),
            ("SHORT_Q95_AGREE_24H", masks["SHORT_Q95_AGREE"], 24),
        ]
        for candidate, mask, short_horizon in candidates:
            short_size = 0.25 if candidate.endswith("25PCT") else 1.0
            trades, summary = run_candidate(
                forecast, row.symbol, int(row.bars_per_year),
                config, candidate, mask, short_horizon, short_size,
            )
            candidate_trades.append(trades)
            candidate_summaries.append(summary)

    paths = pd.concat(all_paths, ignore_index=True)
    path_summary = summarize_paths(paths)
    asset_path_summary = summarize_asset_paths(paths)
    excursions = excursion_summary(paths)
    candidate_summary = pd.DataFrame(candidate_summaries)
    candidate_trade_frame = pd.concat(candidate_trades, ignore_index=True)

    paths.to_csv(report_dir / "trade_pnl_paths.csv", index=False)
    path_summary.to_csv(report_dir / "trade_path_summary.csv", index=False)
    asset_path_summary.to_csv(
        report_dir / "trade_asset_path_summary.csv", index=False
    )
    excursions.to_csv(report_dir / "trade_excursion_summary.csv", index=False)
    candidate_summary.to_csv(report_dir / "short_candidate_summary.csv", index=False)
    candidate_trade_frame.to_csv(
        report_dir / "short_candidate_trades.csv", index=False
    )

    pooled = candidate_summary.groupby("candidate", as_index=False).agg(
        assets=("asset", "nunique"),
        trades=("trades", "sum"),
        long_trades=("long_trades", "sum"),
        short_trades=("short_trades", "sum"),
        asset_equal_avg_net_bps=("avg_net_bps", "mean"),
    )
    pooled_net = candidate_trade_frame.groupby("candidate")["net_return"].mean() * 1e4
    pooled["pooled_avg_net_bps"] = pooled["candidate"].map(pooled_net)

    short_path = path_summary[path_summary["side"] == "SHORT"].set_index("hour")
    short_24 = short_path.loc[24]
    short_72 = short_path.loc[72]
    conservative = pooled[
        pooled["candidate"] == "SHORT_Q90_AGREE_24H_25PCT"
    ].iloc[0]
    conservative_assets = candidate_summary[
        candidate_summary["candidate"] == "SHORT_Q90_AGREE_24H_25PCT"
    ]
    short_observed = conservative_assets["short_avg_net_bps"].notna()
    positive_short_assets = int(
        (conservative_assets.loc[short_observed, "short_avg_net_bps"] > 0).sum()
    )
    total_short_assets = int(short_observed.sum())
    baseline_assets = candidate_summary[
        candidate_summary["candidate"] == "BASELINE"
    ]
    baseline_short_observed = baseline_assets["short_avg_net_bps"].notna()
    baseline_positive_shorts = int(
        (
            baseline_assets.loc[
                baseline_short_observed, "short_avg_net_bps"
            ] > 0
        ).sum()
    )
    baseline_short_assets = int(baseline_short_observed.sum())

    report = [
        "# Entry-Time PnL Path And Conservative Short Report",
        "",
        "## Executive Summary",
        "",
        f"- Baseline short PnL averaged {short_24['avg_pnl_pct']:+.2f}% at 24h "
        f"but deteriorated to {short_72['avg_pnl_pct']:+.2f}% at 72h.",
        f"- Only {short_72['positive_pct']:.1f}% of baseline shorts were positive "
        "after costs at 72h.",
        "- Pooled shorts weakened materially after 24h, so the 72h short "
        "holding period is not supported by this validation sample.",
        "- The conservative research candidate requires q=0.90 confidence, "
        "MULT/PRICE downside agreement, a 24h maximum hold, and 25% short size.",
        f"- That candidate produced {conservative['pooled_avg_net_bps']:+.1f}bp "
        "per completed trade across the combined strategy.",
        f"- Baseline shorts were positive in {baseline_positive_shorts}/"
        f"{baseline_short_assets} assets; conservative shorts were positive in "
        f"{positive_short_assets}/{total_short_assets} assets.",
        "- Therefore the current operational recommendation is to keep shorts "
        "disabled or paper-only until expanded validation is complete.",
        "",
        "## Method",
        "",
        "- Each trade is marked at hourly closes after next-open entry.",
        "- Round-trip cost of 0.20% is deducted at every checkpoint.",
        "- After an early exit, realized PnL is carried forward to avoid survivor bias.",
        "- Short candidates change entry permission only; short signals still close longs.",
        "",
        "## Hourly Checkpoints",
        "",
        markdown_table(path_summary.round(4)),
        "",
        "## Asset And Side Checkpoints",
        "",
        markdown_table(asset_path_summary.round(4)),
        "",
        "## Excursions",
        "",
        markdown_table(excursions.round(4)),
        "",
        "## Conservative Short Candidates",
        "",
        markdown_table(candidate_summary.round(4)),
        "",
        "## Pooled Candidate Comparison",
        "",
        markdown_table(pooled.round(4)),
        "",
        "## Interpretation Rule",
        "",
        "These are validation diagnostics, not a new frozen production rule. "
        "A short restriction may be promoted only after the same behavior appears "
        "across the expanded validation asset set and a new holdout remains unopened.",
        "",
        "## Conservative Short Specification",
        "",
        "Research-only short entry:",
        "",
        "```text",
        "PRICE direction = short",
        "AND confidence >= rolling q=0.90 threshold",
        "AND MULT direction = short",
        "entry = next bar open",
        "maximum hold = 24 bars",
        "opposite high-confidence signal exit remains enabled",
        "short position size = 25% of normal long size",
        "round-trip cost = 0.20% of short notional",
        "```",
        "",
        "Promotion condition: the short leg itself must be positive in a majority "
        "of at least three crypto and two non-crypto validation assets. The current "
        f"candidate is positive in {positive_short_assets}/{total_short_assets} "
        "assets and does not pass. LONG_ONLY_ENTRY remains the lower-risk default.",
        "",
    ]
    (report_dir / "TRADE_PATH_REPORT.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print(f"reports -> {report_dir / 'TRADE_PATH_REPORT.md'}")


if __name__ == "__main__":
    main()
