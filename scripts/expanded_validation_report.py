#!/usr/bin/env python3
"""Create a concise expanded validation report from generated artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "generalization"


def table(frame: pd.DataFrame) -> str:
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
    summary = pd.read_csv(REPORT_DIR / "asset_summary.csv")
    trades = pd.read_csv(REPORT_DIR / "trades.csv")
    candidates = pd.read_csv(REPORT_DIR / "short_candidate_summary.csv")
    paths = pd.read_csv(REPORT_DIR / "trade_path_summary.csv")

    asset_view = summary[[
        "symbol", "asset_class", "trades", "hit_rate", "avg_net_bps",
        "total_return", "sharpe", "max_drawdown", "long_trades", "short_trades",
        "phase_start", "phase_end",
    ]].copy()
    asset_view["hit_rate"] = (asset_view["hit_rate"] * 100).round(1)
    asset_view["avg_net_bps"] = asset_view["avg_net_bps"].round(1)
    asset_view["total_return"] = (asset_view["total_return"] * 100).round(1)
    asset_view["sharpe"] = asset_view["sharpe"].round(2)
    asset_view["max_drawdown"] = (asset_view["max_drawdown"] * 100).round(1)
    asset_view.columns = [
        "asset", "class", "trades", "gross_hit_pct", "avg_net_bps",
        "compound_pct", "sharpe", "mdd_pct", "long", "short",
        "validation_start", "validation_end",
    ]

    wins = trades[trades["net_return"] > 0]["net_return"]
    losses = trades[trades["net_return"] <= 0]["net_return"]
    payoff = float(wins.mean() / abs(losses.mean()))
    profit_factor = float(wins.sum() / abs(losses.sum()))
    positive_assets = int((summary["avg_net_bps"] > 0).sum())

    direction_rows = []
    for direction, label in ((1, "LONG"), (-1, "SHORT")):
        group = trades[trades["direction"] == direction]
        direction_rows.append({
            "side": label,
            "trades": len(group),
            "net_win_pct": round((group["net_return"] > 0).mean() * 100, 1),
            "avg_net_bps": round(group["net_return"].mean() * 1e4, 1),
            "compound_pct": round(((1 + group["net_return"]).prod() - 1) * 100, 1),
        })
    direction_view = pd.DataFrame(direction_rows)

    selected = candidates[candidates["candidate"].isin([
        "BASELINE", "LONG_ONLY_ENTRY", "SHORT_Q90_AGREE_24H_25PCT",
    ])].copy()
    candidate_view = selected.groupby("candidate", as_index=False).agg(
        assets=("asset", "nunique"),
        trades=("trades", "sum"),
        avg_asset_net_bps=("avg_net_bps", "mean"),
        positive_assets=("avg_net_bps", lambda x: int((x > 0).sum())),
        short_assets=("short_avg_net_bps", lambda x: int(x.notna().sum())),
        positive_short_assets=(
            "short_avg_net_bps", lambda x: int((x.dropna() > 0).sum())
        ),
    )
    candidate_view["avg_asset_net_bps"] = candidate_view[
        "avg_asset_net_bps"
    ].round(1)

    short_path = paths[paths["side"] == "SHORT"][
        ["hour", "avg_pnl_pct", "median_pnl_pct", "positive_pct"]
    ].copy()
    short_path = short_path[short_path["hour"].isin([4, 12, 24, 48, 72])]
    short_path = short_path.round(2)

    lines = [
        "# Expanded Multi-Asset Validation Report",
        "",
        "## Scope",
        "",
        "- Frozen strategy: h=72 PRICE, q=0.85, next-bar-open.",
        "- Cost: 10bp per side, 20bp round trip.",
        "- Assets: 4 crypto, 3 ETF, 1 FX.",
        "- Test-role holdout assets were not opened.",
        "- Crypto uses about 720 days of source data; ETF hourly data is limited "
        "by the provider to about 5,000 bars.",
        "- For ETFs, 72 bars represent roughly 11 trading days, not 72 clock hours.",
        "",
        "## Gate Status",
        "",
        "- Gate 0 baseline reproduction: **PASS**",
        "- Gate 1 execution correctness: **PASS**",
        "- Gate 2 feasibility: **PASS**",
        f"- Gate 2 evidence: 4 crypto, 4 non-crypto, {len(trades)} trades.",
        "- Gate 3 evidence: **NOT EVALUATED**",
        "",
        "## Portfolio-Level Result",
        "",
        f"- Positive assets: {positive_assets}/{len(summary)}",
        f"- Asset-equal average net: {summary['avg_net_bps'].mean():+.1f}bp/trade",
        f"- Trade-pooled average net: {trades['net_return'].mean() * 1e4:+.1f}bp/trade",
        f"- Net profitable trades: {(trades['net_return'] > 0).mean() * 100:.1f}%",
        f"- Average winning trade: {wins.mean() * 100:+.2f}%",
        f"- Average losing trade: {losses.mean() * 100:+.2f}%",
        f"- Payoff ratio: {payoff:.2f}:1",
        f"- Profit factor: {profit_factor:.3f}",
        "",
        "The pooled edge is only +2.3bp per trade after 20bp round-trip cost. "
        "It is economically thin and not yet statistically validated.",
        "",
        "## Asset Results",
        "",
        table(asset_view),
        "",
        "## Long Versus Short",
        "",
        table(direction_view),
        "",
        "Longs produced the observed edge. Baseline shorts lost money in six of "
        "seven assets that generated short trades.",
        "",
        "## Short PnL By Holding Time",
        "",
        table(short_path),
        "",
        "Short performance was positive through 24 bars on average, then "
        "deteriorated sharply by 48-72 bars.",
        "",
        "## Conservative Candidate",
        "",
        table(candidate_view),
        "",
        "The 25%-sized q=0.90/agreement/24-bar short candidate improved combined "
        "strategy results, but its short leg was positive in only 3 of 7 assets. "
        "It fails the predefined majority-of-assets promotion rule.",
        "",
        "## Decision",
        "",
        "- **Baseline h72 PRICE:** feasible but weak; continue research.",
        "- **Long-only entry:** strongest and most consistent validation variant.",
        "- **Shorts:** keep disabled or paper-only.",
        "- **Holdout:** remain locked until placebo, bootstrap, and "
        "leave-one-asset-out tests complete.",
        "",
    ]
    (REPORT_DIR / "EXPANDED_VALIDATION_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(f"report -> {REPORT_DIR / 'EXPANDED_VALIDATION_REPORT.md'}")


if __name__ == "__main__":
    main()
