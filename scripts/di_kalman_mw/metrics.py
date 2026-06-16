"""Performance metrics (plan 13, 14)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_metrics(
    trades: pd.DataFrame, bar_returns: pd.Series, bars_per_year: float
) -> dict:
    out: dict = {"num_trades": int(len(trades))}
    n_bars = int(len(bar_returns))
    if n_bars:
        equity = (1.0 + bar_returns).cumprod()
        total = float(equity.iloc[-1] - 1.0)
        out["total_return"] = total
        out["cagr"] = (
            float((1.0 + total) ** (bars_per_year / n_bars) - 1.0)
            if total > -1.0
            else None
        )
        out["max_drawdown"] = float((equity / equity.cummax() - 1.0).min())
        std = float(bar_returns.std())
        out["sharpe"] = (
            float(bar_returns.mean() / std * np.sqrt(bars_per_year))
            if std > 0
            else None
        )
        downside = bar_returns[bar_returns < 0]
        dstd = float(downside.std()) if len(downside) > 1 else 0.0
        out["sortino"] = (
            float(bar_returns.mean() / dstd * np.sqrt(bars_per_year))
            if dstd > 0
            else None
        )
    else:
        out.update(
            total_return=None, cagr=None, max_drawdown=None,
            sharpe=None, sortino=None,
        )
    if len(trades):
        pnl = trades["pnl_pct"].astype(float)
        wins = pnl[pnl > 0]
        losses = pnl[pnl <= 0]
        gross_profit = float(wins.sum())
        gross_loss = float(-losses.sum())
        out["profit_factor"] = (
            float(gross_profit / gross_loss) if gross_loss > 0 else None
        )
        out["win_rate"] = float(len(wins) / len(pnl))
        out["avg_win"] = float(wins.mean()) if len(wins) else 0.0
        out["avg_loss"] = float(-losses.mean()) if len(losses) else 0.0
        out["expectancy"] = float(pnl.mean())
        out["avg_bars_held"] = float(trades["bars_held"].mean())
        out["long_return"] = float(pnl[trades["direction"] == "long"].sum())
        out["short_return"] = float(pnl[trades["direction"] == "short"].sum())
        # plan 14: detect profit concentrated in one or two trades
        out["top2_profit_share"] = (
            float(wins.nlargest(2).sum() / gross_profit)
            if gross_profit > 0
            else None
        )
    else:
        out.update(
            profit_factor=None, win_rate=None, avg_win=None, avg_loss=None,
            expectancy=None, avg_bars_held=None, long_return=0.0,
            short_return=0.0, top2_profit_share=None,
        )
    return out


def split_metrics(
    trades: pd.DataFrame, equity: pd.DataFrame, bars_per_year: float
) -> dict:
    result = {}
    for name in ("train", "validation", "test"):
        split_trades = trades[trades["split"] == name]
        split_returns = equity.loc[equity["split"] == name, "bar_return"]
        result[name] = compute_metrics(split_trades, split_returns, bars_per_year)
    return result


def pressure_alignment_breakdown(trades: pd.DataFrame) -> dict:
    """plan 14: pressure_aligned=True vs False trade performance."""
    result: dict = {}
    if not len(trades):
        return result
    for aligned, grp in trades.groupby("pressure_aligned"):
        pnl = grp["pnl_pct"].astype(float)
        wins = pnl[pnl > 0]
        gross_loss = float(-pnl[pnl <= 0].sum())
        result[f"aligned={bool(aligned)}"] = {
            "num_trades": int(len(grp)),
            "expectancy": float(pnl.mean()),
            "win_rate": float(len(wins) / len(pnl)),
            "profit_factor": (
                float(wins.sum() / gross_loss) if gross_loss > 0 else None
            ),
        }
    return result


def sanitize(obj):
    """Make a nested structure strict-JSON serializable (nan/inf -> null)."""
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if np.isfinite(f) else None
    return obj
