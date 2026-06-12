from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go


def build_markdown_report(
    run_id: str, strategy_id: str, config: dict[str, Any],
    metrics: dict[str, Any], metadata: dict[str, Any],
) -> str:
    return "\n".join([
        f"# Backtest Run {run_id}",
        "",
        "## Configuration",
        "",
        f"- Strategy: `{strategy_id}`",
        f"- Symbol: `{metrics['symbol']}`",
        f"- Phase: `{metrics['phase']}`",
        f"- Horizon: `{config['horizon']}` bars",
        f"- Execution: `{config['execution']}`",
        f"- Fee: `{config['fee_bps_per_side']}` bp per side",
        "",
        "## Performance",
        "",
        f"- Trades: {metrics['trades']}",
        f"- Hit rate: {_percent(metrics['hit_rate'])}",
        f"- Average net: {_number(metrics['avg_net_bps'])} bp/trade",
        f"- Total return: {_percent(metrics['total_return'])}",
        f"- Sharpe: {_number(metrics['sharpe'])}",
        f"- Max drawdown: {_percent(metrics['max_drawdown'])}",
        f"- Long/short trades: {metrics['long_trades']}/{metrics['short_trades']}",
        f"- Long average return: {_percent(metrics.get('long_avg_return'))}",
        f"- Short average return: {_percent(metrics.get('short_avg_return'))}",
        f"- Long close rate: {_percent(metrics.get('long_close_rate'))}",
        f"- Short close rate: {_percent(metrics.get('short_close_rate'))}",
        "",
        "## Data",
        "",
        f"- Raw bars: {metadata['raw_bars']}",
        f"- Forecast bars: {metadata['forecast_bars']}",
        f"- Range: {metadata['start']} to {metadata['end']}",
        "",
        "## Safety",
        "",
        "- This strategy is not live-eligible.",
        "- Live order submission is disabled until a broker adapter is configured.",
        "",
    ])


def build_equity_html(equity: pd.Series, symbol: str) -> str:
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=equity.index, y=equity.values, mode="lines", name="Strategy equity"
    ))
    figure.update_layout(
        title=f"{symbol} strategy equity",
        xaxis_title="Time",
        yaxis_title="Growth of 1.0",
        template="plotly_white",
    )
    return figure.to_html(full_html=True, include_plotlyjs="cdn")


def _number(value: Any) -> str:
    return "n/a" if pd.isna(value) else f"{float(value):.2f}"


def _percent(value: Any) -> str:
    return "n/a" if pd.isna(value) else f"{float(value) * 100:.2f}%"
