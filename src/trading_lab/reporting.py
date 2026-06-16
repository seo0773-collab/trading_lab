from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go


def build_markdown_report(
    run_id: str, strategy_id: str, config: dict[str, Any],
    metrics: dict[str, Any], metadata: dict[str, Any],
) -> str:
    lines = [
        f"# Backtest Run {run_id}",
        "",
        "## Configuration",
        "",
        f"- Strategy: `{strategy_id}`",
        f"- Symbol: `{metrics.get('symbol', 'n/a')}`",
        f"- Phase: `{metrics.get('phase', 'n/a')}`",
    ]
    if "horizon" in config:
        lines.append(f"- Horizon: `{config['horizon']}` bars")
    if "execution" in config:
        lines.append(f"- Execution: `{config['execution']}`")
    if "fee_bps_per_side" in config:
        lines.append(f"- Fee: `{config['fee_bps_per_side']}` bp per side")
    lines += [
        "",
        "## Performance",
        "",
        f"- Trades: {metrics.get('trades', 'n/a')}",
        f"- Hit rate: {_percent(metrics.get('hit_rate'))}",
    ]
    if "avg_net_bps" in metrics:
        lines.append(f"- Average net: {_number(metrics['avg_net_bps'])} bp/trade")
    lines += [
        f"- Total return: {_percent(metrics.get('total_return'))}",
        f"- Sharpe: {_number(metrics.get('sharpe'))}",
        f"- Max drawdown: {_percent(metrics.get('max_drawdown'))}",
        f"- Long/short trades: {metrics.get('long_trades', 'n/a')}/"
        f"{metrics.get('short_trades', 'n/a')}",
        f"- Long average return: {_percent(metrics.get('long_avg_return'))}",
        f"- Short average return: {_percent(metrics.get('short_avg_return'))}",
        f"- Long close rate: {_percent(metrics.get('long_close_rate'))}",
        f"- Short close rate: {_percent(metrics.get('short_close_rate'))}",
        "",
        "## Data",
        "",
    ]
    if "raw_bars" in metadata:
        lines.append(f"- Raw bars: {metadata['raw_bars']}")
    if "forecast_bars" in metadata:
        lines.append(f"- Forecast bars: {metadata['forecast_bars']}")
    if metadata.get("start") and metadata.get("end"):
        lines.append(f"- Range: {metadata['start']} to {metadata['end']}")
    lines += [
        "",
        "## Safety",
        "",
        "- This strategy is not live-eligible.",
        "- Live order submission is disabled until a broker adapter is configured.",
        "",
    ]
    return "\n".join(lines)


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
