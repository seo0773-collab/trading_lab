from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class MarketOption:
    symbol: str
    detail: str
    name: str
    fallback_rank: int
    bars_per_year: int
    market_cap: float | None = None


CATALOG = {
    "crypto": [
        ("BTC-USD", "BTC", "Bitcoin"),
        ("ETH-USD", "ETH", "Ethereum"),
        ("XRP-USD", "XRP", "XRP"),
        ("BNB-USD", "BNB", "BNB"),
        ("SOL-USD", "SOL", "Solana"),
        ("DOGE-USD", "DOGE", "Dogecoin"),
        ("ADA-USD", "ADA", "Cardano"),
        ("TRX-USD", "TRX", "TRON"),
        ("AVAX-USD", "AVAX", "Avalanche"),
        ("LINK-USD", "LINK", "Chainlink"),
    ],
    "stock": [
        ("NVDA", "NVDA", "NVIDIA"),
        ("MSFT", "MSFT", "Microsoft"),
        ("AAPL", "AAPL", "Apple"),
        ("GOOGL", "GOOGL", "Alphabet"),
        ("AMZN", "AMZN", "Amazon"),
        ("META", "META", "Meta"),
        ("AVGO", "AVGO", "Broadcom"),
        ("TSLA", "TSLA", "Tesla"),
        ("BRK-B", "BRK-B", "Berkshire Hathaway"),
        ("SPY", "SPY", "S&P 500 ETF"),
        ("QQQ", "QQQ", "Nasdaq 100 ETF"),
    ],
}


def load_market_options(chart_type: str) -> list[MarketOption]:
    if chart_type not in CATALOG:
        return []
    base = [
        MarketOption(
            symbol=symbol,
            detail=detail,
            name=name,
            fallback_rank=index,
            bars_per_year=8760 if chart_type == "crypto" else 1638,
        )
        for index, (symbol, detail, name) in enumerate(CATALOG[chart_type], 1)
    ]
    caps = _load_market_caps([item.symbol for item in base])
    ranked = [
        MarketOption(
            symbol=item.symbol,
            detail=item.detail,
            name=item.name,
            fallback_rank=item.fallback_rank,
            bars_per_year=item.bars_per_year,
            market_cap=caps.get(item.symbol),
        )
        for item in base
    ]
    return sorted(
        ranked,
        key=lambda item: (
            item.market_cap is None,
            -(item.market_cap or 0.0),
            item.fallback_rank,
        ),
    )


def filter_market_options(
    options: list[MarketOption], query: str,
) -> list[MarketOption]:
    normalized = query.strip().casefold()
    if not normalized:
        return options
    return [
        option for option in options
        if normalized in option.symbol.casefold()
        or normalized in option.detail.casefold()
        or normalized in option.name.casefold()
    ]


def option_label(option: MarketOption) -> str:
    cap = ""
    if option.market_cap:
        cap = f" · 시총 {_compact_money(option.market_cap)}"
    return f"{option.detail} · {option.name}{cap}"


def _load_market_caps(symbols: list[str]) -> dict[str, float]:
    if symbols and all(symbol.endswith("-USD") for symbol in symbols):
        return _load_crypto_market_caps(symbols)
    return _load_stock_market_caps(symbols)


def _load_crypto_market_caps(symbols: list[str]) -> dict[str, float]:
    try:
        response = requests.get(
            "https://api.coingecko.com/api/v3/coins/markets",
            params={
                "vs_currency": "usd",
                "order": "market_cap_desc",
                "per_page": 250,
                "page": 1,
                "sparkline": "false",
            },
            headers={"User-Agent": "trading-lab/0.1"},
            timeout=8,
        )
        response.raise_for_status()
        markets: list[dict[str, Any]] = response.json()
    except (requests.RequestException, TypeError, ValueError):
        return {}
    by_symbol: dict[str, float] = {}
    for market in markets:
        symbol = str(market.get("symbol", "")).upper()
        cap = market.get("market_cap")
        if symbol and cap is not None and symbol not in by_symbol:
            by_symbol[f"{symbol}-USD"] = float(cap)
    return {symbol: by_symbol[symbol] for symbol in symbols if symbol in by_symbol}


def _load_stock_market_caps(symbols: list[str]) -> dict[str, float]:
    try:
        import yfinance as yf
        from yfinance import EquityQuery

        query = EquityQuery("eq", ["region", "us"])
        response = yf.screen(
            query,
            size=250,
            sortField="intradaymarketcap",
            sortAsc=False,
        )
        quotes: list[dict[str, Any]] = response.get("quotes", [])
    except Exception:
        return {}
    requested = set(symbols)
    result: dict[str, float] = {}
    for quote in quotes:
        symbol = str(quote.get("symbol", ""))
        cap = quote.get("marketCap") or quote.get("intradaymarketcap")
        if symbol in requested and cap is not None:
            result[symbol] = float(cap)
    return result


def _compact_money(value: float) -> str:
    for divisor, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if value >= divisor:
            return f"${value / divisor:.1f}{suffix}"
    return f"${value:,.0f}"
