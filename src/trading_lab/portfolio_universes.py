from __future__ import annotations


STOCK_UNIVERSE = [
    "MSFT", "AAPL", "NVDA", "GOOGL", "AMZN", "META", "AVGO", "TSLA",
    "JPM", "BAC", "WFC", "GS", "JNJ", "PFE", "MRK", "UNH",
    "KO", "PEP", "PG", "WMT", "MCD", "COST",
    "CAT", "DE", "HON", "GE", "XOM", "CVX", "NEE", "DIS",
]

CRYPTO_UNIVERSE = [
    "BTC-USD", "ETH-USD", "XRP-USD", "BNB-USD", "SOL-USD",
    "DOGE-USD", "ADA-USD", "TRX-USD", "AVAX-USD", "LINK-USD",
]

MIXED_UNIVERSE = [
    *CRYPTO_UNIVERSE,
    "MSFT", "AAPL", "NVDA", "GOOGL", "AMZN", "META", "AVGO", "TSLA",
    "SPY", "QQQ",
]

PORTFOLIO_UNIVERSES = {
    "crypto": CRYPTO_UNIVERSE,
    "stock": STOCK_UNIVERSE,
    "mixed": MIXED_UNIVERSE,
}


def portfolio_universe(asset_type: str) -> list[str]:
    return list(PORTFOLIO_UNIVERSES.get(asset_type, STOCK_UNIVERSE))
