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

# 종목 → 섹터 지수(레짐 필터용). SPY 단일 시장필터 대신 종목이 자기 섹터 추세로
# 방어하게 한다(yoon1c). SPDR 섹터 ETF(1998~) + 반도체 SOXX(2001~)를 쓴다.
# XLC(통신, 2018~)는 히스토리가 짧아 빅테크/미디어는 XLK·XLY로 대체.
SECTOR_INDEX = {
    "MSFT": "XLK", "AAPL": "XLK", "GOOGL": "XLK", "META": "XLK",
    "NVDA": "SOXX", "AVGO": "SOXX",
    "AMZN": "XLY", "TSLA": "XLY", "MCD": "XLY", "DIS": "XLY",
    "JPM": "XLF", "BAC": "XLF", "WFC": "XLF", "GS": "XLF",
    "JNJ": "XLV", "PFE": "XLV", "MRK": "XLV", "UNH": "XLV",
    "KO": "XLP", "PEP": "XLP", "PG": "XLP", "WMT": "XLP", "COST": "XLP",
    "CAT": "XLI", "DE": "XLI", "HON": "XLI", "GE": "XLI",
    "XOM": "XLE", "CVX": "XLE",
    "NEE": "XLU",
}


def portfolio_universe(asset_type: str) -> list[str]:
    return list(PORTFOLIO_UNIVERSES.get(asset_type, STOCK_UNIVERSE))


def sector_index_tickers() -> list[str]:
    """SECTOR_INDEX에 쓰인 고유 섹터 지수 티커."""
    return sorted(set(SECTOR_INDEX.values()))
