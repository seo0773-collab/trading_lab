from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

import pandas as pd

from .paths import var_dir

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def _safe_component(value: str) -> str:
    component = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    if not component:
        raise ValueError("market data cache key must not be empty")
    return component


def market_data_path(symbol: str, interval: str) -> Path:
    return (
        var_dir()
        / "market_data"
        / _safe_component(interval.lower())
        / f"{_safe_component(symbol.upper())}.parquet"
    )


def normalize_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)
    out = out.rename(columns=lambda value: str(value).lower().strip())
    missing = [column for column in OHLCV_COLUMNS if column not in out]
    if missing:
        raise ValueError(f"OHLCV columns missing: {missing}")

    index = pd.DatetimeIndex(pd.to_datetime(out.index))
    if index.tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    index = pd.DatetimeIndex(index.to_numpy(), name="timestamp")
    out = out.loc[:, OHLCV_COLUMNS]
    out.index = index
    out = out.apply(pd.to_numeric, errors="coerce")
    out = out.dropna(subset=["close"])
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def merge_ohlcv(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    old = normalize_ohlcv(existing)
    new = normalize_ohlcv(incoming)
    return normalize_ohlcv(pd.concat([old, new]))


def _write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    frame.to_parquet(temporary)
    temporary.replace(path)


def load_cumulative_yfinance(
    symbol: str,
    interval: str,
    period: str,
    *,
    downloader: Callable[..., pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Refresh and return a persistent union of Yahoo OHLCV observations.

    A successful download updates duplicate timestamps with the latest vendor
    values. If Yahoo is unavailable, an existing cache remains usable.
    """
    path = market_data_path(symbol, interval)
    cached = normalize_ohlcv(pd.read_parquet(path)) if path.exists() else None

    if downloader is None:
        import yfinance as yf

        downloader = yf.download

    try:
        downloaded = downloader(
            symbol,
            interval=interval,
            period=period,
            auto_adjust=True,
            progress=False,
        )
        if downloaded is None or downloaded.empty:
            raise RuntimeError(f"yfinance returned empty data for {symbol}")
        incoming = normalize_ohlcv(downloaded)
    except Exception:
        if cached is not None and not cached.empty:
            return cached
        raise

    cumulative = incoming if cached is None else merge_ohlcv(cached, incoming)
    _write_parquet_atomic(cumulative, path)
    return cumulative
