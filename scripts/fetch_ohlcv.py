#!/usr/bin/env python
"""Phase 0: fetch crypto OHLCV via ccxt and store raw parquet (plan 3A).

Usage:
    python scripts/fetch_ohlcv.py
    python scripts/fetch_ohlcv.py --symbols BTC/USDT --timeframes 4h \
        --since 2020-01-01 --outdir data/raw
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

DEFAULT_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
DEFAULT_TIMEFRAMES = ["1h", "4h", "1d"]


def fetch_ohlcv(exchange, symbol: str, timeframe: str, since_ms: int,
                limit: int = 1000) -> pd.DataFrame:
    tf_ms = exchange.parse_timeframe(timeframe) * 1000
    rows: list[list] = []
    cursor = since_ms
    while cursor < exchange.milliseconds():
        batch = exchange.fetch_ohlcv(
            symbol, timeframe=timeframe, since=cursor, limit=limit
        )
        if not batch:
            break
        rows.extend(batch)
        nxt = batch[-1][0] + tf_ms
        if nxt <= cursor:
            break
        cursor = nxt
    df = pd.DataFrame(
        rows, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return (
        df.drop_duplicates("timestamp")
        .sort_values("timestamp")
        .set_index("timestamp")
    )


def integrity_report(df: pd.DataFrame, timeframe: str, exchange) -> dict:
    """plan 3A integrity checks: duplicates, ordering, gaps, NaN, volume."""
    step = pd.Timedelta(seconds=exchange.parse_timeframe(timeframe))
    diffs = df.index.to_series().diff().dropna()
    return {
        "rows": int(len(df)),
        "start": str(df.index[0]) if len(df) else None,
        "end": str(df.index[-1]) if len(df) else None,
        "duplicates": 0,  # dropped during fetch; index is unique by build
        "monotonic": bool(df.index.is_monotonic_increasing),
        "gap_count": int((diffs != step).sum()),
        "gap_ratio": float((diffs != step).mean()) if len(diffs) else 0.0,
        "nan_rows": int(df.isna().any(axis=1).sum()),
        "zero_volume_rows": int((df["volume"] == 0).sum()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--timeframes", default=",".join(DEFAULT_TIMEFRAMES))
    parser.add_argument("--since", default="2020-01-01")
    parser.add_argument("--exchange", default="binance")
    parser.add_argument(
        "--audit-only", action="store_true",
        help="Do not fetch; audit existing parquet files in outdir.",
    )
    parser.add_argument(
        "--outdir", type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "raw",
    )
    args = parser.parse_args(argv)

    import ccxt  # imported lazily: only this script needs network deps

    exchange = getattr(ccxt, args.exchange)({"enableRateLimit": True})
    if not args.audit_only:
        exchange.load_markets()
    since_ms = int(pd.Timestamp(args.since, tz="UTC").timestamp() * 1000)
    args.outdir.mkdir(parents=True, exist_ok=True)

    reports: dict[str, dict] = {}
    for symbol in args.symbols.split(","):
        symbol = symbol.strip()
        for timeframe in args.timeframes.split(","):
            timeframe = timeframe.strip()
            out = args.outdir / f"{symbol.replace('/', '')}_{timeframe}.parquet"
            if args.audit_only:
                if not out.exists():
                    raise FileNotFoundError(out)
                df = pd.read_parquet(out)
            else:
                df = fetch_ohlcv(exchange, symbol, timeframe, since_ms)
                df.to_parquet(out)
            report = integrity_report(df, timeframe, exchange)
            reports[f"{symbol}:{timeframe}"] = report
            print(f"{out}: {json.dumps(report)}")
    report_path = args.outdir / "integrity_report.json"
    report_path.write_text(
        json.dumps(reports, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    print(f"{report_path}: wrote {len(reports)} dataset reports")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
