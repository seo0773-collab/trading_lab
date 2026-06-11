"""Network smoke test for yfinance OHLCV downloads."""

import sys


def main() -> int:
    try:
        import yfinance as yf
    except ImportError as exc:
        print(f"ERROR: yfinance import failed: {exc}", file=sys.stderr)
        print("Install with: pip install -r requirements-core.txt", file=sys.stderr)
        return 1

    try:
        data = yf.download(
            "SPY",
            start="2023-01-01",
            end="2024-01-01",
            auto_adjust=False,
            progress=False,
        )
    except Exception as exc:
        print(f"ERROR: SPY download failed: {exc}", file=sys.stderr)
        print("Check network/DNS access and retry.", file=sys.stderr)
        return 1

    if data.empty:
        print("ERROR: yfinance returned no SPY rows.", file=sys.stderr)
        return 1
    print(f"OK: downloaded {len(data)} SPY rows with columns {list(data.columns)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
