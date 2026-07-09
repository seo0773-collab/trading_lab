"""Handler for hole_profile — 계층적 지그재그(L1) 진행-스윙 프로파일 돌파/지속 모멘텀 (단일종목 롱숏).

hole_profile.txt(Pine "Multiscale Market Structure") 이식 + hole_profile_study.py
검증 결과를 바탕으로 한다. 데이터 소스: asset_class='crypto'(기본)면
data/raw/<심볼>_<TF>.parquet 로컬 캐시를 우선 사용하고 없으면 ccxt로 수집,
'equity'면 yfinance.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_lab.paths import ROOT
from trading_lab.strategies.base import StrategyArtifacts

SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from profile_sizing.config import config_from_dict  # noqa: E402
from profile_sizing.engine import performance  # noqa: E402
from profile_sizing.hole_profile import run_hole_profile_pipeline  # noqa: E402
from profile_sizing.run import rebased_equity, slice_window  # noqa: E402
from profile_sizing.synthetic import make_synthetic_ohlcv  # noqa: E402


class HoleProfileHandler:
    def load_data(
        self,
        symbol: str,
        config: dict[str, Any],
        *,
        csv_path: Path | None = None,
        synthetic: bool = False,
    ) -> pd.DataFrame:
        if synthetic:
            return make_synthetic_ohlcv(
                int(config.get("synthetic_bars", 3000)),
                int(config.get("seed", 7)),
                str(config.get("interval", "1d")),
            )
        if csv_path is not None:
            df = pd.read_csv(csv_path, parse_dates=[0], index_col=0)
            df.columns = [c.lower() for c in df.columns]
            return df
        if str(config.get("asset_class", "crypto")) == "crypto":
            return self._load_crypto(symbol, config)
        from run_kalman_pipeline import load_yfinance  # noqa: E402
        return load_yfinance(
            symbol, config.get("interval", "1d"), config.get("period", "max")
        )

    @staticmethod
    def _load_crypto(symbol: str, config: dict[str, Any]) -> pd.DataFrame:
        """로컬 parquet(data/raw) 우선, 없으면 ccxt 수집(네트워크 필요)."""
        interval = str(config.get("interval", "1d"))
        pair = HoleProfileHandler._crypto_pair(symbol)
        local = ROOT / "data" / "raw" / f"{pair.replace('/', '')}_{interval}.parquet"
        if local.exists():
            return pd.read_parquet(local)

        import ccxt  # lazy: 로컬 캐시가 없을 때만

        from fetch_ohlcv import fetch_ohlcv
        from run_kalman_pipeline import load_yfinance  # noqa: E402

        exchange = getattr(ccxt, str(config.get("exchange", "binance")))(
            {"enableRateLimit": True}
        )
        exchange.load_markets()
        since = pd.Timestamp(config.get("since", "2017-08-17"), tz="UTC")
        try:
            return fetch_ohlcv(
                exchange, pair, interval, int(since.timestamp() * 1000)
            )
        except Exception:
            return load_yfinance(
                symbol, interval, str(config.get("period", "max"))
            )

    @staticmethod
    def _crypto_pair(symbol: str) -> str:
        s = str(symbol).strip().upper()
        if "/" in s:
            return s
        if s.endswith("-USD"):
            return f"{s[:-4]}/USDT"
        if s.endswith("USDT"):
            return f"{s[:-4]}/USDT"
        return f"{s}/USDT"

    def build_artifacts(
        self,
        raw: pd.DataFrame,
        config: dict[str, Any],
        *,
        symbol: str,
        phase: str,
        bars_per_year: int,
    ) -> StrategyArtifacts:
        cfg = config_from_dict(config)
        out = run_hole_profile_pipeline(raw, config)

        index = self._naive_index(out["forecast"].index)

        forecast_full = out["forecast"].copy()
        forecast_full.index = index
        window = slice_window(index, phase, cfg)
        forecast = forecast_full.loc[window]

        port_ret = out["bar_return"].copy()
        port_ret.index = index
        equity = rebased_equity(port_ret, window)
        port_ret = port_ret.reindex(window).fillna(0.0)

        bnh_ret = out["buy_hold"].pct_change().fillna(0.0)
        bnh_ret.index = index
        benchmark = rebased_equity(bnh_ret, window)
        bnh_ret = bnh_ret.reindex(window).fillna(0.0)

        perf = performance(equity, port_ret, cfg.interval)
        bnh_perf = performance(benchmark, bnh_ret, cfg.interval)
        trades = self._slice_trades(out["trades"], window)

        metrics = self._metrics(perf, bnh_perf, trades, phase)
        st = config.get("structure") or {}
        metadata = {
            "n_bars": int(len(window)),
            "timeframe": cfg.interval,
            "level": str(st.get("level", "L1")),
            "atr_len": int(st.get("atr_len", 48)),
            "l0_pivots": int(forecast["l0_count"].iloc[-1]) if len(forecast) else 0,
            "l1_pivots": int(forecast["l1_count"].iloc[-1]) if len(forecast) else 0,
            "l2_pivots": int(forecast["l2_count"].iloc[-1]) if len(forecast) else 0,
        }
        extras = {
            "perf_vs_bnh": self._perf_table(perf, bnh_perf),
            "duration_profile": out["duration_profile"],
            "pct_move_profile": out["pct_move_profile"],
        }
        return StrategyArtifacts(
            forecast=forecast,
            trades=trades,
            equity=equity,
            metrics=metrics,
            metadata=metadata,
            horizon=0,
            benchmark=benchmark,
            extras=extras,
        )

    @staticmethod
    def _naive_index(idx: pd.Index) -> pd.DatetimeIndex:
        out = pd.DatetimeIndex(pd.to_datetime(idx))
        return out.tz_localize(None) if out.tz is not None else out

    @staticmethod
    def _slice_trades(trades: pd.DataFrame, window) -> pd.DataFrame:
        if trades.empty or len(window) == 0:
            return trades
        entries = pd.DatetimeIndex(pd.to_datetime(trades["entry_time"]))
        if entries.tz is not None:
            entries = entries.tz_localize(None)
            trades = trades.copy()
            trades["entry_time"] = entries
            exits = pd.DatetimeIndex(
                pd.to_datetime(trades["exit_time"])
            ).tz_localize(None)
            trades["exit_time"] = exits
        mask = np.asarray((entries >= window[0]) & (entries <= window[-1]))
        return trades[mask].reset_index(drop=True)

    @staticmethod
    def _metrics(perf: dict, bnh: dict, trades: pd.DataFrame, phase: str) -> dict:
        n = int(len(trades))
        rets = trades["net_return"].astype(float) if n else pd.Series(dtype=float)
        total = perf.get("total_return")
        bnh_ret = bnh.get("total_return")
        return {
            "trades": n,
            "hit_rate": float((rets > 0).mean()) if n else None,
            "total_return": total,
            "sharpe": perf.get("sharpe"),
            "max_drawdown": perf.get("max_drawdown"),
            "cagr": perf.get("cagr"),
            "volatility": perf.get("volatility"),
            "buy_hold_return": bnh_ret,
            "buy_hold_sharpe": bnh.get("sharpe"),
            "buy_hold_max_drawdown": bnh.get("max_drawdown"),
            "excess_return_vs_bnh": (
                total - bnh_ret
                if total is not None and bnh_ret is not None else None
            ),
            "phase": phase,
        }

    @staticmethod
    def _perf_table(perf: dict, bnh: dict) -> pd.DataFrame:
        def pct(v):
            return None if v is None else round(float(v) * 100.0, 2)

        return pd.DataFrame([
            {"metric": "총수익률 %", "strategy": pct(perf.get("total_return")),
             "buy_hold": pct(bnh.get("total_return"))},
            {"metric": "CAGR %", "strategy": pct(perf.get("cagr")),
             "buy_hold": pct(bnh.get("cagr"))},
            {"metric": "MDD %", "strategy": pct(perf.get("max_drawdown")),
             "buy_hold": pct(bnh.get("max_drawdown"))},
            {"metric": "Sharpe",
             "strategy": None if perf.get("sharpe") is None else round(perf["sharpe"], 3),
             "buy_hold": None if bnh.get("sharpe") is None else round(bnh["sharpe"], 3)},
        ])
