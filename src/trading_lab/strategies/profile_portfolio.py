"""Handler for profile-portfolio-v1 (다종목 포트폴리오).

여러 종목을 병렬로 평가해 상위 K개의 상승 종목을 러프하게 추종하되, 개별 종목의
profile-sizing 방어 로직(regime cap·DEFENSE)이 합산되어 시장 전반 하락 시 현금 비중이
자동으로 올라가는 포트폴리오 전략. 1 run = 1 포트폴리오(가상 심볼 PORTFOLIO)로,
종목별 OHLCV를 MultiIndex wide 프레임으로 모아 단일 NAV equity로 환산해 공통
대시보드 계약에 매핑한다. 벤치마크는 같은 유니버스의 equal-weight buy & hold.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_lab.portfolio_universes import STOCK_UNIVERSE
from trading_lab.strategies.base import StrategyArtifacts

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from profile_sizing.config import config_from_dict  # noqa: E402
from profile_sizing.engine import performance  # noqa: E402
from profile_sizing.portfolio import (  # noqa: E402
    compute_universe, simulate_portfolio,
)
from profile_sizing.run import slice_window  # noqa: E402
from profile_sizing.synthetic import make_synthetic_ohlcv  # noqa: E402

class ProfilePortfolioHandler:
    def load_data(
        self,
        symbol: str,
        config: dict[str, Any],
        *,
        csv_path: Path | None = None,
        synthetic: bool = False,
    ) -> pd.DataFrame:
        cfg = config_from_dict(config)
        if synthetic:
            n = int(config.get("synthetic_symbols", 6))
            panels = {
                f"SYN{i+1}": make_synthetic_ohlcv(cfg.synthetic_bars,
                                                  cfg.seed + i, cfg.interval)
                for i in range(n)
            }
            return self._to_wide(panels)
        from run_kalman_pipeline import load_yfinance  # noqa: E402
        universe = list(config.get("universe") or STOCK_UNIVERSE)
        panels = {}
        for sym in universe:
            try:
                panels[sym] = load_yfinance(sym, cfg.interval, cfg.period)
            except Exception:  # noqa: BLE001 — 한 종목 실패는 건너뛰고 진행.
                continue
        if not panels:
            raise RuntimeError("유니버스 종목 로드 실패")
        return self._to_wide(panels)

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
        top_k = int(config.get("top_k", 10))
        rebal_freq = str(config.get("rebalance_freq", "monthly"))
        panels = self._from_wide(raw)

        scores, prices = compute_universe(panels, cfg)
        if prices.empty:
            raise RuntimeError("유효한 종목 점수/가격이 없습니다")
        synth = self._is_synth(panels)
        # 벤치마크용 시장 종가(SPY): 시장필터 on/off와 무관하게 비교용으로 로드.
        bench_close = None if synth else self._load_close("SPY", cfg, panels)
        mk = self._market_close(config, cfg, panels)   # SPY 단일 시장필터(옵션)
        mf = config.get("market_filter") or {}
        sf = config.get("sector_filter") or {}
        sec_close, sym_sec, sec_off = self._sector_filter(config, cfg, panels, synth)
        ma_len = int(mf.get("ma_len", sf.get("ma_len", 200)))
        sim = simulate_portfolio(
            scores, prices, cfg, top_k=top_k, rebal_freq=rebal_freq,
            market_close=mk,
            market_ma_len=ma_len,
            market_off_scale=float(mf.get("off_scale", 0.5)),
            exposure_gain=float(config.get("exposure_gain", 1.0)),
            sector_close=sec_close, symbol_sector=sym_sec, sector_off_scale=sec_off,
        )

        window = slice_window(prices.index, phase, cfg)
        forecast = sim["forecast"].loc[window]
        nav = sim["nav"].reindex(window)
        equity = (nav / nav.iloc[0]).rename("equity")
        port_ret = nav.pct_change().fillna(0.0)
        trades = self._slice_trades(sim["trades"], window)

        perf = performance(equity, port_ret, cfg.interval)
        # 주 벤치마크 = 시총가중 시장(SPY) — 실제 투자 가능한 패시브 대안.
        # 합성/SPY 미가용 시 EW 지수로 폴백. EW 지수·진짜 buy&hold는 보조로 병기한다.
        ew_perf = self._bench_perf(sim["benchmark_ew"], window, cfg.interval)
        bh_perf = self._bench_perf(sim["benchmark"], window, cfg.interval)
        spy_perf = self._market_perf(bench_close, window, cfg.interval)
        if spy_perf:
            bench_perf, bench_label = spy_perf, "SPY"
            benchmark_raw = pd.Series(bench_close).reindex(window).ffill()
        else:
            bench_perf, bench_label = ew_perf, "EW지수"
            benchmark_raw = sim["benchmark_ew"].reindex(window).ffill().bfill()
        benchmark = (benchmark_raw / benchmark_raw.dropna().iloc[0]).rename(
            "benchmark"
        )
        metrics = self._metrics(perf, bench_perf, trades, phase)
        metrics["benchmark_kind"] = bench_label
        metrics["ew_index_cagr"] = ew_perf.get("cagr")
        metrics["ew_index_sharpe"] = ew_perf.get("sharpe")
        metrics["ew_index_max_drawdown"] = ew_perf.get("max_drawdown")
        metrics["buy_hold_true_cagr"] = bh_perf.get("cagr")
        metrics["buy_hold_true_sharpe"] = bh_perf.get("sharpe")
        metadata = {
            "n_symbols": int(sim["n_symbols"]),
            "top_k": top_k,
            "rebalance_freq": rebal_freq,
            "exposure_gain": float(config.get("exposure_gain", 1.0)),
            "benchmark": bench_label,
            "sector_filter": sec_close is not None,
            "market_filter": mk is not None,
            "timeframe": cfg.interval,
            "avg_exposure": float(forecast["stock_exposure"].mean())
            if len(forecast) else 0.0,
            "avg_holdings": float(forecast["n_holdings"].mean())
            if len(forecast) else 0.0,
            "insufficient_train_data": len(window) < cfg.warmup,
        }
        extras = {
            "perf_vs_bnh": self._perf_table(perf, spy_perf, ew_perf, bh_perf),
            "top_contributors": self._contributors(trades),
        }
        return StrategyArtifacts(
            forecast=forecast, trades=trades, equity=equity,
            metrics=metrics, metadata=metadata, horizon=0,
            benchmark=benchmark, extras=extras,
        )

    @staticmethod
    def _is_synth(panels) -> bool:
        return all(
            str(s).upper().startswith(("SYN", "RANDOM")) for s in panels
        )

    @staticmethod
    def _load_close(symbol, cfg, panels) -> pd.Series | None:
        """심볼 종가 로드. 유니버스에 있으면 재사용, 없으면 yfinance, 실패 시 None."""
        if symbol in panels:
            return panels[symbol]["close"]
        try:
            from run_kalman_pipeline import load_yfinance  # noqa: E402
            return load_yfinance(symbol, cfg.interval, cfg.period)["close"]
        except Exception:  # noqa: BLE001
            return None

    @classmethod
    def _market_close(cls, config, cfg, panels) -> pd.Series | None:
        """시장 레짐 필터용 지수 종가. 합성 유니버스거나 비활성/실패면 None."""
        mf = config.get("market_filter") or {}
        if not mf.get("enabled") or cls._is_synth(panels):
            return None
        return cls._load_close(str(mf.get("symbol", "SPY")), cfg, panels)

    @classmethod
    def _sector_filter(cls, config, cfg, panels, synth):
        """yoon1c용 종목별 섹터 레짐 필터 입력. 비활성/합성/실패면 (None,None,None).

        반환: (섹터지수→종가 dict, 종목→섹터 dict, off_scale)."""
        sf = config.get("sector_filter") or {}
        if not sf.get("enabled") or synth:
            return None, None, None
        from trading_lab.portfolio_universes import SECTOR_INDEX  # noqa: E402
        sym_sec = sf.get("map") or SECTOR_INDEX
        closes = {}
        for tk in sorted(set(sym_sec.values())):
            c = cls._load_close(tk, cfg, panels)
            if c is not None:
                closes[tk] = c
        if not closes:
            return None, None, None
        return closes, sym_sec, float(sf.get("off_scale", 0.5))

    # ----- wide panel <-> dict -----------------------------------------
    @staticmethod
    def _to_wide(panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
        return pd.concat(panels, axis=1)  # MultiIndex columns (symbol, field)

    @staticmethod
    def _from_wide(raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
        if not isinstance(raw.columns, pd.MultiIndex):
            return {"ONLY": raw}
        out = {}
        for sym in raw.columns.get_level_values(0).unique():
            sub = raw[sym].dropna(how="all")
            if not sub.empty:
                out[str(sym)] = sub
        return out

    # ----- helpers ------------------------------------------------------
    @staticmethod
    def _bench_perf(series: pd.Series, window, interval: str) -> dict:
        b = series.reindex(window)
        eq = b / b.iloc[0]
        ret = b.pct_change().fillna(0.0)
        return performance(eq, ret, interval)

    @staticmethod
    def _market_perf(series: pd.Series | None, window, interval: str) -> dict:
        """SPY 등 외부 시장 시리즈 성과. 데이터 존재 구간(dropna)만으로 계산해
        유니버스보다 늦게 시작한 시장의 CAGR이 왜곡(긴 기간에 짧은 성장)되지 않게 한다.
        시리즈가 없거나(합성) 윈도우에 데이터가 없으면 빈 dict → EW 폴백 신호."""
        if series is None:
            return {}
        s = pd.Series(series).reindex(window).dropna()
        if s.empty:
            return {}
        eq = s / s.iloc[0]
        ret = s.pct_change().fillna(0.0)
        return performance(eq, ret, interval)

    @staticmethod
    def _slice_trades(trades: pd.DataFrame, window) -> pd.DataFrame:
        if trades.empty:
            return trades
        entries = pd.DatetimeIndex(pd.to_datetime(trades["entry_time"]))
        lo, hi = window[0], window[-1]
        mask = np.asarray((entries >= lo) & (entries <= hi))
        return trades[mask].reset_index(drop=True)

    @staticmethod
    def _metrics(perf, bnh, trades, phase) -> dict:
        n = int(len(trades))
        rets = trades["net_return"].astype(float) if n else pd.Series(dtype=float)
        total = perf.get("total_return")
        bret = bnh.get("total_return")
        excess = (total - bret) if (total is not None and bret is not None) else None
        return {
            "trades": n,
            "hit_rate": float((rets > 0).mean()) if n else None,
            "total_return": total,
            "sharpe": perf.get("sharpe"),
            "max_drawdown": perf.get("max_drawdown"),
            "cagr": perf.get("cagr"),
            "volatility": perf.get("volatility"),
            "buy_hold_return": bret,
            "buy_hold_sharpe": bnh.get("sharpe"),
            "buy_hold_max_drawdown": bnh.get("max_drawdown"),
            "buy_hold_cagr": bnh.get("cagr"),
            "excess_return_vs_bnh": excess,
            "phase": phase,
        }

    @staticmethod
    def _perf_table(perf, spy, ew, bh) -> pd.DataFrame:
        """전략 vs 시장(SPY, 주 벤치마크) vs EW지수 vs 진짜 buy&hold(보조)."""
        spy = spy or {}

        def pct(v):
            return None if v is None else round(float(v) * 100.0, 2)

        def shp(d):
            return None if d.get("sharpe") is None else round(d["sharpe"], 3)
        rows = [
            {"metric": "CAGR %", "strategy": pct(perf.get("cagr")),
             "market_spy": pct(spy.get("cagr")), "ew_index": pct(ew.get("cagr")),
             "buy_hold_true": pct(bh.get("cagr"))},
            {"metric": "MDD %", "strategy": pct(perf.get("max_drawdown")),
             "market_spy": pct(spy.get("max_drawdown")),
             "ew_index": pct(ew.get("max_drawdown")),
             "buy_hold_true": pct(bh.get("max_drawdown"))},
            {"metric": "Sharpe", "strategy": shp(perf),
             "market_spy": shp(spy), "ew_index": shp(ew), "buy_hold_true": shp(bh)},
        ]
        return pd.DataFrame(rows)

    @staticmethod
    def _contributors(trades: pd.DataFrame) -> pd.DataFrame:
        if trades.empty or "symbol" not in trades:
            return pd.DataFrame(columns=["symbol", "trades", "avg_return", "win_rate"])
        g = trades.groupby("symbol")["net_return"]
        out = pd.DataFrame({
            "trades": g.size(),
            "avg_return": g.mean().round(4),
            "win_rate": g.apply(lambda s: float((s > 0).mean())).round(3),
        }).reset_index().sort_values("avg_return", ascending=False)
        return out.head(15)
