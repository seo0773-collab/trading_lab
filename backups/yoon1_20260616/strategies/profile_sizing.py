"""Handler for profile-sizing-v1 (profile_plan.txt).

scripts/profile_sizing 의 목표비중 사이징 파이프라인을 공통 대시보드 데이터
(yfinance / synthetic / csv)에 돌리고, 결과를 공통 StrategyArtifacts 계약에 매핑한다.

이 전략은 가격 예측이 아니라 **국면별 목표 주식 비중 관리**다. 따라서 성과지표
(total_return/sharpe/max_drawdown)는 lot 거래가 아니라 **평가자산 equity**에서
계산하고, 비중 변화는 lot 기반 trade로 분해해 대시보드 거래 화면에 노출한다.
buy & hold 대비 성과는 metrics·extras(perf_vs_bnh)로 함께 기록한다.

forecast의 보조 컬럼(base_cycle, cm_close, cumulative_percentile, rolling/cumulative_mid_50,
regime_code, target/actual weight)은 대시보드 파형 패널에 자동 노출된다.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_lab.strategies.base import StrategyArtifacts

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from profile_sizing.account import account_summary  # noqa: E402
from profile_sizing.config import config_from_dict  # noqa: E402
from profile_sizing.engine import performance  # noqa: E402
from profile_sizing.run import (  # noqa: E402
    rebased_equity, run_pipeline, slice_window,
)
from profile_sizing.synthetic import make_synthetic_ohlcv  # noqa: E402

_SYNTHETIC_SYMBOLS = {"RANDOM", "SYNTH", "SYNTHETIC"}


class ProfileSizingHandler:
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
            return make_synthetic_ohlcv(cfg.synthetic_bars, cfg.seed, cfg.interval)
        if csv_path is not None:
            df = pd.read_csv(csv_path, parse_dates=[0], index_col=0)
            df.columns = [c.lower() for c in df.columns]
            return df
        from run_kalman_pipeline import load_yfinance  # noqa: E402
        return load_yfinance(symbol, cfg.interval, cfg.period)

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
        out = run_pipeline(raw, cfg)

        index = self._naive_index(raw)
        window = slice_window(index, phase, cfg)

        forecast = out["forecast"].copy()
        forecast.index = index
        forecast = forecast.loc[window]

        equity = rebased_equity(out["port_ret"], window)
        bnh = rebased_equity(out["buy_hold"].pct_change().fillna(0.0), window)
        port_ret = out["port_ret"].reindex(window).fillna(0.0)
        trades = self._slice_trades(out["trades"], window)

        perf = performance(equity, port_ret, cfg.interval)
        bnh_ret = out["buy_hold"].pct_change().fillna(0.0).reindex(window).fillna(0.0)
        bnh_perf = performance(bnh, bnh_ret, cfg.interval)

        metrics = self._metrics(perf, bnh_perf, trades, phase)
        metadata = {
            "n_bars": int(len(window)),
            "timeframe": cfg.interval,
            "base_cycle": f"{cfg.base_cycle.type}{cfg.base_cycle.length}",
            "avg_exposure": float(forecast["actual_weight"].mean())
            if "actual_weight" in forecast and len(forecast) else 0.0,
            "insufficient_train_data": len(window) < cfg.warmup,
        }
        regime_win = pd.Series(
            np.asarray(out["regime"]), index=index
        ).reindex(window)
        extras = {
            "perf_vs_bnh": self._perf_table(perf, bnh_perf),
            "regime_breakdown": self._regime_table(forecast, regime_win),
        }
        if "actual_weight" in forecast and len(forecast):
            extras["account_sim"] = account_summary(
                forecast["close"], forecast["actual_weight"], cfg
            )
        return StrategyArtifacts(
            forecast=forecast,
            trades=trades,
            equity=equity,
            metrics=metrics,
            metadata=metadata,
            horizon=0,
            extras=extras,
        )

    # ----- helpers ------------------------------------------------------
    @staticmethod
    def _naive_index(raw: pd.DataFrame) -> pd.DatetimeIndex:
        idx = pd.DatetimeIndex(pd.to_datetime(raw.index))
        return idx.tz_localize(None) if idx.tz is not None else idx

    @staticmethod
    def _slice_trades(trades: pd.DataFrame, window) -> pd.DataFrame:
        if trades.empty:
            return trades
        entries = pd.DatetimeIndex(pd.to_datetime(trades["entry_time"]))
        lo, hi = window[0], window[-1]
        mask = np.asarray((entries >= lo) & (entries <= hi))
        return trades[mask].reset_index(drop=True)

    @staticmethod
    def _metrics(perf: dict, bnh: dict, trades: pd.DataFrame, phase: str) -> dict:
        n = int(len(trades))
        rets = trades["net_return"].astype(float) if n else pd.Series(dtype=float)
        hit_rate = float((rets > 0).mean()) if n else None
        bnh_ret = bnh.get("total_return")
        total = perf.get("total_return")
        excess = (total - bnh_ret) if (total is not None and bnh_ret is not None) else None
        return {
            "trades": n,
            "hit_rate": hit_rate,
            "total_return": total,
            "sharpe": perf.get("sharpe"),
            "max_drawdown": perf.get("max_drawdown"),
            "cagr": perf.get("cagr"),
            "volatility": perf.get("volatility"),
            "buy_hold_return": bnh_ret,
            "buy_hold_sharpe": bnh.get("sharpe"),
            "buy_hold_max_drawdown": bnh.get("max_drawdown"),
            "excess_return_vs_bnh": excess,
            "phase": phase,
        }

    @staticmethod
    def _perf_table(perf: dict, bnh: dict) -> pd.DataFrame:
        def pct(v):
            return None if v is None else round(float(v) * 100.0, 2)

        rows = [
            {"metric": "총수익률 %", "strategy": pct(perf.get("total_return")),
             "buy_hold": pct(bnh.get("total_return"))},
            {"metric": "CAGR %", "strategy": pct(perf.get("cagr")),
             "buy_hold": pct(bnh.get("cagr"))},
            {"metric": "MDD %", "strategy": pct(perf.get("max_drawdown")),
             "buy_hold": pct(bnh.get("max_drawdown"))},
            {"metric": "Sharpe",
             "strategy": None if perf.get("sharpe") is None else round(perf["sharpe"], 3),
             "buy_hold": None if bnh.get("sharpe") is None else round(bnh["sharpe"], 3)},
            {"metric": "변동성(연) %", "strategy": pct(perf.get("volatility")),
             "buy_hold": pct(bnh.get("volatility"))},
        ]
        return pd.DataFrame(rows)

    @staticmethod
    def _regime_table(forecast: pd.DataFrame, regime_win: pd.Series) -> pd.DataFrame:
        counts = regime_win.value_counts()
        total = int(counts.sum()) or 1
        weights = forecast.get("actual_weight")
        rows = []
        for name in ("NORMAL", "CAUTION", "RECOVERY", "DEFENSE"):
            n = int(counts.get(name, 0))
            if weights is not None and n:
                avg_w = float(weights[np.asarray(regime_win) == name].mean())
            else:
                avg_w = 0.0
            rows.append({
                "regime": name,
                "bars": n,
                "time_pct": round(100.0 * n / total, 1),
                "avg_weight": round(avg_w, 3),
            })
        return pd.DataFrame(rows)
