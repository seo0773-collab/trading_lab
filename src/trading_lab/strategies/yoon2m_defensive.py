"""yoon2m_defensive: yoon2m structure with a defensive 3x3 profile field.

This handler keeps yoon2m's two-sleeve portfolio structure but replaces the
per-asset profile bucket score with a yoon_profile-style 3x3 state field:

    rows = mult_bin 0..2, columns = angle_bin 0..2

It also emits dashboard extras that show how each bar/symbol occupies the field
over time.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_lab.strategies.base import StrategyArtifacts
from trading_lab.strategies.profile_portfolio import ProfilePortfolioHandler
from trading_lab.strategies.yoon1k import Yoon1kHandler

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from profile_sizing.config import config_from_dict  # noqa: E402
from profile_sizing.cycle_state import compute_cycle_state  # noqa: E402
from profile_sizing.engine import performance  # noqa: E402
from profile_sizing.portfolio import simulate_portfolio  # noqa: E402

_TRADE_COLS = [
    "direction", "entry_time", "entry_price", "exit_time", "exit_price",
    "net_return", "exit_reason", "entry_reason", "symbol",
]


class Yoon2mDefensiveHandler(Yoon1kHandler):
    """Hierarchical yoon2m using defensive 3x3 state scores."""

    _base = ProfilePortfolioHandler()
    _cells = [f"M{m}A{a}" for m in range(3) for a in range(3)]

    def build_artifacts(
        self,
        raw: pd.DataFrame,
        config: dict[str, Any],
        *,
        symbol: str,
        phase: str,
        bars_per_year: int,
    ) -> StrategyArtifacts:
        interval = str(config.get("interval", "1d"))
        base_cur = str(config.get("base_currency", "KRW"))
        freq = str(config.get("rebalance_freq", "monthly"))
        fx_sym = str(config.get("fx_symbol", ""))

        navs: list[pd.Series] = []
        weights: list[float] = []
        sleeve_cols: dict[str, pd.Series] = {}
        sleeve_value_cols: list[str] = []
        field_counts: list[pd.DataFrame] = []
        field_summary: list[pd.DataFrame] = []
        field_points: list[pd.DataFrame] = []

        for sub in config["sub_strategies"]:
            sub_id = str(sub["config"])
            sub_cfg = self._sub_config(sub_id, sub.get("overrides"))
            sub_cfg["profile_field"] = config.get("profile_field", {})
            sub_raw = raw[sub_id]
            nav, sim_forecast, counts, summary, points = self._build_sleeve(
                sub_raw, sub_cfg, sub_id=sub_id
            )
            nav = (nav / nav.iloc[0]).rename(sub_id)
            if str(sub.get("currency", base_cur)) != base_cur:
                nav = self._to_base(nav, self._fx(raw, fx_sym))
            navs.append(nav)
            weights.append(float(sub.get("weight", 1.0 / len(config["sub_strategies"]))))
            sleeve_cols[f"sleeve_{sub_id}"] = nav
            sleeve_value_cols.append(sub_id)
            field_counts.append(counts)
            field_summary.append(summary)
            field_points.append(points)

        idx = navs[0].index
        for nav in navs[1:]:
            idx = idx.intersection(nav.index)
        idx = idx.sort_values()

        target_weights = self._target_weight_frame(config, raw, idx, weights)
        combo_values = self._combine_values_dynamic(
            [nav.reindex(idx) for nav in navs],
            target_weights,
            idx,
            freq,
            sleeve_value_cols,
        )
        combo = combo_values.sum(axis=1)

        window = self._slice(idx, phase, config)
        equity = (combo.reindex(window) / combo.reindex(window).iloc[0]).rename("equity")
        ret = equity.pct_change().fillna(0.0)
        perf = performance(equity, ret, interval)

        forecast = pd.DataFrame({"close": equity}, index=window)
        for name, ser in sleeve_cols.items():
            s = ser.reindex(window)
            forecast[name] = (s / s.iloc[0]) if s.notna().any() else s
        for sid in sleeve_value_cols:
            forecast[f"target_w_{sid}"] = target_weights[sid].reindex(window)

        shares = combo_values.reindex(window)
        shares = shares.div(shares.sum(axis=1), axis=0).fillna(0.0)
        cur_label = {
            str(s["config"]): str(s.get("currency", base_cur))
            for s in config["sub_strategies"]
        }
        sleeve_weights = shares.rename(
            columns={sid: f"{cur_label[sid]} ({sid})" for sid in sleeve_value_cols}
        )
        sleeve_weights.index.name = "time"
        sleeve_weights = sleeve_weights.reset_index()
        sleeve_weights["time"] = pd.to_datetime(sleeve_weights["time"]).dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        trades = pd.DataFrame(columns=_TRADE_COLS)
        bench = self._benchmark(raw, config, idx, freq, base_cur, fx_sym)
        bperf = self._bench_perf(bench, window, interval) if bench is not None else {}
        benchmark = None
        if bench is not None:
            b = bench.reindex(window)
            if b.notna().any():
                benchmark = (b / b.dropna().iloc[0]).rename("benchmark")

        metrics = {
            "trades": 0,
            "hit_rate": None,
            "total_return": perf.get("total_return"),
            "sharpe": perf.get("sharpe"),
            "max_drawdown": perf.get("max_drawdown"),
            "cagr": perf.get("cagr"),
            "volatility": perf.get("volatility"),
            "buy_hold_return": bperf.get("total_return"),
            "buy_hold_sharpe": bperf.get("sharpe"),
            "buy_hold_max_drawdown": bperf.get("max_drawdown"),
            "buy_hold_cagr": bperf.get("cagr"),
            "benchmark_kind": str(config.get("benchmark_label", "50/50")),
            "phase": phase,
        }
        metadata = {
            "sub_strategies": sleeve_value_cols,
            "weights": target_weights.iloc[0].to_list(),
            "base_currency": base_cur,
            "fx_symbol": fx_sym,
            "rebalance_freq": freq,
            "profile_matrix": config.get("profile_field", {}).get("state_weights"),
        }

        combined_counts = self._combine_field_counts(field_counts, window)
        heatmap = self._field_heatmap(combined_counts)
        cumulative = self._field_cumulative(combined_counts)
        summary = pd.concat(field_summary, ignore_index=True)
        points = self._slice_points(pd.concat(field_points, ignore_index=True), window)

        return StrategyArtifacts(
            forecast=forecast,
            trades=trades,
            equity=equity,
            metrics=metrics,
            metadata=metadata,
            benchmark=benchmark,
            extras={
                "sleeve_weights": sleeve_weights,
                "regime_field_cumulative": cumulative,
                "regime_field_points": points,
                "regime_field_heatmap": heatmap,
                "regime_field_summary": summary,
            },
        )

    def _build_sleeve(
        self, raw: pd.DataFrame, config: dict[str, Any], *, sub_id: str
    ) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        cfg = config_from_dict(config)
        panels = self._base._from_wide(raw)
        scores, prices, counts, summary, points = self._state_scores(
            panels, config, sub_id=sub_id
        )
        mf = config.get("market_filter") or {}
        ss = config.get("safe_sleeve") or {}
        market_close = self._base._market_close(config, cfg, panels)
        safe_close = None
        if ss.get("enabled"):
            safe = {}
            for safe_symbol in ss.get("symbols", []):
                close = self._base._load_close(str(safe_symbol), cfg, panels)
                if close is not None:
                    safe[str(safe_symbol)] = close
            safe_close = safe or None

        sim = simulate_portfolio(
            scores,
            prices,
            cfg,
            top_k=int(config.get("top_k", 3)),
            rebal_freq=str(config.get("rebalance_freq", "monthly")),
            market_close=market_close,
            market_ma_len=int(mf.get("ma_len", 200)),
            market_off_scale=float(mf.get("off_scale", 0.5)),
            market_mode=str(mf.get("mode", "binary")),
            exposure_gain=float(config.get("exposure_gain", 1.0)),
            max_exposure=float(config.get("max_exposure", 1.0)),
            weight_scheme=str(config.get("weight_scheme", "score")),
            safe_close=safe_close,
            safe_ratio=float(ss.get("ratio", 0.0)),
            safe_max=float(ss.get("max", 0.0)),
            safe_ma_len=int(ss.get("ma_len", 100)),
        )
        return sim["nav"], sim["forecast"], counts, summary, points

    def _state_scores(
        self, panels: dict[str, pd.DataFrame], config: dict[str, Any], *, sub_id: str
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        field = config.get("profile_field") or {}
        matrix = np.asarray(field.get("state_weights"), dtype=float)
        if matrix.shape != (3, 3):
            raise ValueError("profile_field.state_weights must be 3x3")
        warmup_weight = float(field.get("warmup_weight", 1.0))

        cycle = field.get("cycle") or {}
        cycle_kwargs = {
            "cycle_len": int(cycle.get("length", 200)),
            "slope_lookback": int(cycle.get("slope_lookback", 5)),
            "z_window": int(cycle.get("z_window", 252)),
            "grid_bins": int(cycle.get("grid_bins", 3)),
            "min_bin_history": int(cycle.get("min_bin_history", 252)),
        }

        score_cols: dict[str, pd.Series] = {}
        price_cols: dict[str, pd.Series] = {}
        state_cols: dict[str, pd.DataFrame] = {}
        point_frames: list[pd.DataFrame] = []
        for sym, frame in panels.items():
            if frame is None or frame.empty or "close" not in frame:
                continue
            idx = self._naive_index(frame)
            close = pd.Series(frame["close"].to_numpy(dtype=float), index=idx)
            state = compute_cycle_state(close, **cycle_kwargs)
            mb = state["mult_bin"].to_numpy()
            ab = state["angle_bin"].to_numpy()
            valid = (mb >= 0) & (ab >= 0)
            target = np.full(len(state), warmup_weight, dtype=float)
            target[valid] = matrix[mb[valid], ab[valid]]
            score_cols[str(sym)] = pd.Series(
                np.clip(target, 0.0, 1.0), index=idx, name=str(sym)
            )
            price_cols[str(sym)] = close.rename(str(sym))
            state_cols[str(sym)] = state[["mult_bin", "angle_bin"]]
            point_frames.append(
                self._field_points_for_symbol(
                    state, target, sleeve=sub_id, symbol=str(sym)
                )
            )

        idx = None
        for series in price_cols.values():
            idx = series.index if idx is None else idx.union(series.index)
        if idx is None:
            empty = pd.DataFrame()
            return empty, empty, empty, empty, empty
        idx = idx.sort_values()
        scores = pd.DataFrame(score_cols).reindex(idx).fillna(0.0)
        prices = pd.DataFrame(price_cols).reindex(idx)
        counts = self._daily_field_counts(state_cols, idx, sub_id)
        summary = self._field_summary(state_cols, matrix, sub_id)
        points = pd.concat(point_frames, ignore_index=True) if point_frames else pd.DataFrame()
        return scores, prices, counts, summary, points

    @staticmethod
    def _field_points_for_symbol(
        state: pd.DataFrame, target: np.ndarray, *, sleeve: str, symbol: str
    ) -> pd.DataFrame:
        valid = state[(state["mult_bin"] >= 0) & (state["angle_bin"] >= 0)].copy()
        if valid.empty:
            return pd.DataFrame(columns=[
                "time", "sleeve", "symbol", "cell", "mult_bin", "angle_bin",
                "x", "y", "target_weight",
            ])
        pos = np.arange(len(valid), dtype=float)
        # Deterministic jitter spreads repeated observations inside each 3x3 cell
        # without encoding any extra value as color.
        jx = (((pos * 37) % 101) / 100.0 - 0.5) * 0.38
        jy = (((pos * 53) % 101) / 100.0 - 0.5) * 0.38
        mb = valid["mult_bin"].astype(int).to_numpy()
        ab = valid["angle_bin"].astype(int).to_numpy()
        idx_pos = state.index.get_indexer(valid.index)
        return pd.DataFrame({
            "time": pd.DatetimeIndex(valid.index).strftime("%Y-%m-%d %H:%M:%S"),
            "sleeve": sleeve,
            "symbol": symbol,
            "cell": [f"M{m}A{a}" for m, a in zip(mb, ab)],
            "mult_bin": mb,
            "angle_bin": ab,
            "x": ab + jx,
            "y": mb + jy,
            "target_weight": np.asarray(target, dtype=float)[idx_pos],
        })

    @staticmethod
    def _slice_points(points: pd.DataFrame, window: pd.DatetimeIndex) -> pd.DataFrame:
        if points.empty:
            return points
        t = pd.to_datetime(points["time"])
        mask = (t >= window[0]) & (t <= window[-1])
        return points.loc[mask].reset_index(drop=True)

    @classmethod
    def _daily_field_counts(
        cls, states: dict[str, pd.DataFrame], idx: pd.DatetimeIndex, sleeve: str
    ) -> pd.DataFrame:
        out = pd.DataFrame(0.0, index=idx, columns=cls._cells)
        for state in states.values():
            aligned = state.reindex(idx)
            for m in range(3):
                for a in range(3):
                    cell = f"M{m}A{a}"
                    mask = (aligned["mult_bin"] == m) & (aligned["angle_bin"] == a)
                    out[cell] += mask.astype(float)
        out.insert(0, "sleeve", sleeve)
        return out

    @classmethod
    def _combine_field_counts(
        cls, frames: list[pd.DataFrame], window: pd.DatetimeIndex
    ) -> pd.DataFrame:
        total = pd.DataFrame(0.0, index=window, columns=cls._cells)
        for frame in frames:
            total = total.add(frame.reindex(window)[cls._cells].fillna(0.0), fill_value=0.0)
        return total

    @classmethod
    def _field_cumulative(cls, counts: pd.DataFrame) -> pd.DataFrame:
        cum = counts[cls._cells].cumsum()
        denom = cum.sum(axis=1).replace(0.0, np.nan)
        shares = cum.div(denom, axis=0).fillna(0.0)
        shares.insert(0, "time", shares.index.strftime("%Y-%m-%d %H:%M:%S"))
        return shares.reset_index(drop=True)

    @staticmethod
    def _field_heatmap(counts: pd.DataFrame) -> pd.DataFrame:
        total = counts.sum(axis=0)
        denom = float(total.sum()) or 1.0
        rows = []
        for angle in range(3):
            row = {"angle_bin": float(angle)}
            for mult in range(3):
                row[str(mult)] = float(total.get(f"M{mult}A{angle}", 0.0) / denom)
            rows.append(row)
        return pd.DataFrame(rows)

    @classmethod
    def _field_summary(
        cls, states: dict[str, pd.DataFrame], matrix: np.ndarray, sleeve: str
    ) -> pd.DataFrame:
        rows = []
        total_valid = 0
        counts = {cell: 0 for cell in cls._cells}
        for state in states.values():
            valid = state[(state["mult_bin"] >= 0) & (state["angle_bin"] >= 0)]
            total_valid += len(valid)
            for m in range(3):
                for a in range(3):
                    counts[f"M{m}A{a}"] += int(
                        ((valid["mult_bin"] == m) & (valid["angle_bin"] == a)).sum()
                    )
        denom = total_valid or 1
        for m in range(3):
            for a in range(3):
                cell = f"M{m}A{a}"
                rows.append({
                    "sleeve": sleeve,
                    "cell": cell,
                    "mult_bin": m,
                    "angle_bin": a,
                    "target_weight": float(matrix[m, a]),
                    "bars": counts[cell],
                    "share_valid": counts[cell] / denom,
                })
        return pd.DataFrame(rows)

    @staticmethod
    def _naive_index(frame: pd.DataFrame) -> pd.DatetimeIndex:
        idx = pd.DatetimeIndex(pd.to_datetime(frame.index))
        return idx.tz_localize(None) if idx.tz is not None else idx
