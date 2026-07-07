"""yoon1k — 계층 포트폴리오(원/달러 통화 배분 × 통화별 yoon1j).

2단계 fund-of-funds 구조:

    [최상위] 원/달러 통화 배분 리밸런싱 (기본 50/50 월간)
       ├─ 원화 슬리브 → 한국 yoon1j_kr
       └─ 달러 슬리브 → 미국 yoon1j  (원화 환산: USD/KRW 반영)

한국·미국 yoon1j는 일간수익률 상관이 ~0(거의 무상관)이라, 통화 다변화와 함께
50/50으로 결합하면 변동성이 크게 줄어 위험조정 성과(Sharpe)가 개별 슬리브보다
도약한다. 하위 슬리브는 기존 ``ProfilePortfolioHandler`` 를 그대로 위임 호출하고,
이 핸들러는 환율 환산 + 통화 리밸런싱 결합만 담당한다(공통 인프라 무수정).

config 스키마(configs/strategies/yoon1k.json):
    base_currency, fx_symbol, rebalance_freq,
    sub_strategies=[{config, currency, weight}, ...],
    benchmark_symbols={CUR: symbol}, train_frac, validation_frac.
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_lab.strategies.base import StrategyArtifacts
from trading_lab.strategies.profile_portfolio import ProfilePortfolioHandler

ROOT = Path(__file__).resolve().parents[3]

import sys  # noqa: E402
for _p in (str(ROOT / "scripts"), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from profile_sizing.engine import performance  # noqa: E402
from profile_sizing.portfolio import rebalance_dates  # noqa: E402

_TRADE_COLS = ["direction", "entry_time", "entry_price", "exit_time",
               "exit_price", "net_return", "exit_reason", "entry_reason", "symbol"]


class Yoon1kHandler:
    """원/달러 통화 배분 × 통화별 yoon1j 계층 포트폴리오."""

    _base = ProfilePortfolioHandler()

    # ---- 데이터 로딩: 하위 유니버스 + 환율 + 벤치를 3레벨 wide로 묶는다 ----
    def load_data(
        self, symbol: str, config: dict[str, Any], *,
        csv_path: Path | None = None, synthetic: bool = False,
    ) -> pd.DataFrame:
        frames: dict[str, pd.DataFrame] = {}
        for sub in config["sub_strategies"]:
            sub_cfg = self._sub_config(sub["config"], sub.get("overrides"))
            frames[sub["config"]] = self._base.load_data(
                symbol, sub_cfg, synthetic=synthetic)
        if not synthetic:
            from run_kalman_pipeline import load_yfinance  # noqa: E402
            interval = str(config.get("interval", "1d"))
            period = str(config.get("period", "max"))
            fx_sym = str(config["fx_symbol"])
            frames["__FX__"] = pd.concat(
                {fx_sym: load_yfinance(fx_sym, interval, period)}, axis=1)
            bench = config.get("benchmark_symbols") or {}
            bcols = {}
            for sym in bench.values():
                try:
                    bcols[sym] = load_yfinance(sym, interval, period)
                except Exception:  # noqa: BLE001
                    continue
            if bcols:
                frames["__BENCH__"] = pd.concat(bcols, axis=1)
            gate = (config.get("currency_allocation") or {}).get("risk_gate") or {}
            if str(gate.get("source", "trend")).lower() != "trend" and gate.get("enabled", False):
                needed = set(gate.get("symbols", [])) | set(gate.get("vix_symbols", []))
                if gate.get("safe_asset"):
                    needed.add(str(gate["safe_asset"]))
                gcols = {}
                for sym in needed:
                    try:
                        gcols[sym] = load_yfinance(sym, interval, period)
                    except Exception:  # noqa: BLE001
                        continue
                if gcols:
                    frames["__GATE__"] = pd.concat(gcols, axis=1)
        return pd.concat(frames, axis=1)  # 3레벨 (group, symbol, field)

    # ---- 아티팩트: 하위 NAV → 통화 환산 → 50/50 결합 → 메타 성과 ----
    def build_artifacts(
        self, raw: pd.DataFrame, config: dict[str, Any], *,
        symbol: str, phase: str, bars_per_year: int,
    ) -> StrategyArtifacts:
        interval = str(config.get("interval", "1d"))
        base_cur = str(config.get("base_currency", "KRW"))
        freq = str(config.get("rebalance_freq", "monthly"))
        fx_sym = str(config.get("fx_symbol", ""))

        navs, weights, sleeve_cols, sub_trades = [], [], {}, []
        for sub in config["sub_strategies"]:
            sub_id = sub["config"]
            sub_cfg = self._sub_config(sub_id, sub.get("overrides"))
            sub_raw = raw[sub_id]                       # 3→2레벨 복원
            art = self._base.build_artifacts(
                sub_raw, sub_cfg, symbol=symbol, phase="all",
                bars_per_year=bars_per_year)
            nav = (art.equity / art.equity.iloc[0]).rename(sub_id)
            if str(sub.get("currency", base_cur)) != base_cur:
                nav = self._to_base(nav, self._fx(raw, fx_sym))   # 통화 환산
            navs.append(nav)
            weights.append(float(sub.get("weight", 1.0 / len(config["sub_strategies"]))))
            sleeve_cols[f"sleeve_{sub_id}"] = nav
            t = art.trades.copy()
            if not t.empty:
                t["symbol"] = sub_id + ":" + t["symbol"].astype(str)
                sub_trades.append(t)

        # 공통 인덱스 + 가중 리밸런싱 결합(전체 구간).
        # 기본은 기존 그대로 intersection(일봉 등 같은 달력에서 동작하던 방식을
        # 그대로 보존 — 이미 검증된 final0706 등의 수치를 바꾸지 않기 위함).
        # 서로 다른 거래소의 인트라데이 봉(예: 4h)은 실제 봉 시각이 겹치지 않아
        # (미국 4h는 현지장 개장 기준 UTC 13:30/17:39대, 한국은 00:00/04:00대)
        # intersection이 완전히 비어버리는 경우에만, 겹치는 실구간(각 슬리브가
        # 실제로 데이터를 가진 구간의 교집합) 안에서 union+ffill로 폴백한다.
        # 겹치는 실구간 **밖**(한쪽 슬리브가 아예 존재하지 않던 과거, 예: 상장 전)
        # 까지 bfill로 끌어오면 존재하지도 않았던 자산에 가짜 평평한 수익률을
        # 배정해 장기 백테스트를 심각하게 왜곡한다(실측으로 발견) — 그래서 idx를
        # 겹치는 실구간으로 먼저 자른 뒤 union+ffill만 한다(bfill 없음).
        idx = navs[0].index
        for n in navs[1:]:
            idx = idx.intersection(n.index)
        if len(idx) == 0:
            starts = [n.dropna().index.min() for n in navs]
            ends = [n.dropna().index.max() for n in navs]
            overlap_lo, overlap_hi = max(starts), min(ends)
            idx = navs[0].index
            for n in navs[1:]:
                idx = idx.union(n.index)
            idx = idx[(idx >= overlap_lo) & (idx <= overlap_hi)]
        idx = idx.sort_values()
        sub_ids = [s["config"] for s in config["sub_strategies"]]
        target_weights = self._target_weight_frame(config, raw, idx, weights)
        weights = target_weights.iloc[0].to_list()
        combine_navs = [n.reindex(idx).ffill() for n in navs]
        combine_cols = list(sub_ids)
        if "__CASH__" in target_weights.columns:
            combine_navs.append(self._cash_proxy_nav(raw, config, idx))
            combine_cols.append("__CASH__")
        combo_values = self._combine_values_dynamic(
            combine_navs, target_weights, idx, freq, combine_cols)
        combo = combo_values.sum(axis=1)

        window = self._slice(idx, phase, config)
        eq = combo.reindex(window)
        equity = (eq / eq.iloc[0]).rename("equity")
        ret = equity.pct_change().fillna(0.0)
        perf = performance(equity, ret, interval)

        forecast = pd.DataFrame({"close": equity}, index=window)
        for name, ser in sleeve_cols.items():           # 통화별 슬리브(원화 기준) 보조지표
            s = ser.reindex(window)
            forecast[name] = (s / s.iloc[0]) if s.notna().any() else s
        for sid in combine_cols:
            forecast[f"target_w_{sid}"] = target_weights[sid].reindex(window)

        # 자산군(통화 슬리브)별 평가가치 비중(행 합=100%) → 누적 면적 '웨이브'.
        # 평가금액 자체는 초기금액·적립방식 등 대시보드 입력에 종속되므로(금액 환산은
        # 대시보드 책임) 비중으로 노출한다 — 결과 화면이 계좌금액과 곱해 읽으면 된다.
        cur_label = {s["config"]: str(s.get("currency", base_cur))
                     for s in config["sub_strategies"]}
        share_win = combo_values.reindex(window)
        shares = share_win.div(share_win.sum(axis=1), axis=0).fillna(0.0)
        shares = shares.rename(columns={
            sid: f"{cur_label.get(sid, 'CASH')} ({sid})" for sid in combine_cols})
        shares.index.name = "time"
        sleeve_weights = shares.reset_index()

        trades = self._slice_trades(sub_trades, window)
        n = int(len(trades))
        rets = trades["net_return"].astype(float) if n else pd.Series(dtype=float)

        bench = self._benchmark(raw, config, idx, freq, base_cur, fx_sym)
        bperf = self._bench_perf(bench, window, interval) if bench is not None else {}
        metrics = {
            "trades": n,
            "hit_rate": float((rets > 0).mean()) if n else None,
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
        benchmark = None
        if bench is not None:
            b = bench.reindex(window)
            if b.notna().any():
                benchmark = (b / b.dropna().iloc[0]).rename("benchmark")
        metadata = {
            "sub_strategies": [s["config"] for s in config["sub_strategies"]],
            "weights": weights,
            "base_currency": base_cur,
            "fx_symbol": fx_sym,
            "rebalance_freq": freq,
        }
        return StrategyArtifacts(
            forecast=forecast, trades=trades, equity=equity,
            metrics=metrics, metadata=metadata, benchmark=benchmark,
            extras={"sleeve_weights": sleeve_weights})

    # ---------------- helpers ----------------
    @staticmethod
    def _cash_proxy_nav(raw: pd.DataFrame, config: dict[str, Any], idx: pd.DatetimeIndex) -> pd.Series:
        """`__CASH__` 슬롯이 담을 자산의 NAV. 기본은 순수 현금(수익률 0%).

        `risk_gate.safe_asset`(예: "IEF")가 지정되면 게이트 발동 구간의 자금을
        문자 그대로 현금이 아니라 그 안전자산의 실제 가격 수익률로 굴린다 — 신용
        스프레드 신호 자체가 "안전자산이 상대적으로 강해진다"는 뜻이라, 방어
        국면에서 국채 등의 상승분을 그냥 버리지 않기 위함.
        """
        gate = ((config.get("currency_allocation") or {}).get("risk_gate")) or {}
        safe_sym = gate.get("safe_asset")
        if not safe_sym or "__GATE__" not in raw.columns.get_level_values(0):
            return pd.Series(1.0, index=idx, name="__CASH__")
        gate_cols = raw["__GATE__"].columns.get_level_values(0)
        if safe_sym not in gate_cols:
            return pd.Series(1.0, index=idx, name="__CASH__")
        px = raw["__GATE__"][safe_sym]["close"].reindex(idx).ffill()
        if px.dropna().empty:
            return pd.Series(1.0, index=idx, name="__CASH__")
        return (px / px.dropna().iloc[0]).rename("__CASH__")

    @staticmethod
    def _sub_config(sub_id: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        path = ROOT / "configs" / "strategies" / f"{sub_id}.json"
        cfg = json.loads(path.read_text(encoding="utf-8"))
        if overrides:                       # sub_strategies 항목의 inline override(예: top_k)
            cfg.update(overrides)
        return cfg

    @staticmethod
    def _fx(raw: pd.DataFrame, fx_sym: str) -> pd.Series | None:
        if "__FX__" not in raw.columns.get_level_values(0):
            return None
        return raw["__FX__"][fx_sym]["close"]

    @staticmethod
    def _to_base(nav: pd.Series, fx: pd.Series | None) -> pd.Series:
        """로컬통화 NAV(1.0기준)를 기준통화로 환산: × (환율/환율₀)."""
        if fx is None:
            return nav
        f = pd.Series(fx).reindex(nav.index).ffill()
        f0 = f.dropna()
        if f0.empty:
            return nav
        return (nav * (f / f0.iloc[0])).rename(nav.name)

    @classmethod
    def _target_weight_frame(
        cls, config: dict[str, Any], raw: pd.DataFrame, idx: pd.DatetimeIndex,
        weights: list[float],
    ) -> pd.DataFrame:
        sub_ids = [s["config"] for s in config["sub_strategies"]]
        wsum = sum(weights) or 1.0
        static = pd.Series([w / wsum for w in weights], index=sub_ids, dtype=float)
        out = pd.DataFrame(
            np.tile(static.to_numpy(), (len(idx), 1)), index=idx, columns=sub_ids)

        alloc = config.get("currency_allocation") or {}
        kind = str(alloc.get("kind", "")).lower()
        if kind == "relative_momentum_fx_tilt":
            return cls._relative_momentum_fx_weights(config, raw, idx, out, alloc)
        if kind == "relative_momentum_fx2_overlay":
            return cls._relative_momentum_fx2_weights(config, raw, idx, out, alloc)
        if kind != "world_bank_buffett_change_scaled_revert":
            return out
        # 합성 테스트/오프라인 계약에서는 외부 World Bank 호출을 피하고 정적 비중으로 둔다.
        if "__FX__" not in raw.columns.get_level_values(0):
            return out
        if len(config["sub_strategies"]) != 2:
            return out

        k = float(alloc.get("k", 1.0))
        band = alloc.get("band", [0.0, 1.0])
        lo, hi = float(band[0]), float(band[1])
        kr_w = cls._buffett_change_scaled_revert_wkr(idx, k=k, band=(lo, hi))
        if kr_w.empty:
            return out

        dyn = out.copy()
        base_cur = str(config.get("base_currency", "KRW"))
        for sub in config["sub_strategies"]:
            sid = sub["config"]
            cur = str(sub.get("currency", base_cur))
            if cur == "KRW":
                dyn[sid] = kr_w.reindex(idx).ffill().bfill()
            elif cur == "USD":
                dyn[sid] = 1.0 - kr_w.reindex(idx).ffill().bfill()
            else:
                return out
        denom = dyn.sum(axis=1).replace(0.0, np.nan)
        return dyn.div(denom, axis=0).fillna(out)

    @classmethod
    def _relative_momentum_fx_weights(
        cls,
        config: dict[str, Any],
        raw: pd.DataFrame,
        idx: pd.DatetimeIndex,
        static: pd.DataFrame,
        alloc: dict[str, Any],
    ) -> pd.DataFrame:
        """Daily target path for monthly USD/KRW rebalance.

        The signal is shifted by one bar by default, so rebalances use only
        information available before the rebalance date.
        """
        if "__BENCH__" not in raw.columns.get_level_values(0):
            return static
        if "__FX__" not in raw.columns.get_level_values(0):
            return static
        if len(config["sub_strategies"]) != 2:
            return static

        base_cur = str(config.get("base_currency", "KRW"))
        fx_sym = str(config.get("fx_symbol", ""))
        bench_map = config.get("benchmark_symbols") or {}
        usd_sid = krw_sid = None
        for sub in config["sub_strategies"]:
            sid = str(sub["config"])
            cur = str(sub.get("currency", base_cur))
            if cur == "USD":
                usd_sid = sid
            elif cur == "KRW":
                krw_sid = sid
            else:
                return static
        if usd_sid is None or krw_sid is None:
            return static

        usd_sym = bench_map.get("USD")
        krw_sym = bench_map.get("KRW")
        bench_cols = raw["__BENCH__"].columns.get_level_values(0)
        if usd_sym not in bench_cols or krw_sym not in bench_cols:
            return static

        fx = cls._fx(raw, fx_sym)
        if fx is None:
            return static
        lookback = int(alloc.get("lookback", 126))
        fx_lookback = int(alloc.get("fx_lookback", lookback))
        if lookback <= 0 or fx_lookback <= 0:
            return static
        signal_lag = int(alloc.get("signal_lag", 1))
        neutral = float(alloc.get("neutral", 0.5))
        tilt = float(alloc.get("tilt", 0.15))
        scale = max(float(alloc.get("scale", 0.10)), 1e-9)
        fx_weight = float(alloc.get("fx_weight", 0.5))
        band = alloc.get("band", [0.0, 1.0])
        lo, hi = float(band[0]), float(band[1])

        usd_px = raw["__BENCH__"][usd_sym]["close"].reindex(idx).ffill()
        krw_px = raw["__BENCH__"][krw_sym]["close"].reindex(idx).ffill()
        fx = pd.Series(fx).reindex(idx).ffill()
        if usd_px.dropna().empty or krw_px.dropna().empty or fx.dropna().empty:
            return static

        usd_nav = usd_px / usd_px.dropna().iloc[0]
        krw_nav = krw_px / krw_px.dropna().iloc[0]
        fx_nav = fx / fx.dropna().iloc[0]
        usd_base_nav = usd_nav * fx_nav

        relative_momentum = (
            usd_base_nav.pct_change(lookback) - krw_nav.pct_change(lookback)
        )
        fx_trend = fx_nav.pct_change(fx_lookback)
        score = relative_momentum + fx_weight * fx_trend
        usd_w = neutral + tilt * np.tanh(score / scale)
        usd_w = usd_w.clip(lo, hi).shift(signal_lag)
        usd_w = usd_w.reindex(idx).ffill()
        usd_w = usd_w.fillna(static[usd_sid])

        dyn = static.copy()
        dyn[usd_sid] = usd_w
        dyn[krw_sid] = 1.0 - usd_w
        denom = dyn.sum(axis=1).replace(0.0, np.nan)
        return dyn.div(denom, axis=0).fillna(static)

    @classmethod
    def _relative_momentum_fx2_weights(
        cls,
        config: dict[str, Any],
        raw: pd.DataFrame,
        idx: pd.DatetimeIndex,
        static: pd.DataFrame,
        alloc: dict[str, Any],
    ) -> pd.DataFrame:
        """Relative USD/KRW tilt with fast override, vol blend, and cash gate."""
        if "__BENCH__" not in raw.columns.get_level_values(0):
            return static
        if "__FX__" not in raw.columns.get_level_values(0):
            return static
        if len(config["sub_strategies"]) != 2:
            return static

        base_cur = str(config.get("base_currency", "KRW"))
        fx_sym = str(config.get("fx_symbol", ""))
        bench_map = config.get("benchmark_symbols") or {}
        usd_sid = krw_sid = None
        for sub in config["sub_strategies"]:
            sid = str(sub["config"])
            cur = str(sub.get("currency", base_cur))
            if cur == "USD":
                usd_sid = sid
            elif cur == "KRW":
                krw_sid = sid
        if usd_sid is None or krw_sid is None:
            return static

        usd_sym = bench_map.get("USD")
        krw_sym = bench_map.get("KRW")
        bench_cols = raw["__BENCH__"].columns.get_level_values(0)
        if usd_sym not in bench_cols or krw_sym not in bench_cols:
            return static
        fx = cls._fx(raw, fx_sym)
        if fx is None:
            return static

        lookback = int(alloc.get("lookback", 63))
        signal_lag = int(alloc.get("signal_lag", 1))
        neutral = float(alloc.get("neutral", 0.5))
        tilt = float(alloc.get("tilt", 0.125))
        scale = max(float(alloc.get("scale", 0.10)), 1e-9)
        band = alloc.get("band", [0.40, 0.60])
        lo, hi = float(band[0]), float(band[1])

        usd_px = raw["__BENCH__"][usd_sym]["close"].reindex(idx).ffill()
        krw_px = raw["__BENCH__"][krw_sym]["close"].reindex(idx).ffill()
        fx = pd.Series(fx).reindex(idx).ffill()
        if usd_px.dropna().empty or krw_px.dropna().empty or fx.dropna().empty:
            return static

        usd_nav = usd_px / usd_px.dropna().iloc[0]
        krw_nav = krw_px / krw_px.dropna().iloc[0]
        fx_nav = fx / fx.dropna().iloc[0]
        usd_base_nav = usd_nav * fx_nav
        score = usd_base_nav.pct_change(lookback) - krw_nav.pct_change(lookback)
        usd_w = neutral + tilt * np.tanh(score / scale)
        usd_w = usd_w.clip(lo, hi)

        override = alloc.get("fast_override") or {}
        if override.get("enabled", False):
            fast_lb = int(override.get("lookback", 20))
            fast_score = usd_base_nav.pct_change(fast_lb) - krw_nav.pct_change(fast_lb)
            fx_fast = fx_nav.pct_change(fast_lb)
            trigger = (
                (usd_w <= float(override.get("usd_floor_trigger", 0.42)))
                & (
                    (fast_score >= float(override.get("relative_threshold", 0.03)))
                    | (fx_fast >= float(override.get("fx_threshold", 0.03)))
                )
            )
            usd_w = usd_w.where(~trigger, np.maximum(usd_w, float(override.get("min_usd", 0.50))))

        vol_adjust = alloc.get("vol_adjust") or {}
        if vol_adjust.get("enabled", False):
            vol_lb = int(vol_adjust.get("lookback", 63))
            blend = float(vol_adjust.get("blend", 0.35))
            us_vol = usd_base_nav.pct_change().rolling(vol_lb).std()
            kr_vol = krw_nav.pct_change().rolling(vol_lb).std()
            vol_w = (kr_vol / (us_vol + kr_vol).replace(0.0, np.nan)).clip(lo, hi)
            usd_w = ((1.0 - blend) * usd_w + blend * vol_w).clip(lo, hi)

        exposure = pd.Series(1.0, index=idx)
        gate = alloc.get("risk_gate") or {}
        if gate.get("enabled", False):
            source = str(gate.get("source", "trend")).lower()
            off_scale = float(gate.get("off_scale", 0.65))
            severity: pd.Series | None = None
            if source in ("credit_spread", "credit_and_vix"):
                severity = cls._credit_spread_severity(raw, idx, gate)
                weak = severity > 0.0
                if source == "credit_and_vix":
                    vix_flag = cls._vix_weak(raw, idx, gate)
                    weak = weak & vix_flag
                    severity = severity.where(vix_flag, 0.0)
            else:
                lb = int(gate.get("lookback", 63))
                sma_lb = int(gate.get("sma_lookback", 200))
                dd_threshold = float(gate.get("dd_threshold", -0.10))
                vol_threshold = float(gate.get("vol_threshold", 0.24))
                us_ret = usd_base_nav.pct_change(lb)
                kr_ret = krw_nav.pct_change(lb)
                us_sma = usd_base_nav.rolling(sma_lb).mean()
                kr_sma = krw_nav.rolling(sma_lb).mean()
                us_dd = usd_base_nav / usd_base_nav.rolling(lb).max() - 1.0
                kr_dd = krw_nav / krw_nav.rolling(lb).max() - 1.0
                us_vol = usd_base_nav.pct_change().rolling(lb).std() * np.sqrt(252.0)
                kr_vol = krw_nav.pct_change().rolling(lb).std() * np.sqrt(252.0)
                weak = (
                    (us_ret < 0.0) & (kr_ret < 0.0)
                    & (usd_base_nav < us_sma) & (krw_nav < kr_sma)
                    & ((us_dd < dd_threshold) | (kr_dd < dd_threshold) | (np.maximum(us_vol, kr_vol) > vol_threshold))
                )
            weak = weak.reindex(idx).fillna(False)

            min_consecutive = int(gate.get("min_consecutive", 1))
            if min_consecutive > 1:
                confirmed = weak.rolling(min_consecutive).sum() >= min_consecutive
                weak = confirmed & weak
                if severity is not None:
                    severity = severity.where(weak, 0.0)

            continuous = gate.get("continuous") or {}
            if continuous.get("enabled", False) and severity is not None:
                cap = max(float(continuous.get("severity_cap", 2.0)), 1e-9)
                severity_frac = (severity.clip(upper=cap) / cap).clip(0.0, 1.0)
                exposure = 1.0 - severity_frac * (1.0 - off_scale)
            else:
                exposure = exposure.where(~weak, off_scale)

        usd_w = usd_w.shift(signal_lag).reindex(idx).ffill().fillna(static[usd_sid])
        exposure = exposure.shift(signal_lag).reindex(idx).ffill().fillna(1.0).clip(0.0, 1.0)
        dyn = static.copy()
        dyn[usd_sid] = usd_w * exposure
        dyn[krw_sid] = (1.0 - usd_w) * exposure
        dyn["__CASH__"] = 1.0 - exposure
        denom = dyn.sum(axis=1).replace(0.0, np.nan)
        return dyn.div(denom, axis=0).fillna(static)

    @staticmethod
    def _credit_spread_severity(
        raw: pd.DataFrame, idx: pd.DatetimeIndex, gate: dict[str, Any],
    ) -> pd.Series:
        """신용스프레드 대리(예: HYG/IEF) 상대모멘텀 붕괴 강도(0=미달, 1=임계값,
        1보다 크면 더 심함).

        추세확인형 게이트(가격이 200일선 아래로 확정된 뒤 발동)보다 리드타임이
        길다(과거 22개 드로다운 에피소드 중앙값 -48거래일, `yoon2_final_fx_
        leading_signals.md` 참고). 급성 신용경색형 크래시(2020-02 코로나 등)에서
        특히 유효하고, 완만한 약세장에는 별 효과가 없다. `weak = severity > 0`로
        이진화해서 쓰거나, `continuous.enabled`로 세기에 비례한 노출 축소에 쓴다.
        """
        if "__GATE__" not in raw.columns.get_level_values(0):
            return pd.Series(0.0, index=idx)
        symbols = gate.get("symbols", [])
        if len(symbols) != 2:
            return pd.Series(0.0, index=idx)
        sym_a, sym_b = symbols
        gate_cols = raw["__GATE__"].columns.get_level_values(0)
        if sym_a not in gate_cols or sym_b not in gate_cols:
            return pd.Series(0.0, index=idx)
        pa = raw["__GATE__"][sym_a]["close"].reindex(idx).ffill()
        pb = raw["__GATE__"][sym_b]["close"].reindex(idx).ffill()
        ratio = (pa / pb).reindex(idx).ffill()
        lookback = int(gate.get("lookback", 10))
        sma_lookback = int(gate.get("sma_lookback", 20))
        threshold = float(gate.get("threshold", -0.02))
        mom = ratio.pct_change(lookback)
        below_sma = ratio < ratio.rolling(sma_lookback).mean()
        breach = (threshold - mom) / abs(threshold)
        severity = breach.clip(lower=0.0).where(below_sma, 0.0)
        return severity.fillna(0.0)

    @staticmethod
    def _vix_weak(raw: pd.DataFrame, idx: pd.DatetimeIndex, gate: dict[str, Any]) -> pd.Series:
        """VIX 백워데이션(현물 VIX > VIX3M) & VIX가 바닥값 이상 -> 위험회피 플래그.

        신용스프레드 신호와 AND 결합해 서로 다른 신호원이 동시에 확인할 때만
        발동시켜 단독 신호의 노이즈를 줄이는 용도(`source="credit_and_vix"`).
        """
        if "__GATE__" not in raw.columns.get_level_values(0):
            return pd.Series(False, index=idx)
        vix_sym, vix3m_sym = gate.get("vix_symbols", ["^VIX", "^VIX3M"])
        gate_cols = raw["__GATE__"].columns.get_level_values(0)
        if vix_sym not in gate_cols or vix3m_sym not in gate_cols:
            return pd.Series(False, index=idx)
        vix = raw["__GATE__"][vix_sym]["close"].reindex(idx).ffill()
        vix3m = raw["__GATE__"][vix3m_sym]["close"].reindex(idx).ffill()
        floor = float(gate.get("vix_floor", 18.0))
        return ((vix > vix3m) & (vix > floor)).fillna(False)

    @classmethod
    def _buffett_change_scaled_revert_wkr(
        cls, idx: pd.DatetimeIndex, *, k: float, band: tuple[float, float],
    ) -> pd.Series:
        annual = pd.concat({
            "us": cls._fetch_world_bank_buffett("USA"),
            "kr": cls._fetch_world_bank_buffett("KOR"),
        }, axis=1).sort_index()
        growth = annual / annual.shift(1)
        raw = growth["us"] / (growth["kr"] + growth["us"])
        scaled = (0.5 + k * (raw - 0.5)).clip(*band)
        return scaled.reindex(idx.union(scaled.index)).ffill().reindex(idx).dropna()

    @staticmethod
    def _fetch_world_bank_buffett(country: str) -> pd.Series:
        indicator = "CM.MKT.LCAP.GD.ZS"
        url = (
            "https://api.worldbank.org/v2/country/"
            f"{country}/indicator/{indicator}?format=json&per_page=1000"
        )
        last_error: Exception | None = None
        for attempt in range(5):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=20) as resp:
                    payload = json.load(resp)
                rows = payload[1] if len(payload) > 1 else []
                vals = {
                    int(row["date"]): float(row["value"])
                    for row in rows
                    if row.get("value") is not None
                }
                if not vals:
                    raise ValueError(f"World Bank returned no values for {country}")
                # 관측연도 Y 값은 다음 해 1월 1일부터 사용 가능하다고 둔다.
                dates = pd.to_datetime([f"{year + 1}-01-01" for year in vals])
                return pd.Series(list(vals.values()), index=dates, name=country).sort_index()
            except Exception as exc:  # pragma: no cover - network retry guard
                last_error = exc
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"World Bank fetch failed for {country}: {last_error}") from last_error

    @staticmethod
    def _combine_values(navs_weights, idx, freq, columns) -> pd.DataFrame:
        """가중 통화 슬리브를 주기적 목표비중 복원(리밸런싱) 결합하고, 각 슬리브의
        기준통화 평가가치(행 합 = 총 NAV) 행렬을 1.0 기준으로 돌려준다.
        총 NAV는 행 합으로 복원되고, 행별 비중이 곧 자산군 구성(웨이브)이다."""
        reb = rebalance_dates(idx, freq)
        rets = [(n.pct_change().fillna(0.0).to_numpy(), w) for n, w in navs_weights]
        vals = [w for _, w in navs_weights]
        out = np.empty((len(idx), len(rets)))
        for i, t in enumerate(idx):
            for j, (r, _) in enumerate(rets):
                vals[j] *= 1.0 + r[i]
            out[i, :] = vals
            if t in reb:
                tot = sum(vals)
                vals = [tot * w for _, w in navs_weights]
        return pd.DataFrame(out, index=idx, columns=list(columns))

    @staticmethod
    def _combine_values_dynamic(
        navs: list[pd.Series], target_weights: pd.DataFrame,
        idx: pd.DatetimeIndex, freq: str, columns,
    ) -> pd.DataFrame:
        reb = rebalance_dates(idx, freq)
        tw = target_weights.reindex(idx).ffill().bfill()
        tw = tw.div(tw.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
        vals = tw.iloc[0].to_numpy(dtype=float).tolist()
        rets = [n.pct_change().fillna(0.0).to_numpy() for n in navs]
        out = np.empty((len(idx), len(rets)))
        for i, t in enumerate(idx):
            for j, r in enumerate(rets):
                vals[j] *= 1.0 + r[i]
            out[i, :] = vals
            if t in reb:
                total = sum(vals)
                target = tw.iloc[i].to_numpy(dtype=float)
                vals = (total * target).tolist()
        return pd.DataFrame(out, index=idx, columns=list(columns))

    @classmethod
    def _combine(cls, navs_weights, idx, freq) -> pd.Series:
        """가중 통화 슬리브 결합 총 NAV(1.0 기준). 슬리브별 평가가치의 행 합."""
        cols = list(range(len(navs_weights)))
        return cls._combine_values(navs_weights, idx, freq, cols).sum(axis=1)

    def _benchmark(self, raw, config, idx, freq, base_cur, fx_sym) -> pd.Series | None:
        """전략 없는 동일 통화배분 패시브(50/50 시장 buy&hold), 기준통화."""
        if "__BENCH__" not in raw.columns.get_level_values(0):
            return None
        bench_map = config.get("benchmark_symbols") or {}
        legs = []
        n_sub = len(config["sub_strategies"])
        for sub in config["sub_strategies"]:
            cur = str(sub.get("currency", base_cur))
            sym = bench_map.get(cur)
            if sym is None or sym not in raw["__BENCH__"].columns.get_level_values(0):
                return None
            px = raw["__BENCH__"][sym]["close"].reindex(idx).ffill()
            nav = px / px.dropna().iloc[0]
            if cur != base_cur:
                nav = self._to_base(nav, self._fx(raw, fx_sym))
            legs.append((nav, float(sub.get("weight", 1.0 / n_sub))))
        wsum = sum(w for _, w in legs) or 1.0
        legs = [(n, w / wsum) for n, w in legs]
        return self._combine(legs, idx, freq)

    @staticmethod
    def _bench_perf(bench, window, interval) -> dict:
        b = pd.Series(bench).reindex(window).ffill().dropna()
        if b.empty:
            return {}
        eq = b / b.iloc[0]
        return performance(eq, eq.pct_change().fillna(0.0), interval)

    @staticmethod
    def _slice(index, phase, config):
        if phase == "all":
            return index
        n = len(index)
        tf = float(config.get("train_frac", 0.6))
        vf = float(config.get("validation_frac", 0.2))
        te, ve = int(n * tf), int(n * (tf + vf))
        if phase == "train":
            return index[:te]
        if phase == "validation":
            return index[te:ve]
        if phase == "test":
            return index[ve:]
        return index

    @staticmethod
    def _slice_trades(sub_trades, window) -> pd.DataFrame:
        if not sub_trades or len(window) == 0:
            return pd.DataFrame(columns=_TRADE_COLS)
        t = pd.concat(sub_trades, ignore_index=True)
        lo, hi = window[0], window[-1]
        et = pd.to_datetime(t["entry_time"])
        t = t[(et >= lo) & (et <= hi)]
        for c in _TRADE_COLS:
            if c not in t.columns:
                t[c] = None
        return t[_TRADE_COLS].reset_index(drop=True)
