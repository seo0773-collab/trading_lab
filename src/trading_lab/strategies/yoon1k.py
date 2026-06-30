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

        # 공통 인덱스 + 가중 리밸런싱 결합(전체 구간)
        idx = navs[0].index
        for n in navs[1:]:
            idx = idx.intersection(n.index)
        idx = idx.sort_values()
        sub_ids = [s["config"] for s in config["sub_strategies"]]
        target_weights = self._target_weight_frame(config, raw, idx, weights)
        weights = target_weights.iloc[0].to_list()
        combo_values = self._combine_values_dynamic(
            [n.reindex(idx) for n in navs], target_weights, idx, freq, sub_ids)
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
        for sid in sub_ids:
            forecast[f"target_w_{sid}"] = target_weights[sid].reindex(window)

        # 자산군(통화 슬리브)별 평가가치 비중(행 합=100%) → 누적 면적 '웨이브'.
        # 평가금액 자체는 초기금액·적립방식 등 대시보드 입력에 종속되므로(금액 환산은
        # 대시보드 책임) 비중으로 노출한다 — 결과 화면이 계좌금액과 곱해 읽으면 된다.
        cur_label = {s["config"]: str(s.get("currency", base_cur))
                     for s in config["sub_strategies"]}
        share_win = combo_values.reindex(window)
        shares = share_win.div(share_win.sum(axis=1), axis=0).fillna(0.0)
        shares = shares.rename(columns={
            sid: f"{cur_label[sid]} ({sid})" for sid in sub_ids})
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
        if str(alloc.get("kind", "")).lower() != "world_bank_buffett_change_scaled_revert":
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
