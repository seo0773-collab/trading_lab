"""yoon4 — Global Capital Flow Regime Engine (글로벌 자금흐름 국가 로테이션).

`new_layer.txt`의 설계를 1차 구현한다. 경제를 예측하는 게 아니라 **글로벌 자금
이동을 추적해 투자 국가를 고른다**: Global Regime → Country Selection → (리프) Trading.

    [최상위] 글로벌 Risk-On/Off 레짐 게이트 (총 주식노출 조절)
       └─ 국가 선택: Capital Rotation Index(CRI) 상위 N개 국가 ETF 비중 배분
            └─ 리프(v1): 선택 국가 ETF 직접 보유 (섹터 계층은 후속 레이어)

yoon2(미국/한국 50:50 고정)를 일반화한 것 — 고정 2국가 대신 다국가 동적 로테이션.
국가 ETF·안전자산은 전부 **미국 상장**이라 한국투자증권(KIS) 해외주식으로 거래 가능하고,
단일 통화(USD)라 yoon2의 다통화 리밸런싱 복잡도가 사라진다(원화 환산은 표시용 1회).

CRI 구성(국가 간 Z-score 정규화 후 가중합, `config.cri.weights`로 분리해 최적화 가능):
모멘텀 3M/6M/12M · SPY 대비 상대강도 · 52주 신고가 거리 · 거래량 추세 · 변동성 축소 ·
Flow 가속도. (통화 강도는 USD 표시 ETF 수익률에 이미 반영되므로 별도 FX 팩터 없음.)

config 스키마(configs/strategies/yoon4.json):
    base_currency, fx_symbol, rebalance_freq, countries[{symbol,label,exchange}],
    top_n, cri{rs_benchmark, lookbacks, weights}, regime{...}, safe_sleeve{...},
    rebalance{min_weight_change, fee_bps_per_side, slippage_bps},
    benchmark_symbol, train_frac, validation_frac.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_lab.strategies.base import StrategyArtifacts
from trading_lab.strategies.yoon1k import Yoon1kHandler

ROOT = Path(__file__).resolve().parents[3]
for _p in (str(ROOT / "scripts"), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from profile_sizing.engine import performance  # noqa: E402
from profile_sizing.portfolio import rebalance_dates  # noqa: E402
from profile_sizing.synthetic import make_synthetic_ohlcv  # noqa: E402

_TRADE_COLS = ["direction", "entry_time", "entry_price", "exit_time",
               "exit_price", "net_return", "exit_reason", "entry_reason", "symbol"]


class Yoon4Handler(Yoon1kHandler):
    """글로벌 자금흐름(CRI) 국가 로테이션 + Risk-On/Off 레짐 게이트."""

    # ------------------------------------------------------------------ #
    # 데이터 로딩: 국가 ETF + 레짐 + 안전자산 + 환율 + 벤치를 3레벨 wide로.
    # ------------------------------------------------------------------ #
    def load_data(
        self, symbol: str, config: dict[str, Any], *,
        csv_path: Path | None = None, synthetic: bool = False,
    ) -> pd.DataFrame:
        interval = str(config.get("interval", "1d"))
        period = str(config.get("period", "max"))
        country_syms = [str(c["symbol"]).upper() for c in config["countries"]]
        regime = config.get("regime") or {}
        regime_syms = [str(regime.get(k)) for k in ("vix_symbol", "dxy_symbol", "spy_symbol")
                       if regime.get(k)]
        safe_syms = [str(s).upper() for s in (config.get("safe_sleeve") or {}).get("symbols", [])]

        if synthetic:
            return self._synthetic_frame(config, country_syms, regime_syms, safe_syms, interval)

        from run_kalman_pipeline import load_yfinance  # noqa: E402

        def _grp(symbols: list[str]) -> dict[str, pd.DataFrame]:
            out: dict[str, pd.DataFrame] = {}
            for sym in symbols:
                try:
                    out[sym] = load_yfinance(sym, interval, period)
                except Exception:  # noqa: BLE001  — 결측 심볼은 건너뛴다.
                    continue
            return out

        frames: dict[str, pd.DataFrame] = {}
        frames["__COUNTRY__"] = pd.concat(_grp(country_syms), axis=1)
        reg = _grp(regime_syms)
        if reg:
            frames["__REGIME__"] = pd.concat(reg, axis=1)
        if safe_syms:
            safe = _grp(safe_syms)
            if safe:
                frames["__SAFE__"] = pd.concat(safe, axis=1)
        fx_sym = str(config.get("fx_symbol", ""))
        if fx_sym:
            try:
                frames["__FX__"] = pd.concat({fx_sym: load_yfinance(fx_sym, interval, period)}, axis=1)
            except Exception:  # noqa: BLE001
                pass
        bench_sym = str(config.get("benchmark_symbol", ""))
        if bench_sym:
            try:
                frames["__BENCH__"] = pd.concat(
                    {bench_sym: load_yfinance(bench_sym, interval, period)}, axis=1)
            except Exception:  # noqa: BLE001
                pass
        return pd.concat(frames, axis=1)

    @staticmethod
    def _synthetic_frame(config, country_syms, regime_syms, safe_syms, interval) -> pd.DataFrame:
        """재현 가능한 합성 데이터(계약 테스트 경로). 심볼마다 seed offset으로 분산."""
        bars = int(config.get("synthetic_bars", 2000))
        seed = int(config.get("seed", 7))
        frames: dict[str, pd.DataFrame] = {}

        def _grp(symbols, base_off):
            return {s: make_synthetic_ohlcv(bars, seed + base_off + i, interval)
                    for i, s in enumerate(symbols)}

        frames["__COUNTRY__"] = pd.concat(_grp(country_syms, 100), axis=1)
        if regime_syms:
            frames["__REGIME__"] = pd.concat(_grp(regime_syms, 300), axis=1)
        if safe_syms:
            frames["__SAFE__"] = pd.concat(_grp(safe_syms, 400), axis=1)
        # 합성 경로는 FX/벤치를 생략 — _fx()=None이면 원화 환산은 항등(단위 유지).
        return pd.concat(frames, axis=1)

    # ------------------------------------------------------------------ #
    # 아티팩트: CRI 국가선택 + 레짐 게이트 + 안전슬리브 시뮬레이션.
    # ------------------------------------------------------------------ #
    def build_artifacts(
        self, raw: pd.DataFrame, config: dict[str, Any], *,
        symbol: str, phase: str, bars_per_year: int,
    ) -> StrategyArtifacts:
        interval = str(config.get("interval", "1d"))
        base_cur = str(config.get("base_currency", "KRW"))
        freq = str(config.get("rebalance_freq", "monthly"))
        fx_sym = str(config.get("fx_symbol", ""))

        countries = config["countries"]
        country_syms = [str(c["symbol"]).upper() for c in countries]
        labels = {str(c["symbol"]).upper(): str(c.get("label", c["symbol"])) for c in countries}
        top_n = int(config.get("top_n", 4))

        closes = self._wide_field(raw, "__COUNTRY__", country_syms, "close")
        vols = self._wide_field(raw, "__COUNTRY__", country_syms, "volume")
        idx = closes.dropna(how="all").index.sort_values()
        # 원화 환산이 필요하면 idx를 환율 가용 구간으로 정렬한다 — 국가 ETF(예: SPY 1993)가
        # USD/KRW(KRW=X ~2003)보다 먼저 시작하면 환산 NAV 선두가 NaN이 되어 지표가 깨진다.
        fx_ser = self._fx(raw, fx_sym)
        if base_cur != "USD" and fx_ser is not None:
            fx_idx = pd.Series(fx_ser).dropna().index
            if len(fx_idx):
                idx = idx.intersection(fx_idx).sort_values()
        closes = closes.reindex(idx)
        vols = vols.reindex(idx)
        rets = closes.pct_change()

        spy = self._regime_close(raw, config, "spy_symbol", idx)
        factors = self._cri_factors(config, closes, vols, rets, spy)
        equity_scale = self._regime_scale(raw, config, idx)

        safe_cfg = config.get("safe_sleeve") or {}
        safe_syms = [str(s).upper() for s in safe_cfg.get("symbols", [])] if safe_cfg.get("enabled", True) else []
        safe_close = self._wide_field(raw, "__SAFE__", safe_syms, "close").reindex(idx).ffill()
        safe_max = float(safe_cfg.get("max", 0.4))
        safe_ma_len = int(safe_cfg.get("ma_len", 100))
        safe_ma = safe_close.rolling(safe_ma_len, min_periods=1).mean()

        reb = rebalance_dates(idx, freq)
        weights_cfg = dict((config.get("cri") or {}).get("weights", {}))
        targets, episodes = self._rotation_targets(
            idx, reb, factors, weights_cfg, country_syms, top_n,
            equity_scale, safe_syms, safe_close, safe_ma, safe_max, closes, labels)

        assets = country_syms + safe_syms
        ret_mat = pd.concat([rets[country_syms], safe_close[safe_syms].pct_change()],
                            axis=1).reindex(idx).fillna(0.0).to_numpy()
        rcost = float((config.get("rebalance") or {}).get("fee_bps_per_side", 5.0)) \
            + float((config.get("rebalance") or {}).get("slippage_bps", 5.0))
        thr = float((config.get("rebalance") or {}).get("min_weight_change", 0.03))

        values, cash = self._simulate(idx, assets, ret_mat, targets, reb, thr, rcost / 1e4)

        nav_usd = pd.Series(values.sum(axis=1) + cash, index=idx)
        nav_krw = self._to_base(nav_usd / nav_usd.iloc[0], self._fx(raw, fx_sym))

        window = self._slice(idx, phase, config)
        eq = nav_krw.reindex(window)
        equity = (eq / eq.iloc[0]).rename("equity")
        ret = equity.pct_change().fillna(0.0)
        perf = performance(equity, ret, interval)

        forecast = self._forecast(window, equity, values, cash, country_syms,
                                  safe_syms, assets, idx, equity_scale, targets)
        sleeve_weights = self._wave(window, values, cash, assets, labels, safe_syms, idx)
        trades = self._slice_trades([self._episodes_frame(episodes)], window) \
            if episodes else pd.DataFrame(columns=_TRADE_COLS)

        bench, bperf = self._bench(raw, config, idx, window, base_cur, fx_sym, interval)
        n = int(len(trades))
        trets = trades["net_return"].astype(float) if n else pd.Series(dtype=float)
        metrics = {
            "trades": n,
            "hit_rate": float((trets > 0).mean()) if n else None,
            "total_return": perf.get("total_return"),
            "sharpe": perf.get("sharpe"),
            "max_drawdown": perf.get("max_drawdown"),
            "cagr": perf.get("cagr"),
            "volatility": perf.get("volatility"),
            "buy_hold_return": bperf.get("total_return"),
            "buy_hold_sharpe": bperf.get("sharpe"),
            "buy_hold_max_drawdown": bperf.get("max_drawdown"),
            "buy_hold_cagr": bperf.get("cagr"),
            "benchmark_kind": str(config.get("benchmark_label", "B&H")),
            "phase": phase,
        }
        metadata = {
            "countries": country_syms,
            "top_n": top_n,
            "base_currency": base_cur,
            "fx_symbol": fx_sym,
            "rebalance_freq": freq,
            "regime_enabled": bool((config.get("regime") or {}).get("enabled", True)),
        }
        return StrategyArtifacts(
            forecast=forecast, trades=trades, equity=equity,
            metrics=metrics, metadata=metadata, benchmark=bench,
            extras={
                "sleeve_weights": sleeve_weights,
                "perf_vs_bnh": self._perf_table(perf, bperf, label=str(config.get("version", "yoon4"))),
            })

    # ---------------------------- CRI ---------------------------------- #
    @staticmethod
    def _wide_field(raw: pd.DataFrame, group: str, symbols: list[str], field: str) -> pd.DataFrame:
        """3레벨 wide에서 group의 각 symbol의 field 컬럼만 뽑아 [time × symbol] 프레임."""
        cols = {}
        if group in raw.columns.get_level_values(0):
            sub = raw[group]
            for sym in symbols:
                if sym in sub.columns.get_level_values(0) and field in sub[sym].columns:
                    cols[sym] = sub[sym][field]
        if not cols:
            return pd.DataFrame(index=raw.index)
        return pd.DataFrame(cols)

    def _regime_close(self, raw, config, key, idx) -> pd.Series | None:
        sym = str((config.get("regime") or {}).get(key, "")).upper()
        if not sym:
            return None
        for grp in ("__REGIME__", "__COUNTRY__", "__BENCH__"):
            s = self._wide_field(raw, grp, [sym], "close")
            if sym in s.columns:
                return s[sym].reindex(idx).ffill()
        return None

    @staticmethod
    def _cri_factors(config, closes, vols, rets, spy) -> dict[str, pd.DataFrame]:
        L = dict((config.get("cri") or {}).get("lookbacks", {}))
        m3, m6, m12 = int(L.get("mom_3m", 63)), int(L.get("mom_6m", 126)), int(L.get("mom_12m", 252))
        hi_w = int(L.get("hi_window", 252))
        vol_w = int(L.get("vol_window", 63))
        volu_w = int(L.get("volume_window", 63))
        accel_w = int(L.get("accel_window", 21))
        f: dict[str, pd.DataFrame] = {}
        f["mom_3m"] = closes / closes.shift(m3) - 1.0
        f["mom_6m"] = closes / closes.shift(m6) - 1.0
        f["mom_12m"] = closes / closes.shift(m12) - 1.0
        if spy is not None:
            ratio = closes.div(spy, axis=0)
            f["rs"] = ratio / ratio.shift(m6) - 1.0
        else:
            f["rs"] = f["mom_6m"] * 0.0
        f["hi_dist"] = closes / closes.rolling(hi_w, min_periods=hi_w // 2).max() - 1.0
        vshort = vols.rolling(21, min_periods=5).mean()
        vlong = vols.rolling(volu_w, min_periods=volu_w // 2).mean()
        f["vol_trend"] = vshort / vlong - 1.0
        rv_s = rets.rolling(21, min_periods=5).std()
        rv_l = rets.rolling(vol_w, min_periods=vol_w // 2).std()
        f["vol_contraction"] = -(rv_s / rv_l - 1.0)
        f["acceleration"] = f["mom_3m"].diff(accel_w)
        return f

    @staticmethod
    def _zscore(row: pd.Series) -> pd.Series:
        v = row.astype(float)
        sd = v.std(ddof=0)
        if not np.isfinite(sd) or sd <= 0:
            return v * 0.0
        return (v - v.mean()) / sd

    def _rotation_targets(self, idx, reb, factors, weights_cfg, country_syms, top_n,
                          equity_scale, safe_syms, safe_close, safe_ma, safe_max,
                          closes, labels):
        """각 리밸 시점의 자산별 목표비중(dict t->array) + 국가 보유 에피소드."""
        assets = country_syms + safe_syms
        a_idx = {a: i for i, a in enumerate(assets)}
        targets: dict[pd.Timestamp, np.ndarray] = {}
        open_pos: dict[str, tuple] = {}
        episodes: list[dict] = []
        for t in idx:
            if t not in reb:
                continue
            valid = factors["mom_12m"].loc[t].notna()
            sel_w: dict[str, float] = {}
            if valid.any():
                cri = pd.Series(0.0, index=country_syms)
                for name, w in weights_cfg.items():
                    if name not in factors or not w:
                        continue
                    z = self._zscore(factors[name].loc[t][valid]).fillna(0.0)
                    cri.loc[z.index] += float(w) * z
                cri = cri[valid].dropna()
                if not cri.empty:
                    sel = cri.sort_values(ascending=False).head(top_n)
                    pos = sel.clip(lower=0.0)
                    base = (pos / pos.sum()) if pos.sum() > 0 else \
                        pd.Series(1.0 / len(sel), index=sel.index)
                    scale = float(equity_scale.get(t, 1.0))
                    sel_w = {s: float(base[s] * scale) for s in sel.index}

            # 안전 슬리브: 주식에 안 쓴 spare를 추세 ON 안전자산으로 회전.
            spare = max(0.0, 1.0 - sum(sel_w.values()))
            safe_total = min(spare, safe_max)
            safe_on = [s for s in safe_syms
                       if s in safe_close.columns
                       and float(safe_close[s].get(t, np.nan)) > float(safe_ma[s].get(t, np.nan))]
            safe_w = {s: safe_total / len(safe_on) for s in safe_on} if (safe_on and safe_total > 0) else {}

            vec = np.zeros(len(assets))
            for s, w in {**sel_w, **safe_w}.items():
                vec[a_idx[s]] = w
            targets[t] = vec

            # 보유 에피소드(국가 선택 진입/탈락) — trades용.
            selected = set(sel_w)
            for s in selected - set(open_pos):
                open_pos[s] = (t, float(closes[s].get(t, np.nan)))
            for s in set(open_pos) - selected:
                et, ep = open_pos.pop(s)
                xp = float(closes[s].get(t, np.nan))
                episodes.append(self._episode(s, labels, et, ep, t, xp))
        last = idx[-1]
        for s, (et, ep) in open_pos.items():
            episodes.append(self._episode(s, labels, et, ep, last, float(closes[s].get(last, np.nan))))
        return targets, episodes

    @staticmethod
    def _episode(sym, labels, et, ep, xt, xp) -> dict:
        nr = (xp / ep - 1.0) if (ep and np.isfinite(ep) and np.isfinite(xp)) else 0.0
        return {"direction": 1, "entry_time": et, "entry_price": ep,
                "exit_time": xt, "exit_price": xp, "net_return": float(nr),
                "exit_reason": "rotation", "entry_reason": "cri_top", "symbol": labels.get(sym, sym)}

    @staticmethod
    def _episodes_frame(episodes) -> pd.DataFrame:
        return pd.DataFrame(episodes, columns=_TRADE_COLS)

    # -------------------------- regime --------------------------------- #
    def _regime_scale(self, raw, config, idx) -> pd.Series:
        regime = config.get("regime") or {}
        if not regime.get("enabled", True):
            return pd.Series(1.0, index=idx)
        off_scale = float(regime.get("off_scale", 0.4))
        spy = self._regime_close(raw, config, "spy_symbol", idx)
        vix = self._regime_close(raw, config, "vix_symbol", idx)
        dxy = self._regime_close(raw, config, "dxy_symbol", idx)
        off = pd.Series(0.0, index=idx)
        n_sig = 0
        if spy is not None:
            ma = spy.rolling(int(regime.get("spy_ma", 200)), min_periods=20).mean()
            off += (spy < ma).astype(float)
            n_sig += 1
        if vix is not None:
            off += (vix > float(regime.get("vix_high", 25.0))).astype(float)
            n_sig += 1
        if dxy is not None:
            dma = dxy.rolling(100, min_periods=20).mean()
            off += (dxy > dma).astype(float)
            n_sig += 1
        if n_sig == 0:
            return pd.Series(1.0, index=idx)
        frac = (off / n_sig).fillna(0.0)
        return (1.0 - (1.0 - off_scale) * frac).clip(lower=off_scale, upper=1.0)

    # ------------------------- simulation ------------------------------ #
    @staticmethod
    def _simulate(idx, assets, ret_mat, targets, reb, thr, cost):
        """일간 복리 + 리밸 시 목표비중 적용(턴오버 임계·비용). 자산별 평가가치 행렬과
        현금 시계열을 돌려준다(USD, NAV=행합+현금)."""
        T, A = len(idx), len(assets)
        values = np.zeros((T, A))
        cash_arr = np.zeros(T)
        val = np.zeros(A)
        cash = 1.0
        for i, t in enumerate(idx):
            if i > 0:
                val = val * (1.0 + ret_mat[i])
            nav = val.sum() + cash
            if t in reb and t in targets and nav > 0:
                cur = val / nav
                tgt = targets[t]
                new = np.where(np.abs(tgt - cur) < thr, cur, tgt)
                ssum = new.sum()
                if ssum > 1.0:                      # 합>1 방지(현금 음수 차단)
                    new = new / ssum
                    ssum = 1.0
                turnover = float(np.abs(new - cur).sum())
                nav *= (1.0 - turnover * cost)
                val = nav * new
                cash = nav * (1.0 - new.sum())
            values[i] = val
            cash_arr[i] = cash
        return values, cash_arr

    # ------------------------- artifacts ------------------------------- #
    def _forecast(self, window, equity, values, cash, country_syms, safe_syms,
                  assets, idx, equity_scale, targets) -> pd.DataFrame:
        a_idx = {a: i for i, a in enumerate(assets)}
        vdf = pd.DataFrame(values, index=idx, columns=assets)
        nav = vdf.sum(axis=1) + pd.Series(cash, index=idx)
        stock = vdf[country_syms].sum(axis=1) / nav
        safe = (vdf[safe_syms].sum(axis=1) / nav) if safe_syms else pd.Series(0.0, index=idx)
        cash_r = pd.Series(cash, index=idx) / nav
        n_ctry = (vdf[country_syms] > 1e-9).sum(axis=1).astype(float)
        scale = equity_scale.reindex(idx).ffill().fillna(1.0)
        f = pd.DataFrame({"close": equity}, index=window)
        f["stock_exposure"] = stock.reindex(window)
        f["safe_exposure"] = safe.reindex(window)
        f["cash_ratio"] = cash_r.reindex(window)
        f["n_countries"] = n_ctry.reindex(window)
        f["risk_on"] = scale.reindex(window)
        return f

    @staticmethod
    def _wave(window, values, cash, assets, labels, safe_syms, idx) -> pd.DataFrame:
        cols = {labels.get(a, a) if a not in safe_syms else f"안전({a})": values[:, i]
                for i, a in enumerate(assets)}
        vdf = pd.DataFrame(cols, index=idx)
        vdf["현금"] = cash
        vdf = vdf.reindex(window)
        shares = vdf.div(vdf.sum(axis=1), axis=0).fillna(0.0)
        shares.index.name = "time"
        return shares.reset_index()

    def _bench(self, raw, config, idx, window, base_cur, fx_sym, interval):
        bench_sym = str(config.get("benchmark_symbol", "")).upper()
        s = self._wide_field(raw, "__BENCH__", [bench_sym], "close")
        if bench_sym not in s.columns:
            return None, {}
        px = s[bench_sym].reindex(idx).ffill()
        nav = px / px.dropna().iloc[0]
        if base_cur != "USD":
            nav = self._to_base(nav, self._fx(raw, fx_sym))
        b = nav.reindex(window).ffill().dropna()
        if b.empty:
            return None, {}
        bench = (b / b.iloc[0]).rename("benchmark")
        bperf = performance(bench, bench.pct_change().fillna(0.0), interval)
        return bench, bperf

    @staticmethod
    def _perf_table(perf, bench, *, label) -> pd.DataFrame:
        rows = [{"라인": label, "CAGR": perf.get("cagr"), "MDD": perf.get("max_drawdown"),
                 "Sharpe": perf.get("sharpe"), "Total Return": perf.get("total_return")}]
        if bench:
            rows.append({"라인": "벤치 B&H", "CAGR": bench.get("cagr"),
                         "MDD": bench.get("max_drawdown"), "Sharpe": bench.get("sharpe"),
                         "Total Return": bench.get("total_return")})
        return pd.DataFrame(rows)
