"""Test yoon2_final_fx_2 overlay candidates.

The lower yoon2_final sleeves remain frozen. This script only tests top-level
USD/KRW allocation overlays:

1. simultaneous-risk exposure gate into cash
2. fast USD underweight override
3. volatility-based USD/KRW blend

Usage:
    PYTHONPATH=src:scripts python3 scripts/profile_sizing/yoon2_final_fx2_overlay.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from profile_sizing.engine import performance  # noqa: E402
from profile_sizing.yoon2_final_fx_validate import (  # noqa: E402
    CURRENT,
    Context,
    _fmt_num,
    _fmt_pct,
    load_context,
    paired_bootstrap,
    relmom_weights,
    weight_frame,
)
from profile_sizing.portfolio import rebalance_dates  # noqa: E402


FAST = {
    "lookback": 20,
    "usd_floor_trigger": 0.42,
    "relative_threshold": 0.03,
    "fx_threshold": 0.03,
    "min_usd": 0.50,
}
VOL = {"lookback": 63, "blend": 0.35}
GATE = {
    "lookback": 63,
    "sma_lookback": 200,
    "dd_threshold": -0.10,
    "vol_threshold": 0.24,
    "off_scale": 0.65,
}


def _clean_metric(p: dict[str, Any]) -> dict[str, float | None]:
    keys = ["cagr", "max_drawdown", "sharpe", "total_return", "volatility"]
    return {k: (None if p.get(k) is None else float(p[k])) for k in keys}


def perf_nav(nav: pd.Series, window: pd.DatetimeIndex | None = None) -> dict[str, Any]:
    if window is not None:
        nav = nav.reindex(window).ffill().dropna()
    if len(nav) < 60:
        return {}
    eq = nav / nav.iloc[0]
    ret = eq.pct_change().fillna(0.0)
    p = performance(eq, ret, "1d")
    out = _clean_metric(p)
    mdd = out.get("max_drawdown")
    cagr = out.get("cagr")
    out["calmar"] = (
        float(cagr) / abs(float(mdd))
        if cagr is not None and mdd is not None and mdd < 0.0
        else None
    )
    return out


def market_navs(ctx: Context) -> tuple[pd.Series, pd.Series, pd.Series]:
    usd_base = ctx.spy_krw_nav.reindex(ctx.idx).ffill()
    krw = ctx.kodex_nav.reindex(ctx.idx).ffill()
    fx = ctx.fx_nav.reindex(ctx.idx).ffill()
    return usd_base, krw, fx


def current_weight_series(ctx: Context) -> pd.Series:
    w = relmom_weights(ctx, **CURRENT)
    return w[w.columns[0]].reindex(ctx.idx).ffill().fillna(0.5)


def build_weights(
    ctx: Context,
    *,
    fast: dict[str, Any] | None = None,
    vol: dict[str, Any] | None = None,
    gate: dict[str, Any] | None = None,
    gate_source: str = "bench",
) -> pd.DataFrame:
    usd_base, krw, fx = market_navs(ctx)
    usd = current_weight_series(ctx)

    if fast:
        lb = int(fast["lookback"])
        fast_score = usd_base.pct_change(lb) - krw.pct_change(lb)
        fx_fast = fx.pct_change(lb)
        trigger = (
            (usd <= float(fast["usd_floor_trigger"]))
            & (
                (fast_score.shift(1) >= float(fast["relative_threshold"]))
                | (fx_fast.shift(1) >= float(fast["fx_threshold"]))
            )
        )
        usd = usd.where(~trigger, np.maximum(usd, float(fast["min_usd"])))

    if vol:
        lb = int(vol["lookback"])
        blend = float(vol["blend"])
        lo, hi = CURRENT["band"]
        us_vol = usd_base.pct_change().rolling(lb).std().shift(1)
        kr_vol = krw.pct_change().rolling(lb).std().shift(1)
        vol_w = (kr_vol / (us_vol + kr_vol).replace(0.0, np.nan)).clip(lo, hi)
        usd = ((1.0 - blend) * usd + blend * vol_w).clip(lo, hi).fillna(usd)

    exposure = pd.Series(1.0, index=ctx.idx)
    if gate:
        if gate_source == "sleeve":
            us_sig = ctx.navs[0].reindex(ctx.idx).ffill()
            kr_sig = ctx.navs[1].reindex(ctx.idx).ffill()
        else:
            us_sig = usd_base
            kr_sig = krw
        lb = int(gate["lookback"])
        sma_lb = int(gate["sma_lookback"])
        us_ret = us_sig.pct_change(lb)
        kr_ret = kr_sig.pct_change(lb)
        us_sma = us_sig.rolling(sma_lb).mean()
        kr_sma = kr_sig.rolling(sma_lb).mean()
        us_dd = us_sig / us_sig.rolling(lb).max() - 1.0
        kr_dd = kr_sig / kr_sig.rolling(lb).max() - 1.0
        us_vol = us_sig.pct_change().rolling(lb).std() * np.sqrt(252.0)
        kr_vol = kr_sig.pct_change().rolling(lb).std() * np.sqrt(252.0)
        weak = (
            (us_ret < 0.0)
            & (kr_ret < 0.0)
            & (us_sig < us_sma)
            & (kr_sig < kr_sma)
            & (
                (us_dd < float(gate["dd_threshold"]))
                | (kr_dd < float(gate["dd_threshold"]))
                | (np.maximum(us_vol, kr_vol) > float(gate["vol_threshold"]))
            )
        )
        exposure = exposure.where(~weak.shift(1).fillna(False), float(gate["off_scale"]))

    sub_ids = ctx.sub_ids
    out = pd.DataFrame(index=ctx.idx, columns=[*sub_ids, "__CASH__"], dtype=float)
    out[sub_ids[0]] = usd * exposure
    out[sub_ids[1]] = (1.0 - usd) * exposure
    out["__CASH__"] = 1.0 - exposure
    return out.fillna({sub_ids[0]: 0.5, sub_ids[1]: 0.5, "__CASH__": 0.0})


def combine_nav_cash(ctx: Context, weights: pd.DataFrame) -> pd.Series:
    idx = ctx.idx
    tw = weights.reindex(idx).ffill().bfill()
    if "__CASH__" not in tw.columns:
        tw["__CASH__"] = 0.0
    tw = tw[[*ctx.sub_ids, "__CASH__"]]
    tw = tw.div(tw.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    rets = [n.reindex(idx).ffill().pct_change().fillna(0.0).to_numpy() for n in ctx.navs]
    rets.append(np.zeros(len(idx), dtype=float))
    vals = tw.iloc[0].to_numpy(dtype=float).tolist()
    out = np.empty(len(idx))
    reb = set(rebalance_dates(idx, str(ctx.config.get("rebalance_freq", "monthly"))))
    for i, t in enumerate(idx):
        for j, r in enumerate(rets):
            vals[j] *= 1.0 + r[i]
        out[i] = sum(vals)
        if t in reb:
            total = sum(vals)
            vals = (total * tw.iloc[i].to_numpy(dtype=float)).tolist()
    nav = pd.Series(out, index=idx, name="equity")
    return nav / nav.iloc[0]


def static_weights(ctx: Context) -> pd.DataFrame:
    w = weight_frame(ctx, 0.5)
    w["__CASH__"] = 0.0
    return w


def candidate_defs() -> list[tuple[str, dict[str, Any]]]:
    return [
        ("static_50_50", {}),
        ("final_fx_current", {"current": True}),
        ("gate_only", {"gate": GATE}),
        ("fast_override_only", {"fast": FAST}),
        ("vol_adjust_only", {"vol": VOL}),
        ("gate_fast", {"gate": GATE, "fast": FAST}),
        ("gate_vol", {"gate": GATE, "vol": VOL}),
        ("fast_vol", {"fast": FAST, "vol": VOL}),
        ("fx2_all", {"gate": GATE, "fast": FAST, "vol": VOL}),
        ("gate_sleeve_only", {"gate": GATE, "gate_source": "sleeve"}),
        ("fx2_all_sleeve", {"gate": GATE, "fast": FAST, "vol": VOL, "gate_source": "sleeve"}),
    ]


def build_candidates(ctx: Context) -> tuple[pd.DataFrame, dict[str, pd.Series], dict[str, pd.DataFrame]]:
    rows: list[dict[str, Any]] = []
    navs: dict[str, pd.Series] = {}
    weights: dict[str, pd.DataFrame] = {}
    for label, spec in candidate_defs():
        if label == "static_50_50":
            w = static_weights(ctx)
        elif label == "final_fx_current":
            w0 = relmom_weights(ctx, **CURRENT)
            w = w0.copy()
            w["__CASH__"] = 0.0
        else:
            w = build_weights(
                ctx,
                fast=spec.get("fast"),
                vol=spec.get("vol"),
                gate=spec.get("gate"),
                gate_source=spec.get("gate_source", "bench"),
            )
        nav = combine_nav_cash(ctx, w)
        us = w[ctx.sub_ids[0]]
        cash = w["__CASH__"]
        metric = perf_nav(nav)
        rows.append({
            "label": label,
            **metric,
            "usd_min": float(us.min()),
            "usd_mean": float(us.mean()),
            "usd_max": float(us.max()),
            "cash_mean": float(cash.mean()),
            "cash_max": float(cash.max()),
            "cash_days": int((cash > 0.0).sum()),
        })
        navs[label] = nav
        weights[label] = w
    return pd.DataFrame(rows), navs, weights


def objective_score(row: dict[str, Any]) -> float:
    sharpe = row.get("sharpe")
    mdd = row.get("max_drawdown")
    cagr = row.get("cagr")
    if sharpe is None or mdd is None or cagr is None:
        return -999.0
    # Favor Sharpe, but penalize candidates that lose the original defensive identity.
    penalty = max(0.0, abs(float(mdd)) - 0.18) * 2.0
    return float(sharpe) + 0.5 * float(cagr) - penalty


def walk_forward(ctx: Context, navs: dict[str, pd.Series]) -> dict[str, Any]:
    labels = [label for label, _ in candidate_defs()]
    years = sorted({int(d.year) for d in ctx.idx})
    rows: list[dict[str, Any]] = []
    for start in range(years[0], years[-1] - 5 + 1):
        train = ctx.idx[(ctx.idx.year >= start) & (ctx.idx.year <= start + 4)]
        test = ctx.idx[ctx.idx.year == start + 5]
        if len(train) < 750 or len(test) < 60:
            continue
        train_scores = []
        for label in labels:
            m = perf_nav(navs[label], train)
            train_scores.append((objective_score(m), label))
        train_scores.sort(reverse=True)
        chosen = train_scores[0][1]
        rows.append({
            "train": f"{start}-{start + 4}",
            "test": str(start + 5),
            "chosen": chosen,
            "chosen_perf": perf_nav(navs[chosen], test),
            "current_perf": perf_nav(navs["final_fx_current"], test),
            "static_perf": perf_nav(navs["static_50_50"], test),
            "fx2_all_perf": perf_nav(navs["fx2_all"], test),
            "fx2_all_sleeve_perf": perf_nav(navs["fx2_all_sleeve"], test),
            "gate_sleeve_only_perf": perf_nav(navs["gate_sleeve_only"], test),
        })

    def avg(key: str, metric: str) -> float | None:
        vals = [r[key].get(metric) for r in rows if r[key].get(metric) is not None]
        return float(np.mean(vals)) if vals else None

    def wins(a: str, b: str, metric: str, higher: bool = True) -> int:
        n = 0
        for r in rows:
            av = r[a].get(metric)
            bv = r[b].get(metric)
            if av is None or bv is None:
                continue
            n += int(av > bv if higher else av < bv)
        return n

    return {
        "folds": len(rows),
        "rows": rows,
        "selected_counts": dict(pd.Series([r["chosen"] for r in rows]).value_counts()),
        "avg_current": {
            "cagr": avg("current_perf", "cagr"),
            "mdd": avg("current_perf", "max_drawdown"),
            "sharpe": avg("current_perf", "sharpe"),
        },
        "avg_static": {
            "cagr": avg("static_perf", "cagr"),
            "mdd": avg("static_perf", "max_drawdown"),
            "sharpe": avg("static_perf", "sharpe"),
        },
        "avg_fx2_all": {
            "cagr": avg("fx2_all_perf", "cagr"),
            "mdd": avg("fx2_all_perf", "max_drawdown"),
            "sharpe": avg("fx2_all_perf", "sharpe"),
        },
        "avg_fx2_all_sleeve": {
            "cagr": avg("fx2_all_sleeve_perf", "cagr"),
            "mdd": avg("fx2_all_sleeve_perf", "max_drawdown"),
            "sharpe": avg("fx2_all_sleeve_perf", "sharpe"),
        },
        "avg_gate_sleeve_only": {
            "cagr": avg("gate_sleeve_only_perf", "cagr"),
            "mdd": avg("gate_sleeve_only_perf", "max_drawdown"),
            "sharpe": avg("gate_sleeve_only_perf", "sharpe"),
        },
        "wins": {
            "fx2_sharpe_gt_current": wins("fx2_all_perf", "current_perf", "sharpe"),
            "fx2_mdd_shallower_current": wins("fx2_all_perf", "current_perf", "max_drawdown"),
            "fx2_sharpe_gt_static": wins("fx2_all_perf", "static_perf", "sharpe"),
            "fx2_mdd_shallower_static": wins("fx2_all_perf", "static_perf", "max_drawdown"),
            "sleeve_sharpe_gt_current": wins("fx2_all_sleeve_perf", "current_perf", "sharpe"),
            "sleeve_mdd_shallower_current": wins("fx2_all_sleeve_perf", "current_perf", "max_drawdown"),
            "gate_sleeve_sharpe_gt_current": wins("gate_sleeve_only_perf", "current_perf", "sharpe"),
            "gate_sleeve_mdd_shallower_current": wins("gate_sleeve_only_perf", "current_perf", "max_drawdown"),
        },
    }


def drawdown_episodes(nav: pd.Series, thresh: float = -0.06) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, float]]:
    nav = nav.dropna()
    peak = nav.iloc[0]
    peak_date = nav.index[0]
    trough = peak
    trough_date = peak_date
    in_ep = False
    out = []
    for d, v in nav.items():
        if v >= peak:
            if in_ep:
                dd = trough / peak - 1.0
                if dd <= thresh:
                    out.append((peak_date, trough_date, d, float(dd)))
            peak = v
            peak_date = d
            trough = v
            trough_date = d
            in_ep = False
        else:
            in_ep = True
            if v < trough:
                trough = v
                trough_date = d
    return out


def drawdown_table(ctx: Context, navs: dict[str, pd.Series], weights: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    rows = []
    for peak, trough, recover, dd in drawdown_episodes(navs["final_fx_current"]):
        cur = navs["final_fx_current"].loc[peak:recover]
        fx2 = navs["fx2_all"].loc[peak:recover]
        static = navs["static_50_50"].loc[peak:recover]
        sleeve = navs["fx2_all_sleeve"].loc[peak:recover]
        gate_sleeve = navs["gate_sleeve_only"].loc[peak:recover]
        rows.append({
            "peak": str(peak.date()),
            "trough": str(trough.date()),
            "recover": str(recover.date()),
            "current_mdd": float((cur / cur.cummax() - 1.0).min()),
            "fx2_mdd": float((fx2 / fx2.cummax() - 1.0).min()),
            "static_mdd": float((static / static.cummax() - 1.0).min()),
            "sleeve_mdd": float((sleeve / sleeve.cummax() - 1.0).min()),
            "gate_sleeve_mdd": float((gate_sleeve / gate_sleeve.cummax() - 1.0).min()),
            "fx2_minus_current": float((fx2 / fx2.cummax() - 1.0).min() - (cur / cur.cummax() - 1.0).min()),
            "sleeve_minus_current": float((sleeve / sleeve.cummax() - 1.0).min() - (cur / cur.cummax() - 1.0).min()),
            "fx2_cash_mean_to_trough": float(weights["fx2_all"]["__CASH__"].loc[peak:trough].mean()),
            "fx2_usd_mean_to_trough": float(weights["fx2_all"][ctx.sub_ids[0]].loc[peak:trough].mean()),
            "gate_sleeve_cash_mean_to_trough": float(weights["gate_sleeve_only"]["__CASH__"].loc[peak:trough].mean()),
        })
    return rows


def parameter_sensitivity(ctx: Context, current_mdd: float) -> dict[str, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []

    def add(label: str, w: pd.DataFrame) -> None:
        p = perf_nav(combine_nav_cash(ctx, w))
        rows.append({
            "label": label,
            **p,
            "cash_mean": float(w["__CASH__"].mean()) if "__CASH__" in w else 0.0,
            "usd_mean": float(w[ctx.sub_ids[0]].mean()),
        })

    for blend in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.50]:
        add(f"vol_blend_{blend}", build_weights(ctx, vol={**VOL, "blend": blend}))
    for rel in [0.03, 0.05, 0.08, 0.10]:
        for fx in [0.03, 0.05, 0.08]:
            add(
                f"fast_rel_{rel}_fx_{fx}",
                build_weights(ctx, fast={**FAST, "relative_threshold": rel, "fx_threshold": fx}),
            )
    for off in [0.50, 0.65, 0.80]:
        for dd in [-0.08, -0.10, -0.12, -0.15]:
            for vt in [0.24, 0.30, 0.36]:
                add(
                    f"gate_off_{off}_dd_{dd}_vol_{vt}",
                    build_weights(ctx, gate={**GATE, "off_scale": off, "dd_threshold": dd, "vol_threshold": vt}),
                )
    for blend in [0.10, 0.15, 0.20, 0.25]:
        for off in [0.65, 0.80]:
            for dd in [-0.10, -0.12, -0.15]:
                add(
                    f"combo_vol_{blend}_gate_{off}_dd_{dd}",
                    build_weights(
                        ctx,
                        vol={**VOL, "blend": blend},
                        gate={**GATE, "off_scale": off, "dd_threshold": dd, "vol_threshold": 0.30},
                    ),
                )
    sleeve_rows: list[dict[str, Any]] = []

    def add_sleeve(label: str, w: pd.DataFrame) -> None:
        p = perf_nav(combine_nav_cash(ctx, w))
        sleeve_rows.append({
            "label": label,
            **p,
            "cash_mean": float(w["__CASH__"].mean()) if "__CASH__" in w else 0.0,
            "usd_mean": float(w[ctx.sub_ids[0]].mean()),
        })

    for off in [0.50, 0.65, 0.80]:
        for dd in [-0.10, -0.15, -0.20, -0.25]:
            for vt in [0.30, 0.40, 0.50]:
                add_sleeve(
                    f"sleeve_gate_off_{off}_dd_{dd}_vol_{vt}",
                    build_weights(
                        ctx,
                        gate={**GATE, "off_scale": off, "dd_threshold": dd, "vol_threshold": vt},
                        gate_source="sleeve",
                    ),
                )
    df = pd.DataFrame(rows)
    mdd_ok = df[df["max_drawdown"] >= current_mdd].copy()
    sdf = pd.DataFrame(sleeve_rows)
    sleeve_mdd_ok = sdf[sdf["max_drawdown"] >= current_mdd].copy()
    return {
        "top_sharpe_with_mdd_not_worse_than_current": mdd_ok.sort_values(
            ["sharpe", "cagr"], ascending=False
        ).head(8).to_dict(orient="records"),
        "top_mdd": df.sort_values(["max_drawdown", "sharpe"], ascending=False)
        .head(8)
        .to_dict(orient="records"),
        "sleeve_top_sharpe_with_mdd_not_worse_than_current": sleeve_mdd_ok.sort_values(
            ["sharpe", "cagr"], ascending=False
        ).head(8).to_dict(orient="records"),
        "sleeve_top_mdd": sdf.sort_values(["max_drawdown", "sharpe"], ascending=False)
        .head(8)
        .to_dict(orient="records"),
    }


def write_report(result: dict[str, Any]) -> Path:
    out = ROOT / "reports" / "profile_sizing" / "yoon2_final_fx_2_overlay_report.md"
    point = result["point_estimates"]
    lines = [
        "# yoon2_final_fx_2 overlay report",
        "",
        f"- Period: `{result['period']}` ({result['bars']} bars)",
        "- Scope: lower yoon2_final sleeves frozen; only top-level USD/KRW/cash allocation is changed.",
        "",
        "## Candidate Results",
        "",
        "| candidate | CAGR | MDD | Sharpe | Calmar | USD mean | Cash mean/max | Cash days |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in point:
        lines.append(
            f"| {r['label']} | {_fmt_pct(r.get('cagr'))} | {_fmt_pct(r.get('max_drawdown'))} | "
            f"{_fmt_num(r.get('sharpe'))} | {_fmt_num(r.get('calmar'))} | "
            f"{_fmt_pct(r.get('usd_mean'))} | {_fmt_pct(r.get('cash_mean'))}/{_fmt_pct(r.get('cash_max'))} | "
            f"{r.get('cash_days', 0)} |"
        )
    boot = result["bootstrap"]
    lines.extend([
        "",
        "## Bootstrap",
        "",
        "- `fx2_all` vs `final_fx_current`: "
        f"P(Sharpe>) `{boot['fx2_vs_current']['prob']['sharpe_a_gt_b'] * 100:.1f}%`, "
        f"P(CAGR>) `{boot['fx2_vs_current']['prob']['cagr_a_gt_b'] * 100:.1f}%`, "
        f"P(MDD shallower) `{boot['fx2_vs_current']['prob']['mdd_a_shallower'] * 100:.1f}%`",
        "- `fx2_all` vs `static_50_50`: "
        f"P(Sharpe>) `{boot['fx2_vs_static']['prob']['sharpe_a_gt_b'] * 100:.1f}%`, "
        f"P(CAGR>) `{boot['fx2_vs_static']['prob']['cagr_a_gt_b'] * 100:.1f}%`, "
        f"P(MDD shallower) `{boot['fx2_vs_static']['prob']['mdd_a_shallower'] * 100:.1f}%`",
        "- `gate_sleeve_only` vs `final_fx_current`: "
        f"P(Sharpe>) `{boot['gate_sleeve_vs_current']['prob']['sharpe_a_gt_b'] * 100:.1f}%`, "
        f"P(CAGR>) `{boot['gate_sleeve_vs_current']['prob']['cagr_a_gt_b'] * 100:.1f}%`, "
        f"P(MDD shallower) `{boot['gate_sleeve_vs_current']['prob']['mdd_a_shallower'] * 100:.1f}%`",
        "- `fx2_all_sleeve` vs `final_fx_current`: "
        f"P(Sharpe>) `{boot['fx2_sleeve_vs_current']['prob']['sharpe_a_gt_b'] * 100:.1f}%`, "
        f"P(CAGR>) `{boot['fx2_sleeve_vs_current']['prob']['cagr_a_gt_b'] * 100:.1f}%`, "
        f"P(MDD shallower) `{boot['fx2_sleeve_vs_current']['prob']['mdd_a_shallower'] * 100:.1f}%`",
        "- `fx2_all_sleeve` vs `fx2_all` (bench gate): "
        f"P(Sharpe>) `{boot['fx2_sleeve_vs_fx2_bench']['prob']['sharpe_a_gt_b'] * 100:.1f}%`, "
        f"P(CAGR>) `{boot['fx2_sleeve_vs_fx2_bench']['prob']['cagr_a_gt_b'] * 100:.1f}%`, "
        f"P(MDD shallower) `{boot['fx2_sleeve_vs_fx2_bench']['prob']['mdd_a_shallower'] * 100:.1f}%`",
        "",
        "## Walk-Forward",
        "",
        f"- Folds: `{result['walk_forward']['folds']}`",
        f"- Avg current: CAGR `{_fmt_pct(result['walk_forward']['avg_current']['cagr'])}`, "
        f"MDD `{_fmt_pct(result['walk_forward']['avg_current']['mdd'])}`, "
        f"Sharpe `{_fmt_num(result['walk_forward']['avg_current']['sharpe'])}`",
        f"- Avg fx2_all: CAGR `{_fmt_pct(result['walk_forward']['avg_fx2_all']['cagr'])}`, "
        f"MDD `{_fmt_pct(result['walk_forward']['avg_fx2_all']['mdd'])}`, "
        f"Sharpe `{_fmt_num(result['walk_forward']['avg_fx2_all']['sharpe'])}`",
        f"- fx2_all Sharpe wins vs current: `{result['walk_forward']['wins']['fx2_sharpe_gt_current']}/{result['walk_forward']['folds']}`",
        f"- fx2_all MDD wins vs current: `{result['walk_forward']['wins']['fx2_mdd_shallower_current']}/{result['walk_forward']['folds']}`",
        f"- Avg fx2_all_sleeve: CAGR `{_fmt_pct(result['walk_forward']['avg_fx2_all_sleeve']['cagr'])}`, "
        f"MDD `{_fmt_pct(result['walk_forward']['avg_fx2_all_sleeve']['mdd'])}`, "
        f"Sharpe `{_fmt_num(result['walk_forward']['avg_fx2_all_sleeve']['sharpe'])}`",
        f"- fx2_all_sleeve Sharpe wins vs current: `{result['walk_forward']['wins']['sleeve_sharpe_gt_current']}/{result['walk_forward']['folds']}`",
        f"- fx2_all_sleeve MDD wins vs current: `{result['walk_forward']['wins']['sleeve_mdd_shallower_current']}/{result['walk_forward']['folds']}`",
        f"- Avg gate_sleeve_only: CAGR `{_fmt_pct(result['walk_forward']['avg_gate_sleeve_only']['cagr'])}`, "
        f"MDD `{_fmt_pct(result['walk_forward']['avg_gate_sleeve_only']['mdd'])}`, "
        f"Sharpe `{_fmt_num(result['walk_forward']['avg_gate_sleeve_only']['sharpe'])}`",
        f"- gate_sleeve_only Sharpe wins vs current: `{result['walk_forward']['wins']['gate_sleeve_sharpe_gt_current']}/{result['walk_forward']['folds']}`",
        f"- gate_sleeve_only MDD wins vs current: `{result['walk_forward']['wins']['gate_sleeve_mdd_shallower_current']}/{result['walk_forward']['folds']}`",
        "",
        "## Parameter Sensitivity",
        "",
        "Top candidates with MDD not worse than current:",
        "",
        "| candidate | CAGR | MDD | Sharpe | Cash mean |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for r in result["sensitivity"]["top_sharpe_with_mdd_not_worse_than_current"]:
        lines.append(
            f"| {r['label']} | {_fmt_pct(r.get('cagr'))} | {_fmt_pct(r.get('max_drawdown'))} | "
            f"{_fmt_num(r.get('sharpe'))} | {_fmt_pct(r.get('cash_mean'))} |"
        )
    lines.extend([
        "",
        "Top candidates by shallowest MDD:",
        "",
        "| candidate | CAGR | MDD | Sharpe | Cash mean |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for r in result["sensitivity"]["top_mdd"]:
        lines.append(
            f"| {r['label']} | {_fmt_pct(r.get('cagr'))} | {_fmt_pct(r.get('max_drawdown'))} | "
            f"{_fmt_num(r.get('sharpe'))} | {_fmt_pct(r.get('cash_mean'))} |"
        )
    lines.extend([
        "",
        "### Sleeve-based gate sensitivity (`ctx.navs` top-3 sleeves instead of SPY/KODEX200)",
        "",
        "Top candidates with MDD not worse than current:",
        "",
        "| candidate | CAGR | MDD | Sharpe | Cash mean |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for r in result["sensitivity"]["sleeve_top_sharpe_with_mdd_not_worse_than_current"]:
        lines.append(
            f"| {r['label']} | {_fmt_pct(r.get('cagr'))} | {_fmt_pct(r.get('max_drawdown'))} | "
            f"{_fmt_num(r.get('sharpe'))} | {_fmt_pct(r.get('cash_mean'))} |"
        )
    lines.extend([
        "",
        "Top candidates by shallowest MDD:",
        "",
        "| candidate | CAGR | MDD | Sharpe | Cash mean |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for r in result["sensitivity"]["sleeve_top_mdd"]:
        lines.append(
            f"| {r['label']} | {_fmt_pct(r.get('cagr'))} | {_fmt_pct(r.get('max_drawdown'))} | "
            f"{_fmt_num(r.get('sharpe'))} | {_fmt_pct(r.get('cash_mean'))} |"
        )
    lines.extend([
        "",
        "## Drawdown Episodes",
        "",
        "| peak | trough | current MDD | fx2(bench) MDD | fx2(sleeve) MDD | gate_sleeve MDD | static MDD | fx2-current | sleeve-current | gate_sleeve cash mean |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for r in result["drawdowns"]:
        lines.append(
            f"| {r['peak']} | {r['trough']} | {_fmt_pct(r['current_mdd'])} | "
            f"{_fmt_pct(r['fx2_mdd'])} | {_fmt_pct(r['sleeve_mdd'])} | {_fmt_pct(r['gate_sleeve_mdd'])} | "
            f"{_fmt_pct(r['static_mdd'])} | {_fmt_pct(r['fx2_minus_current'])} | "
            f"{_fmt_pct(r['sleeve_minus_current'])} | {_fmt_pct(r['gate_sleeve_cash_mean_to_trough'])} |"
        )
    lines.extend([
        "",
        "## Verdict",
        "",
        result["verdict"],
        "",
    ])
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--block", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    ctx = load_context()
    df, navs, weights = build_candidates(ctx)
    df = df.sort_values("label")
    boot = {
        "fx2_vs_current": paired_bootstrap(
            navs["fx2_all"],
            navs["final_fx_current"],
            label_a="fx2_all",
            label_b="final_fx_current",
            iters=args.iters,
            block=args.block,
            seed=args.seed,
        ),
        "fx2_vs_static": paired_bootstrap(
            navs["fx2_all"],
            navs["static_50_50"],
            label_a="fx2_all",
            label_b="static_50_50",
            iters=args.iters,
            block=args.block,
            seed=args.seed + 1,
        ),
        "gate_sleeve_vs_current": paired_bootstrap(
            navs["gate_sleeve_only"],
            navs["final_fx_current"],
            label_a="gate_sleeve_only",
            label_b="final_fx_current",
            iters=args.iters,
            block=args.block,
            seed=args.seed + 2,
        ),
        "fx2_sleeve_vs_current": paired_bootstrap(
            navs["fx2_all_sleeve"],
            navs["final_fx_current"],
            label_a="fx2_all_sleeve",
            label_b="final_fx_current",
            iters=args.iters,
            block=args.block,
            seed=args.seed + 3,
        ),
        "fx2_sleeve_vs_fx2_bench": paired_bootstrap(
            navs["fx2_all_sleeve"],
            navs["fx2_all"],
            label_a="fx2_all_sleeve",
            label_b="fx2_all",
            iters=args.iters,
            block=args.block,
            seed=args.seed + 4,
        ),
    }
    wf = walk_forward(ctx, navs)
    dd = drawdown_table(ctx, navs, weights)
    fx2 = df[df["label"] == "fx2_all"].iloc[0].to_dict()
    cur = df[df["label"] == "final_fx_current"].iloc[0].to_dict()
    gate_sleeve = df[df["label"] == "gate_sleeve_only"].iloc[0].to_dict()
    fx2_sleeve = df[df["label"] == "fx2_all_sleeve"].iloc[0].to_dict()
    sensitivity = parameter_sensitivity(ctx, float(cur["max_drawdown"]))
    verdict = (
        "fx2_all reduces full-sample MDD versus final_fx_current, but the added cash gate "
        "also lowers CAGR/Sharpe. Treat it as a defensive research variant, not a promotion "
        "candidate, unless the objective is explicitly lower drawdown over return. "
        f"Sleeve-based gate (gate_sleeve_only: CAGR {_fmt_pct(gate_sleeve.get('cagr'))}, "
        f"MDD {_fmt_pct(gate_sleeve.get('max_drawdown'))}, Sharpe {_fmt_num(gate_sleeve.get('sharpe'))}; "
        f"fx2_all_sleeve: CAGR {_fmt_pct(fx2_sleeve.get('cagr'))}, "
        f"MDD {_fmt_pct(fx2_sleeve.get('max_drawdown'))}, Sharpe {_fmt_num(fx2_sleeve.get('sharpe'))}) "
        "is compared against the bench-based gate above; see Bootstrap/Drawdown Episodes sections "
        "for which signal source actually catches the simultaneous-downturn cases."
    )
    if (
        fx2.get("sharpe", -999) > cur.get("sharpe", -999)
        and fx2.get("max_drawdown", -999) > cur.get("max_drawdown", -999)
    ):
        verdict = (
            "fx2_all improves both Sharpe and MDD versus final_fx_current in this run, "
            "so it is a valid promotion candidate after paper forward testing."
        )

    result = {
        "period": f"{ctx.idx[0].date()}~{ctx.idx[-1].date()}",
        "bars": len(ctx.idx),
        "parameters": {"fast": FAST, "vol": VOL, "gate": GATE},
        "point_estimates": df.to_dict(orient="records"),
        "bootstrap": boot,
        "walk_forward": wf,
        "sensitivity": sensitivity,
        "drawdowns": dd,
        "verdict": verdict,
    }
    out_json = ROOT / "reports" / "profile_sizing" / "yoon2_final_fx_2_overlay_report.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    out_md = write_report(result)

    print(f"Period {result['period']} · {result['bars']} bars")
    for label in ["static_50_50", "final_fx_current", "fx2_all"]:
        r = df[df["label"] == label].iloc[0]
        print(
            f"{label}: CAGR {r['cagr'] * 100:.2f}%  "
            f"MDD {r['max_drawdown'] * 100:.2f}%  Sharpe {r['sharpe']:.3f}  "
            f"CashMean {r['cash_mean'] * 100:.2f}%"
        )
    print(f"JSON: {out_json}")
    print(f"Report: {out_md}")


if __name__ == "__main__":
    main()
