"""Validate yoon2_final_fx currency-allocation layer.

This script keeps the frozen yoon2_final sleeves intact and tests only the
top-level USD/KRW allocation layer:

1. point estimates for fixed 50/50, current final_fx, and nearby variants
2. paired moving-block bootstrap against fixed 50/50
3. 5-year train / 1-year test walk-forward parameter selection
4. rolling 3-year current-vs-static stability

Usage:
    PYTHONPATH=src:scripts python3 scripts/profile_sizing/yoon2_final_fx_validate.py
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from profile_sizing.engine import performance  # noqa: E402
from profile_sizing.yoon2a_bootstrap import _stats, block_indices, pct  # noqa: E402
from trading_lab.strategies.yoon1k import Yoon1kHandler  # noqa: E402


CURRENT = {
    "lookback": 63,
    "tilt": 0.125,
    "scale": 0.10,
    "band": (0.40, 0.60),
    "fx_weight": 0.0,
    "lag": 1,
}

GRID_LOOKBACKS = [21, 42, 63, 84, 126]
GRID_TILTS = [0.05, 0.125, 0.20, 0.30]
GRID_SCALES = [0.05, 0.10, 0.20]


@dataclass
class Context:
    config: dict[str, Any]
    raw: pd.DataFrame
    handler: Yoon1kHandler
    idx: pd.DatetimeIndex
    sub_ids: list[str]
    navs: list[pd.Series]
    spy_krw_nav: pd.Series
    kodex_nav: pd.Series
    fx_nav: pd.Series


def _clean_metric(p: dict[str, Any]) -> dict[str, float | int | None]:
    keys = ["cagr", "max_drawdown", "sharpe", "total_return", "volatility"]
    return {k: (None if p.get(k) is None else float(p[k])) for k in keys}


def _fmt_pct(v: float | None) -> str:
    return "-" if v is None else f"{v * 100:.2f}%"


def _fmt_num(v: float | None) -> str:
    return "-" if v is None else f"{v:.3f}"


def load_context() -> Context:
    config = json.loads(
        (ROOT / "configs" / "strategies" / "yoon2_final_fx.json").read_text("utf-8")
    )
    handler = Yoon1kHandler()
    raw = handler.load_data("PORTFOLIO", config, synthetic=False)

    base_cur = str(config.get("base_currency", "KRW"))
    fx_sym = str(config.get("fx_symbol", "KRW=X"))
    navs: list[pd.Series] = []
    sub_ids: list[str] = []
    for sub in config["sub_strategies"]:
        sid = str(sub["config"])
        sub_ids.append(sid)
        sub_cfg = handler._sub_config(sid, sub.get("overrides"))
        art = handler._base.build_artifacts(
            raw[sid], sub_cfg, symbol="PORTFOLIO", phase="all", bars_per_year=252
        )
        nav = (art.equity / art.equity.iloc[0]).rename(sid)
        if str(sub.get("currency", base_cur)) != base_cur:
            nav = handler._to_base(nav, handler._fx(raw, fx_sym))
        navs.append(nav)

    idx = navs[0].index
    for nav in navs[1:]:
        idx = idx.intersection(nav.index)
    idx = idx.sort_values()
    navs = [nav.reindex(idx).ffill() for nav in navs]

    spy = raw["__BENCH__"]["SPY"]["close"].reindex(idx).ffill()
    kodex = raw["__BENCH__"]["069500.KS"]["close"].reindex(idx).ffill()
    fx = pd.Series(handler._fx(raw, fx_sym)).reindex(idx).ffill()
    spy_nav = spy / spy.dropna().iloc[0]
    kodex_nav = kodex / kodex.dropna().iloc[0]
    fx_nav = fx / fx.dropna().iloc[0]

    return Context(
        config=config,
        raw=raw,
        handler=handler,
        idx=idx,
        sub_ids=sub_ids,
        navs=navs,
        spy_krw_nav=spy_nav * fx_nav,
        kodex_nav=kodex_nav,
        fx_nav=fx_nav,
    )


def weight_frame(ctx: Context, usd_weight: pd.Series | float) -> pd.DataFrame:
    if isinstance(usd_weight, float):
        us = pd.Series(usd_weight, index=ctx.idx)
    else:
        us = pd.Series(usd_weight, index=ctx.idx).reindex(ctx.idx)
    us = us.ffill().fillna(0.5)
    out = pd.DataFrame(index=ctx.idx, columns=ctx.sub_ids, dtype=float)
    out[ctx.sub_ids[0]] = us
    out[ctx.sub_ids[1]] = 1.0 - us
    return out


def relmom_weights(
    ctx: Context,
    *,
    lookback: int = 63,
    tilt: float = 0.125,
    scale: float = 0.10,
    band: tuple[float, float] = (0.40, 0.60),
    fx_weight: float = 0.0,
    lag: int = 1,
) -> pd.DataFrame:
    score = (
        ctx.spy_krw_nav.pct_change(lookback)
        - ctx.kodex_nav.pct_change(lookback)
        + fx_weight * ctx.fx_nav.pct_change(lookback)
    )
    usd = 0.5 + tilt * np.tanh(score / max(scale, 1e-9))
    usd = pd.Series(usd, index=ctx.idx).clip(*band).shift(lag)
    return weight_frame(ctx, usd)


def combine_nav(ctx: Context, weights: pd.DataFrame) -> pd.Series:
    combo = ctx.handler._combine_values_dynamic(
        ctx.navs,
        weights,
        ctx.idx,
        str(ctx.config.get("rebalance_freq", "monthly")),
        ctx.sub_ids,
    ).sum(axis=1)
    return (combo / combo.iloc[0]).rename("equity")


def perf_nav(nav: pd.Series, window: pd.DatetimeIndex | None = None) -> dict[str, Any]:
    if window is not None:
        nav = nav.reindex(window).ffill().dropna()
    if len(nav) < 60:
        return {}
    eq = nav / nav.iloc[0]
    ret = eq.pct_change().fillna(0.0)
    return performance(eq, ret, "1d")


def summarize_nav(label: str, nav: pd.Series, weights: pd.DataFrame) -> dict[str, Any]:
    p = perf_nav(nav)
    us = weights[weights.columns[0]]
    return {
        "label": label,
        **_clean_metric(p),
        "usd_min": float(us.min()),
        "usd_mean": float(us.mean()),
        "usd_max": float(us.max()),
        "usd_lt_40": float((us < 0.4).mean()),
        "usd_gt_60": float((us > 0.6).mean()),
    }


def candidate_defs() -> list[tuple[str, dict[str, Any] | None]]:
    rows: list[tuple[str, dict[str, Any] | None]] = [("static_50_50", None)]
    rows.append(("current_L63_t0.125_s0.10_b40_60", dict(CURRENT)))
    for band in [(0.45, 0.55), (0.35, 0.65), (0.0, 1.0)]:
        p = dict(CURRENT)
        p["band"] = band
        rows.append((f"band_{int(band[0] * 100)}_{int(band[1] * 100)}", p))
    for lookback in [21, 42, 84, 126, 189, 252]:
        p = dict(CURRENT)
        p["lookback"] = lookback
        rows.append((f"lookback_{lookback}", p))
    for tilt in [0.05, 0.10, 0.20, 0.30, 0.50]:
        p = dict(CURRENT)
        p["tilt"] = tilt
        rows.append((f"tilt_{tilt}", p))
    for scale in [0.05, 0.20]:
        p = dict(CURRENT)
        p["scale"] = scale
        rows.append((f"scale_{scale}", p))
    for fx_weight in [-1.0, -0.5, 0.5, 1.0]:
        p = dict(CURRENT)
        p["fx_weight"] = fx_weight
        rows.append((f"fx_weight_{fx_weight}", p))
    return rows


def point_estimates(ctx: Context) -> tuple[list[dict[str, Any]], dict[str, pd.Series]]:
    rows: list[dict[str, Any]] = []
    navs: dict[str, pd.Series] = {}
    for label, params in candidate_defs():
        weights = weight_frame(ctx, 0.5) if params is None else relmom_weights(ctx, **params)
        nav = combine_nav(ctx, weights)
        navs[label] = nav
        rows.append(summarize_nav(label, nav, weights))
    return rows, navs


def paired_bootstrap(
    nav_a: pd.Series,
    nav_b: pd.Series,
    *,
    label_a: str,
    label_b: str,
    iters: int,
    block: int,
    seed: int,
) -> dict[str, Any]:
    idx = nav_a.dropna().index.intersection(nav_b.dropna().index).sort_values()
    a = nav_a.reindex(idx).ffill()
    b = nav_b.reindex(idx).ffill()
    ra = a.pct_change().fillna(0.0).to_numpy()
    rb = b.pct_change().fillna(0.0).to_numpy()
    rng = np.random.default_rng(seed)
    bidx = block_indices(len(idx), block, iters, rng)
    sa, ca, ma = _stats(ra[bidx])
    sb, cb, mb = _stats(rb[bidx])
    d_sharpe = sa - sb
    d_cagr = ca - cb
    d_mdd = ma - mb
    return {
        "label_a": label_a,
        "label_b": label_b,
        "period": f"{idx[0].date()}~{idx[-1].date()}",
        "bars": len(idx),
        "iters": iters,
        "block": block,
        "ci95": {
            "excess_sharpe": pct(d_sharpe),
            "excess_cagr": pct(d_cagr),
            "mdd_diff": pct(d_mdd),
        },
        "prob": {
            "sharpe_a_gt_b": float((d_sharpe > 0).mean()),
            "cagr_a_gt_b": float((d_cagr > 0).mean()),
            "mdd_a_shallower": float((d_mdd > 0).mean()),
        },
    }


def grid_params() -> list[dict[str, Any]]:
    out = []
    for lookback in GRID_LOOKBACKS:
        for tilt in GRID_TILTS:
            for scale in GRID_SCALES:
                p = dict(CURRENT)
                p.update({"lookback": lookback, "tilt": tilt, "scale": scale})
                out.append(p)
    return out


def label_params(p: dict[str, Any]) -> str:
    return f"L{p['lookback']}_t{p['tilt']}_s{p['scale']}"


def precompute_grid_navs(ctx: Context) -> dict[str, pd.Series]:
    navs: dict[str, pd.Series] = {"static_50_50": combine_nav(ctx, weight_frame(ctx, 0.5))}
    navs["current"] = combine_nav(ctx, relmom_weights(ctx, **CURRENT))
    for p in grid_params():
        label = label_params(p)
        if label not in navs:
            navs[label] = combine_nav(ctx, relmom_weights(ctx, **p))
    return navs


def _years(ctx: Context) -> list[int]:
    return sorted({int(d.year) for d in ctx.idx})


def walk_forward(ctx: Context, navs: dict[str, pd.Series]) -> dict[str, Any]:
    years = _years(ctx)
    rows: list[dict[str, Any]] = []
    selected_counts: dict[str, int] = {}
    for start in range(years[0], years[-1] - 5 + 1):
        train = ctx.idx[(ctx.idx.year >= start) & (ctx.idx.year <= start + 4)]
        test = ctx.idx[ctx.idx.year == start + 5]
        if len(train) < 750 or len(test) < 60:
            continue

        scored = []
        for p in grid_params():
            label = label_params(p)
            perf = perf_nav(navs[label], train)
            if not perf:
                continue
            scored.append((
                float(perf.get("sharpe") or -999.0),
                float(perf.get("max_drawdown") or -999.0),
                label,
                p,
            ))
        if not scored:
            continue
        scored.sort(reverse=True)
        _, _, chosen_label, chosen_params = scored[0]
        selected_counts[chosen_label] = selected_counts.get(chosen_label, 0) + 1

        chosen_perf = perf_nav(navs[chosen_label], test)
        current_perf = perf_nav(navs["current"], test)
        static_perf = perf_nav(navs["static_50_50"], test)
        rows.append({
            "train": f"{start}-{start + 4}",
            "test": str(start + 5),
            "selected": chosen_label,
            "selected_params": chosen_params,
            "selected_perf": _clean_metric(chosen_perf),
            "current_perf": _clean_metric(current_perf),
            "static_perf": _clean_metric(static_perf),
        })

    def avg(rows_: list[dict[str, Any]], key: str, metric: str) -> float | None:
        vals = [r[key].get(metric) for r in rows_ if r[key].get(metric) is not None]
        return float(np.mean(vals)) if vals else None

    wins = {
        "selected_sharpe_gt_static": sum(
            (r["selected_perf"]["sharpe"] or -999) > (r["static_perf"]["sharpe"] or -999)
            for r in rows
        ),
        "current_sharpe_gt_static": sum(
            (r["current_perf"]["sharpe"] or -999) > (r["static_perf"]["sharpe"] or -999)
            for r in rows
        ),
        "selected_mdd_shallower_static": sum(
            (r["selected_perf"]["max_drawdown"] or -999)
            > (r["static_perf"]["max_drawdown"] or -999)
            for r in rows
        ),
        "current_mdd_shallower_static": sum(
            (r["current_perf"]["max_drawdown"] or -999)
            > (r["static_perf"]["max_drawdown"] or -999)
            for r in rows
        ),
    }
    summary = {
        "folds": len(rows),
        "selected_counts": selected_counts,
        "avg_selected": {
            "cagr": avg(rows, "selected_perf", "cagr"),
            "mdd": avg(rows, "selected_perf", "max_drawdown"),
            "sharpe": avg(rows, "selected_perf", "sharpe"),
        },
        "avg_current": {
            "cagr": avg(rows, "current_perf", "cagr"),
            "mdd": avg(rows, "current_perf", "max_drawdown"),
            "sharpe": avg(rows, "current_perf", "sharpe"),
        },
        "avg_static": {
            "cagr": avg(rows, "static_perf", "cagr"),
            "mdd": avg(rows, "static_perf", "max_drawdown"),
            "sharpe": avg(rows, "static_perf", "sharpe"),
        },
        "wins": wins,
    }
    return {"rows": rows, "summary": summary}


def rolling_3y(ctx: Context, navs: dict[str, pd.Series]) -> dict[str, Any]:
    rows = []
    years = _years(ctx)
    for start in range(years[0], years[-1] - 2 + 1):
        window = ctx.idx[(ctx.idx.year >= start) & (ctx.idx.year <= start + 2)]
        if len(window) < 500:
            continue
        cur = _clean_metric(perf_nav(navs["current"], window))
        sta = _clean_metric(perf_nav(navs["static_50_50"], window))
        rows.append({"window": f"{start}-{start + 2}", "current": cur, "static": sta})
    wins = {
        "sharpe": sum(
            (r["current"]["sharpe"] or -999) > (r["static"]["sharpe"] or -999)
            for r in rows
        ),
        "cagr": sum(
            (r["current"]["cagr"] or -999) > (r["static"]["cagr"] or -999)
            for r in rows
        ),
        "mdd": sum(
            (r["current"]["max_drawdown"] or -999)
            > (r["static"]["max_drawdown"] or -999)
            for r in rows
        ),
        "n": len(rows),
    }
    return {"rows": rows, "wins": wins}


def write_report(result: dict[str, Any]) -> Path:
    out = ROOT / "reports" / "profile_sizing" / "yoon2_final_fx_validation.md"
    points = pd.DataFrame(result["point_estimates"])
    key_labels = [
        "static_50_50",
        "current_L63_t0.125_s0.10_b40_60",
        "band_0_100",
        "lookback_21",
        "lookback_126",
        "tilt_0.3",
        "scale_0.05",
    ]
    key = points[points["label"].isin(key_labels)].copy()
    key["CAGR"] = key["cagr"].map(_fmt_pct)
    key["MDD"] = key["max_drawdown"].map(_fmt_pct)
    key["Sharpe"] = key["sharpe"].map(_fmt_num)
    key["USD range"] = key.apply(
        lambda r: f"{r['usd_min'] * 100:.1f}-{r['usd_max'] * 100:.1f}%", axis=1
    )
    table_rows = key[["label", "CAGR", "MDD", "Sharpe", "USD range"]].to_dict(
        orient="records"
    )
    table = [
        "| label | CAGR | MDD | Sharpe | USD range |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in table_rows:
        table.append(
            f"| {row['label']} | {row['CAGR']} | {row['MDD']} | "
            f"{row['Sharpe']} | {row['USD range']} |"
        )

    boot = result["bootstrap_current_vs_static"]
    wf = result["walk_forward"]["summary"]
    roll = result["rolling_3y"]["wins"]

    lines = [
        "# yoon2_final_fx validation",
        "",
        f"- Period: `{result['period']}` ({result['bars']} bars)",
        "- Scope: top-level USD/KRW allocation only; lower yoon2_final sleeves are unchanged.",
        "",
        "## Point estimates",
        "",
        "\n".join(table),
        "",
        "## Paired moving-block bootstrap",
        "",
        f"- Comparison: `{boot['label_a']}` vs `{boot['label_b']}`",
        f"- Iterations/block: `{boot['iters']}` / `{boot['block']}` trading days",
        f"- P(Sharpe improvement): `{boot['prob']['sharpe_a_gt_b'] * 100:.1f}%`",
        f"- P(CAGR improvement): `{boot['prob']['cagr_a_gt_b'] * 100:.1f}%`",
        f"- P(MDD shallower): `{boot['prob']['mdd_a_shallower'] * 100:.1f}%`",
        "",
        "95% paired CI:",
        "",
        f"- Excess Sharpe: `{boot['ci95']['excess_sharpe'][0]:.3f}` to `{boot['ci95']['excess_sharpe'][2]:.3f}`",
        f"- Excess CAGR: `{boot['ci95']['excess_cagr'][0] * 100:.2f}%` to `{boot['ci95']['excess_cagr'][2] * 100:.2f}%`",
        f"- MDD difference: `{boot['ci95']['mdd_diff'][0] * 100:.2f}%` to `{boot['ci95']['mdd_diff'][2] * 100:.2f}%`",
        "",
        "## Walk-forward",
        "",
        "- Grid: lookback 21/42/63/84/126, tilt 0.05/0.125/0.20/0.30, scale 0.05/0.10/0.20, band 40/60.",
        "- Fold rule: select best train Sharpe over 5 calendar years, test on next calendar year.",
        f"- Folds: `{wf['folds']}`",
        f"- Avg selected: CAGR `{_fmt_pct(wf['avg_selected']['cagr'])}`, MDD `{_fmt_pct(wf['avg_selected']['mdd'])}`, Sharpe `{_fmt_num(wf['avg_selected']['sharpe'])}`",
        f"- Avg current: CAGR `{_fmt_pct(wf['avg_current']['cagr'])}`, MDD `{_fmt_pct(wf['avg_current']['mdd'])}`, Sharpe `{_fmt_num(wf['avg_current']['sharpe'])}`",
        f"- Avg static: CAGR `{_fmt_pct(wf['avg_static']['cagr'])}`, MDD `{_fmt_pct(wf['avg_static']['mdd'])}`, Sharpe `{_fmt_num(wf['avg_static']['sharpe'])}`",
        f"- Current Sharpe wins vs static: `{wf['wins']['current_sharpe_gt_static']}/{wf['folds']}`",
        f"- Current MDD wins vs static: `{wf['wins']['current_mdd_shallower_static']}/{wf['folds']}`",
        "",
        "Most selected train params:",
        "",
    ]
    selected = sorted(
        wf["selected_counts"].items(), key=lambda kv: (-kv[1], kv[0])
    )[:8]
    lines.extend([f"- `{k}`: {v}" for k, v in selected] or ["- none"])
    lines.extend([
        "",
        "## Rolling 3-year stability",
        "",
        f"- Current CAGR wins: `{roll['cagr']}/{roll['n']}`",
        f"- Current Sharpe wins: `{roll['sharpe']}/{roll['n']}`",
        f"- Current MDD wins: `{roll['mdd']}/{roll['n']}`",
        "",
        "## Verdict",
        "",
        "The dynamic layer has real return/Sharpe value versus fixed 50/50, but its MDD defense is not robust. The 40/60 band is a pragmatic guardrail rather than the alpha source. The sensitive knobs are lookback, tilt, and scale; aggressive tilt improves in-sample point estimates but is not suitable for promotion without a stronger out-of-sample gate.",
        "",
    ])
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--block", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    ctx = load_context()
    point, point_navs = point_estimates(ctx)
    grid_navs = precompute_grid_navs(ctx)
    boot = paired_bootstrap(
        point_navs["current_L63_t0.125_s0.10_b40_60"],
        point_navs["static_50_50"],
        label_a="current_L63_t0.125_s0.10_b40_60",
        label_b="static_50_50",
        iters=args.iters,
        block=args.block,
        seed=args.seed,
    )
    wf = walk_forward(ctx, grid_navs)
    roll = rolling_3y(ctx, grid_navs)
    result = {
        "period": f"{ctx.idx[0].date()}~{ctx.idx[-1].date()}",
        "bars": len(ctx.idx),
        "point_estimates": point,
        "bootstrap_current_vs_static": boot,
        "walk_forward": wf,
        "rolling_3y": roll,
    }
    out_json = ROOT / "reports" / "profile_sizing" / "yoon2_final_fx_validation.json"
    out_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8",
    )
    out_md = write_report(result)

    current = next(r for r in point if r["label"] == "current_L63_t0.125_s0.10_b40_60")
    static = next(r for r in point if r["label"] == "static_50_50")
    print(f"Period {result['period']} · {result['bars']} bars")
    print(
        "static_50_50: "
        f"CAGR {static['cagr'] * 100:.2f}%  "
        f"MDD {static['max_drawdown'] * 100:.2f}%  "
        f"Sharpe {static['sharpe']:.3f}"
    )
    print(
        "current_fx:   "
        f"CAGR {current['cagr'] * 100:.2f}%  "
        f"MDD {current['max_drawdown'] * 100:.2f}%  "
        f"Sharpe {current['sharpe']:.3f}"
    )
    print(
        "Bootstrap current vs static: "
        f"P(Sharpe>)={boot['prob']['sharpe_a_gt_b'] * 100:.1f}%  "
        f"P(CAGR>)={boot['prob']['cagr_a_gt_b'] * 100:.1f}%  "
        f"P(MDD shallower)={boot['prob']['mdd_a_shallower'] * 100:.1f}%"
    )
    print(
        "Walk-forward: "
        f"{wf['summary']['folds']} folds, "
        f"current Sharpe wins {wf['summary']['wins']['current_sharpe_gt_static']}/"
        f"{wf['summary']['folds']}, "
        f"current MDD wins {wf['summary']['wins']['current_mdd_shallower_static']}/"
        f"{wf['summary']['folds']}"
    )
    print(f"JSON: {out_json}")
    print(f"Report: {out_md}")


if __name__ == "__main__":
    main()
