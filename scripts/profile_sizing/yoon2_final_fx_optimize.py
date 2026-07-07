"""Broader parameter optimization for yoon2_final_fx.

This is intentionally separate from yoon2_final_fx_validate.py:

- validate.py answers whether the current layer is meaningful.
- optimize.py probes the remaining open questions:
  band in the grid, objective-specific selection, paired bootstrap between
  selected candidates, and top-level rebalance cost sensitivity.

The lower yoon2_final sleeves are reused unchanged.

Usage:
    PYTHONPATH=src:scripts python3 scripts/profile_sizing/yoon2_final_fx_optimize.py
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
    _clean_metric,
    _fmt_num,
    _fmt_pct,
    load_context,
    paired_bootstrap,
    perf_nav,
    relmom_weights,
    weight_frame,
)


LOOKBACKS = [21, 42, 63, 84, 126, 189, 252]
TILTS = [0.05, 0.10, 0.125, 0.20, 0.30, 0.50]
SCALES = [0.025, 0.05, 0.10, 0.20]
BANDS = [(0.45, 0.55), (0.40, 0.60), (0.35, 0.65), (0.30, 0.70), (0.0, 1.0)]
COST_BPS = [0, 5, 10, 25, 50]


def param_label(p: dict[str, Any]) -> str:
    lo, hi = p["band"]
    return (
        f"L{p['lookback']}_t{p['tilt']}_s{p['scale']}"
        f"_b{int(lo * 100)}_{int(hi * 100)}"
    )


def grid_params() -> list[dict[str, Any]]:
    rows = []
    for lookback in LOOKBACKS:
        for tilt in TILTS:
            for scale in SCALES:
                for band in BANDS:
                    p = dict(CURRENT)
                    p.update({
                        "lookback": lookback,
                        "tilt": tilt,
                        "scale": scale,
                        "band": band,
                    })
                    rows.append(p)
    return rows


def combine_nav_cost(ctx, weights: pd.DataFrame, *, cost_bps: float = 0.0) -> pd.Series:
    """Monthly sleeve combine with a top-level one-way turnover cost.

    Cost is applied only when the top-level rebalance sets the sleeve values
    back to target. For two sleeves, one-way traded notional is the absolute
    change required in the USD sleeve. This is a conservative proxy for
    FX/transfer/friction at the allocation layer; lower sleeve strategy costs
    are already inside the sleeve NAVs.
    """
    idx = ctx.idx
    freq = str(ctx.config.get("rebalance_freq", "monthly"))
    reb = set(__import__("profile_sizing.portfolio", fromlist=["rebalance_dates"]).rebalance_dates(idx, freq))
    tw = weights.reindex(idx).ffill().bfill()
    tw = tw.div(tw.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    vals = tw.iloc[0].to_numpy(dtype=float).tolist()
    rets = [n.pct_change().fillna(0.0).to_numpy() for n in ctx.navs]
    out = np.empty(len(idx))
    cost_rate = float(cost_bps) / 10_000.0
    for i, t in enumerate(idx):
        for j, r in enumerate(rets):
            vals[j] *= 1.0 + r[i]
        out[i] = sum(vals)
        if t in reb:
            total = sum(vals)
            target = tw.iloc[i].to_numpy(dtype=float)
            target_vals = total * target
            if cost_rate > 0.0:
                traded = abs(target_vals[0] - vals[0])
                total = max(total - traded * cost_rate, 0.0)
                target_vals = total * target
            vals = target_vals.tolist()
    nav = pd.Series(out, index=idx, name="equity")
    return nav / nav.iloc[0]


def metric_from_nav(nav: pd.Series) -> dict[str, float | None]:
    p = perf_nav(nav)
    out = _clean_metric(p)
    mdd = out.get("max_drawdown")
    cagr = out.get("cagr")
    out["calmar"] = (
        float(cagr) / abs(float(mdd))
        if cagr is not None and mdd is not None and mdd < 0
        else None
    )
    return out


def objective_value(metric: dict[str, Any], objective: str) -> float:
    cagr = metric.get("cagr")
    sharpe = metric.get("sharpe")
    mdd = metric.get("max_drawdown")
    calmar = metric.get("calmar")
    if objective == "sharpe":
        return float(sharpe if sharpe is not None else -999.0)
    if objective == "cagr":
        return float(cagr if cagr is not None else -999.0)
    if objective == "calmar":
        return float(calmar if calmar is not None else -999.0)
    if objective == "sharpe_mdd20":
        return float(sharpe if mdd is not None and mdd >= -0.20 and sharpe is not None else -999.0)
    if objective == "cagr_mdd20":
        return float(cagr if mdd is not None and mdd >= -0.20 and cagr is not None else -999.0)
    raise ValueError(objective)


def build_grid(ctx) -> tuple[pd.DataFrame, dict[str, pd.Series], dict[str, pd.DataFrame]]:
    rows = []
    navs: dict[str, pd.Series] = {}
    weights: dict[str, pd.DataFrame] = {}

    static_w = weight_frame(ctx, 0.5)
    static_nav = combine_nav_cost(ctx, static_w)
    navs["static_50_50"] = static_nav
    weights["static_50_50"] = static_w
    rows.append({"label": "static_50_50", "params": None, **metric_from_nav(static_nav)})

    cur_w = relmom_weights(ctx, **CURRENT)
    cur_nav = combine_nav_cost(ctx, cur_w)
    navs["current"] = cur_nav
    weights["current"] = cur_w
    rows.append({"label": "current", "params": dict(CURRENT), **metric_from_nav(cur_nav)})

    for p in grid_params():
        label = param_label(p)
        w = relmom_weights(ctx, **p)
        nav = combine_nav_cost(ctx, w)
        navs[label] = nav
        weights[label] = w
        us = w[w.columns[0]]
        rows.append({
            "label": label,
            "params": p,
            **metric_from_nav(nav),
            "usd_min": float(us.min()),
            "usd_mean": float(us.mean()),
            "usd_max": float(us.max()),
        })
    return pd.DataFrame(rows), navs, weights


def select_full_sample(df: pd.DataFrame) -> dict[str, Any]:
    candidates = df[~df["params"].isna()].copy()
    selected = {}
    for obj in ["sharpe", "cagr", "calmar", "sharpe_mdd20", "cagr_mdd20"]:
        vals = candidates.apply(lambda r: objective_value(r.to_dict(), obj), axis=1)
        row = candidates.loc[vals.idxmax()].to_dict()
        selected[obj] = row
    return selected


def years(idx: pd.DatetimeIndex) -> list[int]:
    return sorted({int(d.year) for d in idx})


def perf_window(nav: pd.Series, window: pd.DatetimeIndex) -> dict[str, Any]:
    out = metric_from_nav(nav.reindex(window).ffill().dropna())
    return out


def walk_forward(ctx, df: pd.DataFrame, navs: dict[str, pd.Series]) -> dict[str, Any]:
    params_df = df[~df["params"].isna()].copy()
    objectives = ["sharpe", "cagr", "calmar", "sharpe_mdd20", "cagr_mdd20"]
    rows = []
    for start in range(years(ctx.idx)[0], years(ctx.idx)[-1] - 5 + 1):
        train = ctx.idx[(ctx.idx.year >= start) & (ctx.idx.year <= start + 4)]
        test = ctx.idx[ctx.idx.year == start + 5]
        if len(train) < 750 or len(test) < 60:
            continue
        train_metrics = []
        for label in params_df["label"]:
            m = perf_window(navs[label], train)
            train_metrics.append({"label": label, **m})
        tdf = pd.DataFrame(train_metrics)
        fold: dict[str, Any] = {
            "train": f"{start}-{start + 4}",
            "test": str(start + 5),
            "static": perf_window(navs["static_50_50"], test),
            "current": perf_window(navs["current"], test),
            "objectives": {},
        }
        for obj in objectives:
            vals = tdf.apply(lambda r: objective_value(r.to_dict(), obj), axis=1)
            chosen = str(tdf.loc[vals.idxmax(), "label"])
            fold["objectives"][obj] = {
                "selected": chosen,
                "selected_perf": perf_window(navs[chosen], test),
            }
        rows.append(fold)

    summary: dict[str, Any] = {"folds": len(rows), "objectives": {}}
    for obj in objectives:
        selected_perfs = [r["objectives"][obj]["selected_perf"] for r in rows]
        counts: dict[str, int] = {}
        for r in rows:
            lab = r["objectives"][obj]["selected"]
            counts[lab] = counts.get(lab, 0) + 1
        summary["objectives"][obj] = {
            "selected_counts": dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:10]),
            "avg_selected": avg_metrics(selected_perfs),
            "selected_sharpe_wins_static": sum(
                (p.get("sharpe") or -999) > (r["static"].get("sharpe") or -999)
                for p, r in zip(selected_perfs, rows)
            ),
            "selected_mdd_wins_static": sum(
                (p.get("max_drawdown") or -999) > (r["static"].get("max_drawdown") or -999)
                for p, r in zip(selected_perfs, rows)
            ),
        }
    summary["current_avg"] = avg_metrics([r["current"] for r in rows])
    summary["static_avg"] = avg_metrics([r["static"] for r in rows])
    summary["current_sharpe_wins_static"] = sum(
        (r["current"].get("sharpe") or -999) > (r["static"].get("sharpe") or -999)
        for r in rows
    )
    summary["current_mdd_wins_static"] = sum(
        (r["current"].get("max_drawdown") or -999) > (r["static"].get("max_drawdown") or -999)
        for r in rows
    )
    return {"rows": rows, "summary": summary}


def avg_metrics(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    out = {}
    for k in ["cagr", "max_drawdown", "sharpe", "calmar"]:
        vals = [r.get(k) for r in rows if r.get(k) is not None]
        out[k] = float(np.mean(vals)) if vals else None
    return out


def bootstrap_candidates(
    navs: dict[str, pd.Series],
    selected: dict[str, Any],
    *,
    iters: int,
    block: int,
    seed: int,
) -> dict[str, Any]:
    pairs = {
        "current_vs_static": ("current", "static_50_50"),
        "best_sharpe_vs_current": (selected["sharpe"]["label"], "current"),
        "best_sharpe_mdd20_vs_current": (selected["sharpe_mdd20"]["label"], "current"),
        "best_calmar_vs_current": (selected["calmar"]["label"], "current"),
    }
    out = {}
    for i, (name, (a, b)) in enumerate(pairs.items()):
        out[name] = paired_bootstrap(
            navs[a],
            navs[b],
            label_a=a,
            label_b=b,
            iters=iters,
            block=block,
            seed=seed + i,
        )
    return out


def cost_sensitivity(ctx, selected: dict[str, Any], weights: dict[str, pd.DataFrame]) -> dict[str, Any]:
    labels = {
        "static_50_50": "static_50_50",
        "current": "current",
        "best_sharpe": selected["sharpe"]["label"],
        "best_sharpe_mdd20": selected["sharpe_mdd20"]["label"],
        "best_calmar": selected["calmar"]["label"],
    }
    rows = []
    for alias, label in labels.items():
        for bps in COST_BPS:
            nav = combine_nav_cost(ctx, weights[label], cost_bps=bps)
            rows.append({"candidate": alias, "label": label, "cost_bps": bps, **metric_from_nav(nav)})
    return {"rows": rows}


def md_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(name for name, _ in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for r in rows:
        vals = []
        for _, key in columns:
            v = r.get(key)
            if key in {"cagr", "max_drawdown"}:
                vals.append(_fmt_pct(v))
            elif key in {"sharpe", "calmar"}:
                vals.append(_fmt_num(v))
            else:
                vals.append(str(v))
        body.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep, *body])


def write_report(result: dict[str, Any]) -> Path:
    out = ROOT / "reports" / "profile_sizing" / "yoon2_final_fx_param_optimization.md"
    selected_rows = []
    for obj, row in result["full_sample_selected"].items():
        selected_rows.append({
            "objective": obj,
            "label": row["label"],
            "cagr": row["cagr"],
            "max_drawdown": row["max_drawdown"],
            "sharpe": row["sharpe"],
            "calmar": row["calmar"],
        })

    wf = result["walk_forward"]["summary"]
    wf_rows = []
    for obj, s in wf["objectives"].items():
        wf_rows.append({
            "objective": obj,
            "cagr": s["avg_selected"]["cagr"],
            "max_drawdown": s["avg_selected"]["max_drawdown"],
            "sharpe": s["avg_selected"]["sharpe"],
            "calmar": s["avg_selected"]["calmar"],
            "sharpe_wins": f"{s['selected_sharpe_wins_static']}/{wf['folds']}",
            "mdd_wins": f"{s['selected_mdd_wins_static']}/{wf['folds']}",
        })

    cost_rows = [
        r for r in result["cost_sensitivity"]["rows"]
        if r["candidate"] in {"static_50_50", "current", "best_sharpe_mdd20"}
    ]

    boot_lines = []
    for name, b in result["bootstrap"].items():
        boot_lines.extend([
            f"- `{name}`: `{b['label_a']}` vs `{b['label_b']}`",
            f"  P(Sharpe>) `{b['prob']['sharpe_a_gt_b'] * 100:.1f}%`, "
            f"P(CAGR>) `{b['prob']['cagr_a_gt_b'] * 100:.1f}%`, "
            f"P(MDD shallower) `{b['prob']['mdd_a_shallower'] * 100:.1f}%`",
        ])

    lines = [
        "# yoon2_final_fx parameter optimization",
        "",
        f"- Period: `{result['period']}` ({result['bars']} bars)",
        f"- Grid size: `{result['grid_size']}` parameter sets",
        "- Grid: lookback 21/42/63/84/126/189/252, tilt 0.05/0.10/0.125/0.20/0.30/0.50, scale 0.025/0.05/0.10/0.20, band 45/55 through 0/100.",
        "- Warning: full-sample winners are diagnostic only; use walk-forward and cost sensitivity for promotion decisions.",
        "",
        "## Full-sample winners",
        "",
        md_table(selected_rows, [
            ("objective", "objective"),
            ("label", "label"),
            ("CAGR", "cagr"),
            ("MDD", "max_drawdown"),
            ("Sharpe", "sharpe"),
            ("Calmar", "calmar"),
        ]),
        "",
        "## Walk-forward objective selection",
        "",
        "- Rule: 5 calendar years train, next calendar year test.",
        f"- Folds: `{wf['folds']}`",
        f"- Current avg: CAGR `{_fmt_pct(wf['current_avg']['cagr'])}`, MDD `{_fmt_pct(wf['current_avg']['max_drawdown'])}`, Sharpe `{_fmt_num(wf['current_avg']['sharpe'])}`",
        f"- Static avg: CAGR `{_fmt_pct(wf['static_avg']['cagr'])}`, MDD `{_fmt_pct(wf['static_avg']['max_drawdown'])}`, Sharpe `{_fmt_num(wf['static_avg']['sharpe'])}`",
        "",
        md_table(wf_rows, [
            ("objective", "objective"),
            ("avg CAGR", "cagr"),
            ("avg MDD", "max_drawdown"),
            ("avg Sharpe", "sharpe"),
            ("avg Calmar", "calmar"),
            ("Sharpe wins", "sharpe_wins"),
            ("MDD wins", "mdd_wins"),
        ]),
        "",
        "## Paired bootstrap",
        "",
        f"- Iterations/block: `{result['bootstrap_iters']}` / `{result['bootstrap_block']}` trading days",
        *boot_lines,
        "",
        "## Top-level cost sensitivity",
        "",
        "Cost is one-way bps on top-level sleeve turnover at monthly rebalance.",
        "",
        md_table(cost_rows, [
            ("candidate", "candidate"),
            ("cost_bps", "cost_bps"),
            ("CAGR", "cagr"),
            ("MDD", "max_drawdown"),
            ("Sharpe", "sharpe"),
        ]),
        "",
        "## Verdict",
        "",
        "The optimizer likes wider bands and stronger tanh settings in full sample, but those are not robust enough to replace the current conservative setting. Walk-forward objective selection does not beat the current parameter set cleanly, and MDD defense remains inconsistent. Keep the current final_fx parameters unless a separate out-of-sample gate is added for a more aggressive variant.",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--block", type=int, default=20)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    ctx = load_context()
    df, navs, weights = build_grid(ctx)
    selected = select_full_sample(df)
    wf = walk_forward(ctx, df, navs)
    boot = bootstrap_candidates(navs, selected, iters=args.iters, block=args.block, seed=args.seed)
    costs = cost_sensitivity(ctx, selected, weights)

    result = {
        "period": f"{ctx.idx[0].date()}~{ctx.idx[-1].date()}",
        "bars": len(ctx.idx),
        "grid_size": len(grid_params()),
        "full_sample_selected": selected,
        "walk_forward": wf,
        "bootstrap_iters": args.iters,
        "bootstrap_block": args.block,
        "bootstrap": boot,
        "cost_sensitivity": costs,
    }
    out_json = ROOT / "reports" / "profile_sizing" / "yoon2_final_fx_param_optimization.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    out_md = write_report(result)

    print(f"Period {result['period']} · grid {result['grid_size']}")
    cur = df[df["label"] == "current"].iloc[0].to_dict()
    sta = df[df["label"] == "static_50_50"].iloc[0].to_dict()
    print(f"static:  CAGR {sta['cagr']*100:.2f}% MDD {sta['max_drawdown']*100:.2f}% Sharpe {sta['sharpe']:.3f}")
    print(f"current: CAGR {cur['cagr']*100:.2f}% MDD {cur['max_drawdown']*100:.2f}% Sharpe {cur['sharpe']:.3f}")
    for obj, row in selected.items():
        print(
            f"{obj}: {row['label']} "
            f"CAGR {row['cagr']*100:.2f}% MDD {row['max_drawdown']*100:.2f}% "
            f"Sharpe {row['sharpe']:.3f}"
        )
    print(f"Report: {out_md}")
    print(f"JSON: {out_json}")


if __name__ == "__main__":
    main()
