"""yoon2_final_fx 파생: USD/KRW band를 30/70으로 넓힌 후보의 검증.

기존 param_optimization.md에서 grid-search 전체가 in-sample로 고른 최적해는
walk-forward에서 무너진다는 게 확인됐다. 이 스크립트는 사용자가 직접 지정한
"단일" 후보(L63_t0.3_s0.025_b30_70 — optimize.py grid의 sharpe 목적함수 풀샘플
1위였던 조합)를 고정 파라미터로 두고, current(40/60)·static(50/50) 대비
point estimate/paired bootstrap/fixed-param walk-forward/rolling 3y/비용
민감도/과거 드로다운 에피소드까지 재현성 있게 확인한다.

yoon2_final_fx.json 자체는 건드리지 않는다(동결 정책). 승격 여부 판단용 리포트만
생성한다.

Usage:
    PYTHONPATH=src:scripts python3 scripts/profile_sizing/yoon2_final_fx_us_tilt_3070.py
"""
from __future__ import annotations

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

from profile_sizing.yoon2_final_fx_validate import (  # noqa: E402
    CURRENT,
    Context,
    _clean_metric,
    _fmt_num,
    _fmt_pct,
    combine_nav,
    load_context,
    paired_bootstrap,
    perf_nav,
    relmom_weights,
    summarize_nav,
    weight_frame,
)

CANDIDATE = {
    "lookback": 63,
    "tilt": 0.3,
    "scale": 0.025,
    "band": (0.30, 0.70),
    "fx_weight": 0.0,
    "lag": 1,
}
CANDIDATE_LABEL = "us_tilt_30_70 (L63_t0.3_s0.025_b30_70)"
COST_BPS = [0, 5, 10, 25, 50]


def _years(ctx: Context) -> list[int]:
    return sorted({int(d.year) for d in ctx.idx})


def fixed_param_walk_forward(
    ctx: Context, navs: dict[str, pd.Series]
) -> dict[str, Any]:
    """5년 롤링 창마다 test 연도 성과만 비교한다(재선택 없음, 파라미터 고정)."""
    rows = []
    for start in range(_years(ctx)[0], _years(ctx)[-1] - 5 + 1):
        test = ctx.idx[ctx.idx.year == start + 5]
        if len(test) < 60:
            continue
        rows.append({
            "test": str(start + 5),
            "candidate": _clean_metric(perf_nav(navs["candidate"], test)),
            "current": _clean_metric(perf_nav(navs["current"], test)),
            "static": _clean_metric(perf_nav(navs["static_50_50"], test)),
        })

    def avg(key: str, metric: str) -> float | None:
        vals = [r[key].get(metric) for r in rows if r[key].get(metric) is not None]
        return float(np.mean(vals)) if vals else None

    summary = {
        "folds": len(rows),
        "avg_candidate": {m: avg("candidate", m) for m in ["cagr", "max_drawdown", "sharpe"]},
        "avg_current": {m: avg("current", m) for m in ["cagr", "max_drawdown", "sharpe"]},
        "avg_static": {m: avg("static", m) for m in ["cagr", "max_drawdown", "sharpe"]},
        "candidate_sharpe_wins_current": sum(
            (r["candidate"]["sharpe"] or -999) > (r["current"]["sharpe"] or -999) for r in rows
        ),
        "candidate_mdd_wins_current": sum(
            (r["candidate"]["max_drawdown"] or -999) > (r["current"]["max_drawdown"] or -999)
            for r in rows
        ),
        "candidate_cagr_wins_current": sum(
            (r["candidate"]["cagr"] or -999) > (r["current"]["cagr"] or -999) for r in rows
        ),
    }
    return {"rows": rows, "summary": summary}


def rolling_3y(ctx: Context, navs: dict[str, pd.Series]) -> dict[str, Any]:
    rows = []
    for start in range(_years(ctx)[0], _years(ctx)[-1] - 2 + 1):
        window = ctx.idx[(ctx.idx.year >= start) & (ctx.idx.year <= start + 2)]
        if len(window) < 500:
            continue
        rows.append({
            "window": f"{start}-{start + 2}",
            "candidate": _clean_metric(perf_nav(navs["candidate"], window)),
            "current": _clean_metric(perf_nav(navs["current"], window)),
        })
    wins = {
        "sharpe": sum(
            (r["candidate"]["sharpe"] or -999) > (r["current"]["sharpe"] or -999) for r in rows
        ),
        "cagr": sum(
            (r["candidate"]["cagr"] or -999) > (r["current"]["cagr"] or -999) for r in rows
        ),
        "mdd": sum(
            (r["candidate"]["max_drawdown"] or -999) > (r["current"]["max_drawdown"] or -999)
            for r in rows
        ),
        "n": len(rows),
    }
    return {"rows": rows, "wins": wins}


def drawdown_episodes(
    ctx: Context, navs: dict[str, pd.Series]
) -> list[dict[str, Any]]:
    """current 기준 -5% 이상 드로다운 구간들에서 candidate가 얼마나 더/덜 방어했는지."""
    cur = navs["current"]
    cand = navs["candidate"]
    sta = navs["static_50_50"]
    roll_max = cur.cummax()
    dd = cur / roll_max - 1.0
    episodes = []
    in_dd = False
    peak_t = cur.index[0]
    for t, v in dd.items():
        if not in_dd and v < -0.05:
            in_dd = True
            peak_t = roll_max[:t].idxmax()
        if in_dd and v >= -0.001:
            in_dd = False
            window = cur.loc[peak_t:t]
            trough_t = window.idxmin()
            cur_mdd = float(cur.loc[peak_t:t].min() / cur.loc[peak_t] - 1.0)
            cand_mdd = float(cand.loc[peak_t:t].min() / cand.loc[peak_t] - 1.0)
            sta_mdd = float(sta.loc[peak_t:t].min() / sta.loc[peak_t] - 1.0)
            episodes.append({
                "peak": str(peak_t.date()),
                "trough": str(trough_t.date()),
                "current_mdd": cur_mdd,
                "candidate_mdd": cand_mdd,
                "static_mdd": sta_mdd,
                "candidate_minus_current": cand_mdd - cur_mdd,
            })
    return episodes


def cost_sensitivity(
    ctx: Context, weights: dict[str, pd.DataFrame]
) -> list[dict[str, Any]]:
    from profile_sizing.portfolio import rebalance_dates

    def combine_with_cost(w: pd.DataFrame, cost_bps: float) -> pd.Series:
        idx = ctx.idx
        freq = str(ctx.config.get("rebalance_freq", "monthly"))
        reb = rebalance_dates(idx, freq)
        tw = w.reindex(idx).ffill().bfill()
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

    rows = []
    for label, w in weights.items():
        for bps in COST_BPS:
            nav = combine_with_cost(w, bps)
            m = _clean_metric(perf_nav(nav))
            rows.append({"candidate": label, "cost_bps": bps, **m})
    return rows


def main() -> None:
    ctx = load_context()

    w_static = weight_frame(ctx, 0.5)
    w_current = relmom_weights(ctx, **CURRENT)
    w_candidate = relmom_weights(ctx, **CANDIDATE)
    weights = {"static_50_50": w_static, "current": w_current, "candidate": w_candidate}

    navs = {
        "static_50_50": combine_nav(ctx, w_static),
        "current": combine_nav(ctx, w_current),
        "candidate": combine_nav(ctx, w_candidate),
    }

    point_rows = [
        summarize_nav("static_50_50", navs["static_50_50"], w_static),
        summarize_nav("current (band 40/60)", navs["current"], w_current),
        summarize_nav(CANDIDATE_LABEL, navs["candidate"], w_candidate),
    ]

    boot_vs_current = paired_bootstrap(
        navs["candidate"], navs["current"],
        label_a="candidate", label_b="current", iters=5000, block=20, seed=13,
    )
    boot_vs_static = paired_bootstrap(
        navs["candidate"], navs["static_50_50"],
        label_a="candidate", label_b="static_50_50", iters=5000, block=20, seed=17,
    )

    wf = fixed_param_walk_forward(ctx, navs)
    roll = rolling_3y(ctx, navs)
    episodes = drawdown_episodes(ctx, navs)
    costs = cost_sensitivity(ctx, weights)

    result = {
        "period": f"{ctx.idx[0].date()}~{ctx.idx[-1].date()}",
        "bars": len(ctx.idx),
        "candidate_params": CANDIDATE,
        "point_estimates": point_rows,
        "bootstrap_vs_current": boot_vs_current,
        "bootstrap_vs_static": boot_vs_static,
        "walk_forward": wf,
        "rolling_3y": roll,
        "drawdown_episodes": episodes,
        "cost_sensitivity": costs,
    }
    out_json = ROOT / "reports" / "profile_sizing" / "yoon2_final_fx_us_tilt_3070.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    # ---- markdown report ----
    lines = [
        "# yoon2_final_fx band 30/70 (US 쏠림 강화) 후보 검증",
        "",
        f"- Period: `{result['period']}` ({result['bars']} bars)",
        f"- Candidate: `{CANDIDATE_LABEL}` — lookback=63, tilt=0.3, scale=0.025, band=[0.30, 0.70]",
        "- yoon2_final_fx.json은 동결 유지, 이 리포트는 파생 후보의 승격 여부 판단용.",
        "",
        "## Point estimates",
        "",
        "| label | CAGR | MDD | Sharpe | USD range |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for r in point_rows:
        lines.append(
            f"| {r['label']} | {_fmt_pct(r['cagr'])} | {_fmt_pct(r['max_drawdown'])} | "
            f"{_fmt_num(r['sharpe'])} | {r['usd_min']*100:.1f}-{r['usd_max']*100:.1f}% |"
        )

    lines.extend([
        "",
        "## Paired moving-block bootstrap (5000 iters, block=20)",
        "",
        f"- candidate vs current: P(Sharpe>) `{boot_vs_current['prob']['sharpe_a_gt_b']*100:.1f}%`, "
        f"P(CAGR>) `{boot_vs_current['prob']['cagr_a_gt_b']*100:.1f}%`, "
        f"P(MDD shallower) `{boot_vs_current['prob']['mdd_a_shallower']*100:.1f}%`",
        f"- candidate vs static_50_50: P(Sharpe>) `{boot_vs_static['prob']['sharpe_a_gt_b']*100:.1f}%`, "
        f"P(CAGR>) `{boot_vs_static['prob']['cagr_a_gt_b']*100:.1f}%`, "
        f"P(MDD shallower) `{boot_vs_static['prob']['mdd_a_shallower']*100:.1f}%`",
        "",
        "## Fixed-parameter walk-forward (재선택 없음, 5y 학습 스킵·1y 테스트만 집계)",
        "",
        f"- Folds: `{wf['summary']['folds']}`",
        f"- Avg candidate: CAGR `{_fmt_pct(wf['summary']['avg_candidate']['cagr'])}`, "
        f"MDD `{_fmt_pct(wf['summary']['avg_candidate']['max_drawdown'])}`, "
        f"Sharpe `{_fmt_num(wf['summary']['avg_candidate']['sharpe'])}`",
        f"- Avg current: CAGR `{_fmt_pct(wf['summary']['avg_current']['cagr'])}`, "
        f"MDD `{_fmt_pct(wf['summary']['avg_current']['max_drawdown'])}`, "
        f"Sharpe `{_fmt_num(wf['summary']['avg_current']['sharpe'])}`",
        f"- Avg static: CAGR `{_fmt_pct(wf['summary']['avg_static']['cagr'])}`, "
        f"MDD `{_fmt_pct(wf['summary']['avg_static']['max_drawdown'])}`, "
        f"Sharpe `{_fmt_num(wf['summary']['avg_static']['sharpe'])}`",
        f"- candidate Sharpe wins vs current: `{wf['summary']['candidate_sharpe_wins_current']}/{wf['summary']['folds']}`",
        f"- candidate CAGR wins vs current: `{wf['summary']['candidate_cagr_wins_current']}/{wf['summary']['folds']}`",
        f"- candidate MDD wins vs current: `{wf['summary']['candidate_mdd_wins_current']}/{wf['summary']['folds']}`",
        "",
        "## Rolling 3-year stability (candidate vs current)",
        "",
        f"- candidate Sharpe wins: `{roll['wins']['sharpe']}/{roll['wins']['n']}`",
        f"- candidate CAGR wins: `{roll['wins']['cagr']}/{roll['wins']['n']}`",
        f"- candidate MDD wins: `{roll['wins']['mdd']}/{roll['wins']['n']}`",
        "",
        "## 과거 드로다운 에피소드 (current 기준 -5% 이상)",
        "",
        "| peak | trough | current MDD | candidate MDD | static MDD | candidate-current |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ])
    for e in episodes:
        lines.append(
            f"| {e['peak']} | {e['trough']} | {_fmt_pct(e['current_mdd'])} | "
            f"{_fmt_pct(e['candidate_mdd'])} | {_fmt_pct(e['static_mdd'])} | "
            f"{_fmt_pct(e['candidate_minus_current'])} |"
        )

    lines.extend([
        "",
        "## Top-level 리밸런스 비용 민감도",
        "",
        "| candidate | cost_bps | CAGR | MDD | Sharpe |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for r in costs:
        if r["candidate"] in ("current", "candidate", "static_50_50") and r["cost_bps"] in (0, 25, 50):
            lines.append(
                f"| {r['candidate']} | {r['cost_bps']} | {_fmt_pct(r['cagr'])} | "
                f"{_fmt_pct(r['max_drawdown'])} | {_fmt_num(r['sharpe'])} |"
            )

    lines.extend(["", "## 판정", ""])
    sharpe_wins = wf["summary"]["candidate_sharpe_wins_current"]
    mdd_wins = wf["summary"]["candidate_mdd_wins_current"]
    folds = wf["summary"]["folds"]
    if sharpe_wins >= folds * 0.6 and mdd_wins >= folds * 0.5:
        verdict = (
            "band 30/70 후보가 고정 파라미터 워크포워드에서도 안정적으로 current(40/60)를 "
            "능가한다. 승격을 검토할 수 있다."
        )
    else:
        verdict = (
            "band 30/70 후보는 풀샘플/부트스트랩상 current(40/60)보다 우수해 보이지만, "
            "고정 파라미터 워크포워드에서는 승률이 낮아(Sharpe·MDD 각각 충분한 우위 미확보) "
            "특정 구간(최근 US 강세 국면 등)에 의존한 결과일 가능성이 있다. "
            "yoon2_final_fx.json을 이 값으로 바꾸는 것은 권장하지 않는다."
        )
    lines.append(verdict)
    lines.extend([
        "",
        "## 파일",
        "",
        f"- JSON: `{out_json.relative_to(ROOT)}`",
    ])

    out_md = ROOT / "reports" / "profile_sizing" / "yoon2_final_fx_us_tilt_3070.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"JSON: {out_json}")
    print(f"MD:   {out_md}")


if __name__ == "__main__":
    main()
