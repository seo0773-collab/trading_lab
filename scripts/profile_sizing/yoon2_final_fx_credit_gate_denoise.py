"""신용스프레드 게이트 노이즈 필터링(min_consecutive) 검증.

yoon2_final_fx_credit_gate_validate.py에서 확인된 문제: 원안(lookback=10,
sma=20, threshold=-2%, off_scale=0.5)은 19년간 103개의 별도 발동 에피소드를
만드는데, 실질적으로 도움이 된 드로다운은 1~2건뿐이다(대부분 며칠짜리 노이즈).
Yoon1kHandler._credit_spread_weak에 새로 추가한 min_consecutive(N일 연속
발동해야만 게이트 activate) 파라미터를 스윕해서 노이즈를 줄이면서 방어력이
유지되는지 확인한다.

yoon2_final_fx.json은 건드리지 않는다.

Usage:
    PYTHONPATH=src:scripts python3 scripts/profile_sizing/yoon2_final_fx_credit_gate_denoise.py
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

from run_kalman_pipeline import load_yfinance  # noqa: E402
from profile_sizing.portfolio import rebalance_dates  # noqa: E402
from profile_sizing.yoon2_final_fx_validate import (  # noqa: E402
    CURRENT,
    Context,
    _fmt_num,
    _fmt_pct,
    combine_nav,
    load_context,
    paired_bootstrap,
    perf_nav,
    relmom_weights,
    weight_frame,
)

BASE = {"lookback": 10, "sma_lookback": 20, "threshold": -0.02, "off_scale": 0.5}
MIN_CONSECUTIVE_GRID = [1, 2, 3, 5, 7, 10, 15]


def _clean_metric(p: dict[str, Any]) -> dict[str, float | None]:
    keys = ["cagr", "max_drawdown", "sharpe", "total_return", "volatility"]
    return {k: (None if p.get(k) is None else float(p[k])) for k in keys}


def load_ratio(idx: pd.DatetimeIndex) -> pd.Series:
    hyg = load_yfinance("HYG", "1d", "max")["close"].reindex(idx).ffill()
    ief = load_yfinance("IEF", "1d", "max")["close"].reindex(idx).ffill()
    return (hyg / ief).reindex(idx).ffill()


def raw_trigger(ratio: pd.Series, p: dict[str, Any]) -> pd.Series:
    mom = ratio.pct_change(int(p["lookback"]))
    below_sma = ratio < ratio.rolling(int(p["sma_lookback"])).mean()
    return ((mom < float(p["threshold"])) & below_sma).fillna(False)


def denoise(trig: pd.Series, min_consecutive: int) -> pd.Series:
    if min_consecutive <= 1:
        return trig
    confirmed = trig.rolling(min_consecutive).sum() >= min_consecutive
    return (confirmed & trig).fillna(False)


def episode_count(trig: pd.Series) -> int:
    if not trig.any():
        return 0
    runs = (trig != trig.shift()).cumsum()[trig]
    return int(runs.nunique())


def gate_weights(ctx: Context, usd_series: pd.Series, trigger: pd.Series, off_scale: float) -> pd.DataFrame:
    idx = ctx.idx
    exposure = pd.Series(1.0, index=idx).where(~trigger.shift(1).fillna(False), off_scale)
    out = pd.DataFrame(index=idx, columns=[*ctx.sub_ids, "__CASH__"], dtype=float)
    out[ctx.sub_ids[0]] = usd_series * exposure
    out[ctx.sub_ids[1]] = (1.0 - usd_series) * exposure
    out["__CASH__"] = 1.0 - exposure
    return out.fillna(0.0)


def combine_cash(ctx: Context, weights: pd.DataFrame, cost_bps: float = 0.0) -> pd.Series:
    idx = ctx.idx
    tw = weights.reindex(idx).ffill().fillna(0.0)
    cols = [*ctx.sub_ids, "__CASH__"]
    tw = tw[cols]
    tw = tw.div(tw.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    rets = [n.reindex(idx).ffill().pct_change().fillna(0.0).to_numpy() for n in ctx.navs]
    rets.append(np.zeros(len(idx), dtype=float))
    cost_rate = float(cost_bps) / 10_000.0
    reb = set(rebalance_dates(idx, str(ctx.config.get("rebalance_freq", "monthly"))))
    vals = tw.iloc[0].to_numpy(dtype=float).tolist()
    out = np.empty(len(idx))
    for i, t in enumerate(idx):
        for j, r in enumerate(rets):
            vals[j] *= 1.0 + r[i]
        total_before = sum(vals)
        if t in reb:
            target = tw.iloc[i].to_numpy(dtype=float)
            target_vals = total_before * target
            if cost_rate > 0.0:
                traded = float(np.abs(target_vals - np.array(vals)).sum())
                total_after = max(total_before - traded * cost_rate, 0.0)
                target_vals = total_after * target
            vals = target_vals.tolist()
            out[i] = sum(vals)
        else:
            out[i] = total_before
    nav = pd.Series(out, index=idx, name="equity")
    return nav / nav.iloc[0]


def _years(ctx: Context) -> list[int]:
    return sorted({int(d.year) for d in ctx.idx})


def find_episodes(nav: pd.Series, thresh: float = -0.05) -> list[dict[str, Any]]:
    roll_max = nav.cummax()
    dd = nav / roll_max - 1.0
    episodes = []
    in_dd = False
    peak_t = nav.index[0]
    for t, v in dd.items():
        if not in_dd and v < thresh:
            in_dd = True
            peak_t = roll_max[:t].idxmax()
        if in_dd and v >= -0.001:
            in_dd = False
            episodes.append({"peak": peak_t, "trough": dd.loc[peak_t:t].idxmin()})
    return episodes


def main() -> None:
    ctx = load_context()
    idx = ctx.idx
    ratio = load_ratio(idx)
    current_w = relmom_weights(ctx, **CURRENT)
    usd_series = current_w[ctx.sub_ids[0]].reindex(idx).ffill().fillna(0.5)
    current_nav = combine_nav(ctx, current_w)
    static_nav = combine_nav(ctx, weight_frame(ctx, 0.5))
    base_trig = raw_trigger(ratio, BASE)

    rows = []
    navs: dict[str, pd.Series] = {"current": current_nav, "static_50_50": static_nav}
    for mc in MIN_CONSECUTIVE_GRID:
        trig = denoise(base_trig, mc)
        w = gate_weights(ctx, usd_series, trig, BASE["off_scale"])
        nav = combine_cash(ctx, w)
        label = f"min_consec_{mc}"
        navs[label] = nav
        m = _clean_metric(perf_nav(nav))
        rows.append({
            "label": label,
            "min_consecutive": mc,
            **m,
            "trigger_days": int(trig.sum()),
            "trigger_pct": float(trig.mean()),
            "episodes": episode_count(trig),
        })

    cur_m = _clean_metric(perf_nav(current_nav))

    # ---- walk-forward (fixed params per candidate, no reselection) ----
    def perf_window(nav: pd.Series, window: pd.DatetimeIndex) -> dict[str, Any]:
        sub = nav.reindex(window).ffill().dropna()
        return _clean_metric(perf_nav(sub)) if len(sub) >= 60 else {}

    wf_rows = {}
    for mc in MIN_CONSECUTIVE_GRID:
        label = f"min_consec_{mc}"
        per_fold = []
        for start in range(_years(ctx)[0], _years(ctx)[-1] - 5 + 1):
            test = idx[idx.year == start + 5]
            if len(test) < 60:
                continue
            per_fold.append({
                "cand": perf_window(navs[label], test),
                "current": perf_window(current_nav, test),
            })
        sharpe_wins = sum((r["cand"].get("sharpe") or -999) > (r["current"].get("sharpe") or -999) for r in per_fold)
        mdd_wins = sum((r["cand"].get("max_drawdown") or -999) > (r["current"].get("max_drawdown") or -999) for r in per_fold)
        wf_rows[label] = {"folds": len(per_fold), "sharpe_wins": sharpe_wins, "mdd_wins": mdd_wins}

    # ---- bootstrap for the two most promising candidates (chosen after point estimate) ----
    grid_df = pd.DataFrame(rows)
    # candidate that keeps MDD improvement while maximizing Sharpe
    protective = grid_df[grid_df["max_drawdown"] > cur_m["max_drawdown"]]
    best_label = (
        protective.loc[protective["sharpe"].idxmax(), "label"] if not protective.empty else grid_df.loc[grid_df["sharpe"].idxmax(), "label"]
    )
    boot_best = paired_bootstrap(
        navs[best_label], current_nav, label_a=best_label, label_b="current", iters=8000, block=20, seed=41,
    )
    boot_original = paired_bootstrap(
        navs["min_consec_1"], current_nav, label_a="min_consec_1", label_b="current", iters=8000, block=20, seed=43,
    )

    # ---- drawdown episode reproduction for original vs best ----
    episodes = find_episodes(current_nav, thresh=-0.05)
    episode_rows = []
    for e in episodes:
        peak, trough = e["peak"], e["trough"]
        row = {
            "peak": str(peak.date()), "trough": str(trough.date()),
            "current_mdd": float(current_nav.loc[peak:trough].min() / current_nav.loc[peak] - 1.0),
            "original_mdd": float(navs["min_consec_1"].loc[peak:trough].min() / navs["min_consec_1"].loc[peak] - 1.0),
            "best_mdd": float(navs[best_label].loc[peak:trough].min() / navs[best_label].loc[peak] - 1.0),
        }
        episode_rows.append(row)

    result = {
        "period": f"{idx[0].date()}~{idx[-1].date()}",
        "bars": len(idx),
        "base_params": BASE,
        "current_point": cur_m,
        "grid": rows,
        "walk_forward": wf_rows,
        "best_label": best_label,
        "bootstrap_best_vs_current": boot_best,
        "bootstrap_original_vs_current": boot_original,
        "drawdown_episodes": episode_rows,
    }
    out_json = ROOT / "reports" / "profile_sizing" / "yoon2_final_fx_credit_gate_denoise.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    lines = [
        "# yoon2_final_fx 신용스프레드 게이트 노이즈 필터링(min_consecutive) 검증",
        "",
        f"- Period: `{result['period']}` ({result['bars']} bars)",
        f"- 기본 파라미터: `{BASE}` (min_consecutive만 스윕)",
        f"- current: CAGR `{_fmt_pct(cur_m['cagr'])}`, MDD `{_fmt_pct(cur_m['max_drawdown'])}`, Sharpe `{_fmt_num(cur_m['sharpe'])}`",
        "",
        "## min_consecutive 스윕",
        "",
        "| min_consecutive | CAGR | MDD | Sharpe | 발동일수 | 발동비율 | 별도 에피소드 수 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r['min_consecutive']} | {_fmt_pct(r['cagr'])} | {_fmt_pct(r['max_drawdown'])} | "
            f"{_fmt_num(r['sharpe'])} | {r['trigger_days']} | {r['trigger_pct']*100:.1f}% | {r['episodes']} |"
        )

    lines.extend([
        "",
        "## 고정 파라미터 워크포워드 (14폴드, 재선택 없음, vs current)",
        "",
        "| min_consecutive | Sharpe 승 | MDD 승 |",
        "| ---: | ---: | ---: |",
    ])
    for mc in MIN_CONSECUTIVE_GRID:
        w = wf_rows[f"min_consec_{mc}"]
        lines.append(f"| {mc} | {w['sharpe_wins']}/{w['folds']} | {w['mdd_wins']}/{w['folds']} |")

    lines.extend([
        "",
        f"## 최적 후보: `{best_label}`",
        "",
        f"- vs current 부트스트랩(8000회): P(Sharpe>) `{boot_best['prob']['sharpe_a_gt_b']*100:.1f}%`, "
        f"P(CAGR>) `{boot_best['prob']['cagr_a_gt_b']*100:.1f}%`, "
        f"P(MDD shallower) `{boot_best['prob']['mdd_a_shallower']*100:.1f}%`",
        f"- 참고, 원안(min_consecutive=1) vs current: P(Sharpe>) `{boot_original['prob']['sharpe_a_gt_b']*100:.1f}%`, "
        f"P(CAGR>) `{boot_original['prob']['cagr_a_gt_b']*100:.1f}%`, "
        f"P(MDD shallower) `{boot_original['prob']['mdd_a_shallower']*100:.1f}%`",
        "",
        "## 드로다운 에피소드 (current 기준 -5% 이상): 원안 vs 최적 후보",
        "",
        "| peak | trough | current | 원안(min_consec=1) | 최적후보 |",
        "| --- | --- | ---: | ---: | ---: |",
    ])
    for r in episode_rows:
        lines.append(
            f"| {r['peak']} | {r['trough']} | {_fmt_pct(r['current_mdd'])} | "
            f"{_fmt_pct(r['original_mdd'])} | {_fmt_pct(r['best_mdd'])} |"
        )

    lines.extend(["", "## 판정", ""])
    orig_row = next(r for r in rows if r["min_consecutive"] == 1)
    best_row = next(r for r in rows if r["label"] == best_label)
    if best_label != "min_consec_1" and best_row["sharpe"] > orig_row["sharpe"] and best_row["max_drawdown"] >= orig_row["max_drawdown"] - 0.005:
        verdict = (
            f"`{best_label}`가 원안 대비 발동일수/에피소드 수를 줄이면서 Sharpe·MDD를 "
            "동시에 개선(또는 거의 유지)한다. 노이즈 필터링이 실제로 도움이 된다 — 승격 후보 파라미터로 교체 검토."
        )
    else:
        verdict = (
            "min_consecutive를 늘려도 노이즈(발동일수)는 줄지만 성과 개선으로 이어지지 않는다. "
            "원안(min_consecutive=1)을 유지하는 편이 낫다."
        )
    lines.append(verdict)
    lines.extend(["", "## 파일", "", f"- JSON: `{out_json.relative_to(ROOT)}`"])

    out_md = ROOT / "reports" / "profile_sizing" / "yoon2_final_fx_credit_gate_denoise.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"JSON: {out_json}")
    print(f"MD:   {out_md}")


if __name__ == "__main__":
    main()
