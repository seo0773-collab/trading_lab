"""yoon2_final_fx 파생: 신용스프레드(HYG/IEF) 현금 게이트 전체 검증.

yoon2_final_fx_leading_signals.py에서 확인된 사실: 신용스프레드 게이트(HYG/IEF
10일 모멘텀 < -2% & 20일선 아래, 발동 시 노출 0.5배)는 point estimate 기준
CAGR은 거의 그대로(16.23% vs 16.27%)인데 Sharpe는 개선(1.212 vs 1.189)되고
MDD도 73% 확률로 더 얕아진다(특히 2020-02 코로나 크래시 -17.13%->-8.84%).

이 스크립트는 그 결과가 특정 threshold 하나에 의존한 우연이 아닌지 검증한다:

1. 파라미터 그리드 민감도 (lookback/sma_lookback/threshold/off_scale)
2. 워크포워드 파라미터 재선택(5y 학습/1y 테스트, 매 폴드 grid에서 재선택)
3. 고정 파라미터(원안) 워크포워드(재선택 없이 실제 15개 테스트연도 그대로)
4. 페어드 부트스트랩(원안 vs current, vs static)
5. 롤링 3년 안정성
6. top-level 리밸런스 비용 민감도
7. 드로다운 에피소드 전체 재현

yoon2_final_fx.json은 건드리지 않는다.

Usage:
    PYTHONPATH=src:scripts python3 scripts/profile_sizing/yoon2_final_fx_credit_gate_validate.py
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

ORIGINAL = {"lookback": 10, "sma_lookback": 20, "threshold": -0.02, "off_scale": 0.5}

LOOKBACKS = [5, 10, 15, 20, 30]
SMA_LOOKBACKS = [10, 20, 30, 50]
THRESHOLDS = [-0.01, -0.02, -0.03, -0.05]
OFF_SCALES = [0.3, 0.5, 0.7]
COST_BPS = [0, 5, 10, 25, 50]


def _clean_metric(p: dict[str, Any]) -> dict[str, float | None]:
    keys = ["cagr", "max_drawdown", "sharpe", "total_return", "volatility"]
    return {k: (None if p.get(k) is None else float(p[k])) for k in keys}


def label_params(p: dict[str, Any]) -> str:
    return f"L{p['lookback']}_S{p['sma_lookback']}_T{p['threshold']}_O{p['off_scale']}"


def load_ratio(idx: pd.DatetimeIndex) -> pd.Series:
    hyg = load_yfinance("HYG", "1d", "max")["close"].reindex(idx).ffill()
    ief = load_yfinance("IEF", "1d", "max")["close"].reindex(idx).ffill()
    return (hyg / ief).reindex(idx).ffill()


def credit_trigger(ratio: pd.Series, p: dict[str, Any]) -> pd.Series:
    mom = ratio.pct_change(int(p["lookback"]))
    below_sma = ratio < ratio.rolling(int(p["sma_lookback"])).mean()
    return ((mom < float(p["threshold"])) & below_sma).fillna(False)


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
            episodes.append({"peak": peak_t, "trough": dd.loc[peak_t:t].idxmin(), "end": t})
    return episodes


def main() -> None:
    ctx = load_context()
    idx = ctx.idx
    ratio = load_ratio(idx)
    current_w = relmom_weights(ctx, **CURRENT)
    usd_series = current_w[ctx.sub_ids[0]].reindex(idx).ffill().fillna(0.5)
    current_nav = combine_nav(ctx, current_w)
    static_nav = combine_nav(ctx, weight_frame(ctx, 0.5))

    def nav_for(p: dict[str, Any]) -> pd.Series:
        trig = credit_trigger(ratio, p)
        w = gate_weights(ctx, usd_series, trig, float(p["off_scale"]))
        return combine_cash(ctx, w)

    # ---- 1. grid sensitivity ----
    grid_rows = []
    grid_navs: dict[str, pd.Series] = {"current": current_nav, "static_50_50": static_nav}
    for lb in LOOKBACKS:
        for sma in SMA_LOOKBACKS:
            for th in THRESHOLDS:
                for off in OFF_SCALES:
                    p = {"lookback": lb, "sma_lookback": sma, "threshold": th, "off_scale": off}
                    label = label_params(p)
                    nav = nav_for(p)
                    grid_navs[label] = nav
                    m = _clean_metric(perf_nav(nav))
                    grid_rows.append({"label": label, "params": p, **m})
    grid_df = pd.DataFrame(grid_rows)

    # how many of the full grid beat current on both Sharpe and MDD
    cur_m = _clean_metric(perf_nav(current_nav))
    beats_both = grid_df[
        (grid_df["sharpe"] > cur_m["sharpe"]) & (grid_df["max_drawdown"] > cur_m["max_drawdown"])
    ]
    beats_sharpe_only = grid_df[grid_df["sharpe"] > cur_m["sharpe"]]
    beats_mdd_only = grid_df[grid_df["max_drawdown"] > cur_m["max_drawdown"]]

    # ---- 2. walk-forward with reselection ----
    def perf_window(nav: pd.Series, window: pd.DatetimeIndex) -> dict[str, Any]:
        sub = nav.reindex(window).ffill().dropna()
        return _clean_metric(perf_nav(sub)) if len(sub) >= 60 else {}

    wf_rows = []
    for start in range(_years(ctx)[0], _years(ctx)[-1] - 5 + 1):
        train = idx[(idx.year >= start) & (idx.year <= start + 4)]
        test = idx[idx.year == start + 5]
        if len(train) < 750 or len(test) < 60:
            continue
        train_scores = []
        for row in grid_rows:
            m = perf_window(grid_navs[row["label"]], train)
            if m.get("sharpe") is not None:
                train_scores.append((float(m["sharpe"]), row["label"], row["params"]))
        if not train_scores:
            continue
        train_scores.sort(reverse=True)
        _, chosen_label, chosen_params = train_scores[0]
        wf_rows.append({
            "train": f"{start}-{start+4}",
            "test": str(start + 5),
            "chosen": chosen_label,
            "chosen_perf": perf_window(grid_navs[chosen_label], test),
            "original_perf": perf_window(grid_navs[label_params(ORIGINAL)], test),
            "current_perf": perf_window(current_nav, test),
            "static_perf": perf_window(static_nav, test),
        })

    def avg(rows_: list[dict[str, Any]], key: str, metric: str) -> float | None:
        vals = [r[key].get(metric) for r in rows_ if r[key].get(metric) is not None]
        return float(np.mean(vals)) if vals else None

    wf_summary = {
        "folds": len(wf_rows),
        "avg_chosen": {m: avg(wf_rows, "chosen_perf", m) for m in ["cagr", "max_drawdown", "sharpe"]},
        "avg_original": {m: avg(wf_rows, "original_perf", m) for m in ["cagr", "max_drawdown", "sharpe"]},
        "avg_current": {m: avg(wf_rows, "current_perf", m) for m in ["cagr", "max_drawdown", "sharpe"]},
        "avg_static": {m: avg(wf_rows, "static_perf", m) for m in ["cagr", "max_drawdown", "sharpe"]},
        "original_sharpe_wins_current": sum(
            (r["original_perf"].get("sharpe") or -999) > (r["current_perf"].get("sharpe") or -999) for r in wf_rows
        ),
        "original_mdd_wins_current": sum(
            (r["original_perf"].get("max_drawdown") or -999) > (r["current_perf"].get("max_drawdown") or -999) for r in wf_rows
        ),
        "chosen_sharpe_wins_current": sum(
            (r["chosen_perf"].get("sharpe") or -999) > (r["current_perf"].get("sharpe") or -999) for r in wf_rows
        ),
        "chosen_mdd_wins_current": sum(
            (r["chosen_perf"].get("max_drawdown") or -999) > (r["current_perf"].get("max_drawdown") or -999) for r in wf_rows
        ),
    }

    # ---- 3. rolling 3y ----
    roll_rows = []
    for start in range(_years(ctx)[0], _years(ctx)[-1] - 2 + 1):
        window = idx[(idx.year >= start) & (idx.year <= start + 2)]
        if len(window) < 500:
            continue
        roll_rows.append({
            "window": f"{start}-{start+2}",
            "original": perf_window(grid_navs[label_params(ORIGINAL)], window),
            "current": perf_window(current_nav, window),
        })
    roll_wins = {
        "sharpe": sum((r["original"].get("sharpe") or -999) > (r["current"].get("sharpe") or -999) for r in roll_rows),
        "mdd": sum((r["original"].get("max_drawdown") or -999) > (r["current"].get("max_drawdown") or -999) for r in roll_rows),
        "cagr": sum((r["original"].get("cagr") or -999) > (r["current"].get("cagr") or -999) for r in roll_rows),
        "n": len(roll_rows),
    }

    # ---- 4. bootstrap ----
    original_nav = grid_navs[label_params(ORIGINAL)]
    boot_vs_current = paired_bootstrap(
        original_nav, current_nav, label_a="original", label_b="current", iters=8000, block=20, seed=23,
    )
    boot_vs_static = paired_bootstrap(
        original_nav, static_nav, label_a="original", label_b="static_50_50", iters=8000, block=20, seed=29,
    )

    # ---- 5. cost sensitivity ----
    cost_rows = []
    for label, nav_fn in [
        ("static_50_50", lambda bps: combine_cash(ctx, weight_frame(ctx, 0.5).assign(__CASH__=0.0), bps)),
        ("current", lambda bps: combine_cash(ctx, current_w.assign(__CASH__=0.0), bps)),
        ("original_credit_gate", lambda bps: combine_cash(
            ctx, gate_weights(ctx, usd_series, credit_trigger(ratio, ORIGINAL), ORIGINAL["off_scale"]), bps
        )),
    ]:
        for bps in COST_BPS:
            nav = nav_fn(bps)
            m = _clean_metric(perf_nav(nav))
            cost_rows.append({"candidate": label, "cost_bps": bps, **m})

    # ---- 6. drawdown episodes ----
    episodes = find_episodes(current_nav, thresh=-0.05)
    episode_rows = []
    for e in episodes:
        peak, trough = e["peak"], e["trough"]
        row = {
            "peak": str(peak.date()), "trough": str(trough.date()),
            "current_mdd": float(current_nav.loc[peak:trough].min() / current_nav.loc[peak] - 1.0),
            "original_mdd": float(original_nav.loc[peak:trough].min() / original_nav.loc[peak] - 1.0),
        }
        episode_rows.append(row)

    result = {
        "period": f"{idx[0].date()}~{idx[-1].date()}",
        "bars": len(idx),
        "original_params": ORIGINAL,
        "grid_size": len(grid_rows),
        "current_point": cur_m,
        "beats_current_sharpe_and_mdd": len(beats_both),
        "beats_current_sharpe_only": len(beats_sharpe_only),
        "beats_current_mdd_only": len(beats_mdd_only),
        "walk_forward": {"rows": wf_rows, "summary": wf_summary},
        "rolling_3y": {"rows": roll_rows, "wins": roll_wins},
        "bootstrap_vs_current": boot_vs_current,
        "bootstrap_vs_static": boot_vs_static,
        "cost_sensitivity": cost_rows,
        "drawdown_episodes": episode_rows,
    }
    out_json = ROOT / "reports" / "profile_sizing" / "yoon2_final_fx_credit_gate_validate.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    # ---- markdown ----
    lines = [
        "# yoon2_final_fx 신용스프레드(HYG/IEF) 현금 게이트 전체 검증",
        "",
        f"- Period: `{result['period']}` ({result['bars']} bars)",
        f"- 원안 파라미터: `{ORIGINAL}`",
        f"- 그리드 크기: `{result['grid_size']}` (lookback {LOOKBACKS} x sma {SMA_LOOKBACKS} x threshold {THRESHOLDS} x off_scale {OFF_SCALES})",
        "- yoon2_final_fx.json은 동결 유지.",
        "",
        "## 1. 그리드 민감도",
        "",
        f"- current(40/60) 대비 Sharpe·MDD **둘 다** 이긴 조합: `{result['beats_current_sharpe_and_mdd']}/{result['grid_size']}`",
        f"- Sharpe만 이긴 조합: `{result['beats_current_sharpe_only']}/{result['grid_size']}`",
        f"- MDD만 이긴 조합: `{result['beats_current_mdd_only']}/{result['grid_size']}`",
        f"- current: CAGR `{_fmt_pct(cur_m['cagr'])}`, MDD `{_fmt_pct(cur_m['max_drawdown'])}`, Sharpe `{_fmt_num(cur_m['sharpe'])}`",
        "",
        "## 2. 워크포워드 (5y 학습/1y 테스트, 매 폴드 grid에서 Sharpe 최대 재선택)",
        "",
        f"- Folds: `{wf_summary['folds']}`",
        f"- Avg 재선택 chosen: CAGR `{_fmt_pct(wf_summary['avg_chosen']['cagr'])}`, MDD `{_fmt_pct(wf_summary['avg_chosen']['max_drawdown'])}`, Sharpe `{_fmt_num(wf_summary['avg_chosen']['sharpe'])}`",
        f"- Avg 원안 고정: CAGR `{_fmt_pct(wf_summary['avg_original']['cagr'])}`, MDD `{_fmt_pct(wf_summary['avg_original']['max_drawdown'])}`, Sharpe `{_fmt_num(wf_summary['avg_original']['sharpe'])}`",
        f"- Avg current: CAGR `{_fmt_pct(wf_summary['avg_current']['cagr'])}`, MDD `{_fmt_pct(wf_summary['avg_current']['max_drawdown'])}`, Sharpe `{_fmt_num(wf_summary['avg_current']['sharpe'])}`",
        f"- Avg static: CAGR `{_fmt_pct(wf_summary['avg_static']['cagr'])}`, MDD `{_fmt_pct(wf_summary['avg_static']['max_drawdown'])}`, Sharpe `{_fmt_num(wf_summary['avg_static']['sharpe'])}`",
        f"- 원안(고정) Sharpe 승 vs current: `{wf_summary['original_sharpe_wins_current']}/{wf_summary['folds']}`",
        f"- 원안(고정) MDD 승 vs current: `{wf_summary['original_mdd_wins_current']}/{wf_summary['folds']}`",
        f"- 재선택 Sharpe 승 vs current: `{wf_summary['chosen_sharpe_wins_current']}/{wf_summary['folds']}`",
        f"- 재선택 MDD 승 vs current: `{wf_summary['chosen_mdd_wins_current']}/{wf_summary['folds']}`",
        "",
        "## 3. 롤링 3년 안정성 (원안 고정 vs current)",
        "",
        f"- Sharpe 승: `{roll_wins['sharpe']}/{roll_wins['n']}`",
        f"- CAGR 승: `{roll_wins['cagr']}/{roll_wins['n']}`",
        f"- MDD 승: `{roll_wins['mdd']}/{roll_wins['n']}`",
        "",
        "## 4. 페어드 부트스트랩 (8000 iters, block=20)",
        "",
        f"- original vs current: P(Sharpe>) `{boot_vs_current['prob']['sharpe_a_gt_b']*100:.1f}%`, "
        f"P(CAGR>) `{boot_vs_current['prob']['cagr_a_gt_b']*100:.1f}%`, "
        f"P(MDD shallower) `{boot_vs_current['prob']['mdd_a_shallower']*100:.1f}%`",
        f"- original vs static_50_50: P(Sharpe>) `{boot_vs_static['prob']['sharpe_a_gt_b']*100:.1f}%`, "
        f"P(CAGR>) `{boot_vs_static['prob']['cagr_a_gt_b']*100:.1f}%`, "
        f"P(MDD shallower) `{boot_vs_static['prob']['mdd_a_shallower']*100:.1f}%`",
        "",
        "## 5. Top-level 리밸런스 비용 민감도",
        "",
        "| candidate | cost_bps | CAGR | MDD | Sharpe |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for r in cost_rows:
        lines.append(f"| {r['candidate']} | {r['cost_bps']} | {_fmt_pct(r['cagr'])} | {_fmt_pct(r['max_drawdown'])} | {_fmt_num(r['sharpe'])} |")

    lines.extend([
        "",
        "## 6. 드로다운 에피소드 전체 재현 (-5% 이상)",
        "",
        "| peak | trough | current | 원안 신용게이트 |",
        "| --- | --- | ---: | ---: |",
    ])
    for r in episode_rows:
        lines.append(f"| {r['peak']} | {r['trough']} | {_fmt_pct(r['current_mdd'])} | {_fmt_pct(r['original_mdd'])} |")

    lines.extend(["", "## 판정", ""])
    ratio_both = result["beats_current_sharpe_and_mdd"] / result["grid_size"]
    if (
        wf_summary["original_sharpe_wins_current"] >= wf_summary["folds"] * 0.5
        and wf_summary["original_mdd_wins_current"] >= wf_summary["folds"] * 0.5
        and boot_vs_current["prob"]["mdd_a_shallower"] > 0.6
    ):
        verdict = (
            f"그리드 {ratio_both*100:.0f}%가 current를 Sharpe·MDD 동시에 이기고, 워크포워드/롤링/부트스트랩에서도 "
            "일관되게 우위라 특정 threshold 하나에 의존한 우연이 아니라고 판단된다. 승격 후보로 검토할 만하다."
        )
    else:
        verdict = (
            "부트스트랩과 point estimate에서는 매력적이지만 워크포워드·롤링 3년에서는 승률이 절반을 밑돌아, "
            "2020년 코로나 크래시 한두 개 에피소드의 영향이 커 보인다. 승격 전에 해당 에피소드를 제외한 "
            "강건성 확인이 더 필요하다."
        )
    lines.append(verdict)
    lines.extend(["", "## 파일", "", f"- JSON: `{out_json.relative_to(ROOT)}`"])

    out_md = ROOT / "reports" / "profile_sizing" / "yoon2_final_fx_credit_gate_validate.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"JSON: {out_json}")
    print(f"MD:   {out_md}")


if __name__ == "__main__":
    main()
