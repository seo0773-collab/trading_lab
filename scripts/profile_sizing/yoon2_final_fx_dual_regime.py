"""yoon2_final_fx 파생: 양쪽 동반 하락 -> 현금 확대, 양쪽 동반 상승 -> SSO(2x) 일부 편입.

yoon2_final_fx2_overlay.py가 이미 "동반 약세 -> 현금" 게이트를 테스트했다
(gate_only: off_scale=0.65, dd_threshold=-0.10 -> CAGR 16.00%/MDD -17.13%,
current 대비 딱히 개선 없음). 이 스크립트는 그 결론 위에서

1. down_gate를 더 깊게(off_scale 낮춤) 걸었을 때 실제로 MDD가 얕아지는지
2. 새로운 아이디어: 동반 강세 구간에서 US 슬리브의 일부를 SSO(프로셰어스
   2x S&P500)로 바꿔치기했을 때 상승 구간 수익이 얼마나 늘고 MDD가 얼마나
   악화되는지

를 함께 검증한다. yoon2_final_fx.json은 건드리지 않는다(동결 유지).

Usage:
    PYTHONPATH=src:scripts python3 scripts/profile_sizing/yoon2_final_fx_dual_regime.py
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
    load_context,
    paired_bootstrap,
    perf_nav,
    relmom_weights,
    weight_frame,
)

DOWN_GATE = {
    "lookback": 63,
    "sma_lookback": 200,
    "dd_threshold": -0.10,
    "vol_threshold": 0.24,
}
UP_BOOST = {
    "lookback": 63,
    "sma_lookback": 200,
    "near_high_threshold": -0.03,
    "sso_frac": 0.30,
}


def _clean_metric(p: dict[str, Any]) -> dict[str, float | None]:
    keys = ["cagr", "max_drawdown", "sharpe", "total_return", "volatility"]
    return {k: (None if p.get(k) is None else float(p[k])) for k in keys}


def load_sso_nav(idx: pd.DatetimeIndex, fx: pd.Series) -> pd.Series:
    px = load_yfinance("SSO", "1d", "max")["close"].reindex(idx).ffill()
    px_krw = px * fx.reindex(idx).ffill()
    return (px_krw / px_krw.dropna().iloc[0]).rename("SSO")


def market_signals(ctx: Context) -> dict[str, pd.Series]:
    us = ctx.spy_krw_nav.reindex(ctx.idx).ffill()
    kr = ctx.kodex_nav.reindex(ctx.idx).ffill()
    return {"us": us, "kr": kr}


def regime_flags(
    sig: dict[str, pd.Series], *, lookback: int, sma_lookback: int
) -> dict[str, pd.Series]:
    us, kr = sig["us"], sig["kr"]
    us_ret = us.pct_change(lookback)
    kr_ret = kr.pct_change(lookback)
    us_sma = us.rolling(sma_lookback).mean()
    kr_sma = kr.rolling(sma_lookback).mean()
    us_dd = us / us.rolling(lookback).max() - 1.0
    kr_dd = kr / kr.rolling(lookback).max() - 1.0
    us_vol = us.pct_change().rolling(lookback).std() * np.sqrt(252.0)
    kr_vol = kr.pct_change().rolling(lookback).std() * np.sqrt(252.0)
    return {
        "us_ret": us_ret, "kr_ret": kr_ret,
        "us_sma": us_sma, "kr_sma": kr_sma,
        "us_dd": us_dd, "kr_dd": kr_dd,
        "us_vol": us_vol, "kr_vol": kr_vol,
    }


def down_signal(sig: dict[str, pd.Series], f: dict[str, pd.Series], p: dict[str, Any]) -> pd.Series:
    return (
        (f["us_ret"] < 0.0) & (f["kr_ret"] < 0.0)
        & (sig["us"] < f["us_sma"]) & (sig["kr"] < f["kr_sma"])
        & (
            (f["us_dd"] < float(p["dd_threshold"])) | (f["kr_dd"] < float(p["dd_threshold"]))
            | (np.maximum(f["us_vol"], f["kr_vol"]) > float(p["vol_threshold"]))
        )
    )


def up_signal(sig: dict[str, pd.Series], f: dict[str, pd.Series], p: dict[str, Any]) -> pd.Series:
    return (
        (f["us_ret"] > 0.0) & (f["kr_ret"] > 0.0)
        & (sig["us"] > f["us_sma"]) & (sig["kr"] > f["kr_sma"])
        & (f["us_dd"] > float(p["near_high_threshold"])) & (f["kr_dd"] > float(p["near_high_threshold"]))
    )


def build_weights(
    ctx: Context,
    sso_nav: pd.Series,
    *,
    down_gate: dict[str, Any] | None,
    down_off_scale: float | None,
    up_boost: dict[str, Any] | None,
) -> pd.DataFrame:
    sig = market_signals(ctx)
    usd = relmom_weights(ctx, **CURRENT)[ctx.sub_ids[0]].reindex(ctx.idx).ffill().fillna(0.5)

    exposure = pd.Series(1.0, index=ctx.idx)
    if down_gate and down_off_scale is not None:
        f = regime_flags(sig, lookback=int(down_gate["lookback"]), sma_lookback=int(down_gate["sma_lookback"]))
        weak = down_signal(sig, f, down_gate)
        exposure = exposure.where(~weak.shift(1).fillna(False), float(down_off_scale))

    sso_frac = pd.Series(0.0, index=ctx.idx)
    if up_boost:
        f = regime_flags(sig, lookback=int(up_boost["lookback"]), sma_lookback=int(up_boost["sma_lookback"]))
        strong = up_signal(sig, f, up_boost)
        sso_frac = sso_frac.where(~strong.shift(1).fillna(False), float(up_boost["sso_frac"]))

    out = pd.DataFrame(index=ctx.idx, columns=[*ctx.sub_ids, "SSO", "__CASH__"], dtype=float)
    us_total = usd * exposure
    out[ctx.sub_ids[0]] = us_total * (1.0 - sso_frac)
    out["SSO"] = us_total * sso_frac
    out[ctx.sub_ids[1]] = (1.0 - usd) * exposure
    out["__CASH__"] = 1.0 - exposure
    return out.fillna(0.0)


def combine_nav(ctx: Context, sso_nav: pd.Series, weights: pd.DataFrame) -> pd.Series:
    idx = ctx.idx
    tw = weights.reindex(idx).ffill().fillna(0.0)
    cols = [*ctx.sub_ids, "SSO", "__CASH__"]
    tw = tw[cols]
    tw = tw.div(tw.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    rets = [n.reindex(idx).ffill().pct_change().fillna(0.0).to_numpy() for n in ctx.navs]
    rets.append(sso_nav.reindex(idx).ffill().pct_change().fillna(0.0).to_numpy())
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


def _years(ctx: Context) -> list[int]:
    return sorted({int(d.year) for d in ctx.idx})


def fixed_walk_forward(ctx: Context, navs: dict[str, pd.Series], label: str) -> dict[str, Any]:
    rows = []
    for start in range(_years(ctx)[0], _years(ctx)[-1] - 5 + 1):
        test = ctx.idx[ctx.idx.year == start + 5]
        if len(test) < 60:
            continue
        rows.append({
            "test": str(start + 5),
            "candidate": _clean_metric(perf_nav(navs[label], test)),
            "current": _clean_metric(perf_nav(navs["current"], test)),
        })

    def avg(key: str, m: str) -> float | None:
        vals = [r[key].get(m) for r in rows if r[key].get(m) is not None]
        return float(np.mean(vals)) if vals else None

    return {
        "folds": len(rows),
        "avg_candidate": {m: avg("candidate", m) for m in ["cagr", "max_drawdown", "sharpe"]},
        "avg_current": {m: avg("current", m) for m in ["cagr", "max_drawdown", "sharpe"]},
        "sharpe_wins": sum((r["candidate"]["sharpe"] or -999) > (r["current"]["sharpe"] or -999) for r in rows),
        "mdd_wins": sum((r["candidate"]["max_drawdown"] or -999) > (r["current"]["max_drawdown"] or -999) for r in rows),
        "cagr_wins": sum((r["candidate"]["cagr"] or -999) > (r["current"]["cagr"] or -999) for r in rows),
    }


def drawdown_episodes(navs: dict[str, pd.Series], labels: list[str], thresh: float = -0.05) -> list[dict[str, Any]]:
    cur = navs["current"]
    roll_max = cur.cummax()
    dd = cur / roll_max - 1.0
    episodes = []
    in_dd = False
    peak_t = cur.index[0]
    for t, v in dd.items():
        if not in_dd and v < thresh:
            in_dd = True
            peak_t = roll_max[:t].idxmax()
        if in_dd and v >= -0.001:
            in_dd = False
            row: dict[str, Any] = {"peak": str(peak_t.date()), "trough": str(dd.loc[peak_t:t].idxmin().date())}
            for lab in labels:
                nv = navs[lab]
                row[f"{lab}_mdd"] = float(nv.loc[peak_t:t].min() / nv.loc[peak_t] - 1.0)
            episodes.append(row)
    return episodes


def top_rally_windows(navs: dict[str, pd.Series], labels: list[str], n: int = 8) -> list[dict[str, Any]]:
    """current 기준 63일 롤링 최고 수익 구간 상위 n개에서 후보 비교."""
    cur = navs["current"]
    roll_ret = cur.pct_change(63)
    top_ends = roll_ret.dropna().sort_values(ascending=False)
    picked: list[pd.Timestamp] = []
    for end_t in top_ends.index:
        if any(abs((end_t - p).days) < 63 for p in picked):
            continue
        picked.append(end_t)
        if len(picked) >= n:
            break
    rows = []
    for end_t in picked:
        start_t = cur.index[cur.index.get_loc(end_t) - 63]
        row: dict[str, Any] = {"start": str(start_t.date()), "end": str(end_t.date())}
        for lab in labels:
            nv = navs[lab]
            row[f"{lab}_ret"] = float(nv.loc[end_t] / nv.loc[start_t] - 1.0)
        rows.append(row)
    return rows


def main() -> None:
    ctx = load_context()
    fx = ctx.fx_nav.reindex(ctx.idx).ffill()
    sso_nav = load_sso_nav(ctx.idx, fx)

    w_static = weight_frame(ctx, 0.5)
    w_static["SSO"] = 0.0
    w_static["__CASH__"] = 0.0

    w_current = relmom_weights(ctx, **CURRENT).copy()
    w_current["SSO"] = 0.0
    w_current["__CASH__"] = 0.0

    w_down_only = build_weights(ctx, sso_nav, down_gate=DOWN_GATE, down_off_scale=0.65, up_boost=None)
    w_down_deep = build_weights(ctx, sso_nav, down_gate=DOWN_GATE, down_off_scale=0.35, up_boost=None)
    w_up_only = build_weights(ctx, sso_nav, down_gate=None, down_off_scale=None, up_boost=UP_BOOST)
    w_dual = build_weights(ctx, sso_nav, down_gate=DOWN_GATE, down_off_scale=0.35, up_boost=UP_BOOST)

    weights = {
        "static_50_50": w_static,
        "current": w_current,
        "down_gate_0.65": w_down_only,
        "down_gate_0.35": w_down_deep,
        "up_sso_0.30": w_up_only,
        "dual_regime": w_dual,
    }
    navs = {label: combine_nav(ctx, sso_nav, w) for label, w in weights.items()}

    point_rows = []
    for label, w in weights.items():
        nav = navs[label]
        m = _clean_metric(perf_nav(nav))
        us_sleeve = w[ctx.sub_ids[0]]
        sso_w = w["SSO"]
        cash_w = w["__CASH__"]
        point_rows.append({
            "label": label,
            **m,
            "us_sleeve_mean": float(us_sleeve.mean()),
            "sso_mean": float(sso_w.mean()),
            "sso_max": float(sso_w.max()),
            "sso_days": int((sso_w > 0.0).sum()),
            "cash_mean": float(cash_w.mean()),
            "cash_max": float(cash_w.max()),
            "cash_days": int((cash_w > 0.0).sum()),
        })

    boot = {}
    for label in ["down_gate_0.65", "down_gate_0.35", "up_sso_0.30", "dual_regime"]:
        boot[label] = paired_bootstrap(
            navs[label], navs["current"], label_a=label, label_b="current",
            iters=5000, block=20, seed=hash(label) % 1000 + 1,
        )

    wf = {label: fixed_walk_forward(ctx, navs, label) for label in ["down_gate_0.35", "up_sso_0.30", "dual_regime"]}
    episodes = drawdown_episodes(navs, ["current", "down_gate_0.35", "dual_regime"])
    rallies = top_rally_windows(navs, ["current", "up_sso_0.30", "dual_regime"])

    result = {
        "period": f"{ctx.idx[0].date()}~{ctx.idx[-1].date()}",
        "bars": len(ctx.idx),
        "down_gate_params": DOWN_GATE,
        "up_boost_params": UP_BOOST,
        "point_estimates": point_rows,
        "bootstrap_vs_current": boot,
        "walk_forward": wf,
        "drawdown_episodes": episodes,
        "top_rally_windows": rallies,
    }
    out_json = ROOT / "reports" / "profile_sizing" / "yoon2_final_fx_dual_regime.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    lines = [
        "# yoon2_final_fx 파생: 동반하락->현금 확대 + 동반상승->SSO 편입",
        "",
        f"- Period: `{result['period']}` ({result['bars']} bars)",
        "- yoon2_final_fx.json은 동결 유지, top-level 오버레이만 테스트한 파생 리포트.",
        f"- down_gate: lookback={DOWN_GATE['lookback']}, sma={DOWN_GATE['sma_lookback']}, "
        f"dd_threshold={DOWN_GATE['dd_threshold']}, vol_threshold={DOWN_GATE['vol_threshold']} "
        "(양쪽 다 하락추세+SMA 아래+한쪽 이상 드로다운/변동성 트리거일 때 노출 축소)",
        f"- up_boost: lookback={UP_BOOST['lookback']}, sma={UP_BOOST['sma_lookback']}, "
        f"near_high_threshold={UP_BOOST['near_high_threshold']}, sso_frac={UP_BOOST['sso_frac']} "
        "(양쪽 다 상승추세+SMA 위+고점 근접일 때 US 슬리브의 30%를 SSO로 교체)",
        "",
        "## Point estimates",
        "",
        "| label | CAGR | MDD | Sharpe | SSO 평균/최대/일수 | 현금 평균/최대/일수 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in point_rows:
        lines.append(
            f"| {r['label']} | {_fmt_pct(r['cagr'])} | {_fmt_pct(r['max_drawdown'])} | "
            f"{_fmt_num(r['sharpe'])} | {r['sso_mean']*100:.2f}%/{r['sso_max']*100:.1f}%/{r['sso_days']} | "
            f"{r['cash_mean']*100:.2f}%/{r['cash_max']*100:.1f}%/{r['cash_days']} |"
        )

    lines.extend(["", "## Paired bootstrap vs current (5000 iters, block=20)", ""])
    for label, b in boot.items():
        lines.append(
            f"- `{label}` vs current: P(Sharpe>) `{b['prob']['sharpe_a_gt_b']*100:.1f}%`, "
            f"P(CAGR>) `{b['prob']['cagr_a_gt_b']*100:.1f}%`, "
            f"P(MDD shallower) `{b['prob']['mdd_a_shallower']*100:.1f}%`"
        )

    lines.extend(["", "## Fixed-parameter walk-forward (재선택 없음)", ""])
    for label, w in wf.items():
        lines.append(
            f"- `{label}`: folds={w['folds']}, avg candidate CAGR `{_fmt_pct(w['avg_candidate']['cagr'])}`/"
            f"MDD `{_fmt_pct(w['avg_candidate']['max_drawdown'])}`/Sharpe `{_fmt_num(w['avg_candidate']['sharpe'])}` "
            f"vs current CAGR `{_fmt_pct(w['avg_current']['cagr'])}`/MDD `{_fmt_pct(w['avg_current']['max_drawdown'])}`/"
            f"Sharpe `{_fmt_num(w['avg_current']['sharpe'])}` "
            f"(Sharpe승 {w['sharpe_wins']}/{w['folds']}, MDD승 {w['mdd_wins']}/{w['folds']}, CAGR승 {w['cagr_wins']}/{w['folds']})"
        )

    lines.extend([
        "",
        "## 과거 드로다운 에피소드 (current 기준 -5% 이상)",
        "",
        "| peak | trough | current MDD | down_gate_0.35 MDD | dual_regime MDD |",
        "| --- | --- | ---: | ---: | ---: |",
    ])
    for e in episodes:
        lines.append(
            f"| {e['peak']} | {e['trough']} | {_fmt_pct(e['current_mdd'])} | "
            f"{_fmt_pct(e['down_gate_0.35_mdd'])} | {_fmt_pct(e['dual_regime_mdd'])} |"
        )

    lines.extend([
        "",
        "## 상위 랠리(63일 롤링) 구간 포착 비교",
        "",
        "| start | end | current | up_sso_0.30 | dual_regime |",
        "| --- | --- | ---: | ---: | ---: |",
    ])
    for r in rallies:
        lines.append(
            f"| {r['start']} | {r['end']} | {_fmt_pct(r['current_ret'])} | "
            f"{_fmt_pct(r['up_sso_0.30_ret'])} | {_fmt_pct(r['dual_regime_ret'])} |"
        )

    lines.extend(["", "## 판정", ""])
    dg = point_rows[3]  # down_gate_0.35
    us = point_rows[4]  # up_sso_0.30
    dr = point_rows[5]  # dual_regime
    cur_row = point_rows[1]
    lines.append(
        f"- **동반하락 현금게이트 심화(35%까지 축소)**: MDD {_fmt_pct(dg['max_drawdown'])} vs current "
        f"{_fmt_pct(cur_row['max_drawdown'])} — "
        + ("실제로 얕아짐" if dg["max_drawdown"] > cur_row["max_drawdown"] else "거의 개선 없음(신호 지연 문제)")
        + f", CAGR {_fmt_pct(dg['cagr'])} (기존 대비 {_fmt_pct(dg['cagr']-cur_row['cagr'])}p 변화)"
    )
    lines.append(
        f"- **동반상승 SSO 편입(30%)**: CAGR {_fmt_pct(us['cagr'])} vs current {_fmt_pct(cur_row['cagr'])}, "
        f"MDD {_fmt_pct(us['max_drawdown'])} vs {_fmt_pct(cur_row['max_drawdown'])} — "
        "레버리지 도입분만큼 상승/하락 모두 증폭"
    )
    lines.append(
        f"- **둘 다 결합(dual_regime)**: CAGR {_fmt_pct(dr['cagr'])}, MDD {_fmt_pct(dr['max_drawdown'])}, "
        f"Sharpe {_fmt_num(dr['sharpe'])}"
    )
    lines.extend([
        "",
        "## 파일",
        "",
        f"- JSON: `{out_json.relative_to(ROOT)}`",
    ])

    out_md = ROOT / "reports" / "profile_sizing" / "yoon2_final_fx_dual_regime.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"JSON: {out_json}")
    print(f"MD:   {out_md}")


if __name__ == "__main__":
    main()
