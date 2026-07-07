"""yoon2_final_fx 파생: 선행성 있는 매크로 신호(VIX 텀스트럭처, 신용스프레드) 연구.

이전 dual_regime 실험에서 확인된 문제: 추세확인형(63일 모멘텀+200일선+드로다운
임계값) 게이트는 크래시가 이미 상당히 진행된 뒤에야 켜져서 MDD 방어 효과가
거의 없었다(gate_only/down_gate 계열 모두 MDD 불변). 이 스크립트는

Part A. 과거 22개 드로다운 에피소드마다 기존 지연형 게이트 대비
        VIX 백워데이션(^VIX>^VIX3M)과 신용스프레드(HYG/IEF 상대모멘텀 붕괴)가
        실제로 며칠 더 일찍/늦게 켜지는지 측정한다(리드타임 진단).
Part B. 리드타임이 있는 신호로 만든 새 현금 게이트를 실제 오버레이 백테스트에
        넣어 MDD가 이번에는 개선되는지 확인한다.

yoon2_final_fx.json은 건드리지 않는다.

Usage:
    PYTHONPATH=src:scripts python3 scripts/profile_sizing/yoon2_final_fx_leading_signals.py
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

LAGGING_GATE = {
    "lookback": 63, "sma_lookback": 200, "dd_threshold": -0.10, "vol_threshold": 0.24,
}
VIX_GATE = {"floor": 18.0}
CREDIT_GATE = {"lookback": 10, "sma_lookback": 20, "threshold": -0.02}


def _clean_metric(p: dict[str, Any]) -> dict[str, float | None]:
    keys = ["cagr", "max_drawdown", "sharpe", "total_return", "volatility"]
    return {k: (None if p.get(k) is None else float(p[k])) for k in keys}


def market_signals(ctx: Context) -> dict[str, pd.Series]:
    return {
        "us": ctx.spy_krw_nav.reindex(ctx.idx).ffill(),
        "kr": ctx.kodex_nav.reindex(ctx.idx).ffill(),
    }


def lagging_trigger(ctx: Context, sig: dict[str, pd.Series]) -> pd.Series:
    p = LAGGING_GATE
    us, kr = sig["us"], sig["kr"]
    us_ret = us.pct_change(p["lookback"])
    kr_ret = kr.pct_change(p["lookback"])
    us_sma = us.rolling(p["sma_lookback"]).mean()
    kr_sma = kr.rolling(p["sma_lookback"]).mean()
    us_dd = us / us.rolling(p["lookback"]).max() - 1.0
    kr_dd = kr / kr.rolling(p["lookback"]).max() - 1.0
    us_vol = us.pct_change().rolling(p["lookback"]).std() * np.sqrt(252.0)
    kr_vol = kr.pct_change().rolling(p["lookback"]).std() * np.sqrt(252.0)
    return (
        (us_ret < 0.0) & (kr_ret < 0.0) & (us < us_sma) & (kr < kr_sma)
        & (
            (us_dd < p["dd_threshold"]) | (kr_dd < p["dd_threshold"])
            | (np.maximum(us_vol, kr_vol) > p["vol_threshold"])
        )
    )


def vix_trigger(idx: pd.DatetimeIndex) -> pd.Series:
    vix = load_yfinance("^VIX", "1d", "max")["close"].reindex(idx).ffill()
    vix3m = load_yfinance("^VIX3M", "1d", "max")["close"].reindex(idx).ffill()
    backwardation = vix > vix3m
    return (backwardation & (vix > VIX_GATE["floor"])).fillna(False)


def credit_trigger(idx: pd.DatetimeIndex) -> pd.Series:
    hyg = load_yfinance("HYG", "1d", "max")["close"].reindex(idx).ffill()
    ief = load_yfinance("IEF", "1d", "max")["close"].reindex(idx).ffill()
    ratio = (hyg / ief).dropna()
    ratio = ratio.reindex(idx).ffill()
    p = CREDIT_GATE
    mom = ratio.pct_change(p["lookback"])
    below_sma = ratio < ratio.rolling(p["sma_lookback"]).mean()
    return ((mom < p["threshold"]) & below_sma).fillna(False)


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
            trough_t = dd.loc[peak_t:t].idxmin()
            episodes.append({
                "peak": peak_t, "trough": trough_t, "end": t,
                "mdd": float(nav.loc[peak_t:t].min() / nav.loc[peak_t] - 1.0),
            })
    return episodes


def lead_days(trigger: pd.Series, idx: pd.DatetimeIndex, peak: pd.Timestamp, trough: pd.Timestamp, search_before: int = 90) -> int | None:
    """peak 기준 -search_before ~ trough 구간에서 첫 트리거 위치를 bar 오프셋으로."""
    peak_pos = idx.get_loc(peak)
    trough_pos = idx.get_loc(trough)
    lo = max(0, peak_pos - search_before)
    window = trigger.iloc[lo:trough_pos + 1]
    fired = window[window]
    if fired.empty:
        return None
    first_t = fired.index[0]
    return int(idx.get_loc(first_t) - peak_pos)


def main() -> None:
    ctx = load_context()
    idx = ctx.idx
    sig = market_signals(ctx)
    current_nav_w = relmom_weights(ctx, **CURRENT)
    from profile_sizing.yoon2_final_fx_validate import combine_nav
    current_nav = combine_nav(ctx, current_nav_w)

    lag_trig = lagging_trigger(ctx, sig)
    vix_trig = vix_trigger(idx)
    credit_trig = credit_trigger(idx)

    episodes = find_episodes(current_nav, thresh=-0.05)

    rows = []
    for e in episodes:
        rows.append({
            "peak": str(e["peak"].date()),
            "trough": str(e["trough"].date()),
            "mdd": e["mdd"],
            "lag_gate_lead": lead_days(lag_trig, idx, e["peak"], e["trough"]),
            "vix_lead": lead_days(vix_trig, idx, e["peak"], e["trough"]),
            "credit_lead": lead_days(credit_trig, idx, e["peak"], e["trough"]),
        })

    def summarize_lead(key: str) -> dict[str, Any]:
        vals = [r[key] for r in rows if r[key] is not None]
        hit = len(vals)
        return {
            "hit_rate": hit / len(rows) if rows else 0.0,
            "avg_offset": float(np.mean(vals)) if vals else None,
            "median_offset": float(np.median(vals)) if vals else None,
            "leads_before_peak": sum(1 for v in vals if v < 0),
            "n": hit,
        }

    lead_summary = {
        "lagging_gate": summarize_lead("lag_gate_lead"),
        "vix_backwardation": summarize_lead("vix_lead"),
        "credit_spread": summarize_lead("credit_lead"),
    }

    # ---- Part B: backtest overlay using the leading signals ----
    def gate_weights(trigger: pd.Series, off_scale: float) -> pd.DataFrame:
        usd = current_nav_w[ctx.sub_ids[0]].reindex(idx).ffill().fillna(0.5)
        exposure = pd.Series(1.0, index=idx).where(~trigger.shift(1).fillna(False), off_scale)
        out = pd.DataFrame(index=idx, columns=[*ctx.sub_ids, "__CASH__"], dtype=float)
        out[ctx.sub_ids[0]] = usd * exposure
        out[ctx.sub_ids[1]] = (1.0 - usd) * exposure
        out["__CASH__"] = 1.0 - exposure
        return out.fillna(0.0)

    def combine_cash(weights: pd.DataFrame) -> pd.Series:
        tw = weights.reindex(idx).ffill().fillna(0.0)
        cols = [*ctx.sub_ids, "__CASH__"]
        tw = tw[cols]
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

    off_scale = 0.5
    candidates = {
        "current": current_nav,
        "lagging_gate": combine_cash(gate_weights(lag_trig, off_scale)),
        "vix_gate": combine_cash(gate_weights(vix_trig, off_scale)),
        "credit_gate": combine_cash(gate_weights(credit_trig, off_scale)),
        "vix_or_credit_gate": combine_cash(gate_weights(vix_trig | credit_trig, off_scale)),
    }

    point_rows = []
    for label, nav in candidates.items():
        m = _clean_metric(perf_nav(nav))
        point_rows.append({"label": label, **m})

    boot = {}
    for label in ["lagging_gate", "vix_gate", "credit_gate", "vix_or_credit_gate"]:
        boot[label] = paired_bootstrap(
            candidates[label], candidates["current"], label_a=label, label_b="current",
            iters=5000, block=20, seed=hash(label) % 1000 + 3,
        )

    # per-episode MDD comparison
    episode_compare = []
    for e in episodes:
        peak, trough = e["peak"], e["trough"]
        row: dict[str, Any] = {"peak": str(peak.date()), "trough": str(trough.date()), "current_mdd": e["mdd"]}
        for label in ["lagging_gate", "vix_gate", "credit_gate", "vix_or_credit_gate"]:
            nv = candidates[label]
            row[f"{label}_mdd"] = float(nv.loc[peak:trough].min() / nv.loc[peak] - 1.0)
        episode_compare.append(row)

    result = {
        "period": f"{idx[0].date()}~{idx[-1].date()}",
        "bars": len(idx),
        "n_episodes": len(episodes),
        "lead_summary": lead_summary,
        "episode_leads": rows,
        "point_estimates": point_rows,
        "bootstrap_vs_current": boot,
        "episode_mdd_compare": episode_compare,
    }
    out_json = ROOT / "reports" / "profile_sizing" / "yoon2_final_fx_leading_signals.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    # ---- markdown ----
    lines = [
        "# yoon2_final_fx 선행 신호 연구: VIX 텀스트럭처 vs 신용스프레드",
        "",
        f"- Period: `{result['period']}` ({result['bars']} bars), 드로다운 에피소드(현재 40/60 기준 -5% 이상): `{result['n_episodes']}`개",
        "- yoon2_final_fx.json은 동결 유지, 오버레이 연구용.",
        f"- 기존 지연형 게이트: {LAGGING_GATE}",
        f"- VIX 게이트: 백워데이션(VIX>VIX3M) & VIX>{VIX_GATE['floor']}",
        f"- 신용 게이트: HYG/IEF 비율 {CREDIT_GATE['lookback']}일 모멘텀 < {CREDIT_GATE['threshold']*100:.0f}% & {CREDIT_GATE['sma_lookback']}일선 아래",
        "",
        "## Part A. 리드타임 진단 (에피소드 peak 대비 트리거 오프셋, 거래일 기준. 음수=peak 이전에 선행 발동)",
        "",
        "| 신호 | 적중(트리거 있음) 에피소드 | peak 이전 선행 발동 수 | 평균 오프셋 | 중앙값 오프셋 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label, key in [("기존 지연형 게이트", "lagging_gate"), ("VIX 백워데이션", "vix_backwardation"), ("신용스프레드", "credit_spread")]:
        s = lead_summary[key]
        lines.append(
            f"| {label} | {s['n']}/{len(episodes)} | {s['leads_before_peak']} | "
            f"{s['avg_offset']:.1f}일 | {s['median_offset']:.1f}일 |" if s["n"] else
            f"| {label} | 0/{len(episodes)} | 0 | - | - |"
        )

    lines.extend([
        "",
        "### 에피소드별 상세 (거래일 오프셋, 음수=peak 전 선행, 빈칸=미발동)",
        "",
        "| peak | trough | MDD | 지연게이트 | VIX | 신용스프레드 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ])
    for r in rows:
        def fmt_off(v: int | None) -> str:
            return "-" if v is None else f"{v:+d}"
        lines.append(
            f"| {r['peak']} | {r['trough']} | {_fmt_pct(r['mdd'])} | "
            f"{fmt_off(r['lag_gate_lead'])} | {fmt_off(r['vix_lead'])} | {fmt_off(r['credit_lead'])} |"
        )

    lines.extend([
        "",
        "## Part B. 오버레이 백테스트 (트리거 시 top-level 노출 0.5배로 축소)",
        "",
        "| label | CAGR | MDD | Sharpe |",
        "| --- | ---: | ---: | ---: |",
    ])
    for r in point_rows:
        lines.append(f"| {r['label']} | {_fmt_pct(r['cagr'])} | {_fmt_pct(r['max_drawdown'])} | {_fmt_num(r['sharpe'])} |")

    lines.extend(["", "## Paired bootstrap vs current (5000 iters, block=20)", ""])
    for label, b in boot.items():
        lines.append(
            f"- `{label}` vs current: P(Sharpe>) `{b['prob']['sharpe_a_gt_b']*100:.1f}%`, "
            f"P(CAGR>) `{b['prob']['cagr_a_gt_b']*100:.1f}%`, "
            f"P(MDD shallower) `{b['prob']['mdd_a_shallower']*100:.1f}%`"
        )

    lines.extend([
        "",
        "## 에피소드별 MDD 비교",
        "",
        "| peak | trough | current | 지연게이트 | VIX게이트 | 신용게이트 | VIX or 신용 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for r in episode_compare:
        lines.append(
            f"| {r['peak']} | {r['trough']} | {_fmt_pct(r['current_mdd'])} | "
            f"{_fmt_pct(r['lagging_gate_mdd'])} | {_fmt_pct(r['vix_gate_mdd'])} | "
            f"{_fmt_pct(r['credit_gate_mdd'])} | {_fmt_pct(r['vix_or_credit_gate_mdd'])} |"
        )

    out_md = ROOT / "reports" / "profile_sizing" / "yoon2_final_fx_leading_signals.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"JSON: {out_json}")
    print(f"MD:   {out_md}")


if __name__ == "__main__":
    main()
