"""final0706 위험게이트: 기존 신용스프레드(HYG/IEF) 외에 유망한 대체/보완 신호 탐색.

배경: final0706의 신용스프레드 게이트(HYG/IEF 상대모멘텀 붕괴, 2일 연속 확인)는
`yoon2_final_fx_leading_signals.py`에서 VIX 백워데이션·추세확인형 게이트와 이미
비교돼 채택됐다(리드타임 중앙값 -48일, MDD 개선 -17.13→-16.40%, 부트스트랩
P(Sharpe>) 70.5%). 그 연구는 후보를 VIX/추세형 2개만 다뤘다 — 이 스크립트는
같은 방법론(22개 드로다운 에피소드 리드타임 진단, `yoon2_final_fx_leading_signals.py`
재사용)으로 **아직 검토되지 않은** 후보 신호들을 추가로 실측한다:

  - credit_hyg_lqd : HYG/LQD(둘 다 회사채, 신용등급만 다름 — IEF는 국채라 HYG/IEF는
    신용스프레드에 듀레이션 차이(HYG~4y vs IEF~7y)가 섞인다. LQD(~8~9y)로 바꾸면
    "순수 신용등급 스프레드"에 더 가까워진다는 가설.
  - small_cap_relative : IWM/SPY 상대강도 붕괴 — 소형주가 대형주보다 먼저
    꺾이는 현상은 표준적인 리스크오프 조기경보(시장 폭 축소) 신호.
  - dollar_strength : UUP(달러인덱스 ETF) 추세 상향 — 안전자산 선호 시 달러 강세.
  - copper_gold : 구리/금 비율("Dr. Copper") 하락 — 경기민감 vs 안전자산 심리.

기존 config(final0706.json/yoon2_final_fx.json)는 건드리지 않는다 — 순수 리서치.

Usage:
    PYTHONPATH=src:scripts python3 scripts/profile_sizing/final0706_gate_new_signal_search.py
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
)
from profile_sizing.yoon2_final_fx_leading_signals import (  # noqa: E402
    credit_trigger,
    find_episodes,
    lead_days,
)

CREDIT_GATE = {"lookback": 10, "sma_lookback": 20, "threshold": -0.02}


def _ratio_trigger(sym_a: str, sym_b: str, idx: pd.DatetimeIndex, params: dict[str, Any]) -> pd.Series:
    a = load_yfinance(sym_a, "1d", "max")["close"].reindex(idx).ffill()
    b = load_yfinance(sym_b, "1d", "max")["close"].reindex(idx).ffill()
    ratio = (a / b).dropna().reindex(idx).ffill()
    mom = ratio.pct_change(params["lookback"])
    below_sma = ratio < ratio.rolling(params["sma_lookback"]).mean()
    return ((mom < params["threshold"]) & below_sma).fillna(False)


def small_cap_trigger(idx: pd.DatetimeIndex) -> pd.Series:
    return _ratio_trigger("IWM", "SPY", idx, CREDIT_GATE)


def credit_lqd_trigger(idx: pd.DatetimeIndex) -> pd.Series:
    return _ratio_trigger("HYG", "LQD", idx, CREDIT_GATE)


def copper_gold_trigger(idx: pd.DatetimeIndex) -> pd.Series:
    return _ratio_trigger("CPER", "GLD", idx, CREDIT_GATE)


def dollar_strength_trigger(idx: pd.DatetimeIndex) -> pd.Series:
    uup = load_yfinance("UUP", "1d", "max")["close"].reindex(idx).ffill()
    p = CREDIT_GATE
    mom = uup.pct_change(p["lookback"])
    above_sma = uup > uup.rolling(p["sma_lookback"]).mean()
    return ((mom > -p["threshold"]) & above_sma).fillna(False)


def main() -> None:
    ctx = load_context()
    idx = ctx.idx
    current_nav_w = relmom_weights(ctx, **CURRENT)
    current_nav = combine_nav(ctx, current_nav_w)
    episodes = find_episodes(current_nav, thresh=-0.05)

    print(f"기간 {idx[0].date()}~{idx[-1].date()}, 드로다운 에피소드 {len(episodes)}개", flush=True)

    triggers: dict[str, pd.Series] = {}
    print("credit_spread(HYG/IEF, 기존 채택) 로드...", flush=True)
    triggers["credit_spread_hyg_ief"] = credit_trigger(idx)
    print("credit_hyg_lqd(HYG/LQD, 순수신용) 로드...", flush=True)
    triggers["credit_hyg_lqd"] = credit_lqd_trigger(idx)
    print("small_cap_relative(IWM/SPY) 로드...", flush=True)
    triggers["small_cap_relative"] = small_cap_trigger(idx)
    print("dollar_strength(UUP) 로드...", flush=True)
    triggers["dollar_strength"] = dollar_strength_trigger(idx)
    print("copper_gold(CPER/GLD) 로드...", flush=True)
    try:
        triggers["copper_gold"] = copper_gold_trigger(idx)
    except Exception as exc:  # noqa: BLE001
        print(f"  copper_gold 스킵({exc})", flush=True)

    def summarize(trig: pd.Series) -> dict[str, Any]:
        rows = []
        for e in episodes:
            rows.append(lead_days(trig, idx, e["peak"], e["trough"]))
        vals = [v for v in rows if v is not None]
        hit = len(vals)
        return {
            "hit_rate": hit / len(episodes) if episodes else 0.0,
            "avg_offset": float(np.mean(vals)) if vals else None,
            "median_offset": float(np.median(vals)) if vals else None,
            "leads_before_peak": sum(1 for v in vals if v < 0),
            "n": hit,
            "per_episode": rows,
        }

    lead_summary = {label: summarize(trig) for label, trig in triggers.items()}

    print("\n[Part A. 리드타임 진단]", flush=True)
    for label, s in lead_summary.items():
        print(f"  {label}: {s['n']}/{len(episodes)}적중, 선행{s['leads_before_peak']}건, "
              f"평균{_fmt_num(s['avg_offset'])}일 중앙값{_fmt_num(s['median_offset'])}일", flush=True)

    # ---- Part B: 오버레이 백테스트 (트리거 시 top-level 노출 0.5배) ----
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
    candidates: dict[str, pd.Series] = {"current": current_nav}
    for label, trig in triggers.items():
        candidates[label] = combine_cash(gate_weights(trig, off_scale))
    # 신용스프레드가 놓친 에피소드를 보완하는 OR 결합도 같이 실측
    candidates["credit_or_small_cap"] = combine_cash(
        gate_weights(triggers["credit_spread_hyg_ief"] | triggers["small_cap_relative"], off_scale)
    )

    def _clean(p: dict[str, Any]) -> dict[str, float | None]:
        keys = ["cagr", "max_drawdown", "sharpe", "total_return"]
        return {k: (None if p.get(k) is None else float(p[k])) for k in keys}

    point_rows = []
    for label, nav in candidates.items():
        m = _clean(perf_nav(nav))
        point_rows.append({"label": label, **m})

    print("\n[Part B. 오버레이 백테스트]", flush=True)
    for r in point_rows:
        print(f"  {r['label']}: CAGR {_fmt_pct(r['cagr'])} MDD {_fmt_pct(r['max_drawdown'])} "
              f"Sharpe {_fmt_num(r['sharpe'])}", flush=True)

    boot = {}
    for label in candidates:
        if label == "current":
            continue
        boot[label] = paired_bootstrap(
            candidates[label], candidates["current"], label_a=label, label_b="current",
            iters=5000, block=20, seed=hash(label) % 1000 + 3,
        )

    print("\n[페어드 부트스트랩 vs current, 5000 iters block=20]", flush=True)
    for label, b in boot.items():
        print(f"  {label}: P(Sharpe>) {b['prob']['sharpe_a_gt_b']*100:.1f}% "
              f"P(CAGR>) {b['prob']['cagr_a_gt_b']*100:.1f}% "
              f"P(MDD얕음) {b['prob']['mdd_a_shallower']*100:.1f}%", flush=True)

    result = {
        "period": f"{idx[0].date()}~{idx[-1].date()}",
        "n_episodes": len(episodes),
        "episodes": [{"peak": str(e["peak"].date()), "trough": str(e["trough"].date()), "mdd": e["mdd"]} for e in episodes],
        "lead_summary": {k: {kk: vv for kk, vv in v.items() if kk != "per_episode"} for k, v in lead_summary.items()},
        "point_estimates": point_rows,
        "bootstrap_vs_current": boot,
    }
    out_json = ROOT / "reports" / "profile_sizing" / "final0706_gate_new_signal_search.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"\nJSON: {out_json}")


if __name__ == "__main__":
    main()
