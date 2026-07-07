"""off_scale 재그리드 후보의 발동빈도/회전율/비용 민감도 확인.

production Yoon1kHandler._combine_values_dynamic는 월간 리밸런스 시점에만
실제로 비중을 재조정한다(게이트 신호 자체는 매일 계산되지만, 그 신호가 반영되는
건 다음 월간 리밸런스 때 그 시점의 게이트 상태로 한 달치가 결정되는 구조). 즉
게이트가 더 자주/느슨하게(threshold -1%, off_scale 0.3) 발동해도 실제 거래
횟수 자체는 늘지 않고(모두 월간 스케줄을 따름), "그 달의 노출 수준"만 더
자주 낮은 값으로 잡힐 뿐이다. 그래도 top-level 리밸런스 시 실제로 옮겨지는
금액(회전율)은 커질 수 있으므로 비용 민감도를 확인한다.

Usage:
    PYTHONPATH=src:scripts python3 scripts/profile_sizing/yoon2_final_fx_credit_gate_regrid_cost.py
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
    perf_nav,
    relmom_weights,
)

ADOPTED = {"lookback": 10, "sma_lookback": 20, "threshold": -0.02, "off_scale": 0.5}
CANDIDATE = {"lookback": 10, "sma_lookback": 30, "threshold": -0.01, "off_scale": 0.3}
MIN_CONSECUTIVE = 2
COST_BPS = [0, 5, 10, 25, 50]


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


def combine_with_ief_and_cost(
    ctx: Context, weights: pd.DataFrame, ief_nav: pd.Series, cost_bps: float,
) -> pd.Series:
    """월간 리밸런스에만 실제 거래가 일어나는 production 구조를 그대로 재현.

    __CASH__ 슬롯은 문자 그대로 현금이 아니라 IEF NAV로 굴린다(채택된 safe_asset
    로직과 동일). cost_bps는 월간 리밸런스 시점의 전체 자산군 회전(절대값 합)에
    편도로 적용하는 top-level 비용 민감도 실험치다(production 자체는 이 비용을
    모델링하지 않는다).
    """
    idx = ctx.idx
    tw = weights.reindex(idx).ffill().fillna(0.0)
    cols = [*ctx.sub_ids, "__CASH__"]
    tw = tw[cols]
    tw = tw.div(tw.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    rets = [n.reindex(idx).ffill().pct_change().fillna(0.0).to_numpy() for n in ctx.navs]
    rets.append(ief_nav.reindex(idx).ffill().pct_change().fillna(0.0).to_numpy())
    cost_rate = float(cost_bps) / 10_000.0
    reb = set(rebalance_dates(idx, str(ctx.config.get("rebalance_freq", "monthly"))))
    vals = tw.iloc[0].to_numpy(dtype=float).tolist()
    out = np.empty(len(idx))
    turnover_sum = 0.0
    for i, t in enumerate(idx):
        for j, r in enumerate(rets):
            vals[j] *= 1.0 + r[i]
        total_before = sum(vals)
        if t in reb:
            target = tw.iloc[i].to_numpy(dtype=float)
            target_vals = total_before * target
            traded = float(np.abs(target_vals - np.array(vals)).sum())
            turnover_sum += traded / max(total_before, 1e-9)
            if cost_rate > 0.0:
                total_after = max(total_before - traded * cost_rate, 0.0)
                target_vals = total_after * target
            vals = target_vals.tolist()
            out[i] = sum(vals)
        else:
            out[i] = total_before
    nav = pd.Series(out, index=idx, name="equity")
    return nav / nav.iloc[0], turnover_sum


def main() -> None:
    ctx = load_context()
    idx = ctx.idx
    ratio = load_ratio(idx)
    current_w = relmom_weights(ctx, **CURRENT)
    usd_series = current_w[ctx.sub_ids[0]].reindex(idx).ffill().fillna(0.5)
    ief_px = load_yfinance("IEF", "1d", "max")["close"].reindex(idx).ffill()
    ief_nav = (ief_px / ief_px.dropna().iloc[0]).rename("IEF")

    rows = []
    turnovers = {}
    trig_stats = {}
    for label, params in [("adopted", ADOPTED), ("candidate", CANDIDATE)]:
        trig = denoise(raw_trigger(ratio, params), MIN_CONSECUTIVE)
        trig_stats[label] = {
            "trigger_days": int(trig.sum()),
            "trigger_pct": float(trig.mean()),
            "episodes": episode_count(trig),
        }
        w = gate_weights(ctx, usd_series, trig, params["off_scale"])
        for bps in COST_BPS:
            nav, turnover_sum = combine_with_ief_and_cost(ctx, w, ief_nav, bps)
            m = _clean_metric(perf_nav(nav))
            rows.append({"label": label, "cost_bps": bps, **m})
            if bps == 0:
                turnovers[label] = turnover_sum / (len(idx) / 252.0)

    result = {
        "trigger_stats": trig_stats,
        "turnover_per_year": turnovers,
        "cost_sensitivity": rows,
    }
    out_json = ROOT / "reports" / "profile_sizing" / "yoon2_final_fx_credit_gate_regrid_cost.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    lines = [
        "# off_scale 재그리드 후보 발동빈도/회전율/비용 민감도",
        "",
        "## 발동 빈도 (min_consecutive=2 적용 후)",
        "",
        "| | 발동일수 | 발동비율 | 별도 에피소드 수 | 연환산 회전율 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label in ["adopted", "candidate"]:
        s = trig_stats[label]
        lines.append(
            f"| {label} | {s['trigger_days']} | {s['trigger_pct']*100:.1f}% | {s['episodes']} | "
            f"{turnovers[label]:.2f}x |"
        )

    lines.extend([
        "",
        "## 비용 민감도 (top-level 편도 비용, IEF 대피 포함)",
        "",
        "| label | cost_bps | CAGR | MDD | Sharpe |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for r in rows:
        lines.append(f"| {r['label']} | {r['cost_bps']} | {_fmt_pct(r['cagr'])} | {_fmt_pct(r['max_drawdown'])} | {_fmt_num(r['sharpe'])} |")

    out_md = ROOT / "reports" / "profile_sizing" / "yoon2_final_fx_credit_gate_regrid_cost.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"JSON: {out_json}")
    print(f"MD:   {out_md}")


if __name__ == "__main__":
    main()
