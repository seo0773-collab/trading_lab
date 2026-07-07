"""yoon2_final_fx_credit_gate 2차 개선 검증: IEF 대피, VIX AND결합, 연속형 노출,
off_scale 재그리드(min_consecutive=2 고정).

등록된 `yoon2_final_fx_credit_gate`(min_consecutive=2)를 기준선으로 두고
Yoon1kHandler에 새로 추가한 기능들을 검증한다:

1. `risk_gate.safe_asset="IEF"` — 게이트 발동 시 현금 대신 IEF NAV로 대피
2. `risk_gate.source="credit_and_vix"` — 신용스프레드 AND VIX 백워데이션 동시 확인
3. `risk_gate.continuous` — 이진 off_scale 대신 신호 강도 비례 연속 노출 축소
4. off_scale 재그리드(lookback/sma/threshold/off_scale, min_consecutive=2 고정)

Part 1(재그리드)은 가벼운 로컬 재현(sleeve nav 재사용)으로 빠르게, Part 2(신규
기능 3종)는 실제 Yoon1kHandler.build_artifacts를 그대로 호출해 프로덕션 코드와
드리프트 없이 검증한다.

Usage:
    PYTHONPATH=src:scripts python3 scripts/profile_sizing/yoon2_final_fx_credit_gate_v2.py
"""
from __future__ import annotations

import copy
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
from profile_sizing.engine import performance  # noqa: E402
from profile_sizing.portfolio import rebalance_dates  # noqa: E402
from profile_sizing.yoon2_final_fx_validate import (  # noqa: E402
    CURRENT,
    Context,
    _fmt_num,
    _fmt_pct,
    combine_nav,
    load_context,
    paired_bootstrap,
    relmom_weights,
    weight_frame,
)
from trading_lab.strategies.yoon1k import Yoon1kHandler  # noqa: E402

V1_BASE = {"lookback": 10, "sma_lookback": 20, "threshold": -0.02, "off_scale": 0.5, "min_consecutive": 2}
LOOKBACKS = [5, 10, 15, 20]
SMA_LOOKBACKS = [10, 20, 30]
THRESHOLDS = [-0.01, -0.02, -0.03]
OFF_SCALES = [0.3, 0.5, 0.7]


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
    return performance(eq, ret, "1d")


# ---------------- Part 1: off_scale 재그리드 (min_consecutive=2 고정, 가벼운 로컬 재현) ----------------

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


def gate_weights(ctx: Context, usd_series: pd.Series, trigger: pd.Series, off_scale: float) -> pd.DataFrame:
    idx = ctx.idx
    exposure = pd.Series(1.0, index=idx).where(~trigger.shift(1).fillna(False), off_scale)
    out = pd.DataFrame(index=idx, columns=[*ctx.sub_ids, "__CASH__"], dtype=float)
    out[ctx.sub_ids[0]] = usd_series * exposure
    out[ctx.sub_ids[1]] = (1.0 - usd_series) * exposure
    out["__CASH__"] = 1.0 - exposure
    return out.fillna(0.0)


def combine_cash(ctx: Context, weights: pd.DataFrame) -> pd.Series:
    idx = ctx.idx
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


def part1_regrid(ctx: Context, current_nav: pd.Series) -> list[dict[str, Any]]:
    ratio = load_ratio(ctx.idx)
    current_w = relmom_weights(ctx, **CURRENT)
    usd_series = current_w[ctx.sub_ids[0]].reindex(ctx.idx).ffill().fillna(0.5)
    cur_m = _clean_metric(perf_nav(current_nav))

    rows = []
    for lb in LOOKBACKS:
        for sma in SMA_LOOKBACKS:
            for th in THRESHOLDS:
                for off in OFF_SCALES:
                    p = {"lookback": lb, "sma_lookback": sma, "threshold": th, "off_scale": off}
                    trig = denoise(raw_trigger(ratio, p), min_consecutive=2)
                    w = gate_weights(ctx, usd_series, trig, off)
                    nav = combine_cash(ctx, w)
                    m = _clean_metric(perf_nav(nav))
                    rows.append({
                        "label": f"L{lb}_S{sma}_T{th}_O{off}",
                        **p, **m,
                        "beats_sharpe": (m.get("sharpe") or -999) > (cur_m.get("sharpe") or -999),
                        "beats_mdd": (m.get("max_drawdown") or -999) > (cur_m.get("max_drawdown") or -999),
                    })
    return rows


# ---------------- Part 2: 실제 핸들러 기반 변형 검증 ----------------

def build_variant_metrics(base_config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    config["currency_allocation"]["risk_gate"].update(overrides)
    handler = Yoon1kHandler()
    raw = handler.load_data("PORTFOLIO", config, synthetic=False)
    art = handler.build_artifacts(raw, config, symbol="PORTFOLIO", phase="all", bars_per_year=252)
    m = art.metrics
    return {
        "cagr": m["cagr"], "max_drawdown": m["max_drawdown"], "sharpe": m["sharpe"],
        "equity": art.equity,
    }


def main() -> None:
    ctx = load_context()
    current_nav = combine_nav(ctx, relmom_weights(ctx, **CURRENT))
    static_nav = combine_nav(ctx, weight_frame(ctx, 0.5))
    cur_m = _clean_metric(perf_nav(current_nav))

    base_config = json.loads((ROOT / "configs" / "strategies" / "yoon2_final_fx_credit_gate.json").read_text("utf-8"))

    variants = {
        "v1_baseline (등록됨, min_consec=2)": {},
        "v2_safe_ief": {"safe_asset": "IEF"},
        "v3_credit_and_vix": {
            "source": "credit_and_vix",
            "vix_symbols": ["^VIX", "^VIX3M"],
            "vix_floor": 18.0,
        },
        "v3b_credit_and_vix_safe_ief": {
            "source": "credit_and_vix",
            "vix_symbols": ["^VIX", "^VIX3M"],
            "vix_floor": 18.0,
            "safe_asset": "IEF",
        },
        "v4_continuous_cap1.5": {"continuous": {"enabled": True, "severity_cap": 1.5}},
        "v4_continuous_cap2.0": {"continuous": {"enabled": True, "severity_cap": 2.0}},
        "v4_continuous_cap3.0": {"continuous": {"enabled": True, "severity_cap": 3.0}},
        "v5_continuous_cap2_safe_ief": {
            "continuous": {"enabled": True, "severity_cap": 2.0}, "safe_asset": "IEF",
        },
    }

    results = {}
    for label, overrides in variants.items():
        results[label] = build_variant_metrics(base_config, overrides)
        print(f"{label}: CAGR {results[label]['cagr']*100:.2f}% MDD {results[label]['max_drawdown']*100:.2f}% Sharpe {results[label]['sharpe']:.3f}")

    # bootstrap each variant vs v1_baseline and vs current
    boot = {}
    v1_nav = results["v1_baseline (등록됨, min_consec=2)"]["equity"]
    for label in variants:
        if label == "v1_baseline (등록됨, min_consec=2)":
            continue
        nav = results[label]["equity"]
        boot[label] = {
            "vs_v1": paired_bootstrap(nav, v1_nav, label_a=label, label_b="v1", iters=5000, block=20, seed=hash(label) % 1000 + 51),
            "vs_current": paired_bootstrap(nav, current_nav, label_a=label, label_b="current", iters=5000, block=20, seed=hash(label) % 1000 + 71),
        }

    # ---- Part 1 regrid ----
    print("Running Part 1 regrid (off_scale, min_consecutive=2 fixed)...")
    grid_rows = part1_regrid(ctx, current_nav)
    grid_df = pd.DataFrame(grid_rows)
    both = grid_df[grid_df["beats_sharpe"] & grid_df["beats_mdd"]]
    top_sharpe = grid_df.sort_values("sharpe", ascending=False).head(8)

    result = {
        "current_point": cur_m,
        "variants": {k: {kk: vv for kk, vv in v.items() if kk != "equity"} for k, v in results.items()},
        "bootstrap": boot,
        "regrid_grid_size": len(grid_rows),
        "regrid_beats_both": len(both),
        "regrid_top_sharpe": top_sharpe.drop(columns=["beats_sharpe", "beats_mdd"]).to_dict(orient="records"),
    }
    out_json = ROOT / "reports" / "profile_sizing" / "yoon2_final_fx_credit_gate_v2.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    lines = [
        "# yoon2_final_fx_credit_gate 2차 개선 검증",
        "",
        f"- current(고정 50/50 없는 순수 yoon2_final_fx): CAGR `{_fmt_pct(cur_m['cagr'])}`, "
        f"MDD `{_fmt_pct(cur_m['max_drawdown'])}`, Sharpe `{_fmt_num(cur_m['sharpe'])}`",
        "",
        "## Part 2. 신규 기능 3종 (실제 Yoon1kHandler로 실데이터 검증)",
        "",
        "| variant | CAGR | MDD | Sharpe |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, r in result["variants"].items():
        lines.append(f"| {label} | {_fmt_pct(r['cagr'])} | {_fmt_pct(r['max_drawdown'])} | {_fmt_num(r['sharpe'])} |")

    lines.extend(["", "### 부트스트랩 (5000회, vs v1_baseline / vs current)", ""])
    for label, b in boot.items():
        v1b, curb = b["vs_v1"], b["vs_current"]
        lines.append(
            f"- `{label}` vs v1: P(Sharpe>) `{v1b['prob']['sharpe_a_gt_b']*100:.1f}%`, "
            f"P(CAGR>) `{v1b['prob']['cagr_a_gt_b']*100:.1f}%`, P(MDD shallower) `{v1b['prob']['mdd_a_shallower']*100:.1f}%` "
            f"| vs current: P(Sharpe>) `{curb['prob']['sharpe_a_gt_b']*100:.1f}%`, "
            f"P(CAGR>) `{curb['prob']['cagr_a_gt_b']*100:.1f}%`, P(MDD shallower) `{curb['prob']['mdd_a_shallower']*100:.1f}%`"
        )

    lines.extend([
        "",
        "## Part 1. off_scale 재그리드 (min_consecutive=2 고정, "
        f"{len(LOOKBACKS)}x{len(SMA_LOOKBACKS)}x{len(THRESHOLDS)}x{len(OFF_SCALES)}={result['regrid_grid_size']}조합)",
        "",
        f"- current 대비 Sharpe·MDD 동시 개선 조합: `{result['regrid_beats_both']}/{result['regrid_grid_size']}`",
        "",
        "### Sharpe 상위 8개",
        "",
        "| label | lookback | sma | threshold | off_scale | CAGR | MDD | Sharpe |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for r in result["regrid_top_sharpe"]:
        lines.append(
            f"| {r['label']} | {r['lookback']} | {r['sma_lookback']} | {r['threshold']} | {r['off_scale']} | "
            f"{_fmt_pct(r['cagr'])} | {_fmt_pct(r['max_drawdown'])} | {_fmt_num(r['sharpe'])} |"
        )

    out_md = ROOT / "reports" / "profile_sizing" / "yoon2_final_fx_credit_gate_v2.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"JSON: {out_json}")
    print(f"MD:   {out_md}")


if __name__ == "__main__":
    main()
