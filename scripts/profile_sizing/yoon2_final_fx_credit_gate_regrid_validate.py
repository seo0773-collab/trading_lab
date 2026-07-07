"""off_scale 재그리드 1위 후보(L10_S30_T-0.01_O0.3)의 워크포워드/부트스트랩 검증.

yoon2_final_fx_credit_gate_v2.py Part 1에서 min_consecutive=2 고정 상태로
108개 조합을 스윕했을 때 풀샘플 1위였던 후보(lookback=10, sma_lookback=30,
threshold=-0.01, off_scale=0.3)를, 지금 채택된 기준선(v2_safe_ief: lookback=10,
sma_lookback=20, threshold=-0.02, off_scale=0.5, safe_asset=IEF)과 동일하게
safe_asset=IEF를 포함시켜 실제 Yoon1kHandler로 재현하고, 워크포워드/부트스트랩/
롤링 3년/드로다운 에피소드까지 검증한다.

Usage:
    PYTHONPATH=src:scripts python3 scripts/profile_sizing/yoon2_final_fx_credit_gate_regrid_validate.py
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

from profile_sizing.engine import performance  # noqa: E402
from profile_sizing.yoon2_final_fx_validate import paired_bootstrap  # noqa: E402
from trading_lab.strategies.yoon1k import Yoon1kHandler  # noqa: E402

CANDIDATE = {"lookback": 10, "sma_lookback": 30, "threshold": -0.01, "off_scale": 0.3}


def _clean_metric(p: dict[str, Any]) -> dict[str, float | None]:
    keys = ["cagr", "max_drawdown", "sharpe", "total_return", "volatility"]
    return {k: (None if p.get(k) is None else float(p[k])) for k in keys}


def _fmt_pct(v: float | None) -> str:
    return "-" if v is None else f"{v * 100:.2f}%"


def _fmt_num(v: float | None) -> str:
    return "-" if v is None else f"{v:.3f}"


def perf_nav(nav: pd.Series, window: pd.DatetimeIndex | None = None) -> dict[str, Any]:
    if window is not None:
        nav = nav.reindex(window).ffill().dropna()
    if len(nav) < 60:
        return {}
    eq = nav / nav.iloc[0]
    ret = eq.pct_change().fillna(0.0)
    return performance(eq, ret, "1d")


def build_equity(base_config: dict[str, Any], gate_overrides: dict[str, Any]) -> pd.Series:
    config = copy.deepcopy(base_config)
    config["currency_allocation"]["risk_gate"].update(gate_overrides)
    handler = Yoon1kHandler()
    raw = handler.load_data("PORTFOLIO", config, synthetic=False)
    art = handler.build_artifacts(raw, config, symbol="PORTFOLIO", phase="all", bars_per_year=252)
    return art.equity


def years(idx: pd.DatetimeIndex) -> list[int]:
    return sorted({int(d.year) for d in idx})


def fixed_walk_forward(idx: pd.DatetimeIndex, navs: dict[str, pd.Series], label: str, base: str) -> dict[str, Any]:
    rows = []
    for start in range(years(idx)[0], years(idx)[-1] - 5 + 1):
        test = idx[idx.year == start + 5]
        if len(test) < 60:
            continue
        rows.append({
            "test": str(start + 5),
            "cand": _clean_metric(perf_nav(navs[label], test)),
            "base": _clean_metric(perf_nav(navs[base], test)),
        })

    def avg(key: str, m: str) -> float | None:
        vals = [r[key].get(m) for r in rows if r[key].get(m) is not None]
        return float(np.mean(vals)) if vals else None

    return {
        "folds": len(rows),
        "avg_cand": {m: avg("cand", m) for m in ["cagr", "max_drawdown", "sharpe"]},
        "avg_base": {m: avg("base", m) for m in ["cagr", "max_drawdown", "sharpe"]},
        "sharpe_wins": sum((r["cand"].get("sharpe") or -999) > (r["base"].get("sharpe") or -999) for r in rows),
        "mdd_wins": sum((r["cand"].get("max_drawdown") or -999) > (r["base"].get("max_drawdown") or -999) for r in rows),
        "cagr_wins": sum((r["cand"].get("cagr") or -999) > (r["base"].get("cagr") or -999) for r in rows),
    }


def rolling_3y(idx: pd.DatetimeIndex, navs: dict[str, pd.Series], label: str, base: str) -> dict[str, Any]:
    rows = []
    for start in range(years(idx)[0], years(idx)[-1] - 2 + 1):
        window = idx[(idx.year >= start) & (idx.year <= start + 2)]
        if len(window) < 500:
            continue
        rows.append({
            "cand": _clean_metric(perf_nav(navs[label], window)),
            "base": _clean_metric(perf_nav(navs[base], window)),
        })
    return {
        "n": len(rows),
        "sharpe_wins": sum((r["cand"].get("sharpe") or -999) > (r["base"].get("sharpe") or -999) for r in rows),
        "mdd_wins": sum((r["cand"].get("max_drawdown") or -999) > (r["base"].get("max_drawdown") or -999) for r in rows),
        "cagr_wins": sum((r["cand"].get("cagr") or -999) > (r["base"].get("cagr") or -999) for r in rows),
    }


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
    base_config = json.loads((ROOT / "configs" / "strategies" / "yoon2_final_fx_credit_gate.json").read_text("utf-8"))
    current_config = json.loads((ROOT / "configs" / "strategies" / "yoon2_final_fx.json").read_text("utf-8"))

    print("Building equity curves (adopted baseline, candidate, current)...")
    eq_adopted = build_equity(base_config, {})  # 지금 채택된 v2_safe_ief 그대로
    eq_candidate = build_equity(base_config, CANDIDATE)  # off_scale 재그리드 1위 + safe_asset=IEF
    handler = Yoon1kHandler()
    raw_current = handler.load_data("PORTFOLIO", current_config, synthetic=False)
    art_current = handler.build_artifacts(raw_current, current_config, symbol="PORTFOLIO", phase="all", bars_per_year=252)
    eq_current = art_current.equity

    idx = eq_adopted.index.intersection(eq_candidate.index).intersection(eq_current.index).sort_values()
    navs = {
        "adopted": (eq_adopted.reindex(idx) / eq_adopted.reindex(idx).iloc[0]),
        "candidate": (eq_candidate.reindex(idx) / eq_candidate.reindex(idx).iloc[0]),
        "current": (eq_current.reindex(idx) / eq_current.reindex(idx).iloc[0]),
    }

    point = {label: _clean_metric(perf_nav(nav)) for label, nav in navs.items()}
    print("Point estimates:")
    for label, m in point.items():
        print(f"  {label}: CAGR {m['cagr']*100:.2f}% MDD {m['max_drawdown']*100:.2f}% Sharpe {m['sharpe']:.3f}")

    boot_vs_adopted = paired_bootstrap(
        navs["candidate"], navs["adopted"], label_a="candidate", label_b="adopted", iters=8000, block=20, seed=61,
    )
    boot_vs_current = paired_bootstrap(
        navs["candidate"], navs["current"], label_a="candidate", label_b="current", iters=8000, block=20, seed=67,
    )

    wf_vs_adopted = fixed_walk_forward(idx, navs, "candidate", "adopted")
    roll_vs_adopted = rolling_3y(idx, navs, "candidate", "adopted")

    episodes = find_episodes(navs["current"], thresh=-0.05)
    episode_rows = []
    for e in episodes:
        peak, trough = e["peak"], e["trough"]
        episode_rows.append({
            "peak": str(peak.date()), "trough": str(trough.date()),
            "current_mdd": float(navs["current"].loc[peak:trough].min() / navs["current"].loc[peak] - 1.0),
            "adopted_mdd": float(navs["adopted"].loc[peak:trough].min() / navs["adopted"].loc[peak] - 1.0),
            "candidate_mdd": float(navs["candidate"].loc[peak:trough].min() / navs["candidate"].loc[peak] - 1.0),
        })

    result = {
        "period": f"{idx[0].date()}~{idx[-1].date()}",
        "bars": len(idx),
        "candidate_params": CANDIDATE,
        "point_estimates": point,
        "bootstrap_vs_adopted": boot_vs_adopted,
        "bootstrap_vs_current": boot_vs_current,
        "walk_forward_vs_adopted": wf_vs_adopted,
        "rolling_3y_vs_adopted": roll_vs_adopted,
        "drawdown_episodes": episode_rows,
    }
    out_json = ROOT / "reports" / "profile_sizing" / "yoon2_final_fx_credit_gate_regrid_validate.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    lines = [
        "# off_scale 재그리드 1위 후보 워크포워드/부트스트랩 검증",
        "",
        f"- Period: `{result['period']}` ({result['bars']} bars)",
        f"- 후보 파라미터(safe_asset=IEF 포함): `{CANDIDATE}`",
        f"- 채택된 기준선(adopted): lookback=10, sma_lookback=20, threshold=-2%, off_scale=0.5, safe_asset=IEF",
        "",
        "## Point estimates",
        "",
        "| label | CAGR | MDD | Sharpe |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, m in point.items():
        lines.append(f"| {label} | {_fmt_pct(m['cagr'])} | {_fmt_pct(m['max_drawdown'])} | {_fmt_num(m['sharpe'])} |")

    lines.extend([
        "",
        "## 페어드 부트스트랩 (8000회, block=20)",
        "",
        f"- candidate vs adopted: P(Sharpe>) `{boot_vs_adopted['prob']['sharpe_a_gt_b']*100:.1f}%`, "
        f"P(CAGR>) `{boot_vs_adopted['prob']['cagr_a_gt_b']*100:.1f}%`, "
        f"P(MDD shallower) `{boot_vs_adopted['prob']['mdd_a_shallower']*100:.1f}%`",
        f"- candidate vs current: P(Sharpe>) `{boot_vs_current['prob']['sharpe_a_gt_b']*100:.1f}%`, "
        f"P(CAGR>) `{boot_vs_current['prob']['cagr_a_gt_b']*100:.1f}%`, "
        f"P(MDD shallower) `{boot_vs_current['prob']['mdd_a_shallower']*100:.1f}%`",
        "",
        "## 고정 파라미터 워크포워드 (vs adopted, 재선택 없음)",
        "",
        f"- Folds: `{wf_vs_adopted['folds']}`",
        f"- Avg candidate: CAGR `{_fmt_pct(wf_vs_adopted['avg_cand']['cagr'])}`, "
        f"MDD `{_fmt_pct(wf_vs_adopted['avg_cand']['max_drawdown'])}`, "
        f"Sharpe `{_fmt_num(wf_vs_adopted['avg_cand']['sharpe'])}`",
        f"- Avg adopted: CAGR `{_fmt_pct(wf_vs_adopted['avg_base']['cagr'])}`, "
        f"MDD `{_fmt_pct(wf_vs_adopted['avg_base']['max_drawdown'])}`, "
        f"Sharpe `{_fmt_num(wf_vs_adopted['avg_base']['sharpe'])}`",
        f"- Sharpe 승: `{wf_vs_adopted['sharpe_wins']}/{wf_vs_adopted['folds']}`",
        f"- MDD 승: `{wf_vs_adopted['mdd_wins']}/{wf_vs_adopted['folds']}`",
        f"- CAGR 승: `{wf_vs_adopted['cagr_wins']}/{wf_vs_adopted['folds']}`",
        "",
        "## 롤링 3년 (vs adopted)",
        "",
        f"- Sharpe 승: `{roll_vs_adopted['sharpe_wins']}/{roll_vs_adopted['n']}`",
        f"- MDD 승: `{roll_vs_adopted['mdd_wins']}/{roll_vs_adopted['n']}`",
        f"- CAGR 승: `{roll_vs_adopted['cagr_wins']}/{roll_vs_adopted['n']}`",
        "",
        "## 드로다운 에피소드 (current 기준 -5% 이상): adopted vs candidate",
        "",
        "| peak | trough | current | adopted | candidate |",
        "| --- | --- | ---: | ---: | ---: |",
    ])
    for r in episode_rows:
        lines.append(
            f"| {r['peak']} | {r['trough']} | {_fmt_pct(r['current_mdd'])} | "
            f"{_fmt_pct(r['adopted_mdd'])} | {_fmt_pct(r['candidate_mdd'])} |"
        )

    lines.extend(["", "## 판정", ""])
    folds = wf_vs_adopted["folds"]
    if wf_vs_adopted["sharpe_wins"] >= folds * 0.6 and wf_vs_adopted["mdd_wins"] >= folds * 0.5:
        verdict = "워크포워드/롤링에서도 일관되게 채택 기준선을 능가한다 — 채택 검토."
    else:
        verdict = (
            "풀샘플에서는 앞섰지만 워크포워드/롤링 3년에서는 승률이 낮아, 특정 구간에 과적합된 "
            "결과로 보인다. 현재 채택된 파라미터(lookback=10/sma=20/threshold=-2%/off_scale=0.5)를 유지한다."
        )
    lines.append(verdict)
    lines.extend(["", "## 파일", "", f"- JSON: `{out_json.relative_to(ROOT)}`"])

    out_md = ROOT / "reports" / "profile_sizing" / "yoon2_final_fx_credit_gate_regrid_validate.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"JSON: {out_json}")
    print(f"MD:   {out_md}")


if __name__ == "__main__":
    main()
