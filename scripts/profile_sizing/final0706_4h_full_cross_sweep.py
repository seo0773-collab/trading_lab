"""final0706 US+KR 2조 구조 전체 — TF(1h/4h) × cycle_len × rebalance 크로스 스윕.

final0706_4h.json(cycle=100,rebal=42)의 US 슬리브 단독 그리드서치 최적치를 그대로
2조 구조(US+KR+통화틸트+신용게이트) 전체에 적용했더니 Sharpe=0.713으로, US 단독
결과(1.450)보다 훨씬 낮았다(reports/profile_sizing/final0706_4h_backtest.md).
KR 슬리브를 합치면서 최적점이 달라졌을 수 있으므로, 2조 구조 전체 레벨에서
cycle_len × rebalance를 다시 그리드서치한다.

기준(reference): 그 TF가 덮는 구간을 일봉·무스케일 final0706(원본, US+KR 2조
전체)으로 돈 Sharpe.

Usage:
    PYTHONPATH=src:scripts python3 scripts/profile_sizing/final0706_4h_full_cross_sweep.py
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_lab.strategies.yoon1k import Yoon1kHandler  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TRADING_MIN_PER_DAY = 390.0
TRADING_DAYS_PER_YEAR = 252.0
TFS = [("1h", 60), ("4h", 240)]
CYCLE_GRID = (50, 75, 100, 125, 150, 200)
REBAL_GRID = (16, 21, 26, 31, 42)

SCALE_KEYS = [
    ("profile", "rolling_window", 126),
    ("market_filter", "ma_len", 100),
    ("safe_sleeve", "ma_len", 100),
    ("regime_cap", "recovery_stage_bars", 21),
]
ALLOC_SCALE_KEYS = [
    ("lookback", 63),
    (("risk_gate", "lookback"), 10),
    (("risk_gate", "sma_lookback"), 20),
]


def bars_per_year(bar_minutes: float) -> float:
    return (TRADING_MIN_PER_DAY / bar_minutes) * TRADING_DAYS_PER_YEAR


def perf_from_equity(equity: pd.Series, ann: float) -> dict[str, float | None]:
    eq = equity.dropna()
    if len(eq) < 10:
        return {"sharpe": None, "cagr": None, "max_drawdown": None, "total_return": None}
    ret = eq.pct_change().fillna(0.0).to_numpy()
    sd = np.nanstd(ret, ddof=0)
    sharpe = float(np.nanmean(ret) / sd * np.sqrt(ann)) if sd > 0 else None
    n_years = max(len(eq) / ann, 1e-9)
    cagr = float(eq.iloc[-1] ** (1.0 / n_years) - 1.0) if eq.iloc[-1] > 0 else None
    mdd = float((eq / eq.cummax() - 1.0).min())
    return {"sharpe": sharpe, "cagr": cagr, "max_drawdown": mdd,
           "total_return": float(eq.iloc[-1] - 1.0)}


def scale_sub_overrides(overrides: dict[str, Any], cycle_len: int, rebal_bars: int,
                        interval: str, period: str) -> dict[str, Any]:
    ov = copy.deepcopy(overrides)
    ov["interval"] = interval
    ov["period"] = period
    ov["rebalance_freq"] = f"every_{rebal_bars}"
    ov["base_cycle"] = {"type": "SMA", "length": cycle_len, "scale": 1.0}
    ratio = cycle_len / 200.0
    for group, key, default in SCALE_KEYS:
        base = ov.get(group) or {}
        base = copy.deepcopy(base)
        base[key] = max(2, int(round(default * ratio)))
        ov[group] = base
    return ov


def build_config(base_cfg: dict[str, Any], cycle_len: int, rebal_bars: int, tf_label: str) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    cfg["interval"] = tf_label
    cfg["period"] = "729d"
    cfg["rebalance_freq"] = f"every_{rebal_bars}"
    ratio = cycle_len / 200.0
    alloc = cfg["currency_allocation"]
    alloc["lookback"] = max(2, int(round(63 * ratio)))
    gate = alloc["risk_gate"]
    gate["lookback"] = max(2, int(round(10 * ratio)))
    gate["sma_lookback"] = max(2, int(round(20 * ratio)))
    for sub in cfg["sub_strategies"]:
        sub["overrides"] = scale_sub_overrides(sub["overrides"], cycle_len, rebal_bars, tf_label, "729d")
    return cfg


def run_variant(cfg: dict[str, Any]) -> pd.Series | None:
    handler = Yoon1kHandler()
    try:
        raw = handler.load_data("PORTFOLIO", cfg, synthetic=False)
        art = handler.build_artifacts(raw, cfg, symbol="PORTFOLIO", phase="all", bars_per_year=252)
    except Exception as e:  # noqa: BLE001
        print(f"    실패: {type(e).__name__}: {e}")
        return None
    return art.equity


def main() -> None:
    base_cfg = json.loads((ROOT / "configs" / "strategies" / "final0706_4h.json").read_text("utf-8"))

    print("기준(일봉, 무스케일, 원본 final0706) 로드 중...", flush=True)
    ref_cfg = json.loads((ROOT / "configs" / "strategies" / "final0706.json").read_text("utf-8"))
    ref_equity_full = run_variant(ref_cfg)

    all_results = {}
    md_sections = []
    for tf_label, bar_minutes in TFS:
        print(f"\n=== {tf_label} ===", flush=True)
        ann = bars_per_year(bar_minutes)
        rows = []
        for cyc in CYCLE_GRID:
            for rb in REBAL_GRID:
                cfg = build_config(base_cfg, cyc, rb, tf_label)
                eq = run_variant(cfg)
                perf = perf_from_equity(eq, ann) if eq is not None else {}
                sh = perf.get("sharpe")
                if eq is not None and len(eq) > 10:
                    ref_win = ref_equity_full.loc[(ref_equity_full.index >= eq.index.min())
                                                  & (ref_equity_full.index <= eq.index.max())]
                    ref_win = ref_win / ref_win.iloc[0] if len(ref_win) else ref_win
                    ref_perf = perf_from_equity(ref_win, TRADING_DAYS_PER_YEAR)
                else:
                    ref_perf = {}
                ref_sh = ref_perf.get("sharpe")
                pct = f"{sh/ref_sh*100:.0f}%" if (sh is not None and ref_sh) else "-"
                rows.append({"cycle_len": cyc, "rebal_bars": rb, **(perf or {}),
                            "ref_sharpe_same_window": ref_sh})
                print(f"cycle={cyc:<4} rebal={rb:<3} Sharpe={sh} ({pct} of ref={ref_sh}) "
                      f"CAGR={perf.get('cagr')} MDD={perf.get('max_drawdown')}")

        df = pd.DataFrame(rows)
        best = df.loc[df["sharpe"].idxmax()] if df["sharpe"].notna().any() else None
        all_results[tf_label] = {"rows": df.to_dict(orient="records"),
                                 "best": None if best is None else best.to_dict()}

        pivot = df.pivot(index="cycle_len", columns="rebal_bars", values="sharpe")
        md_sections.append(f"## {tf_label}")
        md_sections.append("")
        md_sections.append("| cycle_len \\ rebal |" + "|".join(f" {c} " for c in pivot.columns) + "|")
        md_sections.append("| ---: |" + "|".join(" ---: " for _ in pivot.columns) + "|")
        for idx_, row in pivot.iterrows():
            cells = ["-" if pd.isna(v) else f"{v:.2f}" for v in row]
            md_sections.append(f"| {idx_} |" + "|".join(f" {c} " for c in cells) + "|")
        md_sections.append("")
        if best is not None:
            md_sections.append(f"**최적**: cycle_len={int(best['cycle_len'])}, rebal={int(best['rebal_bars'])}, "
                               f"Sharpe={best['sharpe']:.3f} (기준 동일구간 {best['ref_sharpe_same_window']:.3f} 대비 "
                               f"{best['sharpe']/best['ref_sharpe_same_window']*100:.0f}%), "
                               f"CAGR={best['cagr']*100:.2f}%, MDD={best['max_drawdown']*100:.2f}%")
            md_sections.append("")

    outdir = ROOT / "reports" / "profile_sizing"
    out_json = outdir / "final0706_4h_full_cross_sweep.json"
    out_json.write_text(json.dumps(all_results, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    lines = [
        "# final0706 US+KR 2조 구조 전체 — TF(1h/4h) × cycle_len × rebalance 크로스 스윕",
        "",
        "- US 슬리브 단독 최적치(cycle=100,rebal=42)를 2조 구조 전체에 그대로 적용했더니 Sharpe=0.713으로"
        " US 단독(1.450)보다 훨씬 낮아, 2조 구조 레벨에서 재탐색.",
        "- 기준 = 그 조합의 equity가 덮는 정확히 같은 구간을 자른 일봉·무스케일 원본 final0706(US+KR 2조) Sharpe.",
        "",
    ] + md_sections + ["## 파일", "", f"- JSON: `{out_json.relative_to(ROOT)}`"]
    out_md = outdir / "final0706_4h_full_cross_sweep.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nMD: {out_md}")


if __name__ == "__main__":
    main()
