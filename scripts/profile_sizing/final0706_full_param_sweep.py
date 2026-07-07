"""final0706(일봉, US+KR 2조 원본) — cycle_len(SMA) × rebalance 주기 전면 스윕.

final0706_4h에서 했던 것과 같은 방법론을, 원본 **일봉** 구조에 그대로 적용한다.
현재 확정치(base_cycle=200, rebalance_freq="monthly")가 실제로 이 그리드 안에서
최선인지 확인한다. profile.rolling_window/market_filter.ma_len/safe_sleeve.ma_len/
regime_cap.recovery_stage_bars/통화틸트 lookback/신용게이트 lookback·sma_lookback은
cycle_len에 비례해 같이 스케일(200봉 기준 비율 유지). rebalance는 "monthly"(원본
그대로, 달력 기준)와 봉수 기준 every_N 여러 값을 함께 스윕한다.

Usage:
    PYTHONPATH=src:scripts python3 scripts/profile_sizing/final0706_full_param_sweep.py
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
TRADING_DAYS_PER_YEAR = 252.0
CYCLE_GRID = (50, 75, 100, 125, 150, 200, 250, 300, 400)
REBAL_GRID = ("monthly", 10, 15, 21, 30, 42, 63, 84)

SCALE_KEYS = [
    ("profile", "rolling_window", 126),
    ("market_filter", "ma_len", 100),
    ("safe_sleeve", "ma_len", 100),
    ("regime_cap", "recovery_stage_bars", 21),
]


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


def rebal_label(rb) -> str:
    return "monthly" if rb == "monthly" else f"every_{rb}"


def build_sub_overrides(sub_entry: dict[str, Any], cycle_len: int, rebal_freq: str) -> dict[str, Any]:
    """base 파일(yoon1j_lp/kr.json)을 원본 override와 완전히 병합한 뒤 스케일 적용.

    `Yoon1kHandler._sub_config`는 `cfg.update(overrides)`로 얕은 병합만 하므로,
    "profile": {"rolling_window": X}처럼 그룹의 일부 키만 담은 dict를 넘기면
    weight_mode 등 나머지 형제 키가 통째로 사라진다(실측으로 발견한 버그).
    그래서 여기서는 base 파일 전체를 미리 로드해 완전히 병합한 뒤, 그 안의
    개별 키만 덮어써서 형제 키를 보존한 **완전한** dict를 override로 돌려준다.
    """
    full = Yoon1kHandler._sub_config(sub_entry["config"], sub_entry.get("overrides"))
    full["rebalance_freq"] = rebal_freq
    full["base_cycle"] = {"type": "SMA", "length": cycle_len, "scale": 1.0}
    ratio = cycle_len / 200.0
    for group, key, default in SCALE_KEYS:
        full[group][key] = max(2, int(round(default * ratio)))
    return full


def build_config(base_cfg: dict[str, Any], cycle_len: int, rebal) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    rebal_freq = rebal_label(rebal)
    cfg["rebalance_freq"] = rebal_freq
    ratio = cycle_len / 200.0
    alloc = cfg["currency_allocation"]
    alloc["lookback"] = max(2, int(round(63 * ratio)))
    gate = alloc["risk_gate"]
    gate["lookback"] = max(2, int(round(10 * ratio)))
    gate["sma_lookback"] = max(2, int(round(20 * ratio)))
    for sub in cfg["sub_strategies"]:
        sub["overrides"] = build_sub_overrides(sub, cycle_len, rebal_freq)
    return cfg


def run_variant(cfg: dict[str, Any]) -> pd.Series | None:
    handler = Yoon1kHandler()
    try:
        raw = handler.load_data("PORTFOLIO", cfg, synthetic=False)
        art = handler.build_artifacts(raw, cfg, symbol="PORTFOLIO", phase="all", bars_per_year=252)
    except Exception as e:  # noqa: BLE001
        print(f"    실패 cycle={cfg['base_cycle']['length'] if 'base_cycle' in cfg else '?'}: {type(e).__name__}: {e}")
        return None
    return art.equity


def main() -> None:
    base_cfg = json.loads((ROOT / "configs" / "strategies" / "final0706.json").read_text("utf-8"))

    print("현재 확정치(cycle=200, monthly) 베이스라인 계산 중...", flush=True)
    baseline_cfg = build_config(base_cfg, 200, "monthly")
    baseline_eq = run_variant(baseline_cfg)
    baseline_perf = perf_from_equity(baseline_eq, TRADING_DAYS_PER_YEAR)
    print(f"베이스라인 Sharpe={baseline_perf['sharpe']:.3f} CAGR={baseline_perf['cagr']*100:.2f}% "
          f"MDD={baseline_perf['max_drawdown']*100:.2f}%")

    rows = []
    for cyc in CYCLE_GRID:
        for rb in REBAL_GRID:
            cfg = build_config(base_cfg, cyc, rb)
            eq = run_variant(cfg)
            perf = perf_from_equity(eq, TRADING_DAYS_PER_YEAR) if eq is not None else {}
            sh = perf.get("sharpe")
            rows.append({"cycle_len": cyc, "rebal": rebal_label(rb), **(perf or {})})
            pct = f"{sh/baseline_perf['sharpe']*100:.0f}%" if (sh is not None and baseline_perf['sharpe']) else "-"
            print(f"cycle={cyc:<4} rebal={rebal_label(rb):<10} Sharpe={sh} ({pct}) "
                  f"CAGR={perf.get('cagr')} MDD={perf.get('max_drawdown')}")

    df = pd.DataFrame(rows)
    best = df.loc[df["sharpe"].idxmax()] if df["sharpe"].notna().any() else None

    outdir = ROOT / "reports" / "profile_sizing"
    out_json = outdir / "final0706_full_param_sweep.json"
    out_json.write_text(json.dumps({
        "baseline": {"cycle_len": 200, "rebal": "monthly", **baseline_perf},
        "rows": df.to_dict(orient="records"),
        "best": None if best is None else best.to_dict(),
    }, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    pivot = df.pivot(index="cycle_len", columns="rebal", values="sharpe")
    pivot = pivot[[rebal_label(r) for r in REBAL_GRID]]
    lines = [
        "# final0706(일봉 원본) — cycle_len(SMA) × 리밸런스 주기 전면 스윕",
        "",
        f"- 베이스라인(현재 확정치 cycle=200, monthly): Sharpe={baseline_perf['sharpe']:.3f}, "
        f"CAGR={baseline_perf['cagr']*100:.2f}%, MDD={baseline_perf['max_drawdown']*100:.2f}%",
        "- profile.rolling_window/market_filter.ma_len/safe_sleeve.ma_len/regime_cap.recovery_stage_bars/"
        "통화틸트 lookback/신용게이트 lookback·sma_lookback은 cycle_len에 비례해 같이 스케일.",
        "",
        "## Sharpe 그리드 (행=cycle_len, 열=rebalance)",
        "",
    ]
    header = "| cycle_len \\ rebal |" + "|".join(f" {c} " for c in pivot.columns) + "|"
    sep = "| ---: |" + "|".join(" ---: " for _ in pivot.columns) + "|"
    lines += [header, sep]
    for idx_, row in pivot.iterrows():
        cells = ["-" if pd.isna(v) else f"{v:.3f}" for v in row]
        marker = " **(베이스라인)**" if idx_ == 200 else ""
        lines.append(f"| {idx_}{marker} |" + "|".join(f" {c} " for c in cells) + "|")
    lines.append("")

    if best is not None:
        lines += [
            "## 최적 조합",
            "",
            f"- cycle_len={int(best['cycle_len'])}, rebalance={best['rebal']}",
            f"- Sharpe={best['sharpe']:.3f} (베이스라인 대비 {best['sharpe']/baseline_perf['sharpe']*100:.0f}%), "
            f"CAGR={best['cagr']*100:.2f}%, MDD={best['max_drawdown']*100:.2f}%",
            "",
        ]

    lines += ["## 파일", "", f"- JSON: `{out_json.relative_to(ROOT)}`"]
    out_md = outdir / "final0706_full_param_sweep.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nMD: {out_md}")
    if best is not None:
        print(f"\n=== 최적: cycle={int(best['cycle_len'])} rebal={best['rebal']} "
              f"Sharpe={best['sharpe']:.3f} ({best['sharpe']/baseline_perf['sharpe']*100:.0f}%) ===")


if __name__ == "__main__":
    main()
