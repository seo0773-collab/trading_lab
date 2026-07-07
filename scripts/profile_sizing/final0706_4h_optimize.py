"""final0706 US 슬리브 — 4h 전용 파라미터 최적화(cycle_len × rebalance 2D 그리드).

이전 스윕(v1/v2)은 base_cycle.length와 rebalance_freq를 **같은 ratio**로 묶어서
움직였다(4h ratio=0.75에서 Sharpe=1.156, 기준 1.381의 84%로 최선). 이번엔 두 축을
**독립적으로** 그리드서치해서 실제 최적 조합을 찾는다. profile.rolling_window/
market_filter.ma_len/safe_sleeve.ma_len/regime_cap.recovery_stage_bars는 cycle_len에
비례해 같이 스케일(200봉 기준 비율 유지) — 이 부분은 "국면 정의"의 일부라 cycle_len과
분리할 이유가 약하다. rebalance_freq만 cycle_len과 별개로 독립 스윕한다.

기준(reference): 4h가 덮는 3년 구간을 일봉·무스케일·monthly 리밸런싱으로 돈 Sharpe(≈1.38).

Usage:
    PYTHONPATH=src:scripts python3 scripts/profile_sizing/final0706_4h_optimize.py
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import profile_sizing.portfolio as portfolio_mod  # noqa: E402
from run_kalman_pipeline import load_yfinance  # noqa: E402
from trading_lab.strategies.profile_portfolio import ProfilePortfolioHandler  # noqa: E402
from trading_lab.strategies.yoon1k import Yoon1kHandler  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TRADING_MIN_PER_DAY = 390.0
TRADING_DAYS_PER_YEAR = 252.0
BAR_MINUTES = 240
ANN = (TRADING_MIN_PER_DAY / BAR_MINUTES) * TRADING_DAYS_PER_YEAR

SCALE_KEYS = [
    ("profile", "rolling_window", 126),
    ("market_filter", "ma_len", 100),
    ("safe_sleeve", "ma_len", 100),
    ("regime_cap", "recovery_stage_bars", 21),
]
CYCLE_GRID = (50, 75, 100, 125, 150, 175, 200, 250, 300)
REBAL_GRID = (5, 8, 10, 13, 16, 21, 26, 31, 42)
EXTRA_SYMBOLS = ["SPY", "TLT", "IAU"]

_orig_rebalance_dates = portfolio_mod.rebalance_dates


def _patched_rebalance_dates(index: pd.DatetimeIndex, freq: str) -> set:
    if isinstance(freq, str) and freq.startswith("every_"):
        n = max(1, int(freq.split("_", 1)[1]))
        return set(index[::n])
    return _orig_rebalance_dates(index, freq)


portfolio_mod.rebalance_dates = _patched_rebalance_dates


def load_4h(sym: str) -> pd.DataFrame:
    raw_1h = load_yfinance(sym, "1h", "max")
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in raw_1h:
        agg["volume"] = "sum"
    return raw_1h.resample("4h", origin="start").agg(agg).dropna(subset=["close"])


def us_sleeve_config() -> dict[str, Any]:
    final0706_cfg = json.loads((ROOT / "configs" / "strategies" / "final0706.json").read_text("utf-8"))
    us_sub = next(s for s in final0706_cfg["sub_strategies"] if s["currency"] == "USD")
    return Yoon1kHandler._sub_config(us_sub["config"], us_sub.get("overrides"))


def build_variant_config(base_cfg: dict[str, Any], cycle_len: int, rebal_bars: int) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    cfg["interval"] = "4h"
    cfg["base_cycle"]["length"] = cycle_len
    ratio = cycle_len / 200.0
    for group, key, default in SCALE_KEYS:
        cur = int(cfg.get(group, {}).get(key, default))
        cfg[group][key] = max(2, int(round(default * ratio)))
    cfg["rebalance_freq"] = f"every_{rebal_bars}"
    return cfg


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


def run_variant(panels: dict[str, pd.DataFrame], cfg: dict[str, Any]) -> pd.Series | None:
    handler = ProfilePortfolioHandler()
    wide = handler._to_wide(panels)
    try:
        art = handler.build_artifacts(wide, cfg, symbol="PORTFOLIO", phase="all", bars_per_year=252)
    except Exception as e:  # noqa: BLE001
        print(f"    실패 cycle={cfg['base_cycle']['length']} rebal={cfg['rebalance_freq']}: {type(e).__name__}: {e}")
        return None
    return art.equity


def main() -> None:
    base_cfg = us_sleeve_config()
    universe = list(base_cfg["universe"])
    all_symbols = universe + EXTRA_SYMBOLS

    print("4h 데이터 로드 중...", flush=True)
    panels = {sym: load_4h(sym) for sym in all_symbols}
    min_bars = min(len(panels[s]) for s in universe)
    span_start = min(panels[s].index.min() for s in universe)
    span_end = max(panels[s].index.max() for s in universe)
    print(f"span={span_start}~{span_end}, min_bars={min_bars}")

    print("기준(일봉, 동일 3년 구간) 계산 중...", flush=True)
    daily_panels = {sym: load_yfinance(sym, "1d", "max") for sym in all_symbols}
    daily_window = {sym: df.loc[(df.index >= span_start.normalize() - pd.Timedelta(days=250))
                                & (df.index <= span_end)]
                    for sym, df in daily_panels.items()}
    ref_cfg = copy.deepcopy(base_cfg)
    ref_cfg["interval"] = "1d"
    ref_equity = run_variant(daily_window, ref_cfg)
    ref_equity_win = ref_equity.loc[ref_equity.index >= span_start.normalize()]
    ref_perf = perf_from_equity(ref_equity_win, TRADING_DAYS_PER_YEAR)
    ref_sharpe = ref_perf.get("sharpe")
    print(f"기준 Sharpe={ref_sharpe}")

    max_cycle = int(min_bars * 0.6)
    rows = []
    for cyc in CYCLE_GRID:
        if cyc > max_cycle:
            print(f"cycle={cyc} > 가용한도 {max_cycle} — 스킵")
            continue
        for rb in REBAL_GRID:
            cfg = build_variant_config(base_cfg, cyc, rb)
            eq = run_variant(panels, cfg)
            perf = perf_from_equity(eq, ANN) if eq is not None else {}
            sh = perf.get("sharpe")
            rows.append({"cycle_len": cyc, "rebal_bars": rb, **perf})
            pct = f"{sh/ref_sharpe*100:.0f}%" if (sh is not None and ref_sharpe) else "-"
            print(f"cycle={cyc:<4} rebal={rb:<3} Sharpe={sh if sh is not None else float('nan'):>7.3f} "
                  f"({pct}) CAGR={perf.get('cagr')} MDD={perf.get('max_drawdown')}")

    df = pd.DataFrame(rows)
    best = df.loc[df["sharpe"].idxmax()] if df["sharpe"].notna().any() else None

    outdir = ROOT / "reports" / "profile_sizing"
    out_json = outdir / "final0706_4h_optimize.json"
    out_json.write_text(json.dumps({
        "span": [str(span_start), str(span_end)], "min_bars": min_bars,
        "reference_sharpe": ref_sharpe, "reference_perf": ref_perf,
        "rows": df.to_dict(orient="records"),
        "best": None if best is None else best.to_dict(),
    }, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    pivot = df.pivot(index="cycle_len", columns="rebal_bars", values="sharpe")
    lines = [
        "# final0706 US 슬리브 — 4h 전용 파라미터 최적화(cycle_len × rebalance 2D 그리드)",
        "",
        f"- 커버 구간: `{span_start}~{span_end}` (유니버스 최소 유효봉수 {min_bars})",
        f"- 기준(일봉·무스케일·monthly 리밸런싱·동일 3년 구간) Sharpe: **{ref_sharpe:.3f}**",
        "- profile.rolling_window/market_filter.ma_len/safe_sleeve.ma_len/regime_cap.recovery_stage_bars는"
        " cycle_len에 비례해 같이 스케일(200봉 기준 비율 유지). rebalance_freq만 독립 스윕.",
        "",
        "## Sharpe 그리드 (행=cycle_len(봉), 열=rebalance 간격(봉))",
        "",
    ]
    header = "| cycle_len \\ rebal |" + "|".join(f" {c} " for c in pivot.columns) + "|"
    sep = "| ---: |" + "|".join(" ---: " for _ in pivot.columns) + "|"
    lines += [header, sep]
    for idx, row in pivot.iterrows():
        cells = []
        for v in row:
            cells.append("-" if pd.isna(v) else f"{v:.2f}")
        lines.append(f"| {idx} |" + "|".join(f" {c} " for c in cells) + "|")
    lines.append("")

    if best is not None:
        lines += [
            "## 최적 조합",
            "",
            f"- cycle_len={int(best['cycle_len'])}봉(≈{best['cycle_len']*BAR_MINUTES/TRADING_MIN_PER_DAY:.1f}거래일), "
            f"rebalance={int(best['rebal_bars'])}봉(≈{best['rebal_bars']*BAR_MINUTES/TRADING_MIN_PER_DAY:.1f}거래일)",
            f"- Sharpe={best['sharpe']:.3f} (기준 대비 {best['sharpe']/ref_sharpe*100:.0f}%), "
            f"CAGR={best['cagr']*100:.2f}%, MDD={best['max_drawdown']*100:.2f}%",
            "",
        ]

    lines += ["## 파일", "", f"- JSON: `{out_json.relative_to(ROOT)}`"]
    out_md = outdir / "final0706_4h_optimize.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nMD: {out_md}")
    if best is not None:
        print(f"\n=== 최적: cycle={int(best['cycle_len'])} rebal={int(best['rebal_bars'])} "
              f"Sharpe={best['sharpe']:.3f} ({best['sharpe']/ref_sharpe*100:.0f}%) ===")


if __name__ == "__main__":
    main()
