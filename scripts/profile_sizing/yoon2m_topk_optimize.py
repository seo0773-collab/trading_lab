"""Optimize yoon2m US/KR top_k grid.

This keeps the yoon2m structure fixed:
  - US sleeve yoon1j_lp, KR sleeve yoon1j_kr
  - 50/50 monthly rebalance
  - USD sleeve converted to KRW
  - no SSO leverage

Only the two sleeve `top_k` values are swept.

Usage:
    PYTHONPATH=src:scripts .venv/bin/python scripts/profile_sizing/yoon2m_topk_optimize.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from profile_sizing.engine import performance  # noqa: E402
from profile_sizing.portfolio import rebalance_dates  # noqa: E402
from trading_lab.strategies.profile_portfolio import ProfilePortfolioHandler  # noqa: E402
from trading_lab.strategies.yoon1k import Yoon1kHandler  # noqa: E402
from run_kalman_pipeline import load_yfinance  # noqa: E402


def load_cfg(strategy_id: str) -> dict[str, Any]:
    return json.loads(
        (ROOT / "configs" / "strategies" / f"{strategy_id}.json").read_text(
            encoding="utf-8"
        )
    )


def sleeve_navs(cfg_id: str, top_ks: range) -> dict[int, pd.Series]:
    base = ProfilePortfolioHandler()
    cfg0 = load_cfg(cfg_id)
    raw = base.load_data("SYNTH", cfg0, synthetic=False)
    out: dict[int, pd.Series] = {}
    for k in top_ks:
        cfg = dict(cfg0)
        cfg["top_k"] = int(k)
        art = base.build_artifacts(raw, cfg, symbol="SYNTH", phase="all", bars_per_year=252)
        out[int(k)] = (art.equity / art.equity.iloc[0]).rename(f"{cfg_id}_k{k}")
    return out


def to_base(nav: pd.Series, fx: pd.Series | None) -> pd.Series:
    if fx is None:
        return nav
    f = pd.Series(fx).reindex(nav.index).ffill()
    first = f.dropna()
    if first.empty:
        return nav
    return nav * (f / first.iloc[0])


def combine_5050(us: pd.Series, kr: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    reb = set(rebalance_dates(idx, "monthly"))
    us_ret = us.reindex(idx).ffill().pct_change().fillna(0.0).to_numpy()
    kr_ret = kr.reindex(idx).ffill().pct_change().fillna(0.0).to_numpy()
    v_us, v_kr = 0.5, 0.5
    out = np.empty(len(idx), dtype=float)
    for i, t in enumerate(idx):
        v_us *= 1.0 + us_ret[i]
        v_kr *= 1.0 + kr_ret[i]
        total = v_us + v_kr
        out[i] = total
        if t in reb:
            v_us = total * 0.5
            v_kr = total * 0.5
    return pd.Series(out, index=idx, name="equity")


def metr(eq: pd.Series) -> dict[str, float | None]:
    eq = (eq / eq.iloc[0]).dropna()
    p = performance(eq, eq.pct_change().fillna(0.0), "1d")
    return {
        "total_return": float(eq.iloc[-1] - 1.0),
        "cagr": p["cagr"],
        "max_drawdown": p["max_drawdown"],
        "sharpe": p["sharpe"],
        "volatility": p["volatility"],
    }


def benchmark(idx: pd.DatetimeIndex) -> pd.Series:
    cfg = load_cfg("yoon2m")
    h = Yoon1kHandler()
    raw = h.load_data("SYNTH", cfg, synthetic=False)
    art = h.build_artifacts(raw, cfg, symbol="SYNTH", phase="all", bars_per_year=252)
    if art.benchmark is None:
        raise RuntimeError("yoon2m benchmark not available")
    return art.benchmark.reindex(idx).ffill().dropna()


def score(row: pd.Series, bench_row: pd.Series, mdd_cap: float | None) -> float:
    """Defense-first score. Hard cap optional, then Sharpe primary and CAGR secondary."""
    if mdd_cap is not None and row["max_drawdown"] < mdd_cap:
        return -1e9 + float(row["sharpe"] or 0.0)
    mdd_bonus = (float(row["max_drawdown"]) - float(bench_row["max_drawdown"])) * 0.5
    return float(row["sharpe"] or 0.0) + 0.25 * float(row["cagr"] or 0.0) + mdd_bonus


def write_report(rows: pd.DataFrame, out: Path, mdd_cap: float | None) -> None:
    bench = rows[rows["kind"] == "benchmark"].iloc[0]
    combos = rows[rows["kind"] == "combo"].copy()
    combos["score_defense"] = combos.apply(lambda r: score(r, bench, mdd_cap), axis=1)
    best_sharpe = combos.sort_values(["sharpe", "cagr"], ascending=False).head(10)
    best_cagr = combos.sort_values(["cagr", "sharpe"], ascending=False).head(10)
    best_mdd = combos.sort_values(["max_drawdown", "sharpe"], ascending=False).head(10)
    best_def = combos.sort_values(["score_defense", "sharpe"], ascending=False).head(10)

    def pct(x: Any) -> str:
        return "n/a" if x is None or pd.isna(x) else f"{float(x) * 100:.2f}%"

    def pct_signed(x: Any) -> str:
        return "n/a" if x is None or pd.isna(x) else f"{float(x) * 100:+.2f}%"

    def num(x: Any) -> str:
        return "n/a" if x is None or pd.isna(x) else f"{float(x):.3f}"

    def table(title: str, df: pd.DataFrame) -> list[str]:
        lines = [
            f"## {title}",
            "",
            "| US k | KR k | CAGR | MDD | Sharpe | total | ΔCAGR vs bench | ΔMDD vs bench |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for r in df.itertuples(index=False):
            lines.append(
                f"| {int(r.us_top_k)} | {int(r.kr_top_k)} | {pct(r.cagr)} | "
                f"{pct(r.max_drawdown)} | {num(r.sharpe)} | {pct(r.total_return)} | "
                f"{pct_signed(r.cagr - bench.cagr)} | "
                f"{pct_signed(r.max_drawdown - bench.max_drawdown)} |"
            )
        lines.append("")
        return lines

    selected = best_def.iloc[0]
    current = combos[(combos["us_top_k"] == 3) & (combos["kr_top_k"] == 3)].iloc[0]
    k1 = combos[(combos["us_top_k"] == 1) & (combos["kr_top_k"] == 1)].iloc[0]

    lines = [
        "# yoon2m top_k optimization",
        "",
        "- Structure fixed: yoon1j_lp USD sleeve + yoon1j_kr KRW sleeve, 50/50 monthly, no SSO.",
        "- Sweep: US top_k 1..11, KR top_k 1..8.",
        "- Metrics are ideal continuous-weight backtests, not integer-share execution.",
        f"- Defense score MDD cap: {pct(mdd_cap) if mdd_cap is not None else 'none'}",
        "",
        "## Key Rows",
        "",
        "| row | US k | KR k | CAGR | MDD | Sharpe | total |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| benchmark | - | - | {pct(bench.cagr)} | {pct(bench.max_drawdown)} | "
            f"{num(bench.sharpe)} | {pct(bench.total_return)} |"
        ),
        (
            f"| current yoon2m | 3 | 3 | {pct(current.cagr)} | "
            f"{pct(current.max_drawdown)} | {num(current.sharpe)} | "
            f"{pct(current.total_return)} |"
        ),
        (
            f"| single-per-sleeve | 1 | 1 | {pct(k1.cagr)} | "
            f"{pct(k1.max_drawdown)} | {num(k1.sharpe)} | {pct(k1.total_return)} |"
        ),
        (
            f"| selected defense | {int(selected.us_top_k)} | {int(selected.kr_top_k)} | "
            f"{pct(selected.cagr)} | {pct(selected.max_drawdown)} | "
            f"{num(selected.sharpe)} | {pct(selected.total_return)} |"
        ),
        "",
    ]
    lines.extend(table("Best Defense Score", best_def))
    lines.extend(table("Best Sharpe", best_sharpe))
    lines.extend(table("Best CAGR", best_cagr))
    lines.extend(table("Shallowest MDD", best_mdd))

    lines.extend([
        "## Verdict",
        "",
        (
            f"Current yoon2m 3/3 remains a strong balance point if the objective is "
            f"low-budget concentration with drawdown defense. The defense-score optimum "
            f"is {int(selected.us_top_k)}/{int(selected.kr_top_k)}, but it should be "
            "treated as a research candidate until bootstrap/walk-forward and integer-share "
            "execution are rerun."
        ),
        "",
    ])
    out.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--us-max", type=int, default=11)
    ap.add_argument("--kr-max", type=int, default=8)
    ap.add_argument(
        "--mdd-cap", type=float, default=-0.20,
        help="Optional max drawdown cap for defense score, e.g. -0.20. Use 0 for no cap.",
    )
    ap.add_argument("--outdir", type=Path, default=ROOT / "reports" / "profile_sizing")
    args = ap.parse_args(argv)

    us_range = range(1, args.us_max + 1)
    kr_range = range(1, args.kr_max + 1)
    us = sleeve_navs("yoon1j_lp", us_range)
    kr = sleeve_navs("yoon1j_kr", kr_range)
    fx = load_yfinance("KRW=X", "1d", "max")["close"]
    idx = None
    for s in list(us.values()) + list(kr.values()):
        idx = s.index if idx is None else idx.intersection(s.index)
    assert idx is not None
    idx = idx.sort_values()
    bench = benchmark(idx)
    idx = idx.intersection(bench.index).sort_values()

    rows: list[dict[str, Any]] = []
    b = bench.reindex(idx).ffill()
    rows.append({"kind": "benchmark", "name": "50/50 bench", "us_top_k": np.nan,
                 "kr_top_k": np.nan, **metr(b)})
    for uk, unav in us.items():
        unav_base = to_base(unav.reindex(idx).ffill(), fx)
        for kk, knav in kr.items():
            eq = combine_5050(unav_base, knav.reindex(idx).ffill(), idx)
            rows.append({
                "kind": "combo",
                "name": f"us{uk}_kr{kk}",
                "us_top_k": uk,
                "kr_top_k": kk,
                **metr(eq),
            })

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    csv_path = outdir / "yoon2m_topk_optimize.csv"
    json_path = outdir / "yoon2m_topk_optimize.json"
    md_path = outdir / "yoon2m_topk_optimize.md"
    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    mdd_cap = None if args.mdd_cap == 0.0 else float(args.mdd_cap)
    write_report(df, md_path, mdd_cap)
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
