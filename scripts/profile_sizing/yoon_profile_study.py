"""yoon_profile 사전 검증: 멀티플 × 기울기 2D 상태공간 조건부 forward return.

profile_plan/profile_plan.txt의 분석 단계를 그대로 구현한다:
  1. expanding percentile 3분위 그리드(look-ahead 금지)
  2. forward 5/20/60일 수익률 셀별 mean/median/승률/샘플 + Newey-West t-stat
  3. 정보 추가성: multiple 단독 vs 2D (같은 mult_bin 안에서 angle 상/하위 delta)
  4. walk-forward 안정성: 3개 사이클 구간별 셀 엣지 부호 일관성
  5. H1(눌림목 알파)/H2(떨어지는 칼)/H3(감속 익절) 판정
다중비교는 Bonferroni + Benjamini-Hochberg FDR 둘 다 표기. 수수료 제외(플랜).

Usage:
    PYTHONPATH=scripts .venv/bin/python scripts/profile_sizing/yoon_profile_study.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from profile_sizing.cycle_state import (
    compute_cycle_state, forward_returns, newey_west_mean_test,
)

ROOT = Path(__file__).resolve().parents[2]
HORIZONS = (5, 20, 60)
GRID = 3
MIN_CELL_SAMPLES = 50
# 워밍업(SMA200 + z 252 + expanding bin 252 ≈ 2019-07)이 ~2017 사이클을 소진하므로
# 플랜의 (~2017 / 2018~2021 / 2022~) 대신 유효구간을 3개 사이클로 나눈다.
WF_SPLITS = (("cycle1_~2021", None, "2021-12-31"),
             ("cycle2_2022~2023", "2022-01-01", "2023-12-31"),
             ("cycle3_2024~", "2024-01-01", None))


def cell_stats(df: pd.DataFrame, horizons=HORIZONS, grid=GRID) -> pd.DataFrame:
    rows = []
    for h in horizons:
        fwd_col = f"fwd_{h}"
        for mb in range(grid):
            for ab in range(grid):
                mask = (df["mult_bin"] == mb) & (df["angle_bin"] == ab)
                vals = df.loc[mask, fwd_col].to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                n = int(vals.size)
                mean, se, t = newey_west_mean_test(vals, lag=h)
                rows.append({
                    "horizon": h, "mult_bin": mb, "angle_bin": ab, "n": n,
                    "mean": mean, "median": float(np.median(vals)) if n else np.nan,
                    "win_rate": float((vals > 0).mean()) if n else np.nan,
                    "nw_se": se, "t_stat": t,
                    "p_value": float(2 * (1 - norm.cdf(abs(t))))
                    if np.isfinite(t) else np.nan,
                    "eligible": n >= MIN_CELL_SAMPLES,
                })
    out = pd.DataFrame(rows)
    return _apply_multiple_testing(out)


def _apply_multiple_testing(table: pd.DataFrame) -> pd.DataFrame:
    """유효 셀에 대해 Bonferroni + BH-FDR(α=0.05) 판정 컬럼 추가."""
    table = table.copy()
    table["sig_bonferroni"] = False
    table["sig_fdr"] = False
    elig = table["eligible"] & table["p_value"].notna()
    m = int(elig.sum())
    if m == 0:
        return table
    table.loc[elig, "sig_bonferroni"] = table.loc[elig, "p_value"] < 0.05 / m
    p = table.loc[elig, "p_value"].sort_values()
    thresh = 0.0
    for rank, (idx, pv) in enumerate(p.items(), start=1):
        if pv <= 0.05 * rank / m:
            thresh = pv
    table.loc[elig, "sig_fdr"] = table.loc[elig, "p_value"] <= thresh
    return table


def marginal_stats(df: pd.DataFrame, axis: str, horizons=HORIZONS) -> pd.DataFrame:
    """단독 축(mult_bin 또는 angle_bin) 조건부 통계 — 정보 추가성 비교 기준선."""
    rows = []
    for h in horizons:
        for b in range(GRID):
            vals = df.loc[df[axis] == b, f"fwd_{h}"].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            mean, se, t = newey_west_mean_test(vals, lag=h)
            rows.append({"horizon": h, axis: b, "n": int(vals.size),
                         "mean": mean, "nw_se": se, "t_stat": t})
    return pd.DataFrame(rows)


def additivity_stats(cells: pd.DataFrame, horizons=HORIZONS) -> pd.DataFrame:
    """같은 mult_bin 안에서 angle 상위(2) − 하위(0) delta와 t.

    delta t는 NW se 결합(sqrt(se_up² + se_dn²)) — 기울기 축이 실제로
    분리력을 추가하면 delta가 유의해야 한다.
    """
    rows = []
    for h in horizons:
        sub = cells[cells["horizon"] == h]
        for mb in range(GRID):
            up = sub[(sub["mult_bin"] == mb) & (sub["angle_bin"] == GRID - 1)]
            dn = sub[(sub["mult_bin"] == mb) & (sub["angle_bin"] == 0)]
            if up.empty or dn.empty:
                continue
            delta = float(up["mean"].iloc[0] - dn["mean"].iloc[0])
            se = float(np.sqrt(up["nw_se"].iloc[0] ** 2 + dn["nw_se"].iloc[0] ** 2))
            t = delta / se if se > 0 else np.nan
            rows.append({
                "horizon": h, "mult_bin": mb,
                "delta_up_minus_down": delta, "se": se, "t_stat": t,
                "n_up": int(up["n"].iloc[0]), "n_down": int(dn["n"].iloc[0]),
                "eligible": bool(up["eligible"].iloc[0] and dn["eligible"].iloc[0]),
            })
    return pd.DataFrame(rows)


def walk_forward(df: pd.DataFrame, horizon: int = 20) -> pd.DataFrame:
    rows = []
    for name, start, end in WF_SPLITS:
        seg = df
        if start is not None:
            seg = seg[seg.index >= start]
        if end is not None:
            seg = seg[seg.index <= end]
        for mb in range(GRID):
            for ab in range(GRID):
                vals = seg.loc[
                    (seg["mult_bin"] == mb) & (seg["angle_bin"] == ab),
                    f"fwd_{horizon}",
                ].to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                mean, _, t = newey_west_mean_test(vals, lag=horizon)
                rows.append({"split": name, "mult_bin": mb, "angle_bin": ab,
                             "n": int(vals.size), "mean": mean, "t_stat": t})
    return pd.DataFrame(rows)


def h3_stats(df: pd.DataFrame, horizons=HORIZONS) -> pd.DataFrame:
    """H3: 기울기 극단 bin 도달 후 감속 시 높은 멀티플 구간의 forward 악화."""
    recent_extreme = (
        (df["angle_bin"] == GRID - 1).rolling(20, min_periods=1).max() > 0
    )
    decel = df["angle_deg"] < df["angle_deg"].shift(5)
    high_mult = df["mult_bin"] == GRID - 1
    cond = high_mult & recent_extreme & decel
    base = high_mult & ~(recent_extreme & decel)
    rows = []
    for h in horizons:
        a = df.loc[cond, f"fwd_{h}"].to_numpy(dtype=float)
        b = df.loc[base, f"fwd_{h}"].to_numpy(dtype=float)
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        mean_a, se_a, _ = newey_west_mean_test(a, lag=h)
        mean_b, se_b, _ = newey_west_mean_test(b, lag=h)
        delta = mean_a - mean_b
        se = float(np.sqrt(se_a ** 2 + se_b ** 2))
        rows.append({
            "horizon": h, "n_decel": int(a.size), "n_other": int(b.size),
            "mean_decel": mean_a, "mean_other": mean_b,
            "delta": delta, "t_stat": delta / se if se > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def judge_hypotheses(cells: pd.DataFrame, h3: pd.DataFrame) -> dict:
    """H1/H2/H3 판정 (기준: FDR 유의=채택, |t|>1.65 방향 일치=보류, 그 외 기각)."""
    def verdict(row, want_positive: bool) -> str:
        if row is None or not row["eligible"]:
            return "보류(샘플 부족)"
        t = row["t_stat"]
        right_sign = (row["mean"] > 0) == want_positive
        if row["sig_fdr"] and right_sign:
            return "채택"
        if np.isfinite(t) and abs(t) > 1.65 and right_sign:
            return "보류(방향 일치, FDR 미달)"
        return "기각"

    def cell(h, mb, ab):
        sub = cells[(cells["horizon"] == h) & (cells["mult_bin"] == mb)
                    & (cells["angle_bin"] == ab)]
        return sub.iloc[0] if len(sub) else None

    out = {}
    for h in HORIZONS:
        h1 = cell(h, 0, GRID - 1)   # 낮은 멀티플 × 상승 기울기 → 양(+)
        h2 = cell(h, 0, 0)          # 낮은 멀티플 × 하락 기울기 → 나쁨
        out[f"H1_fwd{h}"] = {
            "verdict": verdict(h1, want_positive=True),
            "mean": None if h1 is None else h1["mean"],
            "t": None if h1 is None else h1["t_stat"], "n": 0 if h1 is None else int(h1["n"]),
        }
        # H2는 '유의하게 나쁜'(비교 기준: 음수 또는 그리드 최하위 평균)
        sub = cells[cells["horizon"] == h]
        worst = sub.loc[sub["eligible"], "mean"].min() if sub["eligible"].any() else np.nan
        h2_verdict = verdict(h2, want_positive=False)
        if h2 is not None and h2["eligible"] and np.isfinite(worst):
            h2_is_worst = bool(np.isclose(h2["mean"], worst) or h2["mean"] <= worst + 1e-12)
        else:
            h2_is_worst = False
        out[f"H2_fwd{h}"] = {
            "verdict": h2_verdict, "is_worst_cell": h2_is_worst,
            "mean": None if h2 is None else h2["mean"],
            "t": None if h2 is None else h2["t_stat"], "n": 0 if h2 is None else int(h2["n"]),
        }
    for _, row in h3.iterrows():
        h = int(row["horizon"])
        t = row["t_stat"]
        if not np.isfinite(t) or min(row["n_decel"], row["n_other"]) < MIN_CELL_SAMPLES:
            v = "보류(샘플 부족)"
        elif row["delta"] < 0 and t < -1.96:
            v = "채택"
        elif row["delta"] < 0 and t < -1.65:
            v = "보류(방향 일치)"
        else:
            v = "기각"
        out[f"H3_fwd{h}"] = {"verdict": v, "delta": row["delta"], "t": t,
                             "n_decel": int(row["n_decel"])}
    return out


def render_heatmaps(cells: pd.DataFrame, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, len(HORIZONS), figsize=(4.2 * len(HORIZONS), 7.2))
    for j, h in enumerate(HORIZONS):
        sub = cells[cells["horizon"] == h]
        mean_grid = np.full((GRID, GRID), np.nan)
        n_grid = np.zeros((GRID, GRID))
        for _, r in sub.iterrows():
            mean_grid[int(r["mult_bin"]), int(r["angle_bin"])] = r["mean"] * 100
            n_grid[int(r["mult_bin"]), int(r["angle_bin"])] = r["n"]
        for i, (grid, title, cmap) in enumerate(
            [(mean_grid, f"fwd {h}d mean (%)", "RdYlGn"),
             (n_grid, f"fwd {h}d n", "Blues")]
        ):
            ax = axes[i, j]
            vmax = np.nanmax(np.abs(grid)) if i == 0 else None
            im = ax.imshow(grid, cmap=cmap, origin="lower",
                           vmin=-vmax if i == 0 else None, vmax=vmax)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("angle_bin (0=down)")
            ax.set_ylabel("mult_bin (0=low)")
            ax.set_xticks(range(GRID))
            ax.set_yticks(range(GRID))
            for mb in range(GRID):
                for ab in range(GRID):
                    v = grid[mb, ab]
                    if np.isfinite(v):
                        ax.text(ab, mb, f"{v:.2f}" if i == 0 else f"{int(v)}",
                                ha="center", va="center", fontsize=9)
            fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        default=ROOT / "data" / "raw" / "BTCUSDT_1d.parquet")
    parser.add_argument("--outdir", type=Path,
                        default=ROOT / "reports" / "profile_sizing")
    args = parser.parse_args(argv)

    raw = pd.read_parquet(args.data)
    close = raw["close"]
    state = compute_cycle_state(close)
    df = state.join(forward_returns(close, HORIZONS))
    df = df[(df["mult_bin"] >= 0) & (df["angle_bin"] >= 0)]

    cells = cell_stats(df)
    marg_mult = marginal_stats(df, "mult_bin")
    marg_angle = marginal_stats(df, "angle_bin")
    addit = additivity_stats(cells)
    wf = walk_forward(df)
    h3 = h3_stats(df)
    judgments = judge_hypotheses(cells, h3)

    args.outdir.mkdir(parents=True, exist_ok=True)
    cells.to_csv(args.outdir / "yoon_profile_cell_stats.csv", index=False)
    png = args.outdir / "yoon_profile_heatmaps.png"
    render_heatmaps(cells, png)

    payload = {
        "data": str(args.data),
        "rows_effective": int(len(df)),
        "range": [str(df.index[0]), str(df.index[-1])],
        "params": {"cycle_len": 200, "slope_lookback": 5, "z_window": 252,
                   "grid_bins": GRID, "min_bin_history": 252,
                   "min_cell_samples": MIN_CELL_SAMPLES,
                   "horizons": list(HORIZONS)},
        "cells": cells.to_dict(orient="records"),
        "marginal_multiple": marg_mult.to_dict(orient="records"),
        "marginal_angle": marg_angle.to_dict(orient="records"),
        "additivity": addit.to_dict(orient="records"),
        "walk_forward_fwd20": wf.to_dict(orient="records"),
        "h3": h3.to_dict(orient="records"),
        "judgments": judgments,
    }
    out_json = args.outdir / "yoon_profile_study.json"
    out_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8",
    )
    print(f"wrote {out_json}")
    print(f"wrote {png}")
    print(json.dumps(judgments, indent=2, ensure_ascii=False, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
