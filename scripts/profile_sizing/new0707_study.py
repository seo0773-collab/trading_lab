"""new0707 사전 검증: 10봉 가속도×거래량 상태그리드 + 10봉 프렉탈(analog) 투영.

2026-07-07 설계 대화의 두 축을 방향 가정 없이 검증한다:
  (a) state grid — accel_bin(SMA10 기울기 가속도) × volsurge_bin(거래량 서지)
      expanding percentile 3×3, 셀별 forward return NW t + Bonferroni/BH-FDR.
      (yoon_profile_study.py와 동형 방법론, cycle_state.py 유틸 재사용)
  (b) pattern_score — 직전 10봉 z-정규화 로그가격과 가장 유사한 과거 10봉 구간
      top-M '직후 H봉' 실현수익 중앙값(analog). self(자기 패턴 연장)도 비교.
      (yoon_pattern_study.py와 동형 방법론, pattern_projection.py 함수 재사용)
      주의: yoon_pattern은 L=300(연 단위 구조)에서 analog가 유의했다 — L=10(2주
      캔들 모양)에서도 통한다는 보장은 없으므로 IC를 반드시 재검증한다.
  (c) 두 점수의 상관 — 중복 신호인지 확인(중복이면 앙상블 의미 없음).

Usage:
    PYTHONPATH=scripts .venv/bin/python scripts/profile_sizing/new0707_study.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from profile_sizing.cycle_state import forward_returns, newey_west_mean_test
from profile_sizing.new0707_features import compute_new0707_state
from profile_sizing.pattern_projection import analog_scores, self_scores

ROOT = Path(__file__).resolve().parents[2]
HORIZONS = (5, 10, 20)
GRID = 3
MIN_CELL_SAMPLES = 50
L = 10          # 프렉탈 창 길이(봉)
H_PATTERN = 10  # 프렉탈 투영 horizon(=주 horizon)
STEP = 1        # 10봉은 짧아 서브샘플 없이 전부 사용
MATCHES = (30, 50)


def cell_stats(df: pd.DataFrame, horizons=HORIZONS, grid=GRID) -> pd.DataFrame:
    rows = []
    for h in horizons:
        fwd_col = f"fwd_{h}"
        for ab in range(grid):
            for vb in range(grid):
                mask = (df["accel_bin"] == ab) & (df["volsurge_bin"] == vb)
                vals = df.loc[mask, fwd_col].to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                n = int(vals.size)
                mean, se, t = newey_west_mean_test(vals, lag=h)
                rows.append({
                    "horizon": h, "accel_bin": ab, "volsurge_bin": vb, "n": n,
                    "mean": mean, "median": float(np.median(vals)) if n else np.nan,
                    "win_rate": float((vals > 0).mean()) if n else np.nan,
                    "nw_se": se, "t_stat": t,
                    "p_value": float(2 * (1 - norm.cdf(abs(t))))
                    if np.isfinite(t) else np.nan,
                    "eligible": n >= MIN_CELL_SAMPLES,
                })
    return _apply_multiple_testing(pd.DataFrame(rows))


def _apply_multiple_testing(table: pd.DataFrame) -> pd.DataFrame:
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


def evaluate_pattern(score: np.ndarray, fwd_ret: np.ndarray) -> dict:
    d = pd.DataFrame({"s": score, "f": fwd_ret}).dropna()
    if len(d) < 500:
        return {"n": int(len(d))}
    thirds = np.array_split(d.index, 3)
    q = pd.qcut(d["s"], 5, labels=False, duplicates="drop")
    return {
        "n": int(len(d)),
        "ic_spearman": float(d["s"].corr(d["f"], method="spearman")),
        "ic_thirds": [float(d.loc[ix, "s"].corr(d.loc[ix, "f"], method="spearman"))
                      for ix in thirds],
        "quintile_bps": [float(d["f"][q == i].mean() * 1e4) for i in range(5)],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        default=ROOT / "data" / "raw" / "BTCUSDT_1d.parquet")
    parser.add_argument("--outdir", type=Path,
                        default=ROOT / "reports" / "profile_sizing")
    args = parser.parse_args(argv)

    raw = pd.read_parquet(args.data)
    close = raw["close"]
    volume = raw["volume"]
    logp = np.log(close.to_numpy(dtype=np.float64))
    n = len(logp)
    print(f"BTC {close.index[0].date()}~{close.index[-1].date()} ({n}봉)")

    # (a) state grid
    state = compute_new0707_state(close, volume)
    df = state.join(forward_returns(close, HORIZONS))
    df_valid = df[(df["accel_bin"] >= 0) & (df["volsurge_bin"] >= 0)]
    cells = cell_stats(df_valid)
    n_sig_fdr = int(cells["sig_fdr"].sum())
    print(f"\n[state grid] 유효표본 {len(df_valid)}봉, FDR 유의 셀 {n_sig_fdr}/{len(cells)}")
    for h in HORIZONS:
        sub = cells[(cells["horizon"] == h) & cells["eligible"]]
        for _, r in sub.sort_values("t_stat").iterrows():
            flag = "FDR유의" if r["sig_fdr"] else ("|t|>1.65" if abs(r["t_stat"]) > 1.65 else "")
            if flag:
                print(f"  h={h:2d} accel={int(r['accel_bin'])} vol={int(r['volsurge_bin'])} "
                      f"n={int(r['n']):4d} mean={r['mean']*100:+.2f}% t={r['t_stat']:+.2f} {flag}")

    # (b) pattern_score (analog / self), 여러 M 비교
    pattern_results = []
    fwd_future = np.full(n, np.nan)
    fwd_future[: n - H_PATTERN] = logp[H_PATTERN:] - logp[: n - H_PATTERN]
    for m in MATCHES:
        a = analog_scores(logp, L, H_PATTERN, step=STEP, matches=m)
        res_a = evaluate_pattern(a, fwd_future)
        pattern_results.append({"method": "analog", "matches": m, **res_a})
        if "ic_spearman" in res_a:
            print(f"\n[pattern L={L} H={H_PATTERN} M={m}] analog IC {res_a['ic_spearman']:+.4f} · "
                  f"1/3 {['%+.3f' % x for x in res_a['ic_thirds']]} · "
                  f"5분위(bps) {['%.0f' % x for x in res_a['quintile_bps']]}")
        else:
            print(f"[pattern M={m}] analog 표본 부족 (n={res_a['n']})")
    s = self_scores(logp, L, H_PATTERN)
    res_s = evaluate_pattern(s, fwd_future)
    pattern_results.append({"method": "self", "matches": None, **res_s})
    if "ic_spearman" in res_s:
        print(f"[pattern L={L} H={H_PATTERN}] self   IC {res_s['ic_spearman']:+.4f} · "
              f"1/3 {['%+.3f' % x for x in res_s['ic_thirds']]} · "
              f"5분위(bps) {['%.0f' % x for x in res_s['quintile_bps']]}")

    # (c) 상관: pattern_score(analog, 첫 M) vs state 축(accel_deg, vol_z)
    best_m = MATCHES[0]
    pattern_best = analog_scores(logp, L, H_PATTERN, step=STEP, matches=best_m)
    corr_df = pd.DataFrame({
        "pattern": pattern_best,
        "accel_deg": state["accel_deg"].to_numpy(),
        "vol_z": state["vol_z"].to_numpy(),
    }, index=close.index).dropna()
    corr_accel = float(corr_df["pattern"].corr(corr_df["accel_deg"], method="spearman"))
    corr_vol = float(corr_df["pattern"].corr(corr_df["vol_z"], method="spearman"))
    print(f"\n[상관, n={len(corr_df)}] pattern vs accel_deg={corr_accel:+.3f} · "
          f"pattern vs vol_z={corr_vol:+.3f}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    cells.to_csv(args.outdir / "new0707_cell_stats.csv", index=False)
    payload = {
        "data": str(args.data),
        "rows_effective_state": int(len(df_valid)),
        "range": [str(close.index[0]), str(close.index[-1])],
        "params": {
            "sma_len": 10, "accel_lookback": 5, "vol_base_window": 60,
            "z_window": 60, "grid_bins": GRID, "min_bin_history": 120,
            "min_cell_samples": MIN_CELL_SAMPLES, "horizons": list(HORIZONS),
            "pattern_length": L, "pattern_horizon": H_PATTERN, "pattern_step": STEP,
            "pattern_matches": list(MATCHES),
        },
        "state_grid_n_sig_fdr": n_sig_fdr,
        "cells": cells.to_dict(orient="records"),
        "pattern_results": pattern_results,
        "correlation": {"pattern_vs_accel_deg": corr_accel, "pattern_vs_vol_z": corr_vol,
                        "n": int(len(corr_df))},
    }
    path = args.outdir / "new0707_study.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nJSON: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
