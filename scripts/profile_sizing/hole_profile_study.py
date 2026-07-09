"""hole_profile 사전 검증: 계층적 지그재그(L0/L1/L2) 진행-스윙 백분위의 방향성 가설.

hole_profile.txt(Pine "Multiscale Market Structure")의 익스트림 프로파일(스윙
지속시간·%변동 히스토그램) 아이디어를 "돌파/지속 모멘텀"으로 해석한다: 선택
레벨에서 마지막 확정 피벗 이후 진행 중인 스윙의 |%변동|이, 그 레벨에서 과거
완결된 스윙들의 분포 중 상위(희소, "hole") 구간에 있을수록 — 즉 이미 보기 드문
크기로 확장된 스윙일수록 — 그 방향으로 더 간다는 가설이다(소진/반전이 아님).

검증:
  (a) L0/L1/L2 각각에서 direction-signed forward return을 profile_rank
      10분위(decile)로 나눠 Newey-West t + Bonferroni/BH-FDR(new0707_study.py와
      동형 방법론, cycle_state.py 유틸 재사용).
  (b) 상위 분위(>=0.85/0.90/0.95) 진입 게이트만 뽑아 baseline(전체 방향 confirmed
      표본)과 평균 비교 — 실제 전략이 쓰는 게이트에 가장 가까운 형태.

Usage:
    PYTHONPATH=scripts .venv/bin/python scripts/profile_sizing/hole_profile_study.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from profile_sizing.cycle_state import forward_returns, newey_west_mean_test
from profile_sizing.hole_profile_features import compute_hole_profile_state

ROOT = Path(__file__).resolve().parents[2]
HORIZONS = (5, 10, 20)
LEVELS = ("L0", "L1", "L2")
DECILE_EDGES = np.linspace(0.0, 1.0, 11)
GATE_THRESHOLDS = (0.70, 0.85, 0.90, 0.95)
MIN_BUCKET_SAMPLES = 30


def _signed_forward(df: pd.DataFrame, horizons=HORIZONS) -> pd.DataFrame:
    close = pd.Series(np.asarray(df["close"], dtype=float), index=df.index)
    fwd = forward_returns(close, horizons)
    return fwd


def decile_stats(state: pd.DataFrame, fwd: pd.DataFrame, level: str) -> pd.DataFrame:
    rank = state["profile_rank"].to_numpy()
    direction = state["direction"].to_numpy()
    valid = (direction != 0) & np.isfinite(rank)
    bucket = np.full(len(state), -1, dtype=int)
    bucket[valid] = np.clip(
        np.digitize(rank[valid], DECILE_EDGES[1:-1]), 0, 9
    )
    rows = []
    for h in HORIZONS:
        signed = fwd[f"fwd_{h}"].to_numpy() * direction
        for b in range(10):
            mask = valid & (bucket == b) & np.isfinite(signed)
            vals = signed[mask]
            n = int(vals.size)
            mean, se, t = newey_west_mean_test(vals, lag=h)
            rows.append({
                "level": level, "horizon": h, "decile": b,
                "rank_lo": round(float(DECILE_EDGES[b]), 2),
                "rank_hi": round(float(DECILE_EDGES[b + 1]), 2),
                "n": n, "mean": mean,
                "median": float(np.median(vals)) if n else np.nan,
                "win_rate": float((vals > 0).mean()) if n else np.nan,
                "nw_se": se, "t_stat": t,
                "p_value": float(2 * (1 - norm.cdf(abs(t)))) if np.isfinite(t) else np.nan,
                "eligible": n >= MIN_BUCKET_SAMPLES,
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


def gate_stats(state: pd.DataFrame, fwd: pd.DataFrame, level: str) -> list[dict]:
    rank = state["profile_rank"].to_numpy()
    direction = state["direction"].to_numpy()
    base_valid = (direction != 0) & np.isfinite(rank)
    rows = []
    for h in HORIZONS:
        signed_all = fwd[f"fwd_{h}"].to_numpy() * direction
        base_mask = base_valid & np.isfinite(signed_all)
        base_mean, base_se, base_t = newey_west_mean_test(signed_all[base_mask], lag=h)
        for thr in GATE_THRESHOLDS:
            gate_mask = base_mask & (rank >= thr)
            vals = signed_all[gate_mask]
            n = int(vals.size)
            mean, se, t = newey_west_mean_test(vals, lag=h)
            rows.append({
                "level": level, "horizon": h, "threshold": thr,
                "n": n, "n_baseline": int(base_mask.sum()),
                "mean": mean, "baseline_mean": base_mean,
                "edge_vs_baseline": (mean - base_mean) if np.isfinite(mean) else np.nan,
                "win_rate": float((vals > 0).mean()) if n else np.nan,
                "nw_se": se, "t_stat": t,
                "p_value": float(2 * (1 - norm.cdf(abs(t)))) if np.isfinite(t) else np.nan,
                "eligible": n >= MIN_BUCKET_SAMPLES,
            })
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        default=ROOT / "data" / "raw" / "BTCUSDT_1d.parquet")
    parser.add_argument("--outdir", type=Path,
                        default=ROOT / "reports" / "profile_sizing")
    parser.add_argument("--atr-len", type=int, default=48)
    parser.add_argument("--atr-mult", type=float, default=1.0)
    parser.add_argument("--min-profile-samples", type=int, default=30)
    args = parser.parse_args(argv)

    raw = pd.read_parquet(args.data)
    print(f"BTC {raw.index[0].date()}~{raw.index[-1].date()} ({len(raw)}봉)")
    fwd = _signed_forward(raw)

    all_deciles = []
    all_gates = []
    for level in LEVELS:
        state = compute_hole_profile_state(
            raw, atr_len=args.atr_len, atr_mult=args.atr_mult, level=level,
            min_profile_samples=args.min_profile_samples,
        )
        n_pivots = int(state["l0_count"].iloc[-1]), int(state["l1_count"].iloc[-1]), int(state["l2_count"].iloc[-1])
        print(f"\n=== level={level} (최종 L0/L1/L2 확정 피벗수={n_pivots}) ===")

        deciles = decile_stats(state, fwd, level)
        all_deciles.append(deciles)
        n_sig_fdr = int(deciles["sig_fdr"].sum())
        print(f"[decile] FDR 유의 {n_sig_fdr}/{len(deciles)}")
        for h in HORIZONS:
            sub = deciles[deciles["horizon"] == h].sort_values("decile")
            means = " ".join(f"{r.mean*100:+.2f}%" if np.isfinite(r.mean) else "  n/a "
                              for r in sub.itertuples())
            print(f"  h={h:2d} decile 0->9 mean: {means}")

        gates = gate_stats(state, fwd, level)
        all_gates.extend(gates)
        for row in gates:
            flag = ""
            if row["eligible"] and np.isfinite(row["p_value"]):
                if row["p_value"] < 0.05:
                    flag = "p<0.05"
            print(f"  gate h={row['horizon']:2d} thr={row['threshold']:.2f} "
                  f"n={row['n']:4d}/{row['n_baseline']:4d} mean={row['mean']*100:+.2f}% "
                  f"base={row['baseline_mean']*100:+.2f}% edge={row['edge_vs_baseline']*100:+.2f}%p "
                  f"t={row['t_stat']:+.2f} {flag}")

    deciles_df = pd.concat(all_deciles, ignore_index=True)
    gates_df = pd.DataFrame(all_gates)

    args.outdir.mkdir(parents=True, exist_ok=True)
    deciles_df.to_csv(args.outdir / "hole_profile_decile_stats.csv", index=False)
    gates_df.to_csv(args.outdir / "hole_profile_gate_stats.csv", index=False)
    payload = {
        "data": str(args.data),
        "range": [str(raw.index[0]), str(raw.index[-1])],
        "params": {
            "atr_len": args.atr_len, "atr_mult": args.atr_mult,
            "min_profile_samples": args.min_profile_samples,
            "horizons": list(HORIZONS), "levels": list(LEVELS),
            "gate_thresholds": list(GATE_THRESHOLDS),
        },
        "decile_stats": deciles_df.to_dict(orient="records"),
        "gate_stats": gates_df.to_dict(orient="records"),
    }
    path = args.outdir / "hole_profile_study.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nJSON: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
