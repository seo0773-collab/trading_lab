"""yoon_profile 경로 틸트(path_tilt) 정식 검증 — 셀 × 진입 경로 조건부 forward return.

reports/profile_sizing/state_field_trend_analysis.md §6의 사전 등록 가설을
yoon_profile_study.py와 같은 방법론(NW t + Bonferroni/BH-FDR + walk-forward
3사이클)으로 검정한다.

사전 등록 (스윕 금지 — 아래 값은 스터디 실행 전에 고정):
  vel        = mult_pct.diff(10)   # 인과적 필드 속도, 창 10봉 고정
  arrival    = up(vel >= 0) / down(vel < 0)
  검정 family = 셀 {(1,1), (2,1)} × 경로 {down, up} × horizon {5,20,60} = 12개 평균 검정
  PA (M1A1 분리): delta(up − down) > 0. 채택 = fwd20 또는 fwd60에서 t > 1.96
      이고 WF 3/3 부호 양수. 보류 = t > 1.65 & WF ≥ 2/3. 그 외 기각.
  PB (M2A1 하강 방어): down 평균 fwd20 < 0. 채택 = t < −1.96 & WF 3/3 음수.
      보류 = t < −1.65 & WF ≥ 2/3. 그 외 기각.
  가중치(백테스트 게이트용, 튜닝 아님): M1A1 up→1.0 / down→0.5, M2A1 down→0.0.
  강건성 보고(판정 미사용): vel 창 {5, 20}.

백테스트 게이트: base vs path_tilt(위 고정 가중치)를 all/validation/test로 비교.
채택 조건 = 스터디 판정(PA 채택/보류 이상) AND test phase Sharpe·MDD 비열위.

Usage:
    PYTHONPATH=scripts .venv/bin/python scripts/profile_sizing/yoon_profile_path_study.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from profile_sizing.config import config_from_dict
from profile_sizing.cycle_state import (
    compute_cycle_state, forward_returns, newey_west_mean_test,
)
from profile_sizing.engine import performance
from profile_sizing.run import rebased_equity, slice_window
from profile_sizing.yoon_profile import run_yoon_profile_pipeline

ROOT = Path(__file__).resolve().parents[2]
HORIZONS = (5, 20, 60)
VEL_WINDOW = 10
ROBUST_WINDOWS = (5, 20)
CELLS = ((1, 1), (2, 1))
MIN_SAMPLES = 50
WF_SPLITS = (("cycle1_~2021", None, "2021-12-31"),
             ("cycle2_2022~2023", "2022-01-01", "2023-12-31"),
             ("cycle3_2024~", "2024-01-01", None))
# 백테스트 게이트용 고정 가중치 (제안 §6 그대로 — 스윕 아님)
PA_OVERRIDES = [
    {"mult_bin": 1, "angle_bin": 1, "arrival": "up", "weight": 1.0},
    {"mult_bin": 1, "angle_bin": 1, "arrival": "down", "weight": 0.5},
]
PB_OVERRIDES = [
    {"mult_bin": 2, "angle_bin": 1, "arrival": "down", "weight": 0.0},
]


def build_frame(close: pd.Series, vel_window: int) -> pd.DataFrame:
    state = compute_cycle_state(close)
    df = state.join(forward_returns(close, HORIZONS))
    df["vel"] = df["mult_pct"].diff(vel_window)
    return df[(df["mult_bin"] >= 0) & (df["angle_bin"] >= 0) & df["vel"].notna()]


def path_cell_stats(df: pd.DataFrame) -> pd.DataFrame:
    """사전 등록 family: 셀 × 경로 × horizon 평균 검정 + 다중비교 판정."""
    rows = []
    for mb, ab in CELLS:
        cell_mask = (df["mult_bin"] == mb) & (df["angle_bin"] == ab)
        for arrival, arr_mask in (("down", df["vel"] < 0), ("up", df["vel"] >= 0)):
            mask = cell_mask & arr_mask
            for h in HORIZONS:
                vals = df.loc[mask, f"fwd_{h}"].to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                mean, se, t = newey_west_mean_test(vals, lag=h)
                rows.append({
                    "mult_bin": mb, "angle_bin": ab, "arrival": arrival,
                    "horizon": h, "n": int(vals.size), "mean": mean,
                    "nw_se": se, "t_stat": t,
                    "p_value": float(2 * (1 - norm.cdf(abs(t))))
                    if np.isfinite(t) else np.nan,
                    "eligible": int(vals.size) >= MIN_SAMPLES,
                })
    table = pd.DataFrame(rows)
    table["sig_bonferroni"] = False
    table["sig_fdr"] = False
    elig = table["eligible"] & table["p_value"].notna()
    m = int(elig.sum())
    if m:
        table.loc[elig, "sig_bonferroni"] = table.loc[elig, "p_value"] < 0.05 / m
        p = table.loc[elig, "p_value"].sort_values()
        thresh = 0.0
        for rank, (_, pv) in enumerate(p.items(), start=1):
            if pv <= 0.05 * rank / m:
                thresh = pv
        table.loc[elig, "sig_fdr"] = table.loc[elig, "p_value"] <= thresh
    return table


def delta_stats(stats: pd.DataFrame) -> pd.DataFrame:
    """PA: 같은 셀 up − down delta와 결합 NW t (additivity_stats와 동일 방식)."""
    rows = []
    for mb, ab in CELLS:
        for h in HORIZONS:
            sub = stats[(stats["mult_bin"] == mb) & (stats["angle_bin"] == ab)
                        & (stats["horizon"] == h)]
            up = sub[sub["arrival"] == "up"]
            dn = sub[sub["arrival"] == "down"]
            if up.empty or dn.empty:
                continue
            delta = float(up["mean"].iloc[0] - dn["mean"].iloc[0])
            se = float(np.sqrt(up["nw_se"].iloc[0] ** 2 + dn["nw_se"].iloc[0] ** 2))
            rows.append({
                "mult_bin": mb, "angle_bin": ab, "horizon": h,
                "delta_up_minus_down": delta, "se": se,
                "t_stat": delta / se if se > 0 else np.nan,
                "n_up": int(up["n"].iloc[0]), "n_down": int(dn["n"].iloc[0]),
                "eligible": bool(up["eligible"].iloc[0] and dn["eligible"].iloc[0]),
            })
    return pd.DataFrame(rows)


def walk_forward(df: pd.DataFrame) -> pd.DataFrame:
    """WF 3사이클: PA delta 부호(fwd20/60), PB down 평균 부호(fwd20)."""
    rows = []
    for name, start, end in WF_SPLITS:
        seg = df
        if start is not None:
            seg = seg[seg.index >= start]
        if end is not None:
            seg = seg[seg.index <= end]
        for mb, ab in CELLS:
            cell = (seg["mult_bin"] == mb) & (seg["angle_bin"] == ab)
            for h in (20, 60):
                res = {}
                for arrival, arr in (("down", seg["vel"] < 0), ("up", seg["vel"] >= 0)):
                    vals = seg.loc[cell & arr, f"fwd_{h}"].to_numpy(dtype=float)
                    vals = vals[np.isfinite(vals)]
                    mean, _, t = newey_west_mean_test(vals, lag=h)
                    res[arrival] = (mean, t, int(vals.size))
                rows.append({
                    "split": name, "mult_bin": mb, "angle_bin": ab, "horizon": h,
                    "mean_down": res["down"][0], "n_down": res["down"][2],
                    "mean_up": res["up"][0], "n_up": res["up"][2],
                    "delta_up_minus_down": res["up"][0] - res["down"][0],
                })
    return pd.DataFrame(rows)


def judge(stats: pd.DataFrame, deltas: pd.DataFrame, wf: pd.DataFrame) -> dict:
    out = {}
    # PA: (1,1) delta > 0, fwd20/60
    wf11 = wf[(wf["mult_bin"] == 1) & (wf["angle_bin"] == 1)]
    for h in (20, 60):
        row = deltas[(deltas["mult_bin"] == 1) & (deltas["angle_bin"] == 1)
                     & (deltas["horizon"] == h)].iloc[0]
        signs = wf11[wf11["horizon"] == h]["delta_up_minus_down"]
        wf_pos = int((signs > 0).sum())
        t = float(row["t_stat"])
        if not row["eligible"]:
            v = "보류(샘플 부족)"
        elif t > 1.96 and wf_pos == len(signs):
            v = "채택"
        elif t > 1.65 and wf_pos >= 2:
            v = "보류(방향 일치)"
        else:
            v = "기각"
        out[f"PA_M1A1_delta_fwd{h}"] = {
            "verdict": v, "delta": float(row["delta_up_minus_down"]),
            "t": t, "wf_positive": f"{wf_pos}/{len(signs)}",
        }
    # PB: (2,1) down 평균 < 0, fwd20
    row = stats[(stats["mult_bin"] == 2) & (stats["angle_bin"] == 1)
                & (stats["arrival"] == "down") & (stats["horizon"] == 20)].iloc[0]
    wf21 = wf[(wf["mult_bin"] == 2) & (wf["angle_bin"] == 1)
              & (wf["horizon"] == 20)]["mean_down"]
    wf_neg = int((wf21 < 0).sum())
    t = float(row["t_stat"])
    if not row["eligible"]:
        v = "보류(샘플 부족)"
    elif t < -1.96 and wf_neg == len(wf21):
        v = "채택"
    elif t < -1.65 and wf_neg >= 2:
        v = "보류(방향 일치)"
    else:
        v = "기각"
    out["PB_M2A1_down_fwd20"] = {
        "verdict": v, "mean": float(row["mean"]), "t": t,
        "wf_negative": f"{wf_neg}/{len(wf21)}",
    }
    return out


def backtest_gate(raw: pd.DataFrame, base_cfg: dict) -> list[dict]:
    """base vs path_tilt phase별 성과 비교 — 전체 제안(PA+PB)과 채택분(PA)만 둘 다."""
    def tilt(overrides):
        cfg = dict(base_cfg)
        cfg["path_tilt"] = {
            "enabled": True, "vel_window": VEL_WINDOW, "overrides": overrides,
        }
        return cfg

    psc = config_from_dict(base_cfg)
    idx = pd.DatetimeIndex(raw.index)
    idx = idx.tz_localize(None) if idx.tz is not None else idx
    rows = []
    for name, cfg in (
        ("base", base_cfg),
        ("tilt_PA+PB", tilt(PA_OVERRIDES + PB_OVERRIDES)),
        ("tilt_PA_only", tilt(PA_OVERRIDES)),
    ):
        out = run_yoon_profile_pipeline(raw, cfg)
        port_ret = out["port_ret"].copy()
        port_ret.index = idx
        exposure = out["forecast"]["actual_weight"]
        for phase in ("all", "validation", "test"):
            window = slice_window(idx, phase, psc)
            equity = rebased_equity(port_ret, window)
            ret = port_ret.reindex(window).fillna(0.0)
            perf = performance(equity, ret, psc.interval)
            rows.append({
                "variant": name, "phase": phase,
                "total_return": perf.get("total_return"),
                "sharpe": perf.get("sharpe"),
                "max_drawdown": perf.get("max_drawdown"),
                "avg_exposure": float(
                    exposure.iloc[[idx.get_loc(t) for t in window]].mean()
                ) if len(window) else np.nan,
            })
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        default=ROOT / "data" / "raw" / "BTCUSDT_1d.parquet")
    parser.add_argument("--outdir", type=Path,
                        default=ROOT / "reports" / "profile_sizing")
    args = parser.parse_args(argv)

    raw = pd.read_parquet(args.data)
    close = pd.Series(np.asarray(raw["close"], dtype=float), index=raw.index)

    df = build_frame(close, VEL_WINDOW)
    stats = path_cell_stats(df)
    deltas = delta_stats(stats)
    wf = walk_forward(df)
    judgments = judge(stats, deltas, wf)

    robustness = {}
    for w in ROBUST_WINDOWS:
        dfw = build_frame(close, w)
        dw = delta_stats(path_cell_stats(dfw))
        r11 = dw[(dw["mult_bin"] == 1) & (dw["angle_bin"] == 1)]
        robustness[f"vel_window_{w}"] = {
            f"delta_fwd{int(r.horizon)}": {"delta": float(r.delta_up_minus_down),
                                           "t": float(r.t_stat)}
            for r in r11.itertuples()
        }

    base_cfg = json.loads(
        (ROOT / "configs" / "strategies" / "yoon_profile.json").read_text()
    )
    gate = backtest_gate(raw, base_cfg)

    payload = {
        "data": str(args.data),
        "rows_effective": int(len(df)),
        "range": [str(df.index[0]), str(df.index[-1])],
        "preregistered": {
            "vel_window": VEL_WINDOW, "cells": list(CELLS),
            "horizons": list(HORIZONS), "family_size": int(len(stats)),
            "pa_overrides": PA_OVERRIDES, "pb_overrides": PB_OVERRIDES,
        },
        "path_cells": stats.to_dict(orient="records"),
        "deltas": deltas.to_dict(orient="records"),
        "walk_forward": wf.to_dict(orient="records"),
        "robustness_vel_window": robustness,
        "judgments": judgments,
        "backtest_gate": gate,
    }
    args.outdir.mkdir(parents=True, exist_ok=True)
    out_json = args.outdir / "yoon_profile_path_study.json"
    out_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8",
    )
    print(f"wrote {out_json}")
    print(json.dumps(judgments, indent=2, ensure_ascii=False, default=float))
    print(pd.DataFrame(gate).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
