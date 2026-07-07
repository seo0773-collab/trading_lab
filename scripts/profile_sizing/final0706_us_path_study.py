"""final0706 US 슬리브(yoon1j_lp) 경로 틸트(path_gate) 사전 스크리닝.

`yoon_profile_path_study.py`(state_field mult×angle 그리드, BTC 단일종목)와 같은
방법론(NW t + Bonferroni/BH-FDR + walk-forward 3사이클)을, final0706 US 슬리브가
실제로 쓰는 **거래량 프로파일 percentile 버킷 축**(1차원, `cumulative_percentile`)
에 US 섹터 ETF 유니버스(11종, top_k=3 후보군) 전체로 이식한 것이다.

state_field 스터디는 사전에 궤적 기하로 M1A1/M2A1이라는 구체적 셀 가설을 세우고
검정했지만, 여기는 그런 기하적 사전지식이 없으므로 **5개 버킷 전부**를 스크리닝
family에 넣는다(m = 5버킷 × 2경로 × 3horizon = 30, 더 보수적인 다중비교 보정).

사전 등록 (스윕 금지):
  vel        = cumulative_percentile.diff(10)   # 인과적, 창 10봉 고정
  arrival    = up(vel >= 0) / down(vel < 0)
  horizons   = 5, 20, 60
  채택 = FDR 유의(across 30) AND horizon in {20,60} AND WF 3사이클 중 >=2 부호 일치
  보류 = Bonferroni 미달이나 FDR 유의 & WF>=2, 또는 그 역
  풀링: 종목별 NW(mean, se, n) 산출 후 표본가중 결합
        pooled_mean = sum(n_i*mean_i)/sum(n_i)
        pooled_se   = sqrt(sum((n_i/N)^2 * se_i^2))   (종목 간 독립 가정 — 한계로 보고)

Usage:
    PYTHONPATH=src:scripts python3 scripts/profile_sizing/final0706_us_path_study.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from profile_sizing.config import config_from_dict  # noqa: E402
from profile_sizing.cycle_state import forward_returns, newey_west_mean_test  # noqa: E402
from profile_sizing.run import run_pipeline  # noqa: E402
from run_kalman_pipeline import load_yfinance  # noqa: E402
from trading_lab.strategies.yoon1k import Yoon1kHandler  # noqa: E402

VEL_WINDOW = 10
HORIZONS = (5, 20, 60)
MIN_SAMPLES = 50
WF_SPLITS = (("cycle1_~2021", None, "2021-12-31"),
             ("cycle2_2022~2023", "2022-01-01", "2023-12-31"),
             ("cycle3_2024~", "2024-01-01", None))


def load_us_universe() -> tuple[dict[str, pd.DataFrame], dict]:
    final0706_cfg = json.loads((ROOT / "configs" / "strategies" / "final0706.json").read_text("utf-8"))
    us_sub = next(s for s in final0706_cfg["sub_strategies"] if s["currency"] == "USD")
    sub_cfg = Yoon1kHandler._sub_config(us_sub["config"], us_sub.get("overrides"))
    universe = list(sub_cfg["universe"])
    panels = {}
    for sym in universe:
        panels[sym] = load_yfinance(sym, sub_cfg.get("interval", "1d"), sub_cfg.get("period", "max"))
    return panels, sub_cfg


def build_symbol_frame(raw: pd.DataFrame, cfg) -> pd.DataFrame:
    out = run_pipeline(raw, cfg)
    fc = out["forecast"]
    idx = pd.DatetimeIndex(pd.to_datetime(raw.index))
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    close = pd.Series(np.asarray(fc["close"], dtype=float), index=idx)
    pct = pd.Series(np.asarray(fc["cumulative_percentile"], dtype=float), index=idx)
    df = forward_returns(close, HORIZONS)
    df["pct"] = pct
    df["vel"] = pct.diff(VEL_WINDOW)
    df["pct100"] = df["pct"] * 100.0
    return df[df["vel"].notna() & df["pct"].notna()]


def bucket_mask(df: pd.DataFrame, lo: float, hi: float) -> pd.Series:
    if hi >= 100.0:
        return (df["pct100"] >= lo) & (df["pct100"] <= hi)
    return (df["pct100"] >= lo) & (df["pct100"] < hi)


def pooled_nw(frames: dict[str, pd.DataFrame], mask_fn, col: str) -> tuple[float, float, int]:
    means, ses, ns = [], [], []
    for df in frames.values():
        vals = df.loc[mask_fn(df), col].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) < 8:
            continue
        mean, se, _ = newey_west_mean_test(vals, lag=int(col.split("_")[1]))
        means.append(mean)
        ses.append(se)
        ns.append(len(vals))
    if not ns or sum(ns) == 0:
        return float("nan"), float("nan"), 0
    n_tot = sum(ns)
    pooled_mean = sum(m * n for m, n in zip(means, ns)) / n_tot
    pooled_var = sum((n / n_tot) ** 2 * se ** 2 for se, n in zip(ses, ns))
    pooled_se = float(np.sqrt(pooled_var))
    return pooled_mean, pooled_se, n_tot


def main() -> None:
    print("US 유니버스 로드 중...", flush=True)
    panels, sub_cfg = load_us_universe()
    cfg = config_from_dict(sub_cfg)
    buckets = list(cfg.buckets)

    print("종목별 profile pipeline 실행 중...", flush=True)
    frames = {sym: build_symbol_frame(raw, cfg) for sym, raw in panels.items()}

    rows = []
    for lo, hi, weight in buckets:
        label = f"B[{int(lo)},{int(hi)})" if hi < 100 else f"B[{int(lo)},{int(hi)}]"
        for arrival in ("down", "up"):
            def mask_fn(df, lo=lo, hi=hi, arrival=arrival):
                bm = bucket_mask(df, lo, hi)
                return bm & ((df["vel"] < 0) if arrival == "down" else (df["vel"] >= 0))
            for h in HORIZONS:
                mean, se, n = pooled_nw(frames, mask_fn, f"fwd_{h}")
                t = mean / se if se and se > 0 else float("nan")
                rows.append({
                    "bucket": label, "bucket_weight": weight, "arrival": arrival,
                    "horizon": h, "n": n, "mean": mean, "se": se, "t_stat": t,
                    "p_value": float(2 * (1 - norm.cdf(abs(t)))) if np.isfinite(t) else np.nan,
                    "eligible": n >= MIN_SAMPLES,
                })
    stats = pd.DataFrame(rows)

    stats["sig_bonferroni"] = False
    stats["sig_fdr"] = False
    elig = stats["eligible"] & stats["p_value"].notna()
    m = int(elig.sum())
    if m:
        stats.loc[elig, "sig_bonferroni"] = stats.loc[elig, "p_value"] < 0.05 / m
        p = stats.loc[elig, "p_value"].sort_values()
        thresh = 0.0
        for rank, (_, pv) in enumerate(p.items(), start=1):
            if pv <= 0.05 * rank / m:
                thresh = pv
        stats.loc[elig, "sig_fdr"] = stats.loc[elig, "p_value"] <= thresh

    delta_rows = []
    for lo, hi, weight in buckets:
        label = f"B[{int(lo)},{int(hi)})" if hi < 100 else f"B[{int(lo)},{int(hi)}]"
        for h in HORIZONS:
            up = stats[(stats["bucket"] == label) & (stats["arrival"] == "up") & (stats["horizon"] == h)].iloc[0]
            dn = stats[(stats["bucket"] == label) & (stats["arrival"] == "down") & (stats["horizon"] == h)].iloc[0]
            delta = float(up["mean"] - dn["mean"])
            se = float(np.sqrt(up["se"] ** 2 + dn["se"] ** 2))
            t = delta / se if se > 0 else float("nan")
            delta_rows.append({
                "bucket": label, "bucket_weight": weight, "horizon": h,
                "delta_up_minus_down": delta, "se": se, "t_stat": t,
                "p_value": float(2 * (1 - norm.cdf(abs(t)))) if np.isfinite(t) else np.nan,
                "n_up": int(up["n"]), "n_down": int(dn["n"]),
                "eligible": bool(up["eligible"] and dn["eligible"]),
            })
    deltas = pd.DataFrame(delta_rows)

    # FDR/Bonferroni는 delta 검정 자체(패밀리=버킷×horizon) 위에서 계산해야 한다.
    # (up 또는 down 각각의 평균이 0과 다른지는 시장 우상향 표류만 반영해 무의미하다.)
    deltas["sig_bonferroni"] = False
    deltas["sig_fdr"] = False
    delig = deltas["eligible"] & deltas["p_value"].notna()
    dm = int(delig.sum())
    if dm:
        deltas.loc[delig, "sig_bonferroni"] = deltas.loc[delig, "p_value"] < 0.05 / dm
        dp = deltas.loc[delig, "p_value"].sort_values()
        dthresh = 0.0
        for rank, (_, pv) in enumerate(dp.items(), start=1):
            if pv <= 0.05 * rank / dm:
                dthresh = pv
        deltas.loc[delig, "sig_fdr"] = deltas.loc[delig, "p_value"] <= dthresh

    wf_rows = []
    for name, start, end in WF_SPLITS:
        seg_frames = {}
        for sym, df in frames.items():
            seg = df
            if start is not None:
                seg = seg[seg.index >= start]
            if end is not None:
                seg = seg[seg.index <= end]
            seg_frames[sym] = seg
        for lo, hi, weight in buckets:
            label = f"B[{int(lo)},{int(hi)})" if hi < 100 else f"B[{int(lo)},{int(hi)}]"
            for h in (20, 60):
                def mk(arrival, lo=lo, hi=hi):
                    def fn(df, arrival=arrival, lo=lo, hi=hi):
                        bm = bucket_mask(df, lo, hi)
                        return bm & ((df["vel"] < 0) if arrival == "down" else (df["vel"] >= 0))
                    return fn
                mean_up, _, n_up = pooled_nw(seg_frames, mk("up"), f"fwd_{h}")
                mean_dn, _, n_dn = pooled_nw(seg_frames, mk("down"), f"fwd_{h}")
                wf_rows.append({
                    "split": name, "bucket": label, "horizon": h,
                    "mean_up": mean_up, "n_up": n_up,
                    "mean_down": mean_dn, "n_down": n_dn,
                    "delta_up_minus_down": mean_up - mean_dn if np.isfinite(mean_up) and np.isfinite(mean_dn) else np.nan,
                })
    wf = pd.DataFrame(wf_rows)

    judgments = {}
    for lo, hi, weight in buckets:
        label = f"B[{int(lo)},{int(hi)})" if hi < 100 else f"B[{int(lo)},{int(hi)}]"
        for h in (20, 60):
            row = deltas[(deltas["bucket"] == label) & (deltas["horizon"] == h)].iloc[0]
            wf_seg = wf[(wf["bucket"] == label) & (wf["horizon"] == h)]
            same_sign = int((np.sign(wf_seg["delta_up_minus_down"]) == np.sign(row["delta_up_minus_down"])).sum())
            fdr_sig = bool(row["sig_fdr"])
            if not row["eligible"]:
                v = "보류(샘플 부족)"
            elif fdr_sig and same_sign >= 2:
                v = "채택"
            elif (abs(row["t_stat"]) > 1.65 if np.isfinite(row["t_stat"]) else False) and same_sign >= 2:
                v = "보류(방향 일치)"
            else:
                v = "기각"
            judgments[f"{label}_delta_fwd{h}"] = {
                "verdict": v, "delta": float(row["delta_up_minus_down"]),
                "t": float(row["t_stat"]) if np.isfinite(row["t_stat"]) else None,
                "fdr_sig": fdr_sig, "wf_same_sign": f"{same_sign}/{len(wf_seg)}",
            }

    payload = {
        "universe": list(panels.keys()),
        "preregistered": {"vel_window": VEL_WINDOW, "horizons": list(HORIZONS),
                          "buckets": buckets, "family_size": int(len(stats))},
        "stats": stats.to_dict(orient="records"),
        "deltas": deltas.to_dict(orient="records"),
        "walk_forward": wf.to_dict(orient="records"),
        "judgments": judgments,
    }
    outdir = ROOT / "reports" / "profile_sizing"
    out_json = outdir / "final0706_us_path_study.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    accepted = [k for k, v in judgments.items() if v["verdict"] == "채택"]
    held = [k for k, v in judgments.items() if v["verdict"].startswith("보류")]
    print(f"채택: {accepted}")
    print(f"보류: {held}")
    print(f"wrote {out_json}")

    lines = [
        "# final0706 US 슬리브 경로 틸트(path_gate) 사전 스크리닝",
        "",
        f"- 유니버스: {', '.join(panels.keys())} (final0706 US 슬리브 top_k=3 후보 11종)",
        "- 방법: `yoon_profile_path_study.py`와 동일 방법론(NW t + Bonferroni/BH-FDR + WF 3사이클)"
        "을 거래량 프로파일 percentile 버킷 축(1차원)에 이식. 사전 기하 가설이 없어 5개 버킷 전부를"
        " family에 포함(m=30, 더 보수적인 다중비교 보정).",
        f"- vel = cumulative_percentile.diff({VEL_WINDOW}) (사전고정, 스윕 금지). arrival: up=vel≥0 / down=vel<0.",
        "- 풀링: 종목별 NW(mean,se,n) 산출 후 표본가중 결합(종목 간 독립 가정 — 한계로 보고, 실제로는 섹터 ETF 간 일부 상관 존재).",
        "",
        "## 버킷×horizon delta(up−down) 판정",
        "",
        "| bucket(base weight) | horizon | delta(up-down) | t | FDR유의 | WF 부호일치 | 판정 |",
        "| --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for k, v in judgments.items():
        bucket, h = k.rsplit("_delta_fwd", 1)
        lines.append(
            f"| {bucket} | {h} | {v['delta']*100:.2f}%p | {v['t']:.2f} | {v['fdr_sig']} | {v['wf_same_sign']} | **{v['verdict']}** |"
            if v["t"] is not None else
            f"| {bucket} | {h} | {v['delta']*100:.2f}%p | - | {v['fdr_sig']} | {v['wf_same_sign']} | **{v['verdict']}** |"
        )
    lines.append("")
    lines.append("## 파일")
    lines.append("")
    lines.append(f"- JSON: `{out_json.relative_to(ROOT)}`")

    out_md = outdir / "final0706_us_path_study.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"MD:   {out_md}")


if __name__ == "__main__":
    main()
