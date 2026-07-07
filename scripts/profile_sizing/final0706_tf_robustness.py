"""final0706 US 슬리브 percentile 버킷의 컨트래리언 경향성 — 1시간봉 강건성 검증.

5분봉은 yfinance 한도가 60일(단일 레짐, SMA200 재현 불가)이라 검증이 불가능해
1시간봉(최대 730일 신규 + 캐시 누적분, 실측 2~3년)으로 대체한다.

state_field_trend_analysis.md §5(BTC 4h/1h, SPY 1h 강건성 검증)와 같은 원칙 —
"같은 값"이 아니라 "같은 봉-수 비율"로 파라미터를 스케일한다:
    base_cycle.length: 200일봉 → 200×6.5(하루 거래시간) ≈ 1300 시간봉
    horizons(일봉 5/20/60) → 시간봉 33/130/390 (동일 wall-clock 기간)

검정: final0706 US 유니버스(11종) 전체 풀링, 5개 버킷(0-20/20-40/40-60/60-80/80-100)
각각의 forward return을 NW t-test(vs 0)로 산출하고, 버킷 가중치(0.8→0.1)와 같은
방향으로 forward return이 단조감소하는지 확인한다. 같은 방법을 "1시간봉 데이터가
덮는 동일 기간의 일봉"에도 적용해 TF 자체의 효과와 레짐(최근 2~3년) 효과를 분리한다.

Usage:
    PYTHONPATH=src:scripts python3 scripts/profile_sizing/final0706_tf_robustness.py
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

from profile_sizing.config import BaseCycle, Profile  # noqa: E402
from profile_sizing.cycle_state import newey_west_mean_test  # noqa: E402
from profile_sizing.indicators import compute_cycle  # noqa: E402
from profile_sizing.profile import compute_profile  # noqa: E402
from run_kalman_pipeline import load_yfinance  # noqa: E402
from trading_lab.strategies.yoon1k import Yoon1kHandler  # noqa: E402

BUCKETS = [(0, 20, 0.8), (20, 40, 0.6), (40, 60, 0.5), (60, 80, 0.3), (80, 100, 0.1)]
HOURS_PER_DAY = 6.5
MIN_SAMPLES = 30


def load_universe() -> list[str]:
    final0706_cfg = json.loads((ROOT / "configs" / "strategies" / "final0706.json").read_text("utf-8"))
    us_sub = next(s for s in final0706_cfg["sub_strategies"] if s["currency"] == "USD")
    sub_cfg = Yoon1kHandler._sub_config(us_sub["config"], us_sub.get("overrides"))
    return list(sub_cfg["universe"]), sub_cfg


def forward_returns(close: pd.Series, horizons: tuple[int, ...]) -> pd.DataFrame:
    out = pd.DataFrame(index=close.index)
    for h in horizons:
        out[f"fwd_{h}"] = close.shift(-h) / close - 1.0
    return out


def build_frame(raw: pd.DataFrame, cycle_len: int, profile_cfg: Profile,
                horizons: tuple[int, ...]) -> pd.DataFrame:
    cycle = compute_cycle(raw, BaseCycle(type="SMA", length=cycle_len, scale=1.0))
    profile = compute_profile(cycle, raw, profile_cfg)
    close = pd.Series(np.asarray(raw["close"], dtype=float), index=raw.index)
    df = forward_returns(close, horizons)
    df["pct100"] = np.asarray(profile["cumulative_percentile"], dtype=float) * 100.0
    return df[df["pct100"].notna()]


def pooled_nw(frames: dict[str, pd.DataFrame], lo: float, hi: float, col: str,
             lag: int) -> tuple[float, float, int]:
    means, ses, ns = [], [], []
    for df in frames.values():
        mask = (df["pct100"] >= lo) & (df["pct100"] < hi if hi < 100 else df["pct100"] <= hi)
        vals = df.loc[mask, col].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) < 8:
            continue
        mean, se, _ = newey_west_mean_test(vals, lag=lag)
        means.append(mean); ses.append(se); ns.append(len(vals))
    if not ns or sum(ns) == 0:
        return float("nan"), float("nan"), 0
    n_tot = sum(ns)
    pooled_mean = sum(m * n for m, n in zip(means, ns)) / n_tot
    pooled_var = sum((n / n_tot) ** 2 * se ** 2 for se, n in zip(ses, ns))
    return pooled_mean, float(np.sqrt(pooled_var)), n_tot


def run_scale(frames: dict[str, pd.DataFrame], horizons: tuple[int, ...], label: str) -> pd.DataFrame:
    rows = []
    for lo, hi, weight in BUCKETS:
        blabel = f"B[{lo},{hi})" if hi < 100 else f"B[{lo},{hi}]"
        for h in horizons:
            mean, se, n = pooled_nw(frames, lo, hi, f"fwd_{h}", lag=h)
            t = mean / se if se and se > 0 else float("nan")
            rows.append({
                "scale": label, "bucket": blabel, "bucket_weight": weight, "horizon": h,
                "n": n, "mean": mean, "t_stat": t,
                "p_value": float(2 * (1 - norm.cdf(abs(t)))) if np.isfinite(t) else np.nan,
                "eligible": n >= MIN_SAMPLES,
            })
    return pd.DataFrame(rows)


def monotonic_check(stats: pd.DataFrame, horizons: tuple[int, ...]) -> dict:
    """버킷 가중치(0.8→0.1) 순서로 정렬했을 때 평균 forward return이 단조감소하는지."""
    out = {}
    order = [f"B[{lo},{hi})" if hi < 100 else f"B[{lo},{hi}]" for lo, hi, _ in BUCKETS]
    for h in horizons:
        sub = stats[stats["horizon"] == h].set_index("bucket").loc[order]
        means = sub["mean"].to_numpy()
        diffs = np.diff(means)
        spearman_ok = int((diffs <= 0).sum())
        top = sub.loc[order[0]]
        bottom = sub.loc[order[-1]]
        delta = float(top["mean"] - bottom["mean"])
        se = float(np.sqrt(top.get("se", 0) ** 2 + bottom.get("se", 0) ** 2)) if "se" in sub else np.nan
        out[f"fwd_{h}"] = {
            "means_by_bucket": {b: float(m) for b, m in zip(order, means)},
            "monotonic_steps_ok": f"{spearman_ok}/{len(diffs)}",
            "top_minus_bottom": delta,
        }
    return out


def main() -> None:
    universe, sub_cfg = load_universe()
    print(f"유니버스: {universe}", flush=True)

    profile_cfg = Profile(
        min_mult=float(sub_cfg["profile"]["min_mult"]), max_mult=float(sub_cfg["profile"]["max_mult"]),
        bin_count=int(sub_cfg["profile"]["bin_count"]), rolling_window=int(sub_cfg["profile"]["rolling_window"]),
        weight_mode=str(sub_cfg["profile"]["weight_mode"]), accumulation_mode=str(sub_cfg["profile"]["accumulation_mode"]),
        percentile_value=float(sub_cfg["profile"]["percentile_value"]),
    )
    daily_cycle_len = int(sub_cfg["base_cycle"]["length"])  # 200
    hourly_cycle_len = int(round(daily_cycle_len * HOURS_PER_DAY))  # ~1300
    daily_horizons = (5, 20, 60)
    hourly_horizons = tuple(int(round(h * HOURS_PER_DAY)) for h in daily_horizons)  # ~33,130,390
    print(f"base_cycle: daily={daily_cycle_len} -> hourly={hourly_cycle_len}, horizons: daily={daily_horizons} -> hourly={hourly_horizons}")

    print("1시간봉 로드 중...", flush=True)
    hourly_frames = {}
    for sym in universe:
        raw = load_yfinance(sym, "1h", "max")
        hourly_frames[sym] = build_frame(raw, hourly_cycle_len, profile_cfg, hourly_horizons)
    span = pd.DatetimeIndex(sorted(set().union(*[f.index for f in hourly_frames.values()])))
    print(f"1시간봉 커버 구간: {span.min()} ~ {span.max()}, 유니버스 중 최소 유효봉수={min(len(f) for f in hourly_frames.values())}")

    print("동일 구간 일봉 로드 중 (레짐 vs TF 분리용)...", flush=True)
    daily_frames = {}
    start, end = span.min().normalize(), span.max().normalize()
    for sym in universe:
        raw_d = load_yfinance(sym, "1d", "max")
        raw_d = raw_d.loc[(raw_d.index >= start - pd.Timedelta(days=daily_cycle_len * 2)) & (raw_d.index <= end)]
        df = build_frame(raw_d, daily_cycle_len, profile_cfg, daily_horizons)
        daily_frames[sym] = df[df.index >= start]

    stats_hourly = run_scale(hourly_frames, hourly_horizons, "hourly_scaled")
    stats_daily_same_window = run_scale(daily_frames, daily_horizons, "daily_same_window")

    mono_hourly = monotonic_check(stats_hourly, hourly_horizons)
    mono_daily = monotonic_check(stats_daily_same_window, daily_horizons)

    payload = {
        "universe": universe,
        "hourly_span": [str(span.min()), str(span.max())],
        "params": {"daily_cycle_len": daily_cycle_len, "hourly_cycle_len": hourly_cycle_len,
                   "daily_horizons": daily_horizons, "hourly_horizons": hourly_horizons},
        "stats_hourly": stats_hourly.to_dict(orient="records"),
        "stats_daily_same_window": stats_daily_same_window.to_dict(orient="records"),
        "monotonic_hourly": mono_hourly,
        "monotonic_daily_same_window": mono_daily,
    }
    outdir = ROOT / "reports" / "profile_sizing"
    out_json = outdir / "final0706_tf_robustness.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    print(pd.concat([stats_hourly, stats_daily_same_window]).to_string(index=False))
    print(json.dumps({"hourly": mono_hourly, "daily_same_window": mono_daily}, indent=2, ensure_ascii=False, default=float))

    lines = [
        "# final0706 US 슬리브 percentile 버킷 컨트래리언 경향성 — 1시간봉 강건성 검증",
        "",
        f"- 유니버스: {', '.join(universe)} (final0706 US 슬리브 top_k=3 후보 11종, 풀링)",
        f"- 1시간봉 커버 구간: `{span.min()}~{span.max()}` (심볼별 최소 유효봉수 {min(len(f) for f in hourly_frames.values())})",
        "- 5분봉은 yfinance 한도가 60일(단일 레짐, SMA200 미충족)이라 검증 불가 — 1시간봉으로 대체(사용자 승인).",
        f"- 봉-수 비율 스케일: base_cycle.length 일봉 {daily_cycle_len} → 시간봉 {hourly_cycle_len}(하루 {HOURS_PER_DAY}시간 기준),"
        f" horizon 일봉{daily_horizons} → 시간봉{hourly_horizons}(동일 wall-clock 기간).",
        "- `daily_same_window` = 1시간봉이 덮는 것과 똑같은 최근 기간만 잘라 일봉으로 재검정(TF 효과와 레짐 효과 분리용 대조군).",
        "",
        "## 버킷별 forward return (pooled NW, vs 0)",
        "",
        "| scale | bucket(base weight) | horizon | n | mean fwd return | t | 유의(|t|>1.96) |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for _, r in pd.concat([stats_hourly, stats_daily_same_window]).iterrows():
        sig = "Y" if np.isfinite(r["t_stat"]) and abs(r["t_stat"]) > 1.96 else ""
        lines.append(f"| {r['scale']} | {r['bucket']}({r['bucket_weight']}) | {r['horizon']} | {r['n']} | "
                     f"{r['mean']*100:.2f}% | {r['t_stat']:.2f} | {sig} |")
    lines += ["", "## 단조성(버킷 가중치 0.8→0.1 순서로 forward return이 단조감소하는가)", ""]
    for label, mono in (("1시간봉(스케일)", mono_hourly), ("일봉(동일 최근기간)", mono_daily)):
        lines.append(f"### {label}")
        lines.append("")
        for h, v in mono.items():
            means_str = ", ".join(f"{b}={m*100:.2f}%" for b, m in v["means_by_bucket"].items())
            lines.append(f"- **{h}**: {means_str} — 단조감소 스텝 {v['monotonic_steps_ok']}/4, top−bottom={v['top_minus_bottom']*100:.2f}%p")
        lines.append("")

    lines += ["## 파일", "", f"- JSON: `{out_json.relative_to(ROOT)}`"]
    out_md = outdir / "final0706_tf_robustness.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"MD: {out_md}")


if __name__ == "__main__":
    main()
