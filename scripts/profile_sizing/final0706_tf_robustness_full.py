"""final0706 US 슬리브 percentile 버킷 컨트래리언 경향성 — 5분~4시간봉 전체 스캔.

`final0706_tf_robustness.py`(1시간봉 단일)를 5m/15m/30m/1h/4h 다섯 타임프레임으로
일반화한다. TF마다 데이터 가용량이 다르므로 파라미터 스케일 방식이 갈린다:

  - **wall_clock_scaled**: 일봉 파라미터(base_cycle=200, horizon=5/20/60)를 하루
    거래시간(6.5h) 기준 봉-수 비율로 스케일 — 같은 wall-clock 기간을 재현.
    가용 봉수가 충분한 TF(1h, 4h)에서만 가능.
  - **compressed_fallback**: wall-clock 스케일이 가용 데이터보다 크면(5m/15m/30m —
    yfinance 60일 한도), base_cycle을 가용 데이터에 맞게 줄이되 **원래 비율**
    (horizon = cycle_len × {5,20,60}/200)은 유지한다. 이건 "같은 기간"이 아니라
    "같은 구조를 압축한" 다른 성격의 테스트임을 리포트에 명시한다.

4h는 yfinance에 없는 interval이라 1h 데이터를 리샘플링해서 만든다.

Usage:
    PYTHONPATH=src:scripts python3 scripts/profile_sizing/final0706_tf_robustness_full.py
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
BUCKET_ORDER = [f"B[{lo},{hi})" if hi < 100 else f"B[{lo},{hi}]" for lo, hi, _ in BUCKETS]
TRADING_MIN_PER_DAY = 390.0  # 6.5h
DAILY_CYCLE_LEN = 200
DAILY_HORIZONS = (5, 20, 60)
MIN_SAMPLES = 30

TFS = [
    ("5m", 5),
    ("15m", 15),
    ("30m", 30),
    ("1h", 60),
    ("4h", 240),
]


def load_universe() -> tuple[list[str], dict]:
    final0706_cfg = json.loads((ROOT / "configs" / "strategies" / "final0706.json").read_text("utf-8"))
    us_sub = next(s for s in final0706_cfg["sub_strategies"] if s["currency"] == "USD")
    sub_cfg = Yoon1kHandler._sub_config(us_sub["config"], us_sub.get("overrides"))
    return list(sub_cfg["universe"]), sub_cfg


def load_raw(sym: str, tf_label: str) -> pd.DataFrame:
    if tf_label == "4h":
        raw_1h = load_yfinance(sym, "1h", "max")
        agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
        if "volume" in raw_1h:
            agg["volume"] = "sum"
        out = raw_1h.resample("4h", origin="start").agg(agg).dropna(subset=["close"])
        return out
    return load_yfinance(sym, tf_label, "max")


def scale_params(bar_minutes: float, available_bars: int) -> tuple[int, tuple[int, ...], str]:
    bars_per_day = TRADING_MIN_PER_DAY / bar_minutes
    desired_cycle = int(round(DAILY_CYCLE_LEN * bars_per_day))
    desired_horizons = tuple(int(round(h * bars_per_day)) for h in DAILY_HORIZONS)
    max_usable = max(available_bars // 4, 30)
    if desired_cycle <= max_usable:
        return desired_cycle, desired_horizons, "wall_clock_scaled"
    compressed_cycle = max(20, min(desired_cycle, max_usable))
    ratios = tuple(h / DAILY_CYCLE_LEN for h in DAILY_HORIZONS)
    compressed_horizons = tuple(max(3, int(round(compressed_cycle * r))) for r in ratios)
    return compressed_cycle, compressed_horizons, "compressed_fallback"


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


def pooled_nw(frames: dict[str, pd.DataFrame], lo: float, hi: float, col: str, lag: int) -> tuple[float, float, int]:
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


def run_scale(frames: dict[str, pd.DataFrame], horizons: tuple[int, ...]) -> pd.DataFrame:
    rows = []
    for (lo, hi, weight), blabel in zip(BUCKETS, BUCKET_ORDER):
        for h in horizons:
            mean, se, n = pooled_nw(frames, lo, hi, f"fwd_{h}", lag=h)
            t = mean / se if se and se > 0 else float("nan")
            rows.append({
                "bucket": blabel, "bucket_weight": weight, "horizon": h,
                "n": n, "mean": mean, "se": se, "t_stat": t,
                "p_value": float(2 * (1 - norm.cdf(abs(t)))) if np.isfinite(t) else np.nan,
            })
    return pd.DataFrame(rows)


def top_bottom_delta(stats: pd.DataFrame, horizons: tuple[int, ...]) -> dict:
    out = {}
    for h in horizons:
        sub = stats[stats["horizon"] == h].set_index("bucket").loc[BUCKET_ORDER]
        means = sub["mean"].to_numpy()
        diffs = np.diff(means)
        top, bottom = sub.loc[BUCKET_ORDER[0]], sub.loc[BUCKET_ORDER[-1]]
        delta = float(top["mean"] - bottom["mean"])
        se = float(np.sqrt(top["se"] ** 2 + bottom["se"] ** 2))
        t = delta / se if se > 0 else float("nan")
        out[f"h={h}"] = {
            "monotonic_steps_ok": f"{int((diffs <= 0).sum())}/{len(diffs)}",
            "top_minus_bottom_pct": delta * 100, "t": t,
        }
    return out


def main() -> None:
    universe, sub_cfg = load_universe()
    profile_cfg = Profile(
        min_mult=float(sub_cfg["profile"]["min_mult"]), max_mult=float(sub_cfg["profile"]["max_mult"]),
        bin_count=int(sub_cfg["profile"]["bin_count"]), rolling_window=int(sub_cfg["profile"]["rolling_window"]),
        weight_mode=str(sub_cfg["profile"]["weight_mode"]), accumulation_mode=str(sub_cfg["profile"]["accumulation_mode"]),
        percentile_value=float(sub_cfg["profile"]["percentile_value"]),
    )

    all_results = {}
    md_sections = []
    for tf_label, bar_minutes in TFS:
        print(f"\n=== {tf_label} ===", flush=True)
        raws = {}
        for sym in universe:
            try:
                raws[sym] = load_raw(sym, tf_label)
            except Exception as e:  # noqa: BLE001
                print(f"  {sym} 로드 실패: {e}")
        if not raws:
            continue
        min_bars = min(len(r) for r in raws.values())
        cycle_len, horizons, mode = scale_params(bar_minutes, min_bars)
        print(f"  span={min(r.index.min() for r in raws.values())}~{max(r.index.max() for r in raws.values())}, "
              f"min_bars={min_bars}, cycle_len={cycle_len}, horizons={horizons}, mode={mode}")

        frames = {sym: build_frame(raw, cycle_len, profile_cfg, horizons) for sym, raw in raws.items()}
        stats = run_scale(frames, horizons)
        delta = top_bottom_delta(stats, horizons)
        print(stats.to_string(index=False))
        print(delta)

        all_results[tf_label] = {
            "bar_minutes": bar_minutes, "min_bars": min_bars, "cycle_len": cycle_len,
            "horizons": horizons, "mode": mode,
            "span": [str(min(r.index.min() for r in raws.values())), str(max(r.index.max() for r in raws.values()))],
            "stats": stats.to_dict(orient="records"), "top_bottom_delta": delta,
        }

        md_sections.append(f"## {tf_label} (mode={mode}, cycle_len={cycle_len}, horizons={horizons}, min_bars={min_bars})")
        md_sections.append("")
        md_sections.append(f"- 커버 구간: `{all_results[tf_label]['span'][0]}~{all_results[tf_label]['span'][1]}`")
        md_sections.append("")
        md_sections.append("| bucket(weight) | horizon | n | mean fwd return | t |")
        md_sections.append("| --- | ---: | ---: | ---: | ---: |")
        for _, r in stats.iterrows():
            md_sections.append(f"| {r['bucket']}({r['bucket_weight']}) | {r['horizon']} | {r['n']} | {r['mean']*100:.2f}% | {r['t_stat']:.2f} |")
        md_sections.append("")
        md_sections.append("**top(B[0,20)) − bottom(B[80,100]) 스프레드**")
        md_sections.append("")
        for h, v in delta.items():
            md_sections.append(f"- {h}: {v['top_minus_bottom_pct']:.2f}%p (t={v['t']:.2f}), 단조감소 {v['monotonic_steps_ok']}")
        md_sections.append("")

    outdir = ROOT / "reports" / "profile_sizing"
    out_json = outdir / "final0706_tf_robustness_full.json"
    out_json.write_text(json.dumps(all_results, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    lines = [
        "# final0706 US 슬리브 percentile 버킷 컨트래리언 경향성 — 5분~4시간봉 전체 스캔",
        "",
        f"- 유니버스: {', '.join(universe)}",
        "- `wall_clock_scaled` = 일봉과 같은 wall-clock 기간을 봉-수 비율로 재현(유효 비교).",
        "- `compressed_fallback` = 가용 데이터가 부족해(yfinance TF별 한도) base_cycle을 데이터에 맞게 줄이되"
        " horizon:cycle 비율(5/20/60 : 200)은 유지 — **다른 시간축을 압축 테스트한 것**이라 일봉과 직접 비교 불가.",
        "- 4h는 yfinance에 없는 interval이라 1h 데이터를 리샘플링해서 만듦.",
        "",
    ] + md_sections + ["## 파일", "", f"- JSON: `{out_json.relative_to(ROOT)}`"]
    out_md = outdir / "final0706_tf_robustness_full.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nMD: {out_md}")


if __name__ == "__main__":
    main()
