
"""
flat_chart.py
=============
Pine "평면차트" 엔진의 Python 포팅 + Kalman용 피처 추출.

구조
----
1. 가격 -> 사이클 배수(price / cycle EMA) 변환
2. 배수 공간에서 fast / slow 2중 rolling volume profile (증분 업데이트)
3. 매 봉 피처 추출:
   - m_fast   : fast profile 질량중심 (관측 z의 핵심)
   - w_fast   : upper - lower percentile 폭 (분포 응축/확산)
   - s_fast   : 질량 비대칭도 (위/아래 질량 불균형, -1 ~ +1)
   - dc       : 이번 봉 유입 질량 무게중심 - m_fast (질량 유입 방향, ṁ의 선행 관측치)
   - m_slow   : slow profile 질량중심 -> OU 평균회귀 목표점 μ(t)
   - lower/upper percentile 배수 및 가격

사용 예
-------
    from indicators.flat_chart import FlatChartConfig, compute_features
    feats = compute_features(df, FlatChartConfig())   # df: OHLCV DataFrame
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
@dataclass
class FlatChartConfig:
    # Cycle (Pine: baseLen / baseType / baseScale)
    cycle_len: int = 200          # 1h봉 기준 EMA200 ≈ 8일
    cycle_scale: float = 1.0

    # Rolling windows (bars)
    fast_window: int = 120        # 1h봉 × 5일
    slow_window: int = 720        # 1h봉 × 30일  -> μ(t)

    # Profile bins (Pine: profileBinCount / Min / Max Multiple)
    n_bins: int = 120
    min_mult: float = 0.0
    max_mult: float = 5.0

    # Percentile (Pine: profilePercentileValue)
    percentile: float = 20.0      # lower 20% / upper 80% cut

    # Weight mode: "volume_fallback" | "volume" | "time"
    weight_mode: str = "volume_fallback"

    @property
    def bin_width(self) -> float:
        return (self.max_mult - self.min_mult) / self.n_bins

    @property
    def bin_mids(self) -> np.ndarray:
        return self.min_mult + (np.arange(self.n_bins) + 0.5) * self.bin_width


# ----------------------------------------------------------------------
# Cycle
# ----------------------------------------------------------------------
def ema(x: np.ndarray, length: int) -> np.ndarray:
    """Pine ta.ema 와 동일한 재귀 EMA (초기값: 첫 length개 SMA)."""
    out = np.full_like(x, np.nan, dtype=float)
    alpha = 2.0 / (length + 1.0)
    valid = np.where(~np.isnan(x))[0]
    if len(valid) < length:
        return out
    start = valid[0]
    if start + length > len(x):
        return out
    seed = np.nanmean(x[start : start + length])
    out[start + length - 1] = seed
    for i in range(start + length, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out


# ----------------------------------------------------------------------
# Incremental range-uniform profile (Pine f_add_range_to_profile 포팅)
# ----------------------------------------------------------------------
def _range_contribution(
    lo: float, hi: float, weight: float, cfg: FlatChartConfig
) -> tuple[int, int, np.ndarray] | None:
    """캔들 [lo, hi] 배수 구간을 겹친 bin들에 균등 분배한 기여분을 반환.

    Returns (start_idx, end_idx, values) or None.
    """
    if not np.isfinite(lo) or not np.isfinite(hi) or weight <= 0:
        return None
    lo, hi = min(lo, hi), max(lo, hi)
    c_lo = max(lo, cfg.min_mult)
    c_hi = min(hi, cfg.max_mult)
    if c_hi < c_lo:
        return None

    bw = cfg.bin_width
    if c_hi == c_lo:  # 한 점
        idx = int(np.clip((c_lo - cfg.min_mult) // bw, 0, cfg.n_bins - 1))
        return idx, idx, np.array([weight])

    i0 = int(np.clip((c_lo - cfg.min_mult) // bw, 0, cfg.n_bins - 1))
    i1 = int(np.clip((c_hi - cfg.min_mult) // bw, 0, cfg.n_bins - 1))
    rng = c_hi - c_lo

    bins_lo = cfg.min_mult + np.arange(i0, i1 + 1) * bw
    overlap = np.minimum(c_hi, bins_lo + bw) - np.maximum(c_lo, bins_lo)
    overlap = np.clip(overlap, 0.0, None)
    vals = weight * overlap / rng
    return i0, i1, vals


def _profile_stats(bins: np.ndarray, cfg: FlatChartConfig) -> dict:
    """질량중심 m, percentile lower/upper, 폭 w, 비대칭도 s."""
    total = bins.sum()
    out = dict(m=np.nan, lower=np.nan, upper=np.nan, w=np.nan, s=np.nan, total=total)
    if total <= 0:
        return out

    mids = cfg.bin_mids
    m = float((bins * mids).sum() / total)
    out["m"] = m

    # percentile cuts (Pine f_calc_profile_stats: bin 내 선형 보간)
    target = total * cfg.percentile / 100.0
    cum = np.cumsum(bins)
    bw = cfg.bin_width

    i_lo = int(np.searchsorted(cum, target))
    i_lo = min(i_lo, cfg.n_bins - 1)
    before = cum[i_lo] - bins[i_lo]
    ratio = (target - before) / bins[i_lo] if bins[i_lo] > 0 else 1.0
    out["lower"] = cfg.min_mult + i_lo * bw + np.clip(ratio, 0, 1) * bw

    cum_r = np.cumsum(bins[::-1])
    j = int(np.searchsorted(cum_r, target))
    j = min(j, cfg.n_bins - 1)
    i_hi = cfg.n_bins - 1 - j
    before_r = cum_r[j] - bins[i_hi]
    ratio_r = (target - before_r) / bins[i_hi] if bins[i_hi] > 0 else 1.0
    out["upper"] = cfg.min_mult + (i_hi + 1) * bw - np.clip(ratio_r, 0, 1) * bw

    out["w"] = max(out["upper"] - out["lower"], 1e-6)

    # 비대칭도: m 기준 위/아래 질량 불균형 (-1 ~ +1)
    above = bins[mids > m].sum()
    below = bins[mids <= m].sum()
    out["s"] = float((above - below) / total)
    return out


# ----------------------------------------------------------------------
# Main feature pipeline
# ----------------------------------------------------------------------
def compute_features(df: pd.DataFrame, cfg: FlatChartConfig | None = None) -> pd.DataFrame:
    """OHLCV DataFrame -> 피처 DataFrame.

    df 필수 컬럼: open, high, low, close. volume은 없으면 time-weight로 대체.
    인덱스는 DatetimeIndex 권장.
    """
    cfg = cfg or FlatChartConfig()
    n = len(df)

    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    if "volume" in df.columns:
        v = np.nan_to_num(df["volume"].to_numpy(float), nan=0.0)
    else:
        v = np.zeros(n)

    if cfg.weight_mode == "time":
        weight = np.ones(n)
    elif cfg.weight_mode == "volume":
        weight = v.copy()
    else:  # volume_fallback (Pine 기본)
        weight = np.where(v > 0, v, 1.0)

    cycle = ema(c, cfg.cycle_len) * cfg.cycle_scale
    valid_cycle = np.isfinite(cycle) & (cycle > 0)

    # 봉별 배수 구간 (cycle은 해당 봉 시점 값 사용 -> 제거 시에도 동일 값으로 정확히 제거)
    lo_mult = np.where(valid_cycle, l / cycle, np.nan)
    hi_mult = np.where(valid_cycle, h / cycle, np.nan)
    cl_mult = np.where(valid_cycle, c / cycle, np.nan)

    fast_bins = np.zeros(cfg.n_bins)
    slow_bins = np.zeros(cfg.n_bins)

    cols = [
        "m_fast", "w_fast", "s_fast", "dc",
        "m_slow", "lower_mult", "upper_mult",
        "mult_close", "cycle",
    ]
    out = np.full((n, len(cols)), np.nan)

    contribs: list = [None] * n  # 제거를 위해 각 봉의 기여분 캐시

    for i in range(n):
        contrib = _range_contribution(lo_mult[i], hi_mult[i], weight[i], cfg)
        contribs[i] = contrib

        # add current bar
        if contrib is not None:
            i0, i1, vals = contrib
            fast_bins[i0 : i1 + 1] += vals
            slow_bins[i0 : i1 + 1] += vals

        # subtract bars leaving each window
        j = i - cfg.fast_window
        if j >= 0 and contribs[j] is not None:
            i0, i1, vals = contribs[j]
            fast_bins[i0 : i1 + 1] = np.clip(fast_bins[i0 : i1 + 1] - vals, 0.0, None)
        k = i - cfg.slow_window
        if k >= 0 and contribs[k] is not None:
            i0, i1, vals = contribs[k]
            slow_bins[i0 : i1 + 1] = np.clip(slow_bins[i0 : i1 + 1] - vals, 0.0, None)
            contribs[k] = None  # 메모리 해제

        # 워밍업: fast window가 다 차기 전엔 피처 생략
        if i < cfg.fast_window or not valid_cycle[i]:
            continue

        fstats = _profile_stats(fast_bins, cfg)
        sstats = _profile_stats(slow_bins, cfg)
        if not np.isfinite(fstats["m"]):
            continue

        # dc: 이번 봉 유입 질량 무게중심 - m_fast (질량 유입 방향)
        dc = np.nan
        if contrib is not None:
            i0, i1, vals = contrib
            wsum = vals.sum()
            if wsum > 0:
                c_in = float((vals * cfg.bin_mids[i0 : i1 + 1]).sum() / wsum)
                dc = c_in - fstats["m"]

        out[i] = [
            fstats["m"], fstats["w"], fstats["s"], dc,
            sstats["m"], fstats["lower"], fstats["upper"],
            cl_mult[i], cycle[i],
        ]

    feats = pd.DataFrame(out, columns=cols, index=df.index)
    feats["lower_price"] = feats["lower_mult"] * feats["cycle"]
    feats["upper_price"] = feats["upper_mult"] * feats["cycle"]
    feats["open"] = o
    feats["close"] = c
    return feats


# ----------------------------------------------------------------------
# 빠른 자가 테스트
# ----------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 2000
    ret = rng.normal(0, 0.01, n)
    close = 100 * np.exp(np.cumsum(ret))
    high = close * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.004, n)))
    open_ = np.roll(close, 1); open_[0] = close[0]
    vol = rng.lognormal(10, 1, n)
    df = pd.DataFrame(
        dict(open=open_, high=high, low=low, close=close, volume=vol),
        index=pd.date_range("2024-01-01", periods=n, freq="1h"),
    )
    f = compute_features(df)
    print(f.dropna().describe().T[["mean", "std", "min", "max"]])
