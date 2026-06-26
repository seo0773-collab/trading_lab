"""rolling / cumulative profile + percentile (profile_plan.txt §4·§5).

가격 자체가 아니라 cycle_multiple (price / base_cycle) 을 bin에 누적한다.
- rolling profile: 최근 N봉만 유지(window 밖 봉 제거).
- cumulative profile: 시작부터 현재 봉까지 누적(제거 없음 → 과거·현재만 사용 = 무누수).

각 봉의 분포에서 다음을 산출:
- cumulative_percentile: 현재 cm_close 가 누적분포에서 차지하는 하위 누적비율(0~1).
- cumulative_mid_50 / rolling_mid_50: weighted median에 해당하는 multiple.
- cumulative_lower_percentile: 아래에서 percentile_value%만큼 누적된 위치의 multiple.

구현은 (bars × bins) 기여행렬을 만들어 cumsum으로 rolling/cumulative를 한 번에 얻는다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Profile


def _bar_weights(raw: pd.DataFrame, mode: str) -> np.ndarray:
    n = len(raw)
    if mode == "time":
        return np.ones(n, dtype=float)
    vol = np.asarray(raw["volume"], dtype=float) if "volume" in raw else np.ones(n)
    vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)
    if mode == "volume_fallback":
        vol = np.where(vol > 0.0, vol, 1.0)
    return vol


def _contrib_matrix(cycle: pd.DataFrame, cfg: Profile, weights: np.ndarray) -> np.ndarray:
    """봉마다 bin별 기여 weight (bars × bins). base_cycle 무효봉은 0행."""
    edges = np.linspace(cfg.min_mult, cfg.max_mult, cfg.bin_count + 1)
    lo_e, hi_e = edges[:-1], edges[1:]

    cm_low = np.asarray(cycle["cm_low"], dtype=float)
    cm_high = np.asarray(cycle["cm_high"], dtype=float)
    cm_open = np.asarray(cycle["cm_open"], dtype=float)
    cm_close = np.asarray(cycle["cm_close"], dtype=float)

    valid = np.isfinite(cm_close) & np.isfinite(cm_high) & np.isfinite(cm_low)
    bars = len(cm_close)
    contrib = np.zeros((bars, cfg.bin_count), dtype=float)

    mode = cfg.accumulation_mode.lower()
    if mode in ("range_uniform", "range_close"):
        a = np.clip(np.minimum(cm_low, cm_high), cfg.min_mult, cfg.max_mult)
        b = np.clip(np.maximum(cm_low, cm_high), cfg.min_mult, cfg.max_mult)
        span = np.clip(b - a, 1e-9, None)
        overlap = np.clip(
            np.minimum(b[:, None], hi_e[None, :]) - np.maximum(a[:, None], lo_e[None, :]),
            0.0, None,
        )
        frac = overlap / span[:, None]
        contrib = frac * weights[:, None]
        if mode == "range_close":
            # 절반은 range uniform, 절반은 close bin에 집중.
            contrib *= 0.5
            close_idx = _bin_index(cm_close, edges, cfg.bin_count)
            for i in range(bars):
                if valid[i] and close_idx[i] >= 0:
                    contrib[i, close_idx[i]] += 0.5 * weights[i]
    elif mode == "ohlc":
        for arr in (cm_open, cm_high, cm_low, cm_close):
            idx = _bin_index(arr, edges, cfg.bin_count)
            for i in range(bars):
                if valid[i] and idx[i] >= 0:
                    contrib[i, idx[i]] += 0.25 * weights[i]
    else:
        raise ValueError(f"지원하지 않는 accumulation_mode: {cfg.accumulation_mode}")

    contrib[~valid] = 0.0
    return contrib


def _bin_index(mult: np.ndarray, edges: np.ndarray, bin_count: int) -> np.ndarray:
    idx = np.searchsorted(edges, mult, side="right") - 1
    out_of_range = ~np.isfinite(mult) | (mult < edges[0]) | (mult > edges[-1])
    idx = np.clip(idx, 0, bin_count - 1)
    idx[out_of_range] = -1
    return idx


def _percentile_at(profile: np.ndarray, edges: np.ndarray, mult: np.ndarray) -> np.ndarray:
    """각 봉에서 query multiple 의 하위 누적비율(0~1). 분포 비어있으면 NaN."""
    bars, bins = profile.shape
    total = profile.sum(axis=1)
    cum = np.cumsum(profile, axis=1)
    idx = np.clip(np.searchsorted(edges, mult, side="right") - 1, 0, bins - 1)
    rows = np.arange(bars)
    below = np.where(idx > 0, cum[rows, np.maximum(idx - 1, 0)], 0.0)
    width = edges[idx + 1] - edges[idx]
    frac = np.clip((mult - edges[idx]) / np.where(width > 0, width, 1.0), 0.0, 1.0)
    in_bin = profile[rows, idx] * frac
    pct = np.divide(below + in_bin, total, out=np.full(bars, np.nan), where=total > 0)
    return np.clip(pct, 0.0, 1.0)


def _mult_at_quantile(profile: np.ndarray, edges: np.ndarray, q: float) -> np.ndarray:
    """각 봉에서 하위 누적비율 q에 도달하는 multiple (weighted quantile)."""
    bars, bins = profile.shape
    total = profile.sum(axis=1)
    cum = np.cumsum(profile, axis=1)
    target = q * total
    reached = cum >= target[:, None]
    idx = np.where(reached.any(axis=1), reached.argmax(axis=1), bins - 1)
    rows = np.arange(bars)
    prev_cum = np.where(idx > 0, cum[rows, np.maximum(idx - 1, 0)], 0.0)
    bin_w = profile[rows, idx]
    need = target - prev_cum
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(bin_w > 0, need / bin_w, 0.0)
    frac = np.clip(np.nan_to_num(ratio, nan=0.0), 0.0, 1.0)
    mult = edges[idx] + frac * (edges[idx + 1] - edges[idx])
    return np.where(total > 0, mult, np.nan)


def _poc_va(
    profile: np.ndarray, edges: np.ndarray, va_pct: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """각 봉의 프로파일에서 POC와 Value Area 상·하단 multiple (yoon1h 매물대 사이징).

    - POC(Point of Control) = volume 최대 bin 중심 multiple.
    - VA = POC에서 좌우로 *더 큰 이웃*을 흡수하며 누적 volume이 total*va_pct 에
      도달할 때까지 확장한 구간 → 하단 edge=VAL, 상단 edge=VAH.

    프로파일은 cumulative/rolling 어느 쪽이든 t 이하 데이터만 누적한 행렬이므로
    룩어헤드가 없다. 빈 분포(total=0, warmup 포함)는 NaN.
    """
    bars, bins = profile.shape
    centers = (edges[:-1] + edges[1:]) / 2.0
    total = profile.sum(axis=1)
    poc_idx = profile.argmax(axis=1)
    poc = np.where(total > 0, centers[poc_idx], np.nan)
    val = np.full(bars, np.nan)
    vah = np.full(bars, np.nan)
    for i in range(bars):
        if total[i] <= 0:
            continue
        row = profile[i]
        target = total[i] * va_pct
        lo = hi = int(poc_idx[i])
        acc = row[lo]
        while acc < target and (lo > 0 or hi < bins - 1):
            left = row[lo - 1] if lo > 0 else -1.0
            right = row[hi + 1] if hi < bins - 1 else -1.0
            if right >= left:
                hi += 1
                acc += row[hi]
            else:
                lo -= 1
                acc += row[lo]
        val[i] = edges[lo]
        vah[i] = edges[hi + 1]
    return poc, vah, val


def _va_position(
    cm_close: np.ndarray, poc: np.ndarray, vah: np.ndarray, val: np.ndarray
) -> np.ndarray:
    """현재가의 VA 대비 위치(0~1, 연속 외삽). percentile 대체 사이징 입력.

    POC=0.5 중립, VAL→0(싸다), VAH→1(비싸다). VA 밖 이탈은 같은 방향 기울기로
    **외삽**(이탈 강도 반영) 후 [0,1] 클립. POC가 VA 가운데가 아닐 수 있어 상·하
    반폭을 따로 정규화한다. percentile 자리에 들어가 동일 bucket 가중을 통과한다.
    """
    cm = np.asarray(cm_close, dtype=float)
    up = np.maximum(vah - poc, 1e-9)
    dn = np.maximum(poc - val, 1e-9)
    raw = np.where(
        cm >= poc, 0.5 + 0.5 * (cm - poc) / up, 0.5 - 0.5 * (poc - cm) / dn
    )
    pos = np.clip(raw, 0.0, 1.0)
    valid = np.isfinite(poc) & np.isfinite(cm)
    return np.where(valid, pos, np.nan)


def compute_profile(cycle: pd.DataFrame, raw: pd.DataFrame, cfg: Profile) -> pd.DataFrame:
    """봉별 percentile/mid_50 시리즈 프레임을 cycle 인덱스로 반환."""
    weights = _bar_weights(raw, cfg.weight_mode.lower())
    contrib = _contrib_matrix(cycle, cfg, weights)
    edges = np.linspace(cfg.min_mult, cfg.max_mult, cfg.bin_count + 1)

    cum_profile = np.cumsum(contrib, axis=0)
    window = max(1, cfg.rolling_window)
    roll_profile = cum_profile.copy()
    if window < len(contrib):
        roll_profile[window:] = cum_profile[window:] - cum_profile[:-window]

    cm_close = np.asarray(cycle["cm_close"], dtype=float)
    pv = cfg.percentile_value / 100.0

    out = pd.DataFrame(index=cycle.index)
    out["cumulative_percentile"] = _percentile_at(cum_profile, edges, cm_close)
    out["cumulative_mid_50"] = _mult_at_quantile(cum_profile, edges, 0.5)
    out["rolling_mid_50"] = _mult_at_quantile(roll_profile, edges, 0.5)
    out["cumulative_lower_percentile"] = _mult_at_quantile(cum_profile, edges, pv)
    out["cumulative_upper_percentile"] = _mult_at_quantile(cum_profile, edges, 1.0 - pv)
    if cfg.compute_va:
        # 매물대(VA)는 "최근 거래 밀집" 개념 → rolling profile 사용.
        poc, vah, val = _poc_va(roll_profile, edges, cfg.va_pct)
        out["poc"] = poc
        out["vah"] = vah
        out["val"] = val
        out["va_position"] = _va_position(cm_close, poc, vah, val)
    return out
