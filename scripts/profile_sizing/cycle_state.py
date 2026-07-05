"""SMA200 멀티플 × 기울기 각도 2D 상태공간 (profile_plan/profile_plan.txt).

지표 스펙(플랜 고정 파라미터, 인자로만 노출 — 스윕 금지):
    cycle      = SMA(close, 200)
    multiple   = close / cycle
    slope_pct  = (cycle - cycle.shift(5)) / cycle.shift(5)
    slope_z    = slope_pct / rolling_std(slope_pct, 252)
    angle_deg  = degrees(arctan(slope_z))          # z=1 → 45°

bin 경계는 expanding window percentile로 산출한다(전 기간 percentile 금지).
시점 t의 경계는 t까지(포함)의 데이터만 사용 → look-ahead 없음.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_cycle_state(
    close: pd.Series,
    *,
    cycle_len: int = 200,
    slope_lookback: int = 5,
    z_window: int = 252,
    grid_bins: int = 3,
    min_bin_history: int = 252,
) -> pd.DataFrame:
    """멀티플/기울기 지표 + expanding percentile 그리드 상태 프레임.

    반환 컬럼: base_cycle, multiple, slope_pct, slope_z, angle_deg,
    mult_bin, angle_bin (0..grid_bins-1, 유효 이전 구간은 -1),
    mult_pct, angle_pct (expanding percentile rank 0..1 — bin의 연속 버전).
    """
    s = pd.Series(np.asarray(close, dtype=float), index=close.index)
    cycle = s.rolling(cycle_len, min_periods=cycle_len).mean()
    cycle = cycle.where(cycle > 0.0)

    prev = cycle.shift(slope_lookback)
    slope_pct = (cycle - prev) / prev
    slope_sd = slope_pct.rolling(z_window, min_periods=z_window).std()
    slope_z = slope_pct / slope_sd.where(slope_sd > 0.0)
    angle_deg = np.degrees(np.arctan(slope_z))

    out = pd.DataFrame(index=s.index)
    out["base_cycle"] = cycle
    out["multiple"] = s / cycle
    out["slope_pct"] = slope_pct
    out["slope_z"] = slope_z
    out["angle_deg"] = angle_deg
    out["mult_bin"] = expanding_percentile_bin(
        out["multiple"], grid_bins, min_bin_history
    )
    out["angle_bin"] = expanding_percentile_bin(
        out["angle_deg"], grid_bins, min_bin_history
    )
    out["mult_pct"] = expanding_percentile_rank(out["multiple"], min_bin_history)
    out["angle_pct"] = expanding_percentile_rank(out["angle_deg"], min_bin_history)
    return out


def expanding_percentile_bin(
    values: pd.Series, n_bins: int, min_history: int
) -> pd.Series:
    """expanding percentile 경계로 n분위 bin index(0..n-1)를 매긴다.

    시점 t의 경계 = values[:t+1]의 분위수(인과적). 유효 표본이
    ``min_history`` 미만이거나 값이 NaN이면 -1.
    """
    v = pd.Series(np.asarray(values, dtype=float), index=values.index)
    qs = [k / n_bins for k in range(1, n_bins)]
    edges = [
        v.expanding(min_periods=min_history).quantile(q) for q in qs
    ]
    out = np.full(len(v), -1, dtype=int)
    val = v.to_numpy()
    edge_mat = np.column_stack([e.to_numpy() for e in edges])
    valid = np.isfinite(val) & np.isfinite(edge_mat).all(axis=1)
    # 경계 초과 개수 = bin index (경계 포함: value <= edge → 아래 bin)
    out[valid] = (val[valid, None] > edge_mat[valid]).sum(axis=1)
    return pd.Series(out, index=v.index)


def expanding_percentile_rank(values: pd.Series, min_history: int) -> pd.Series:
    """expanding window 안에서 현재 값의 percentile rank(0..1, 인과적).

    ``expanding_percentile_bin``의 연속 버전 — 시점 t의 값은 values[:t+1]만
    사용한다(look-ahead 없음). 유효 표본이 ``min_history`` 미만이면 NaN.
    2D 상태 필드에 각 봉의 위치를 점으로 찍을 때 좌표로 쓴다.
    """
    v = pd.Series(np.asarray(values, dtype=float), index=values.index)
    return v.expanding(min_periods=min_history).rank(pct=True)


def zigzag_moves(
    close: pd.Series, *, threshold: float = 0.3, reversal: float | None = None,
) -> pd.DataFrame:
    """종가 zigzag로 큰 상승/하락 leg를 라벨한다 (시각화 전용 라벨).

    현재 방향의 극값 대비 ``reversal``(기본 threshold/2) 이상 되돌리면 그
    극값을 pivot으로 확정하고 방향을 뒤집는다. 그렇게 나눈 leg 중 진폭이
    ``threshold`` 이상인 것만 라벨한다 — reversal < threshold라서 작은 파동
    leg는 라벨 없이 남고(전 구간이 칠해지는 것 방지), 큰 파동만 선으로 이어진다.
    pivot 확정에 미래 봉을 쓰므로 **신호에 사용 금지**(forward_returns와 같은
    라벨 전용 함수).

    반환 컬럼: move_dir(+1 상승 / -1 하락 / 0 없음), move_id(leg 번호, 없음 -1).
    인접 leg가 공유하는 pivot 봉은 새 leg에 귀속된다.
    """
    s = pd.Series(np.asarray(close, dtype=float), index=close.index)
    px = s.to_numpy()
    move_dir = np.zeros(len(s), dtype=int)
    move_id = np.full(len(s), -1, dtype=int)
    finite = np.flatnonzero(np.isfinite(px))
    rev = 0.5 * threshold if reversal is None else float(reversal)
    if len(finite) < 2 or threshold <= 0.0 or rev <= 0.0:
        return pd.DataFrame(
            {"move_dir": move_dir, "move_id": move_id}, index=s.index
        )

    pivots: list[int] = []
    direction = 0
    ext_i, ext_p = finite[0], px[finite[0]]        # 현재 방향의 극값
    lo_i, lo_p = ext_i, ext_p                      # 방향 미확정 구간용 최저/최고
    hi_i, hi_p = ext_i, ext_p
    for i in finite[1:]:
        p = px[i]
        if direction == 0:
            if p > hi_p:
                hi_i, hi_p = i, p
            if p < lo_p:
                lo_i, lo_p = i, p
            if p <= hi_p * (1.0 - rev):
                # 시작~첫 고점 구간도 되돌림 기준을 넘으면 leg 후보로 포함
                if lo_i < hi_i and hi_p >= lo_p * (1.0 + rev):
                    pivots.append(lo_i)
                pivots.append(hi_i)
                direction, ext_i, ext_p = -1, i, p
            elif p >= lo_p * (1.0 + rev):
                if hi_i < lo_i and hi_p >= lo_p * (1.0 + rev):
                    pivots.append(hi_i)
                pivots.append(lo_i)
                direction, ext_i, ext_p = 1, i, p
        elif direction > 0:
            if p > ext_p:
                ext_i, ext_p = i, p
            elif p <= ext_p * (1.0 - rev):
                pivots.append(ext_i)
                direction, ext_i, ext_p = -1, i, p
        else:
            if p < ext_p:
                ext_i, ext_p = i, p
            elif p >= ext_p * (1.0 + rev):
                pivots.append(ext_i)
                direction, ext_i, ext_p = 1, i, p
    # 마지막 미완성 leg: 현재 극값까지를 leg 후보로 포함(극값 이후 되돌림 제외).
    if pivots and ext_i > pivots[-1]:
        pivots.append(ext_i)

    leg = 0
    for k in range(len(pivots) - 1):
        a, b = pivots[k], pivots[k + 1]
        amp = px[b] / px[a] - 1.0
        if abs(amp) < threshold:
            continue                       # 작은 파동 leg는 라벨하지 않는다
        move_dir[a:b + 1] = 1 if amp >= 0.0 else -1
        move_id[a:b + 1] = leg
        leg += 1
    return pd.DataFrame({"move_dir": move_dir, "move_id": move_id}, index=s.index)


def forward_returns(close: pd.Series, horizons: tuple[int, ...]) -> pd.DataFrame:
    """h봉 뒤 종가 수익률(shift(-h) — 라벨 전용, 신호에 쓰면 look-ahead)."""
    s = pd.Series(np.asarray(close, dtype=float), index=close.index)
    out = pd.DataFrame(index=s.index)
    for h in horizons:
        out[f"fwd_{h}"] = s.shift(-int(h)) / s - 1.0
    return out


def newey_west_mean_test(x: np.ndarray, lag: int) -> tuple[float, float, float]:
    """overlapping 표본 평균의 (mean, NW se, t-stat). 표본<8이면 NaN."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 8:
        return float("nan"), float("nan"), float("nan")
    mu = float(x.mean())
    e = x - mu
    lag = max(0, min(int(lag), n - 1))
    s = float(e @ e) / n
    for l in range(1, lag + 1):
        w = 1.0 - l / (lag + 1.0)
        s += 2.0 * w * float(e[l:] @ e[:-l]) / n
    se = float(np.sqrt(max(s, 0.0) / n))
    t = mu / se if se > 0.0 else float("nan")
    return mu, se, t
