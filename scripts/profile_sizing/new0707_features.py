"""new0707 사전검증 피처: 최근 10봉 SMA10 기울기 가속도 × 거래량 서지 2D 상태.

가설(2026-07-07 설계 대화, angle.txt/yoon_profile 교훈 반영):
    sma10       = SMA(close, 10)
    slope_pct   = sma10.pct_change(1)                       # 1봉 기울기(%)
    accel_pct   = slope_pct - slope_pct.shift(accel_lookback)  # 기울기의 변화량(가속도)
    accel_z     = accel_pct / rolling_std(accel_pct, z_window)   # 고정 스케일 금지(angle.txt 반례) → 자체 rolling std로 정규화
    accel_deg   = degrees(arctan(accel_z))                  # 부호 보존(상승/하락 가속 구분)

    vol_ratio   = volume / rolling_mean(volume, vol_base_window)   # 거래량 서지 배율
    vol_z       = (vol_ratio - 1.0) / rolling_std(vol_ratio - 1.0, z_window)

bin 경계는 cycle_state.py와 동일하게 expanding percentile(무누수, 인과적)로 산출한다.
z_window/vol_base_window은 200봉 계열(z_window=252)보다 훨씬 짧게 잡는다 — 10봉짜리
빠른 신호라 최근 변동성 레짐에 더 빨리 맞춰져야 한다(스터디로 확정 전 잠정값).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .cycle_state import expanding_percentile_bin, expanding_percentile_rank


def compute_new0707_state(
    close: pd.Series,
    volume: pd.Series,
    *,
    sma_len: int = 10,
    accel_lookback: int = 5,
    vol_base_window: int = 60,
    z_window: int = 60,
    grid_bins: int = 3,
    min_bin_history: int = 120,
) -> pd.DataFrame:
    """SMA10 기울기 가속도 × 거래량 서지 2D 상태공간 (cycle_state.py와 동형 구조).

    반환 컬럼: sma10, slope_pct, accel_pct, accel_z, accel_deg, vol_ratio, vol_z,
    accel_bin, volsurge_bin(0..grid_bins-1, 유효 이전 구간은 -1),
    accel_rank, volsurge_rank(expanding percentile rank 0..1 — bin의 연속 버전).
    """
    px = pd.Series(np.asarray(close, dtype=float), index=close.index)
    vol = pd.Series(np.asarray(volume, dtype=float), index=close.index)

    sma10 = px.rolling(sma_len, min_periods=sma_len).mean()
    slope_pct = sma10.pct_change(1)
    accel_pct = slope_pct - slope_pct.shift(accel_lookback)
    accel_sd = accel_pct.rolling(z_window, min_periods=z_window).std()
    accel_z = accel_pct / accel_sd.where(accel_sd > 0.0)
    accel_deg = np.degrees(np.arctan(accel_z))

    vol_base = vol.rolling(vol_base_window, min_periods=vol_base_window).mean()
    vol_ratio = vol / vol_base.where(vol_base > 0.0)
    vol_dev = vol_ratio - 1.0
    vol_sd = vol_dev.rolling(z_window, min_periods=z_window).std()
    vol_z = vol_dev / vol_sd.where(vol_sd > 0.0)

    out = pd.DataFrame(index=px.index)
    out["sma10"] = sma10
    out["slope_pct"] = slope_pct
    out["accel_pct"] = accel_pct
    out["accel_z"] = accel_z
    out["accel_deg"] = accel_deg
    out["vol_ratio"] = vol_ratio
    out["vol_z"] = vol_z
    out["accel_bin"] = expanding_percentile_bin(accel_deg, grid_bins, min_bin_history)
    out["volsurge_bin"] = expanding_percentile_bin(vol_z, grid_bins, min_bin_history)
    out["accel_rank"] = expanding_percentile_rank(accel_deg, min_bin_history)
    out["volsurge_rank"] = expanding_percentile_rank(vol_z, min_bin_history)
    return out
