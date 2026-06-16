"""base_cycle + cycle multiple (profile_plan.txt §3).

base_cycle = moving_average(close, length, type) * scale 기준선.
cycle_multiple_x = x / base_cycle 로 가격을 기준선 대비 배수로 정규화한다.
base_cycle 이 NaN/0 이하인 구간은 profile에서 제외하기 위해 NaN으로 남긴다.
미래 데이터를 쓰지 않는다(전부 과거·현재 봉만).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import BaseCycle


def moving_average(
    values: pd.Series, length: int, kind: str, volume: pd.Series | None = None
) -> pd.Series:
    s = pd.Series(np.asarray(values, dtype=float), index=values.index)
    kind = kind.upper()
    if length <= 1:
        return s
    if kind == "SMA":
        return s.rolling(length, min_periods=length).mean()
    if kind == "EMA":
        return s.ewm(span=length, adjust=False, min_periods=length).mean()
    if kind == "RMA":  # Wilder smoothing
        return s.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    if kind == "WMA":
        weights = np.arange(1, length + 1, dtype=float)
        denom = weights.sum()
        return s.rolling(length, min_periods=length).apply(
            lambda x: float(np.dot(x, weights) / denom), raw=True
        )
    if kind == "VWMA":
        if volume is None:
            raise ValueError("VWMA는 volume이 필요합니다")
        v = pd.Series(np.asarray(volume, dtype=float), index=values.index)
        num = (s * v).rolling(length, min_periods=length).sum()
        den = v.rolling(length, min_periods=length).sum()
        return num / den.replace(0.0, np.nan)
    raise ValueError(f"지원하지 않는 base_type: {kind}")


def compute_cycle(raw: pd.DataFrame, cfg: BaseCycle) -> pd.DataFrame:
    """OHLC + base_cycle + cycle_multiple_{open,high,low,close} 프레임."""
    close = pd.Series(np.asarray(raw["close"], dtype=float), index=raw.index)
    volume = raw["volume"] if "volume" in raw else None
    base = moving_average(close, cfg.length, cfg.type, volume=volume) * cfg.scale
    base = base.where(base > 0.0)  # 0/음수는 무효 → NaN

    out = pd.DataFrame(index=raw.index)
    for col in ("open", "high", "low", "close"):
        if col in raw:
            out[col] = np.asarray(raw[col], dtype=float)
    out["base_cycle"] = base
    for col in ("open", "high", "low", "close"):
        if col in raw:
            out[f"cm_{col}"] = np.asarray(raw[col], dtype=float) / base
    return out
