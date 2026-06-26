"""heatmap2 HVN 지지/저항 → 매수/매도 기대값 게이트 (yoon1i).

heatmap2의 `rolling_sr_levels`(현재가에 인접한 고볼륨 노드 지지/저항)로 종목별
**상방여지/하방위험 비율**을 구해 [g_min, 1] 게이트로 yoon1b 점수에 곱한다(블렌드).

정규화 기대값:
    EV = (저항 − 종가) / (저항 − 지지)  ∈ [0, 1]
- 지지 근처(상방 큼)  → EV↑ → 게이트 열림(매수 기대 반영, 비중 유지)
- 저항 근처(상방 작음) → EV↓ → 게이트 닫힘(매도 기대 반영, 비중 억제)
한쪽 노드만 있으면: 위 막힘 없음=상방 무제한(EV=1), 아래 받침 없음(EV=0).

무누수: rolling_sr_levels가 각 t에서 t 이하만 사용 + 엔진 simulate_portfolio가
점수를 shift(1). warmup/노드 부재는 1.0(불변). 기본 off(yoon1b 동작 불변).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPTS = Path(__file__).resolve().parents[1]  # scripts/
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from volume_profile import rolling_sr_levels  # noqa: E402


def sr_gate(ohlc: pd.DataFrame, gate_cfg: dict) -> pd.Series:
    """OHLCV → [g_min, 1.0] 지지/저항 기대값 게이트 (ohlc.index 정렬)."""
    g_min = float(gate_cfg.get("g_min", 0.5))
    levels = rolling_sr_levels(
        ohlc,
        lookback=int(gate_cfg.get("lookback", 120)),
        bins=int(gate_cfg.get("profile_bins", 80)),
        scale=str(gate_cfg.get("price_scale", "log")),
        top_n=int(gate_cfg.get("node_top_n", 4)),
        min_strength=float(gate_cfg.get("node_min_strength", 0.3)),
        min_gap_bins=int(gate_cfg.get("node_min_gap_bins", 3)),
        va_pct=float(gate_cfg.get("va_pct", 0.70)),
        cumulative=bool(gate_cfg.get("cumulative", False)),
    )
    close = ohlc["close"].astype(float).to_numpy()
    sup = levels["val"].to_numpy(float)   # 지지
    res = levels["vah"].to_numpy(float)   # 저항
    n = len(close)

    ev = np.full(n, np.nan)
    both = np.isfinite(sup) & np.isfinite(res) & (res > sup)
    ev[both] = np.clip(
        (res[both] - close[both]) / (res[both] - sup[both]), 0.0, 1.0
    )
    ev[np.isfinite(sup) & ~np.isfinite(res)] = 1.0  # 위 막힘 없음 = 상방 무제한
    ev[np.isfinite(res) & ~np.isfinite(sup)] = 0.0  # 아래 받침 없음

    gate = g_min + (1.0 - g_min) * ev
    out = pd.Series(gate, index=ohlc.index).fillna(1.0)  # warmup/둘다 부재 = 불변
    return out.rename("sr_gate")
