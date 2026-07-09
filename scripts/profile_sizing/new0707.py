"""new0707: SMA10 기울기 가속도 × 거래량 서지 2D 상태 가중치 사이징 (long-only 단일종목).

2026-07-07 설계·검증 대화의 결론을 가중치 행렬로 옮긴다. 프렉탈(유사 10봉 패턴
투영/자기연장) 축은 스터디에서 기각(analog IC 부호 불일치, self는 이상치
구동으로 재현 불가 — reports/profile_sizing/new0707_study.json)되어 폐기하고,
accel×vol 상태그리드만 사용한다.

가중치 근거(BTC/USDT 일봉 2017-08~2026-07, new0707_study.json):
  27칸(9셀×3horizon) 전부 forward return 부호는 양수(BTC 강세 드리프트 지배) —
  숏/현금 대상 셀 없음. FDR 유의 셀은 0/27 — **개별 셀 유의성은 미확정**이며,
  아래는 h=5/10/20 평균 fwd return의 상대 순위(3horizon 전부 방향 일관)로만
  정한 tilt다:
    (accel=0,vol=0) 감속+한산(눌림목류) 최고 → 1.00
    (accel=2,vol=2) 가속+거래량 확인(모멘텀+볼륨) 2위 → 0.925
    (accel=0,vol=2) 감속+거래량 급증(소진/캐피튤레이션 추정) 최저 → 0.40
  두 대각(0,0)/(2,2)이 상위, "가속상태와 거래량이 불일치"하는 셀들이 하위 —
  방향은 일관되나 FDR 미달이므로 live_eligible=False, 페이퍼/추가 검증 전 보류.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import config_from_dict
from .engine import (
    buy_hold_equity, equity_from_returns, lot_trades, performance,
    portfolio_returns,
)
from .new0707_features import compute_new0707_state

# 행=accel_bin(0=최대 감속), 열=volsurge_bin(0=한산). new0707_study.json 3horizon
# 평균 fwd return 순위(1~9)를 0.4~1.0 구간에 선형 매핑(스윕 금지, 플랜 고정값).
DEFAULT_STATE_WEIGHTS: tuple[tuple[float, ...], ...] = (
    (1.000, 0.700, 0.400),   # 감속: 한산(최고) → 보통 → 거래량 급증(최저)
    (0.550, 0.775, 0.475),   # 중립 가속
    (0.625, 0.850, 0.925),   # 가속: 한산 → 보통 → 거래량 확인(2위)
)


@dataclass(frozen=True)
class New0707Config:
    sma_len: int = 10
    accel_lookback: int = 5
    vol_base_window: int = 60
    z_window: int = 60
    grid_bins: int = 3
    min_bin_history: int = 120
    state_weights: tuple[tuple[float, ...], ...] = DEFAULT_STATE_WEIGHTS
    max_weight: float = 1.0
    min_weight: float = 0.0
    rebalance_threshold: float = 0.05
    max_trade_weight_per_bar: float = 0.25
    warmup_weight: float = 0.0


def new0707_config(raw: dict[str, Any]) -> New0707Config:
    base = New0707Config()
    cy = raw.get("cycle") or {}
    sz = raw.get("sizing") or {}
    weights = raw.get("state_weights")
    if weights:
        weights = tuple(tuple(float(w) for w in row) for row in weights)
    else:
        weights = base.state_weights
    grid = int(cy.get("grid_bins", base.grid_bins))
    if len(weights) != grid or any(len(r) != grid for r in weights):
        raise ValueError("state_weights는 grid_bins × grid_bins 행렬이어야 합니다")
    return New0707Config(
        sma_len=int(cy.get("sma_len", base.sma_len)),
        accel_lookback=int(cy.get("accel_lookback", base.accel_lookback)),
        vol_base_window=int(cy.get("vol_base_window", base.vol_base_window)),
        z_window=int(cy.get("z_window", base.z_window)),
        grid_bins=grid,
        min_bin_history=int(cy.get("min_bin_history", base.min_bin_history)),
        state_weights=weights,
        max_weight=float(sz.get("max_weight", base.max_weight)),
        min_weight=float(sz.get("min_weight", base.min_weight)),
        rebalance_threshold=float(
            sz.get("rebalance_threshold", base.rebalance_threshold)
        ),
        max_trade_weight_per_bar=float(
            sz.get("max_trade_weight_per_bar", base.max_trade_weight_per_bar)
        ),
        warmup_weight=float(sz.get("warmup_weight", base.warmup_weight)),
    )


def state_target_weights(state: pd.DataFrame, ncfg: New0707Config) -> pd.Series:
    """(accel_bin, volsurge_bin) → config 가중치. 상태 미확정(-1) 구간은 warmup_weight."""
    w = np.asarray(ncfg.state_weights, dtype=float)
    ab = state["accel_bin"].to_numpy()
    vb = state["volsurge_bin"].to_numpy()
    valid = (ab >= 0) & (vb >= 0)
    out = np.full(len(state), ncfg.warmup_weight, dtype=float)
    out[valid] = w[ab[valid], vb[valid]]
    out = np.clip(out, ncfg.min_weight, ncfg.max_weight)
    return pd.Series(out, index=state.index, name="target_weight")


def gradual_weights(target: pd.Series, ncfg: New0707Config) -> pd.Series:
    """rebalance threshold + 봉당 체결 상한으로 목표 비중을 점진 추종."""
    actual = np.zeros(len(target), dtype=float)
    prev = 0.0
    step_cap = float(ncfg.max_trade_weight_per_bar)
    for i, tgt in enumerate(np.asarray(target, dtype=float)):
        diff = tgt - prev
        if abs(diff) < float(ncfg.rebalance_threshold):
            new = prev
        else:
            new = prev + float(np.clip(diff, -step_cap, step_cap))
        actual[i] = float(np.clip(new, ncfg.min_weight, ncfg.max_weight))
        prev = actual[i]
    return pd.Series(actual, index=target.index, name="actual_weight")


def run_new0707_pipeline(raw: pd.DataFrame, raw_cfg: dict[str, Any]) -> dict:
    psc = config_from_dict(raw_cfg)   # costs / interval / split fraction 재사용
    ncfg = new0707_config(raw_cfg)

    close = pd.Series(np.asarray(raw["close"], dtype=float), index=raw.index)
    volume = pd.Series(np.asarray(raw["volume"], dtype=float), index=raw.index)
    state = compute_new0707_state(
        close, volume,
        sma_len=ncfg.sma_len,
        accel_lookback=ncfg.accel_lookback,
        vol_base_window=ncfg.vol_base_window,
        z_window=ncfg.z_window,
        grid_bins=ncfg.grid_bins,
        min_bin_history=ncfg.min_bin_history,
    )
    target = state_target_weights(state, ncfg)
    actual = gradual_weights(target, ncfg)

    port_ret = portfolio_returns(close, actual, psc)
    equity = equity_from_returns(port_ret)
    bnh = buy_hold_equity(close)

    regime = (
        "A" + state["accel_bin"].astype(str) + "V" + state["volsurge_bin"].astype(str)
    ).where((state["accel_bin"] >= 0) & (state["volsurge_bin"] >= 0), "WARMUP")
    trades = lot_trades(close, actual, regime, psc)

    forecast = pd.DataFrame(index=raw.index)
    for col in ("open", "high", "low", "close"):
        if col in raw:
            forecast[col] = np.asarray(raw[col], dtype=float)
    if "volume" in raw:
        forecast["volume"] = np.asarray(raw["volume"], dtype=float)
    forecast = forecast.join(state)
    forecast["target_weight"] = target
    forecast["actual_weight"] = actual

    return {
        "forecast": forecast,
        "state_label": regime,
        "port_ret": port_ret,
        "equity": equity,
        "buy_hold": bnh,
        "trades": trades,
        "metrics": performance(equity, port_ret, psc.interval),
    }
