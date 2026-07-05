"""yoon_profile: 멀티플 × 기울기 2D 상태 가중치 사이징 (long-only 기본).

profile_plan/profile_plan.txt 검증(yoon_profile_study)에서 확인된 셀 구조를
가중치 행렬로 옮긴다. 상태 결정은 시점 t까지의 expanding percentile만 사용하고,
비중은 t+1 수익률에 적용(engine.portfolio_returns가 1봉 지연)하므로 무누수.

가중치 근거(BTC/USDT 일봉, reports/profile_sizing/yoon_profile_study.json):
  (0,0) 떨어지는 칼(전 horizon 음수, WF 일관 음수)      → 0.0
  (0,1) 눌림목 알파(fwd60 +28% FDR 유의, WF 3/3 양수)   → 1.0
  (2,2) 모멘텀 지속(fwd5/20 FDR 유의, WF 3/3 양수)      → 1.0
  (2,1) 고멀티플 기울기 이탈(fwd20 −3.5%)               → 축소

숏 확장(yoon_profile_1): config ``short.enabled`` 가 켜지면 음수 셀 가중치를
허용하고, 스퀴즈 가드(숏 에피소드 최저 종가 대비 +trail_pct 반등 시 에피소드
종료까지 숏 금지)를 적용한다. 근거: yoon_profile_squeeze_guard_study.md
(trail15만 전 구간 개선, ma20·fast_exit 기각). 기본 off → 기존 동작 불변.

경로 틸트(``path_tilt`` 블록, 기본 off): 같은 셀이라도 필드 진입 경로(인과적
필드 속도 vel = mult_pct.diff(vel_window)의 부호)에 따라 목표 비중을 오버라이드
한다. 근거: reports/profile_sizing/state_field_trend_analysis.md §6 (큰 하락
leg가 중간 행을 경유 — M1A1 상승 진입 fwd60 +21% vs 하강 진입 +7.6%) 및
yoon_profile_path_study.md 정식 검증.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import config_from_dict
from .cycle_state import compute_cycle_state
from .engine import (
    buy_hold_equity, equity_from_returns, performance, portfolio_returns,
)

# 행=mult_bin(0=저평가), 열=angle_bin(0=하락 기울기). 스터디 셀 부호/유의성 반영.
DEFAULT_STATE_WEIGHTS: tuple[tuple[float, ...], ...] = (
    (0.0, 1.0, 1.0),   # 저멀티플: 칼 회피 / 눌림목 / (희소 셀 → 눌림목 준용)
    (0.5, 0.8, 0.3),   # 중간
    (0.0, 0.3, 1.0),   # 고멀티플: (희소 셀 → 방어) / 감속 축소 / 모멘텀 지속
)


@dataclass(frozen=True)
class PathOverride:
    """경로 조건부 비중 오버라이드: (셀, 진입 방향) → 목표 비중."""
    mult_bin: int
    angle_bin: int
    arrival: str          # "up"(vel >= 0) | "down"(vel < 0)
    weight: float


@dataclass(frozen=True)
class YoonProfileConfig:
    cycle_len: int = 200
    slope_lookback: int = 5
    z_window: int = 252
    grid_bins: int = 3
    min_bin_history: int = 252
    state_weights: tuple[tuple[float, ...], ...] = DEFAULT_STATE_WEIGHTS
    max_weight: float = 1.0
    rebalance_threshold: float = 0.05
    max_trade_weight_per_bar: float = 0.25
    warmup_weight: float = 0.0
    short_enabled: bool = False
    short_trail_pct: float = 0.15
    path_tilt_enabled: bool = False
    path_vel_window: int = 10
    path_overrides: tuple[PathOverride, ...] = ()

    @property
    def min_weight(self) -> float:
        return -self.max_weight if self.short_enabled else 0.0


def yoon_profile_config(raw: dict[str, Any]) -> YoonProfileConfig:
    base = YoonProfileConfig()
    cy = raw.get("cycle") or {}
    sz = raw.get("sizing") or {}
    sh = raw.get("short") or {}
    pt = raw.get("path_tilt") or {}
    overrides = tuple(
        PathOverride(
            mult_bin=int(o["mult_bin"]), angle_bin=int(o["angle_bin"]),
            arrival=str(o["arrival"]), weight=float(o["weight"]),
        )
        for o in (pt.get("overrides") or ())
    )
    if any(o.arrival not in ("up", "down") for o in overrides):
        raise ValueError("path_tilt arrival은 'up' 또는 'down'이어야 합니다")
    weights = raw.get("state_weights")
    if weights:
        weights = tuple(tuple(float(w) for w in row) for row in weights)
    else:
        weights = base.state_weights
    grid = int(cy.get("grid_bins", base.grid_bins))
    if len(weights) != grid or any(len(r) != grid for r in weights):
        raise ValueError("state_weights는 grid_bins × grid_bins 행렬이어야 합니다")
    return YoonProfileConfig(
        cycle_len=int(cy.get("length", base.cycle_len)),
        slope_lookback=int(cy.get("slope_lookback", base.slope_lookback)),
        z_window=int(cy.get("z_window", base.z_window)),
        grid_bins=grid,
        min_bin_history=int(cy.get("min_bin_history", base.min_bin_history)),
        state_weights=weights,
        max_weight=float(sz.get("max_weight", base.max_weight)),
        rebalance_threshold=float(
            sz.get("rebalance_threshold", base.rebalance_threshold)
        ),
        max_trade_weight_per_bar=float(
            sz.get("max_trade_weight_per_bar", base.max_trade_weight_per_bar)
        ),
        warmup_weight=float(sz.get("warmup_weight", base.warmup_weight)),
        short_enabled=bool(sh.get("enabled", base.short_enabled)),
        short_trail_pct=float(sh.get("trail_pct", base.short_trail_pct)),
        path_tilt_enabled=bool(pt.get("enabled", base.path_tilt_enabled)),
        path_vel_window=int(pt.get("vel_window", base.path_vel_window)),
        path_overrides=overrides,
    )


def state_target_weights(state: pd.DataFrame, ycfg: YoonProfileConfig) -> pd.Series:
    """(mult_bin, angle_bin) → config 가중치. 상태 미확정(-1) 구간은 warmup_weight.

    short 비활성 시 음수 가중치는 0으로 클립(안전장치).
    """
    w = np.asarray(ycfg.state_weights, dtype=float)
    mb = state["mult_bin"].to_numpy()
    ab = state["angle_bin"].to_numpy()
    valid = (mb >= 0) & (ab >= 0)
    out = np.full(len(state), ycfg.warmup_weight, dtype=float)
    out[valid] = w[mb[valid], ab[valid]]
    out = np.clip(out, ycfg.min_weight, ycfg.max_weight)
    return pd.Series(out, index=state.index, name="target_weight")


def apply_path_tilt(
    target: pd.Series, state: pd.DataFrame, ycfg: YoonProfileConfig
) -> tuple[pd.Series, pd.Series]:
    """경로 조건부 오버라이드: (셀, 필드 속도 부호)가 맞는 봉의 목표 비중 교체.

    vel = mult_pct.diff(vel_window) — 시점 t까지의 percentile만 사용(인과적).
    vel이 NaN인 워밍업 구간은 오버라이드하지 않는다. 반환: (목표, vel 시리즈).
    """
    vel = state["mult_pct"].diff(ycfg.path_vel_window)
    if not ycfg.path_tilt_enabled or not ycfg.path_overrides:
        return target, vel
    out = target.to_numpy(dtype=float).copy()
    mb = state["mult_bin"].to_numpy()
    ab = state["angle_bin"].to_numpy()
    v = vel.to_numpy(dtype=float)
    for ov in ycfg.path_overrides:
        mask = (
            (mb == ov.mult_bin) & (ab == ov.angle_bin) & np.isfinite(v)
            & ((v >= 0.0) if ov.arrival == "up" else (v < 0.0))
        )
        out[mask] = ov.weight
    out = np.clip(out, ycfg.min_weight, ycfg.max_weight)
    return pd.Series(out, index=target.index, name="target_weight"), vel


def apply_squeeze_guard(
    target: pd.Series, close: pd.Series, ycfg: YoonProfileConfig
) -> tuple[pd.Series, pd.Series]:
    """숏 스퀴즈 가드(trail): 에피소드 최저 종가 대비 +trail_pct 반등 시
    그 숏 에피소드(연속 음수 목표 구간)가 끝날 때까지 숏 목표를 0으로.

    인과적: 당일 종가까지만 사용, 비중은 다음 봉 적용. 반환: (목표, 가드 발동 마스크).
    """
    tgt = target.to_numpy(dtype=float).copy()
    px = close.to_numpy(dtype=float)
    guard = np.zeros(len(tgt), dtype=bool)
    if not ycfg.short_enabled or ycfg.short_trail_pct <= 0.0:
        return target, pd.Series(guard, index=target.index)
    episode_min = np.nan
    tripped = False
    for i in range(len(tgt)):
        if tgt[i] >= 0.0:              # 숏 목표 아님 → 에피소드 리셋
            episode_min = np.nan
            tripped = False
            continue
        episode_min = px[i] if np.isnan(episode_min) else min(episode_min, px[i])
        if px[i] > episode_min * (1.0 + ycfg.short_trail_pct):
            tripped = True
        if tripped:
            tgt[i] = 0.0
            guard[i] = True
    return (
        pd.Series(tgt, index=target.index, name="target_weight"),
        pd.Series(guard, index=target.index, name="squeeze_guard"),
    )


def gradual_weights(target: pd.Series, ycfg: YoonProfileConfig) -> pd.Series:
    """rebalance threshold + 봉당 체결 상한으로 목표 비중을 점진 추종."""
    actual = np.zeros(len(target), dtype=float)
    prev = 0.0
    step_cap = float(ycfg.max_trade_weight_per_bar)
    for i, tgt in enumerate(np.asarray(target, dtype=float)):
        diff = tgt - prev
        if abs(diff) < float(ycfg.rebalance_threshold):
            new = prev
        else:
            new = prev + float(np.clip(diff, -step_cap, step_cap))
        actual[i] = float(np.clip(new, ycfg.min_weight, ycfg.max_weight))
        prev = actual[i]
    return pd.Series(actual, index=target.index, name="actual_weight")


def signed_lot_trades(
    close: pd.Series,
    weight: pd.Series,
    state_label: pd.Series,
    guard_active: pd.Series,
    fee_per_side: float,
) -> pd.DataFrame:
    """롱/숏 비중 증감을 FIFO lot으로 분해 (engine.lot_trades의 양방향 확장).

    |비중| 증가 = 신규 lot, 감소 = 오래된 lot부터 청산. 부호가 바뀌면 기존 방향을
    전량 청산 후 새 방향 lot을 연다. 숏 net_return = −(가격수익) − 양편 비용.
    숏 청산이 가드 발동 봉과 겹치면 exit_reason="squeeze_guard".
    """
    idx = close.index
    px = close.to_numpy(dtype=float)
    w = weight.to_numpy(dtype=float)
    lab = np.asarray(state_label, dtype=object)
    guard = np.asarray(guard_active, dtype=bool)

    open_lots: list[dict] = []   # {entry_time, entry_price, qty>0, direction}
    rows: list[dict] = []

    def close_qty(qty_to_close: float, t: int, reason: str) -> None:
        remaining = qty_to_close
        while remaining > 1e-9 and open_lots:
            lot = open_lots[0]
            take = min(lot["qty"], remaining)
            gross = lot["direction"] * (px[t] / lot["entry_price"] - 1.0)
            rows.append({
                "direction": lot["direction"],
                "entry_time": lot["entry_time"],
                "entry_price": lot["entry_price"],
                "exit_time": idx[t],
                "exit_price": px[t],
                "weight": take,
                "net_return": gross - 2.0 * fee_per_side,
                "exit_reason": reason,
                "entry_reason": lot["entry_reason"],
            })
            lot["qty"] -= take
            remaining -= take
            if lot["qty"] <= 1e-9:
                open_lots.pop(0)

    def open_lot(qty: float, direction: int, t: int) -> None:
        side = "롱" if direction > 0 else "숏"
        open_lots.append({
            "entry_time": idx[t], "entry_price": px[t], "qty": qty,
            "direction": direction,
            "entry_reason": f"{lab[t]} {side} 진입 · 비중 {direction * qty:+.2f}",
        })

    prev = 0.0
    for t in range(len(idx)):
        cur = w[t]
        if abs(cur - prev) <= 1e-9:
            prev = cur
            continue
        prev_dir = 0 if abs(prev) <= 1e-9 else (1 if prev > 0 else -1)
        cur_dir = 0 if abs(cur) <= 1e-9 else (1 if cur > 0 else -1)

        def exit_reason() -> str:
            return "squeeze_guard" if prev_dir < 0 and guard[t] else "rebalance"

        if prev_dir != 0 and cur_dir != 0 and prev_dir != cur_dir:
            close_qty(abs(prev), t, exit_reason())     # 방향 전환: 전량 청산 후
            open_lot(abs(cur), cur_dir, t)             # 반대 방향 신규
        elif abs(cur) > abs(prev):
            open_lot(abs(cur) - abs(prev), cur_dir, t)
        else:
            close_qty(abs(prev) - abs(cur), t, exit_reason())
        prev = cur

    if open_lots:                                       # 잔여분 평가 청산
        close_qty(sum(l["qty"] for l in open_lots), len(idx) - 1, "end_of_data")

    columns = [
        "direction", "entry_time", "entry_price", "exit_time", "exit_price",
        "weight", "net_return", "exit_reason", "entry_reason",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows)[columns]


def run_yoon_profile_pipeline(raw: pd.DataFrame, raw_cfg: dict[str, Any]) -> dict:
    psc = config_from_dict(raw_cfg)   # costs / interval / split fraction 재사용
    ycfg = yoon_profile_config(raw_cfg)

    close = pd.Series(np.asarray(raw["close"], dtype=float), index=raw.index)
    state = compute_cycle_state(
        close,
        cycle_len=ycfg.cycle_len,
        slope_lookback=ycfg.slope_lookback,
        z_window=ycfg.z_window,
        grid_bins=ycfg.grid_bins,
        min_bin_history=ycfg.min_bin_history,
    )
    target = state_target_weights(state, ycfg)
    target, path_vel = apply_path_tilt(target, state, ycfg)
    target, guard_active = apply_squeeze_guard(target, close, ycfg)
    actual = gradual_weights(target, ycfg)

    port_ret = portfolio_returns(close, actual, psc)
    equity = equity_from_returns(port_ret)
    bnh = buy_hold_equity(close)

    state_label = (
        "M" + state["mult_bin"].astype(str) + "A" + state["angle_bin"].astype(str)
    ).where((state["mult_bin"] >= 0) & (state["angle_bin"] >= 0), "WARMUP")
    fee_per_side = (psc.costs.fee_bps_per_side + psc.costs.slippage_bps) / 10_000.0
    trades = signed_lot_trades(close, actual, state_label, guard_active, fee_per_side)

    forecast = pd.DataFrame(index=raw.index)
    for col in ("open", "high", "low", "close"):
        if col in raw:
            forecast[col] = np.asarray(raw[col], dtype=float)
    if "volume" in raw:
        forecast["volume"] = np.asarray(raw["volume"], dtype=float)
    forecast = forecast.join(state)
    forecast["target_weight"] = target
    forecast["actual_weight"] = actual
    if ycfg.short_enabled:
        forecast["squeeze_guard"] = guard_active.astype(float)
    if ycfg.path_tilt_enabled:
        forecast["path_vel"] = path_vel

    return {
        "forecast": forecast,
        "state_label": state_label,
        "port_ret": port_ret,
        "equity": equity,
        "buy_hold": bnh,
        "trades": trades,
        "metrics": performance(equity, port_ret, psc.interval),
    }
