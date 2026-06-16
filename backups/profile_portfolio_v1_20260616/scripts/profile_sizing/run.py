"""profile-sizing 파이프라인 오케스트레이터.

raw OHLCV → 지표/프로파일/국면/비중 → equity·buy&hold·trades 아티팩트 dict.
핸들러(src/trading_lab/strategies/profile_sizing.py)와 batch.py가 이 함수를 쓴다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ProfileSizingConfig, config_from_dict  # noqa: F401
from .engine import (
    buy_hold_equity, equity_from_returns, lot_trades, performance,
    portfolio_returns,
)
from .indicators import compute_cycle
from .profile import compute_profile
from .regime import classify
from .sizing import build_weights
from .synthetic import make_synthetic_ohlcv  # noqa: F401


def run_pipeline(raw: pd.DataFrame, cfg: ProfileSizingConfig) -> dict:
    """전 구간 지표/비중/평가자산 산출. 결과는 raw 인덱스 정렬."""
    cycle = compute_cycle(raw, cfg.base_cycle)
    profile = compute_profile(cycle, raw, cfg.profile)
    regimes = classify(cycle, profile)

    # 추세 강도: 가격이 base_cycle 위(cm_close>1)이고 기준선 자체가 상승 중일 때만 양수.
    cm_close = cycle["cm_close"].to_numpy()
    base_up = np.asarray(cycle["base_cycle"].diff() > 0)
    trend_strength = np.where(base_up, np.clip(cm_close - 1.0, 0.0, None), 0.0)

    weights = build_weights(
        profile["cumulative_percentile"].to_numpy(),
        regimes["regime"].to_numpy(),
        regimes["recovery_bars"].to_numpy(),
        cfg,
        trend_strength=trend_strength,
    )
    weights.index = raw.index

    close = pd.Series(np.asarray(raw["close"], dtype=float), index=raw.index)
    port_ret = portfolio_returns(close, weights["actual_weight"], cfg)
    equity = equity_from_returns(port_ret)
    bnh = buy_hold_equity(close)
    trades = lot_trades(close, weights["actual_weight"], regimes["regime"], cfg)

    forecast = cycle.join(profile).join(regimes[["regime_code"]]).join(
        weights[["base_target_weight", "regime_cap",
                 "final_target_weight", "actual_weight"]]
    )
    return {
        "forecast": forecast,
        "regime": regimes["regime"],
        "port_ret": port_ret,
        "equity": equity,
        "buy_hold": bnh,
        "trades": trades,
    }


def slice_window(index: pd.DatetimeIndex, phase: str,
                 cfg: ProfileSizingConfig) -> pd.DatetimeIndex:
    """train/validation/test 시간순 분할. warmup 이후를 기준으로 자른다."""
    if phase == "all":
        return index
    n = len(index)
    t_end = int(n * cfg.train_frac)
    v_end = int(n * (cfg.train_frac + cfg.validation_frac))
    if phase == "validation":
        return index[t_end:v_end]
    if phase == "test":
        return index[v_end:]
    return index


def rebased_equity(port_ret: pd.Series, window: pd.DatetimeIndex) -> pd.Series:
    """phase 구간 시작을 1.0으로 재정규화한 equity."""
    seg = port_ret.reindex(window).fillna(0.0)
    return (1.0 + seg).cumprod().rename("equity")
