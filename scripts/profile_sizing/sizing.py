"""목표 비중 계산 + regime cap + rebalancer (profile_plan.txt §7~§10).

매 봉:
1. base_target_weight  = weight_model(cumulative_percentile)   §7
2. max_allowed_weight  = regime_cap(regime, recovery_stage)     §8
3. final_target_weight = min(base, cap), DEFENSE면 증액 금지     §9
4. actual_weight       = rebalancer(현재→목표, threshold/step)   §10

actual_weight는 직전 봉 비중에서 출발해 점진 조정하므로 순차 루프로 만든다.
모든 비중은 봉 t의 정보로 결정되어 t+1 수익률에 적용된다(engine에서 1봉 지연).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ProfileSizingConfig
from .regime import DEFENSE


def base_target_weight(percentile: np.ndarray, cfg: ProfileSizingConfig) -> np.ndarray:
    """cumulative_percentile(0~1) → 기본 주식 비중."""
    pct = np.asarray(percentile, dtype=float)
    out = np.zeros_like(pct)
    if cfg.weight_model == "exponential":
        w = cfg.exp_max_weight * np.exp(-cfg.exp_k * pct)
        out = np.clip(w, 0.0, 1.0)
    else:  # bucket
        pct100 = pct * 100.0
        for lo, hi, weight in cfg.buckets:
            mask = (pct100 >= lo) & (pct100 < hi)
            out[mask] = weight
        # 상단 경계(100%) 포함.
        top = cfg.buckets[-1]
        out[pct100 >= top[1]] = top[2]
    out[~np.isfinite(pct)] = 0.0  # warmup 등 무효 → 비중 0
    return out


def regime_cap(regime: np.ndarray, recovery_bars: np.ndarray,
               cfg: ProfileSizingConfig) -> np.ndarray:
    rc = cfg.regime_cap
    base_caps = {
        "NORMAL": rc.NORMAL, "CAUTION": rc.CAUTION,
        "DEFENSE": rc.DEFENSE, "RECOVERY": rc.RECOVERY,
    }
    caps = np.array([base_caps.get(str(r), 1.0) for r in regime], dtype=float)
    # RECOVERY 단계별 상향: 회복 지속 봉수 / recovery_stage_bars 로 단계 결정.
    stage_caps = rc.recovery_stage_caps
    for i, r in enumerate(regime):
        if str(r) == "RECOVERY" and rc.recovery_stage_bars > 0:
            stage = int(recovery_bars[i] // rc.recovery_stage_bars)
            stage = min(stage, len(stage_caps) - 1)
            caps[i] = stage_caps[stage]
    return caps


def trend_boost(
    regime: np.ndarray, trend_strength: np.ndarray, cfg: ProfileSizingConfig
) -> np.ndarray:
    """추세 가산(profile-sizing-trend 변형). off면 0 배열.

    trend_strength(=clip(cm_close-1, 0, ∞), 기준선 상승 시에만 양수)에 비례한
    보너스를 apply_regimes(기본 NORMAL/RECOVERY)에서만 더한다.
    """
    to = cfg.trend_overlay
    n = len(regime)
    if not to.enabled:
        return np.zeros(n, dtype=float)
    ts = np.nan_to_num(np.asarray(trend_strength, dtype=float), nan=0.0)
    boost = np.clip(ts * to.boost_gain, 0.0, to.max_boost)
    applies = np.array([str(r) in to.apply_regimes for r in regime])
    return np.where(applies, boost, 0.0)


def build_weights(
    percentile: np.ndarray,
    regime: np.ndarray,
    recovery_bars: np.ndarray,
    cfg: ProfileSizingConfig,
    trend_strength: np.ndarray | None = None,
) -> pd.DataFrame:
    """base/cap/final target + 점진 rebalance 후 actual_weight 산출."""
    base = base_target_weight(percentile, cfg)
    to = cfg.trend_overlay
    if trend_strength is not None and to.enabled:
        base = np.clip(base + trend_boost(regime, trend_strength, cfg), 0.0, 1.0)
        if to.floor > 0.0:
            ts = np.nan_to_num(np.asarray(trend_strength, dtype=float), nan=0.0)
            strong = (ts > 0.0) & np.array(
                [str(r) in to.apply_regimes for r in regime]
            )
            base = np.where(strong, np.maximum(base, to.floor), base)
    cap = regime_cap(regime, recovery_bars, cfg)
    n = len(base)
    rb = cfg.rebalance

    final = np.zeros(n, dtype=float)
    actual = np.zeros(n, dtype=float)
    prev = 0.0
    for i in range(n):
        is_defense = str(regime[i]) == DEFENSE
        if is_defense:
            # 방어장: 싸도 증액 금지. cap 초과분만 축소.
            target = min(prev, cap[i])
        else:
            target = min(base[i], cap[i])
        final[i] = target

        diff = target - prev
        if abs(diff) < rb.threshold:
            new = prev
        else:
            step = float(np.clip(diff, -rb.max_trade_weight_per_bar,
                                 rb.max_trade_weight_per_bar))
            new = prev + step
        # 방어장에서는 매수(증액) 금지.
        if is_defense and not rb.defense_buy_allowed and new > prev:
            new = prev
        new = float(np.clip(new, 0.0, 1.0))
        actual[i] = new
        prev = new

    out = pd.DataFrame(index=range(n))
    out["base_target_weight"] = base
    out["regime_cap"] = cap
    out["final_target_weight"] = final
    out["actual_weight"] = actual
    return out
