"""yoon_pattern 피처·파이프라인: L틱 패턴 투영 → 예상수익 점수 → 기대값 사이징.

가설(2026-07-02 yoon_pattern_study로 검증): 직전 L틱(기본 300 일봉) 패턴과 가장
유사한 과거 구간들의 '직후 H틱' 실현수익 중앙값(analog 투영)은 선행수익률과 양의
IC를 갖는다(SPY L300/H21: IC +0.056, 1/3구간 전부 양수, 5분위 단조). 자기 패턴
연장(self)은 L=300에서 일관된 음의 IC로 기각 — 비교용으로만 유지.

체인: pattern score → walk_forward_expectancy(무누수 버킷 기대값, rolling_expectancy
재사용, 확장창·굵은 버킷·min_samples 교훈 반영) → expectancy_weights → equity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ProfileSizingConfig
from .engine import (
    buy_hold_equity, equity_from_returns, lot_trades, performance,  # noqa: F401
    portfolio_returns,
)
from .rolling_expectancy import expectancy_weights, walk_forward_expectancy


def pattern_matrix(logp: np.ndarray, length: int, step: int) -> np.ndarray:
    """행 t = [t−L+step, ..., t] 서브샘플 로그가격의 z-정규화 벡터 (미달 행 NaN)."""
    n = len(logp)
    k = max(2, length // step)
    offs = np.arange(k) * step - (k - 1) * step
    G = np.full((n, k), np.nan, dtype=np.float32)
    for j, o in enumerate(offs):
        G[-o:, j] = logp[: n + o] if o < 0 else logp
    mu = np.nanmean(G, axis=1, keepdims=True)
    sd = np.nanstd(G, axis=1, keepdims=True)
    return (G - mu) / np.where(sd > 1e-12, sd, np.nan)


def analog_scores(
    logp: np.ndarray, length: int, horizon: int, *,
    step: int = 5, matches: int = 50, block: int = 512,
) -> np.ndarray:
    """무누수 유사 패턴 앙상블 점수: top-M 유사 창 '직후 H틱' 로그수익 중앙값.

    후보 창 끝 s는 s+H ≤ t (직후 경로가 t 이전에 완결된 것만) — 무누수.
    선택은 유사도 내림차순 그리디 + 최소 간격 gap(중복 에피소드 방지).
    """
    n = len(logp)
    G = pattern_matrix(logp, length, step)
    valid = ~np.isnan(G).any(axis=1)
    k = G.shape[1]
    Gz = np.where(valid[:, None], G, 0.0).astype(np.float32)
    fwd = np.full(n, np.nan)
    fwd[: n - horizon] = logp[horizon:] - logp[: n - horizon]
    scores = np.full(n, np.nan)
    gap = max(horizon, 5)
    topn = matches * 4
    for b0 in range(0, n, block):
        b1 = min(b0 + block, n)
        S = Gz[b0:b1] @ Gz.T / k
        for t in range(b0, b1):
            if not valid[t]:
                continue
            smax = t - horizon
            if smax < length:
                continue
            row = S[t - b0, : smax + 1].copy()
            row[~valid[: smax + 1]] = -9.9
            if smax + 1 <= topn:
                cand = np.argsort(row)[::-1]
            else:
                part = np.argpartition(row, -topn)[-topn:]
                cand = part[np.argsort(row[part])[::-1]]
            picked: list[int] = []
            for s in cand:
                if row[s] <= -9.0:
                    break
                if all(abs(int(s) - q) >= gap for q in picked):
                    picked.append(int(s))
                    if len(picked) >= matches:
                        break
            if len(picked) >= max(10, matches // 5):
                scores[t] = float(np.median(fwd[picked]))
    return scores


def self_scores(logp: np.ndarray, length: int, horizon: int) -> np.ndarray:
    """자기 패턴 연장 점수(주기 L 반복 가정) — 스터디에서 기각, 비교용."""
    n = len(logp)
    s = np.full(n, np.nan)
    if n > length:
        s[length:] = logp[horizon : n - length + horizon] - logp[: n - length]
    return s


def run_pattern_expectancy_pipeline(
    raw: pd.DataFrame, cfg: ProfileSizingConfig, pattern: dict,
    risk_overlay: dict | None = None,
) -> dict:
    """raw OHLCV → 패턴 투영 점수 → 무누수 기대값 → 비중 → equity 아티팩트 dict.

    risk_overlay(선택): 확장창 기대값 테이블이 급락 국면 전환을 지연 학습하는 공백을
    외생 리스크 신호로 메운다. {"enabled": true, "vol_target": 0.15, "vol_lookback": 21}
    이면 비중 × min(1, 목표변동성/실현 연변동성(전봉)) — 무누수, warmup은 1.0.
    실측 근거: yoon_pattern_sweep(2026-07-02) test MDD -23.3→-16.4%·Sharpe 개선.
    """
    close = pd.Series(np.asarray(raw["close"], dtype=float), index=raw.index)
    logp = np.log(close.to_numpy())
    length = int(pattern.get("length", 300))
    horizon = int(pattern.get("horizon", cfg.expectancy.primary_horizon))
    method = str(pattern.get("method", "analog"))
    if method == "self":
        score = self_scores(logp, length, horizon)
    else:
        score = analog_scores(
            logp, length, horizon,
            step=int(pattern.get("step", 5)),
            matches=int(pattern.get("matches", 50)),
        )

    features = pd.DataFrame(
        {"pattern_score": score, "profile_score": score}, index=raw.index)
    exp, exp_tables = walk_forward_expectancy(features, close, cfg)
    weights = expectancy_weights(exp["expectancy_bps"], cfg)

    vol_scale = None
    ro = dict(risk_overlay or {})
    if ro.get("enabled"):
        vt = float(ro.get("vol_target", 0.15))
        lb = int(ro.get("vol_lookback", 21))
        rv = close.pct_change().rolling(lb, min_periods=lb).std() * np.sqrt(252.0)
        vol_scale = (vt / rv).clip(upper=1.0).shift(1).fillna(1.0)  # 전봉 신호(무누수)
        weights = weights.copy()
        weights["actual_weight"] = weights["actual_weight"] * vol_scale

    port_ret = portfolio_returns(close, weights["actual_weight"], cfg)
    equity = equity_from_returns(port_ret)
    bnh = buy_hold_equity(close)
    regime = pd.Series(["PATTERN"] * len(raw), index=raw.index)
    trades = lot_trades(close, weights["actual_weight"], regime, cfg)

    forecast = pd.DataFrame({"close": close}, index=raw.index)
    forecast["pattern_score"] = features["pattern_score"]
    forecast = forecast.join(exp).join(weights)
    if vol_scale is not None:
        forecast["vol_scale"] = vol_scale
    if "volume" in raw:
        forecast["volume"] = np.asarray(raw["volume"], dtype=float)
    return {
        "forecast": forecast,
        "expectancy_tables": exp_tables,
        "port_ret": port_ret,
        "equity": equity,
        "buy_hold": bnh,
        "trades": trades,
        "metrics": performance(equity, port_ret, cfg.interval),
    }
