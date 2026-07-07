"""Equity 시리즈 성과지표 공통 유틸 (연환산은 달력 기준이 권위).

배경(2026-07-07 final0706 평가에서 실측): 봉 수 기준 연환산(`len(eq)/ann`)은
인덱스에 결측 구간이 있으면 경과 연수를 과소평가해 CAGR을 부풀린다. final0706은
KR 데이터 갭(2007-02~2009-04) 탓에 19.4년이 16.2"년"으로 계산돼 공표 CAGR
16.84%가 달력 기준 13.96%였다. 그래서 DatetimeIndex면 **달력 경과일 / 365.25**를
연수로 쓴다. 봉 수 기준은 DatetimeIndex가 아닐 때(합성 정수 인덱스 등)의 폴백.

Sharpe/변동성의 연환산 계수(ann)는 봉 주기의 속성이므로 그대로 둔다 — 인트라데이
(4h 등)는 호출부가 실제 연간 봉 수를 넘겨야 한다(기존 관행 유지). 단, 결측 갭을
가로지르는 봉의 수익률이 return 계열에 남는 문제는 연환산으로 못 고친다 — 평가
시작점을 데이터 연속 구간으로 자르는 것(개선계획 A2)이 근본 처치다.

계획: docs/strategy/plans/final0706_improvement.md (A1)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def years_elapsed(index: pd.Index, *, ann: float = 252.0) -> float:
    """경과 연수. DatetimeIndex면 달력 기준, 아니면 봉 수/ann 폴백."""
    if isinstance(index, pd.DatetimeIndex) and len(index) >= 2:
        days = (index[-1] - index[0]).days
        if days > 0:
            return days / 365.25
    return max(len(index) / ann, 1e-9)


def equity_metrics(equity: pd.Series, *, ann: float = 252.0,
                   min_bars: int = 10) -> dict[str, float | None]:
    """정규화 여부와 무관하게 동작(첫 유효값으로 나눠서 계산)."""
    eq = equity.dropna()
    n = int(len(eq))
    if n < min_bars or eq.iloc[0] <= 0 or eq.iloc[-1] <= 0:
        return {"sharpe": None, "cagr": None, "max_drawdown": None,
                "total_return": None, "volatility": None, "n": n, "years": None}
    growth = eq / eq.iloc[0]
    ret = growth.pct_change().to_numpy()[1:]
    sd = float(np.nanstd(ret, ddof=0))
    sharpe = float(np.nanmean(ret) / sd * np.sqrt(ann)) if sd > 0 else None
    vol = float(sd * np.sqrt(ann)) if sd > 0 else None
    yrs = years_elapsed(eq.index, ann=ann)
    cagr = float(growth.iloc[-1] ** (1.0 / yrs) - 1.0)
    mdd = float((growth / growth.cummax() - 1.0).min())
    return {"sharpe": sharpe, "cagr": cagr, "max_drawdown": mdd,
            "total_return": float(growth.iloc[-1] - 1.0), "volatility": vol,
            "n": n, "years": float(yrs)}


def coverage_label(index: pd.Index, *, max_gap_days: int = 30) -> str:
    """구간 라벨 + 결측 갭 명시. 예: '2007-01-29~2026-07-06 (결측 2007-02-09~2009-04-09)'.

    final0706처럼 라벨상 시작일과 실제 데이터 밀도가 다른 경우, 리포트가
    '19.5년 검증'으로 오독되는 것을 막기 위해 max_gap_days를 넘는 갭을 병기한다.
    """
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
        return f"{len(index)} bars"
    idx = index.sort_values()
    base = f"{idx[0].date()}~{idx[-1].date()}"
    deltas = idx.to_series().diff()
    gaps = deltas[deltas > pd.Timedelta(days=max_gap_days)]
    if gaps.empty:
        return base
    parts = [f"{(end - d).date()}~{end.date()}" for end, d in gaps.items()]
    return f"{base} (결측 {', '.join(parts)})"
