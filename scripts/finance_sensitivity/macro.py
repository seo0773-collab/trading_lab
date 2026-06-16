"""미국 금리(거시) 사이드로드 — 합성/실데이터 (finance_plan.txt 보강).

금리 '발표' 효과를 종목별 민감도 학습 피처로 쓰기 위한 **일별 금리 시계열**을
제공한다. 가격(OHLCV)·분기재무와 동일하게 "사이드로드" 데이터 계층이며,
모델 피처는 dataset 단계에서 이벤트(발표 사용가능일)에 as-of로 정렬한다.

- 실데이터: yfinance ``^IRX``(13주 T-bill 수익률, 연방기금금리 프록시) 종가.
  (핸들러가 로드 — 이 모듈은 합성/신호 계산만 담당해 오프라인 테스트 가능.)
- 합성: ``(가격 인덱스, seed)``만의 함수로 결정적 생성. 합성 OHLCV의 인과
  주입(synthetic.py)과 같은 금리 시계열을 재구성하므로 학습 신호가 정합한다.

Point-in-time: 금리는 발표 즉시 공개되므로 거래일 종가를 그대로 쓴다(누수 없음).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FinSensitivityConfig


def _naive(index: pd.Index) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(pd.to_datetime(index))
    return idx.tz_localize(None) if idx.tz is not None else idx


def make_synthetic_rates(
    index: pd.DatetimeIndex, cfg: FinSensitivityConfig
) -> pd.Series:
    """가격 범위에 정렬된 결정적 일별 정책금리 경로(연 % 단위).

    완만한 추세 + 작은 일별 충격의 누적(정책금리의 끈적임을 흉내). seed에만
    의존하므로 같은 (index, seed)면 항상 같은 시계열 — 합성 인과와 정합한다.
    """
    idx = _naive(index)
    n = len(idx)
    rng = np.random.default_rng(cfg.seed * 17 + 3)
    steps = rng.normal(0.0, 0.01, size=n)
    rate = 2.0 + np.cumsum(steps)
    return pd.Series(np.clip(rate, 0.0, None), index=idx, name="rate")


def rate_change_signal(
    rates: pd.Series, lookback: int
) -> np.ndarray:
    """일별 금리변화(lookback 영업일)의 표준화 신호 — 합성 인과 주입용.

    dataset가 이벤트에 붙이는 ``d_rate``(금리 변화)와 같은 정의를 일별로 편 것.
    """
    change = rates - rates.shift(lookback)
    std = change.std(ddof=0)
    if std and not np.isnan(std):
        change = change / std
    return change.fillna(0.0).to_numpy()
