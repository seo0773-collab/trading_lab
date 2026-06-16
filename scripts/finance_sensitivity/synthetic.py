"""재현 가능한 합성 가격 + 분기재무 생성 (finance_plan.txt §21).

계약 테스트와 누수 테스트가 오프라인에서 도는 데 필요하다. 두 가지를 보장한다.
1. **결정성**: 같은 (날짜 범위, seed)면 항상 같은 재무를 만든다 — 그래서 핸들러의
   load_data(합성 OHLCV)와 build_artifacts(재무 재생성)가 일치한다.
2. **주입된 인과**: 각 분기 발표 사용가능일 이후 forward 구간의 가격 드리프트를
   그 분기의 팩터 변화(주로 매출성장률 변화)에 비례시켜, rolling Ridge가 학습할
   실제 신호가 존재하게 한다.

재무는 (가격 범위, seed)만의 함수이므로 OHLCV 인덱스로부터 재구성 가능하다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .availability import (
    AVAILABLE_DATE, PERIOD_END, REPORT_TYPE, compute_available_date,
)
from .config import FinSensitivityConfig

# 주입 신호 강도(분기 표준화 시그널 1단위당 forward 일간 초과 드리프트).
# 합성 검증용이라 모델이 학습 가능하도록 SNR을 충분히 높게 둔다.
_SIGNAL_BETA = 0.005
# 금리 인과: 금리 상승(양의 변화)일수록 forward 수익률이 낮아지도록 음의 주입.
_RATE_BETA = 0.0015
_FORWARD_BARS = 60  # 드리프트가 작용하는 영업일 수


def _business_index(n_bars: int, start: str = "2008-01-02") -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=int(n_bars), name="timestamp")


def _quarter_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    # "QE"(pandas>=2.2) / "Q"(이전) 둘 다 지원 — 버전 경고·오류 회피.
    try:
        return pd.date_range(index[0], index[-1], freq="QE")
    except ValueError:
        return pd.date_range(index[0], index[-1], freq="Q")


def make_synthetic_fundamentals(
    index: pd.DatetimeIndex, cfg: FinSensitivityConfig
) -> pd.DataFrame:
    """가격 범위에 정렬된 결정적 분기재무를 만든다(seed=cfg.seed)."""
    qends = _quarter_ends(index)
    n = len(qends)
    rng = np.random.default_rng(cfg.seed)

    # 매출: 양의 추세 + AR(1) 충격. 분기 성장률 변동이 신호원.
    shocks = rng.normal(0.0, 0.05, size=n)
    growth = 0.02 + np.cumsum(rng.normal(0.0, 0.01, size=n)) * 0.1 + shocks
    revenue = 1000.0 * np.cumprod(1.0 + np.clip(growth, -0.3, 0.5))

    op_margin = 0.15 + rng.normal(0.0, 0.02, size=n)
    operating_income = revenue * op_margin
    net_income = operating_income * (0.75 + rng.normal(0.0, 0.03, size=n))
    operating_cashflow = net_income * (1.05 + rng.normal(0.0, 0.05, size=n))

    total_equity = 2000.0 + np.cumsum(net_income) * 0.5
    total_debt = total_equity * (0.6 + rng.normal(0.0, 0.05, size=n))
    inventory = revenue * (0.2 + rng.normal(0.0, 0.02, size=n))
    shares = np.full(n, 100.0)
    eps = net_income / shares

    fundamentals = pd.DataFrame({
        PERIOD_END: qends,
        REPORT_TYPE: ["quarter"] * n,
        "revenue": revenue,
        "operating_income": operating_income,
        "net_income": net_income,
        "operating_cashflow": operating_cashflow,
        "total_equity": total_equity,
        "total_debt": total_debt,
        "inventory": inventory,
        "shares_outstanding": shares,
        "eps": eps,
    })
    return fundamentals


def _quarter_signal(fundamentals: pd.DataFrame) -> pd.Series:
    """주입할 분기 신호 = 표준화한 매출성장률 변화(인덱스=period_end)."""
    revenue = fundamentals["revenue"].to_numpy()
    growth = pd.Series(revenue).pct_change()
    signal = growth.diff()
    std = signal.std(ddof=0)
    norm = signal / std if std and not np.isnan(std) else signal.fillna(0.0)
    return pd.Series(
        norm.fillna(0.0).to_numpy(),
        index=pd.to_datetime(fundamentals[PERIOD_END]),
    )


def make_synthetic_ohlcv(
    n_bars: int, seed: int, interval: str = "1d",
    cfg: FinSensitivityConfig | None = None,
) -> pd.DataFrame:
    """주입 인과를 담은 합성 일봉 OHLCV.

    각 분기 사용가능일부터 _FORWARD_BARS 동안 일간 드리프트에 그 분기 신호를
    더한다 → '팩터 변화 → 발표 후 수익률' 관계가 데이터에 존재한다.
    """
    cfg = cfg or FinSensitivityConfig(seed=seed)
    if cfg.seed != seed:
        from dataclasses import replace
        cfg = replace(cfg, seed=seed)

    index = _business_index(n_bars)
    fundamentals = make_synthetic_fundamentals(index, cfg)
    avail = compute_available_date(fundamentals, cfg)
    signal = _quarter_signal(fundamentals).to_numpy()

    # 금리 인과: 발표 시점 금리 변화(표준화)를 forward 드리프트에 음으로 주입해
    # rate 피처(d_rate)가 학습 가능한 실제 신호를 갖게 한다(dataset와 같은 정의).
    from .macro import make_synthetic_rates, rate_change_signal
    rate_sig = rate_change_signal(
        make_synthetic_rates(index, cfg), cfg.rate_change_lookback
    )

    rng = np.random.default_rng(seed * 31 + 1)
    drift = rng.normal(0.0002, 0.005, size=n_bars)  # 기저 일간 수익률

    pos = index.get_indexer(
        pd.DatetimeIndex(avail).tz_localize(None), method="bfill"
    )
    for k, start in enumerate(pos):
        if start < 0:
            continue
        end = min(start + _FORWARD_BARS, n_bars)
        drift[start:end] += _SIGNAL_BETA * signal[k]
        if cfg.use_rate_feature:
            drift[start:end] -= _RATE_BETA * float(rate_sig[start])

    close = 100.0 * np.cumprod(1.0 + drift)
    close = pd.Series(close, index=index)
    intra = np.abs(rng.normal(0.0, 0.006, size=n_bars))
    high = close * (1.0 + intra)
    low = close * (1.0 - intra)
    open_ = close.shift(1).fillna(close.iloc[0])
    return pd.DataFrame(
        {
            "open": open_.to_numpy(),
            "high": np.maximum.reduce([high.to_numpy(), open_.to_numpy(), close.to_numpy()]),
            "low": np.minimum.reduce([low.to_numpy(), open_.to_numpy(), close.to_numpy()]),
            "close": close.to_numpy(),
            "volume": rng.integers(1_000, 10_000, size=n_bars).astype(float),
        },
        index=index,
    )
