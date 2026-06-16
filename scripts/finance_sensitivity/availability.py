"""Point-in-time 사용가능일 계산과 as-of 결합 (finance_plan.txt §5·§21).

이 모듈이 **누수 방지의 경계**다. 재무제표 기준일(period_end)이 아니라
투자자가 실제로 알 수 있었던 날(available_date) 이후부터만 데이터가 보이도록,
일봉 인덱스에 ``merge_asof(direction="backward")``로 재무 팩터를 붙인다.

규칙(§5):
- 발표일(announce_date)이 있으면 그 날을 사용가능일로 쓴다.
- 없으면 보수적으로 분기보고서는 분기말 + 45일, 사업보고서는 결산말 + 90일.
"""
from __future__ import annotations

import pandas as pd

from .config import FinSensitivityConfig

PERIOD_END = "period_end"
AVAILABLE_DATE = "available_date"
REPORT_TYPE = "report_type"
ANNOUNCE_DATE = "announce_date"


def _naive(series: pd.Series) -> pd.Series:
    """tz 제거 — merge_asof는 키 dtype이 일치해야 한다."""
    dt = pd.to_datetime(series)
    if getattr(dt.dt, "tz", None) is not None:
        dt = dt.dt.tz_localize(None)
    return dt


def _naive_index(index: pd.Index) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(pd.to_datetime(index))
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx


def compute_available_date(
    fundamentals: pd.DataFrame, cfg: FinSensitivityConfig
) -> pd.Series:
    """각 재무 행의 사용가능일을 계산해 Series로 반환.

    ``announce_date``가 있고 결측이 아니면 그것을, 아니면 report_type별 보수
    지연(분기 45일 / 연간 90일)을 period_end에 더한다.
    """
    if PERIOD_END not in fundamentals:
        raise KeyError(f"fundamentals에 '{PERIOD_END}' 컬럼이 필요합니다")
    period_end = _naive(fundamentals[PERIOD_END])

    report_type = (
        fundamentals[REPORT_TYPE]
        if REPORT_TYPE in fundamentals
        else pd.Series("quarter", index=fundamentals.index)
    )
    lag_days = report_type.map(
        lambda r: cfg.availability_lag_annual_days
        if str(r).startswith("annual")
        else cfg.availability_lag_quarter_days
    ).astype("int64")
    conservative = period_end + pd.to_timedelta(lag_days, unit="D")

    if ANNOUNCE_DATE in fundamentals:
        announced = _naive(fundamentals[ANNOUNCE_DATE])
        return announced.fillna(conservative)
    return conservative


def with_available_date(
    fundamentals: pd.DataFrame, cfg: FinSensitivityConfig
) -> pd.DataFrame:
    """``available_date`` 컬럼을 부여하고 그 기준으로 정렬한 사본을 반환."""
    out = fundamentals.copy()
    out[AVAILABLE_DATE] = compute_available_date(out, cfg)
    out[PERIOD_END] = _naive(out[PERIOD_END])
    return out.sort_values(AVAILABLE_DATE).reset_index(drop=True)


def asof_join(
    daily_index: pd.Index,
    fundamentals: pd.DataFrame,
    value_cols: list[str],
    cfg: FinSensitivityConfig,
) -> pd.DataFrame:
    """일봉 인덱스에 "그날까지 공개된" 최신 재무 팩터를 backward as-of로 붙인다.

    반환 프레임의 index는 ``daily_index``와 동일하고, 각 ``value_cols`` 컬럼은
    available_date <= 그 거래일 인 가장 최근 재무행 값이다. 아직 어떤 발표도
    사용 불가한 초기 구간은 NaN으로 남는다(결측에도 비중단 — §13).
    또한 사용된 발표의 ``available_date``/``period_end``를 함께 실어 누수 검증과
    forward-return 타깃 정렬에 쓴다.
    """
    fund = with_available_date(fundamentals, cfg)
    idx = _naive_index(daily_index)
    left = pd.DataFrame({"_t": idx}).sort_values("_t")

    keep = [AVAILABLE_DATE, PERIOD_END] + [
        c for c in value_cols if c in fund.columns
    ]
    merged = pd.merge_asof(
        left,
        fund[keep],
        left_on="_t",
        right_on=AVAILABLE_DATE,
        direction="backward",
        allow_exact_matches=True,
    )
    merged.index = left.index
    merged = merged.sort_index()
    merged.index = pd.DatetimeIndex(idx)
    return merged.drop(columns=["_t"], errors="ignore")
