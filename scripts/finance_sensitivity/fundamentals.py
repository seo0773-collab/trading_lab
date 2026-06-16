"""재무 원시 데이터 → 팩터 도출 + 변화량(Δ) 피처 (finance_plan.txt §3·§4·§22).

핵심은 절대 수준이 아니라 **직전 분기 대비 변화**다(§3). 원시 분기재무에서
파생 팩터(매출성장률·ROE·부채비율)를 만들고, 설정된 모든 팩터에 대해 전분기
대비 변화 피처 ``d_<factor>``를 생성한다. 비율형은 차분, 수준/흐름형은 매출
규모로 정규화한 차분을 써서 음수·단위 문제를 피한다(이후 rolling 표준화가
스케일을 마저 맞춘다).

밸류에이션(PER/PBR/PSR)은 주가가 필요하므로 여기서는 1주당 기준값(eps_ttm,
bvps, spr)만 만들고, 실제 비율·z-score는 주가가 결합되는 dataset 단계에서 낸다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .availability import ANNOUNCE_DATE, PERIOD_END, REPORT_TYPE
from .config import FinSensitivityConfig

# 변화 피처를 차분으로 낼 비율형 팩터(나머지는 매출규모 정규화 차분).
RATIO_FACTORS = frozenset({"revenue_growth", "roe", "debt_ratio"})
# 원시 입력으로 기대하는 컬럼(없으면 결측 → 비중단).
RAW_NUMERIC = (
    "revenue", "operating_income", "net_income", "total_equity",
    "total_debt", "operating_cashflow", "inventory", "shares_outstanding",
    "eps",
)
_EPS = 1e-9


def derive_factors(raw: pd.DataFrame) -> pd.DataFrame:
    """원시 분기재무에서 파생 팩터 컬럼을 채운 사본을 반환(period_end 정렬)."""
    out = raw.copy()
    out[PERIOD_END] = pd.to_datetime(out[PERIOD_END])
    if getattr(out[PERIOD_END].dt, "tz", None) is not None:
        out[PERIOD_END] = out[PERIOD_END].dt.tz_localize(None)
    out = out.sort_values(PERIOD_END).reset_index(drop=True)

    for col in RAW_NUMERIC:
        if col not in out:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")

    revenue = out["revenue"]
    equity = out["total_equity"].replace(0.0, np.nan)

    out["revenue_growth"] = revenue.pct_change()
    out["roe"] = out["net_income"] / equity
    out["debt_ratio"] = out["total_debt"] / equity

    # 밸류에이션 1주당 기준값(주가는 dataset에서 곱함).
    shares = out["shares_outstanding"].replace(0.0, np.nan)
    ni_ttm = out["net_income"].rolling(4, min_periods=1).sum()
    rev_ttm = revenue.rolling(4, min_periods=1).sum()
    out["eps_ttm"] = np.where(
        out["eps"].notna(),
        out["eps"].rolling(4, min_periods=1).sum(),
        ni_ttm / shares,
    )
    out["bvps"] = out["total_equity"] / shares       # 주당 순자산
    out["spr"] = rev_ttm / shares                     # 주당 매출(TTM)
    return out


# 서프라이즈(YoY 가속) 프록시를 만들 흐름 항목 — 애널리스트 컨센서스가 없으므로
# "YoY 성장률의 최근 변화"로 어닝 서프라이즈를 근사한다(PEAD 동인).
_SURPRISE_BASE = ("revenue", "net_income", "operating_cashflow")


def factor_changes(
    raw: pd.DataFrame, cfg: FinSensitivityConfig
) -> pd.DataFrame:
    """팩터별 변화 피처를 만든다(전분기 Δ·YoY Δ·서프라이즈).

    - ``d_<factor>``: 전분기 대비 변화(§6).
    - ``y_<factor>``: 전년 동기 대비 변화(YoY) — 계절성 제거.
    - ``s_<base>``: YoY 성장률의 가속(컨센서스 부재 시 어닝 서프라이즈 프록시).
    모두 과거(shift)만 쓰므로 누수가 없다. 반환은 fundamentals 행 단위.
    """
    df = derive_factors(raw)
    revenue_scale = df["revenue"].abs().replace(0.0, np.nan)

    change_cols: list[str] = []
    for factor in cfg.factors:
        if factor not in df:
            df[factor] = np.nan
        if factor in RATIO_FACTORS:
            df[f"d_{factor}"] = df[factor].diff()
            df[f"y_{factor}"] = df[factor].diff(4)
        else:
            df[f"d_{factor}"] = df[factor].diff() / revenue_scale
            df[f"y_{factor}"] = df[factor].diff(4) / revenue_scale
        change_cols += [f"d_{factor}", f"y_{factor}"]

    for base in _SURPRISE_BASE:
        if base in df:
            yoy_growth = df[base].pct_change(4)
            df[f"s_{base}"] = yoy_growth.diff()  # YoY 성장률의 분기 변화 = 가속
            change_cols.append(f"s_{base}")

    df["missing_ratio"] = df[change_cols].isna().mean(axis=1)

    carry = [PERIOD_END]
    for col in (REPORT_TYPE, ANNOUNCE_DATE):
        if col in df:
            carry.append(col)
    value_cols = change_cols + [
        "eps_ttm", "bvps", "spr", "missing_ratio",
        "operating_income", "net_income", "operating_cashflow",
        "total_equity", "inventory", "revenue", "debt_ratio",
    ]
    keep = carry + [c for c in dict.fromkeys(value_cols) if c in df]
    return df[keep]


# valuation_z 와의 상호작용 피처(dataset에서 생성). plan §3: 고평가면 반응 둔화.
INTERACTION_FEATURES = {
    "ix_roe_val": "y_roe",
    "ix_rev_val": "s_revenue",
}


def feature_columns(cfg: FinSensitivityConfig) -> list[str]:
    """모델이 피처로 쓰는 컬럼 목록(feature_set에 따라).

    "qoq" = 전분기 Δ만(원안). "redesign" = YoY 변화 + 서프라이즈 + 밸류에이션과
    그 상호작용. d_revenue_growth는 두 세트 모두에 포함(합성 인과 검증 호환).
    """
    if cfg.feature_set == "qoq":
        return [f"d_{factor}" for factor in cfg.factors]
    feats = [
        "d_revenue_growth", "y_revenue_growth", "y_roe",
        "y_operating_cashflow", "y_debt_ratio",
        "s_revenue", "s_net_income", "valuation_z",
    ]
    feats += list(INTERACTION_FEATURES)
    if cfg.use_rate_feature:
        # 미국 금리 수준(z)과 변화 — 거시(금리 발표) 민감도 학습 피처.
        feats += ["rate_level", "d_rate"]
    return feats
