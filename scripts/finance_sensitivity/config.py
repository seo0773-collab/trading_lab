"""Configuration for fin-sensitivity-v1 (finance_plan.txt §4·§5·§22·§23).

The dashboard funnels its TF/period widgets through ``interval``/``period`` and
its tunable panel through the flat JSON keys; ``config_from_dict`` maps that flat
config onto a typed structure the research modules consume.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

# 기본 재무 팩터(변화량으로 사용) — config["factors"]가 우선.
DEFAULT_FACTORS = (
    "revenue_growth",
    "operating_income",
    "net_income",
    "roe",
    "operating_cashflow",
    "debt_ratio",
    "total_equity",
)
DEFAULT_VALUATION = ("per", "pbr", "psr")


@dataclass(frozen=True)
class MarketFilter:
    symbol: str = "SPY"
    ma_len: int = 200
    fallback_self_ma: bool = True


@dataclass(frozen=True)
class Exclusions:
    """plan §9 제외 조건 임계값."""
    operating_loss_streak: int = 2   # 영업이익 적자 지속 분기수
    ocf_decline_streak: int = 2      # 영업현금흐름 지속 악화 분기수
    debt_ratio_jump: float = 0.20    # 부채비율 급등(분기 Δ) 한도
    valuation_overheat_z: float = 2.0  # 밸류에이션 과거평균 대비 과열 z
    max_missing_ratio: float = 0.4   # 팩터 결측 비율 상한


@dataclass(frozen=True)
class FinSensitivityConfig:
    interval: str = "1d"
    period: str = "max"
    factors: tuple[str, ...] = DEFAULT_FACTORS
    valuation_factors: tuple[str, ...] = DEFAULT_VALUATION

    # 민감도 학습(§22)
    train_quarters: int = 16
    min_train_quarters: int = 12
    ridge_alpha: float = 1.0
    horizons: tuple[int, ...] = (20, 60)
    # 피처 세트: "redesign"(YoY 변화 + 서프라이즈 + 밸류 상호작용) | "qoq"(전분기 Δ만).
    feature_set: str = "redesign"

    # Point-in-time 사용가능일(§21) — 발표일 부재 시 보수 룰.
    availability_lag_quarter_days: int = 45
    availability_lag_annual_days: int = 90

    # 미국 금리(거시) 피처(보강) — 금리 '발표' 효과를 민감도 학습 피처로 포함한다.
    # 금리는 발표 즉시 공개되므로 거래일 종가를 그대로 PIT as-of로 쓴다(누수 없음).
    use_rate_feature: bool = True
    rate_symbol: str = "^IRX"        # 13주 T-bill 수익률(연방기금금리 프록시)
    rate_change_lookback: int = 63   # 금리 변화 측정 영업일(≈1분기)

    # 타깃 재정의(§8) — forward 수익률을 시장 대비 초과(abnormal)로 학습.
    # raw 수익률은 시장 베타·종목 추세가 펀더멘털 신호를 압도하므로 기본 활성.
    target_excess: bool = True

    # 진입/청산(§23)
    pred20_min: float = 0.0
    pred60_min: float = 0.0
    stop_loss_pct: float = 0.25
    max_hold_days: int = 60
    fee_bps_per_side: float = 5.0
    execution: str = "next_open"
    long_only: bool = True

    market_filter: MarketFilter = field(default_factory=MarketFilter)
    exclusions: Exclusions = field(default_factory=Exclusions)

    # 검증 분할(§8)
    train_frac: float = 0.6
    validation_frac: float = 0.2

    # 합성 경로(§21·계약 테스트)
    synthetic_bars: int = 2600
    seed: int = 7

    @property
    def horizon_20(self) -> int:
        return int(self.horizons[0])

    @property
    def horizon_60(self) -> int:
        return int(self.horizons[-1])


def _as_tuple(value: Any, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return default
    return tuple(str(v) for v in value)


def _as_bool(value: Any, default: bool) -> bool:
    """JSON bool 또는 대시보드 select의 문자열("true"/"off" 등)을 안전하게 파싱."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"true", "1", "on", "yes"}


def config_from_dict(raw: dict[str, Any]) -> FinSensitivityConfig:
    """Flat dashboard/JSON config → typed FinSensitivityConfig.

    Unknown keys are ignored; missing keys fall back to dataclass defaults so a
    partial tunable override (config_overrides) stays valid.
    """
    base = FinSensitivityConfig()
    mf_raw = raw.get("market_filter") or {}
    ex_raw = raw.get("exclusions") or {}
    market_filter = replace(
        base.market_filter,
        symbol=str(mf_raw.get("symbol", base.market_filter.symbol)),
        ma_len=int(mf_raw.get("ma_len", base.market_filter.ma_len)),
        fallback_self_ma=bool(
            mf_raw.get("fallback_self_ma", base.market_filter.fallback_self_ma)
        ),
    )
    exclusions = replace(
        base.exclusions,
        operating_loss_streak=int(
            ex_raw.get("operating_loss_streak", base.exclusions.operating_loss_streak)
        ),
        ocf_decline_streak=int(
            ex_raw.get("ocf_decline_streak", base.exclusions.ocf_decline_streak)
        ),
        debt_ratio_jump=float(
            ex_raw.get("debt_ratio_jump", base.exclusions.debt_ratio_jump)
        ),
        valuation_overheat_z=float(
            ex_raw.get("valuation_overheat_z", base.exclusions.valuation_overheat_z)
        ),
        max_missing_ratio=float(
            ex_raw.get("max_missing_ratio", base.exclusions.max_missing_ratio)
        ),
    )
    horizons = raw.get("horizons")
    return replace(
        base,
        interval=str(raw.get("interval", base.interval)),
        period=str(raw.get("period", base.period)),
        factors=_as_tuple(raw.get("factors"), base.factors),
        valuation_factors=_as_tuple(
            raw.get("valuation_factors"), base.valuation_factors
        ),
        train_quarters=int(raw.get("train_quarters", base.train_quarters)),
        min_train_quarters=int(
            raw.get("min_train_quarters", base.min_train_quarters)
        ),
        ridge_alpha=float(raw.get("ridge_alpha", base.ridge_alpha)),
        feature_set=str(raw.get("feature_set", base.feature_set)),
        horizons=tuple(int(h) for h in horizons) if horizons else base.horizons,
        availability_lag_quarter_days=int(
            raw.get("availability_lag_quarter_days",
                    base.availability_lag_quarter_days)
        ),
        availability_lag_annual_days=int(
            raw.get("availability_lag_annual_days",
                    base.availability_lag_annual_days)
        ),
        use_rate_feature=_as_bool(
            raw.get("use_rate_feature"), base.use_rate_feature
        ),
        rate_symbol=str(raw.get("rate_symbol", base.rate_symbol)),
        rate_change_lookback=int(
            raw.get("rate_change_lookback", base.rate_change_lookback)
        ),
        target_excess=_as_bool(raw.get("target_excess"), base.target_excess),
        pred20_min=float(raw.get("pred20_min", base.pred20_min)),
        pred60_min=float(raw.get("pred60_min", base.pred60_min)),
        stop_loss_pct=float(raw.get("stop_loss_pct", base.stop_loss_pct)),
        max_hold_days=int(raw.get("max_hold_days", base.max_hold_days)),
        fee_bps_per_side=float(raw.get("fee_bps_per_side", base.fee_bps_per_side)),
        execution=str(raw.get("execution", base.execution)),
        long_only=_as_bool(raw.get("long_only"), base.long_only),
        market_filter=market_filter,
        exclusions=exclusions,
        train_frac=float(raw.get("train_frac", base.train_frac)),
        validation_frac=float(raw.get("validation_frac", base.validation_frac)),
        synthetic_bars=int(raw.get("synthetic_bars", base.synthetic_bars)),
        seed=int(raw.get("seed", base.seed)),
    )
