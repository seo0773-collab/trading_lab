"""profile-sizing-v1 설정 (profile_plan.txt §14).

대시보드는 TF/기간 위젯을 ``interval``/``period`` 로, 튜너블 패널을 평평한 JSON
키로 흘려보낸다. ``config_from_dict`` 가 그 평평한 config를 타입드 구조로 매핑한다.
모르는 키는 무시하고, 없는 키는 dataclass 기본값으로 떨어지므로 부분 override도
안전하다.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

# (low_pct, high_pct, weight) — cumulative percentile(%) 구간별 기본 주식 비중.
# percentile이 낮을(=싸다) 수록 비중 ↑. plan §7.
DEFAULT_BUCKETS: tuple[tuple[float, float, float], ...] = (
    (0.0, 20.0, 0.80),
    (20.0, 40.0, 0.60),
    (40.0, 60.0, 0.50),
    (60.0, 80.0, 0.30),
    (80.0, 100.0, 0.10),
)


@dataclass(frozen=True)
class BaseCycle:
    type: str = "SMA"        # SMA | EMA | RMA | WMA | VWMA
    length: int = 200
    scale: float = 1.0


@dataclass(frozen=True)
class Profile:
    min_mult: float = 0.0
    max_mult: float = 3.0
    bin_count: int = 120
    rolling_window: int = 126
    weight_mode: str = "time"        # time | volume | volume_fallback
    accumulation_mode: str = "range_uniform"  # range_uniform | ohlc | range_close
    percentile_value: float = 20.0   # lower/upper percentile 산출용(%)


@dataclass(frozen=True)
class RegimeCap:
    NORMAL: float = 1.00
    CAUTION: float = 0.60
    DEFENSE: float = 0.30
    RECOVERY: float = 0.50
    # RECOVERY 단계별 상향(plan §8). 회복 지속 봉수마다 단계 ↑.
    recovery_stage_bars: int = 21
    recovery_stage_caps: tuple[float, ...] = (0.50, 0.70, 1.00)


@dataclass(frozen=True)
class Rebalance:
    threshold: float = 0.03           # 비중 차이가 이보다 작으면 거래 안 함
    max_trade_weight_per_bar: float = 0.20
    defense_buy_allowed: bool = False  # 방어장 신규 매수 금지(plan §10)


@dataclass(frozen=True)
class Costs:
    fee_bps_per_side: float = 5.0     # 편도 수수료(bp)
    slippage_bps: float = 5.0         # 슬리피지(bp)


@dataclass(frozen=True)
class TrendOverlay:
    """추세 가산(profile-sizing-trend 변형). 기본 off → baseline 동작 불변.

    percentile 기반 contrarian 비중을 강한 상승추세(가격이 base_cycle 위 + 기준선
    자체 상승)에서 깎지 않도록 보너스를 더한다. 방어장(CAUTION/DEFENSE)에는
    적용하지 않아 하락장 방어 엣지는 보존한다.
    """
    enabled: bool = False
    boost_gain: float = 0.8          # trend_strength(=cm_close-1) 단위당 가산
    max_boost: float = 0.40          # 가산 상한
    apply_regimes: tuple[str, ...] = ("NORMAL", "RECOVERY")
    # 상승추세 목표비중 바닥(0=비활성). >0이면 강한 상승추세(추세강도>0) +
    # apply_regimes 국면에서 percentile 감점을 무시하고 비중을 floor 이상으로 보장한다.
    # 단 regime cap에는 여전히 종속(추세 꺾이면 해제 → DEFENSE가 받아냄).
    floor: float = 0.0


@dataclass(frozen=True)
class ProfileSizingConfig:
    interval: str = "1d"
    period: str = "max"

    base_cycle: BaseCycle = field(default_factory=BaseCycle)
    profile: Profile = field(default_factory=Profile)
    regime_cap: RegimeCap = field(default_factory=RegimeCap)
    rebalance: Rebalance = field(default_factory=Rebalance)
    costs: Costs = field(default_factory=Costs)
    trend_overlay: TrendOverlay = field(default_factory=TrendOverlay)

    weight_model: str = "bucket"      # bucket | exponential
    buckets: tuple[tuple[float, float, float], ...] = DEFAULT_BUCKETS
    # exponential 모델 파라미터: w = max_w * exp(-k * percentile)
    exp_max_weight: float = 1.0
    exp_k: float = 2.0

    # 검증 분할(다른 전략과 동일 관례)
    train_frac: float = 0.6
    validation_frac: float = 0.2

    # 합성 경로(계약 테스트)
    synthetic_bars: int = 3000
    seed: int = 7

    @property
    def warmup(self) -> int:
        """base_cycle·rolling profile이 의미를 갖기 전까지의 봉 수."""
        return max(self.base_cycle.length, self.profile.rolling_window)


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"true", "1", "on", "yes"}


def _buckets_from(value: Any, default) -> tuple[tuple[float, float, float], ...]:
    if not value:
        return default
    out = []
    for row in value:
        lo, hi, w = row
        out.append((float(lo), float(hi), float(w)))
    return tuple(out)


def config_from_dict(raw: dict[str, Any]) -> ProfileSizingConfig:
    base = ProfileSizingConfig()
    bc_raw = raw.get("base_cycle") or {}
    pf_raw = raw.get("profile") or {}
    rc_raw = raw.get("regime_cap") or {}
    rb_raw = raw.get("rebalance") or {}
    co_raw = raw.get("costs") or {}

    base_cycle = replace(
        base.base_cycle,
        type=str(bc_raw.get("type", base.base_cycle.type)),
        length=int(bc_raw.get("length", base.base_cycle.length)),
        scale=float(bc_raw.get("scale", base.base_cycle.scale)),
    )
    profile = replace(
        base.profile,
        min_mult=float(pf_raw.get("min_mult", base.profile.min_mult)),
        max_mult=float(pf_raw.get("max_mult", base.profile.max_mult)),
        bin_count=int(pf_raw.get("bin_count", base.profile.bin_count)),
        rolling_window=int(pf_raw.get("rolling_window", base.profile.rolling_window)),
        weight_mode=str(pf_raw.get("weight_mode", base.profile.weight_mode)),
        accumulation_mode=str(
            pf_raw.get("accumulation_mode", base.profile.accumulation_mode)
        ),
        percentile_value=float(
            pf_raw.get("percentile_value", base.profile.percentile_value)
        ),
    )
    regime_cap = replace(
        base.regime_cap,
        NORMAL=float(rc_raw.get("NORMAL", base.regime_cap.NORMAL)),
        CAUTION=float(rc_raw.get("CAUTION", base.regime_cap.CAUTION)),
        DEFENSE=float(rc_raw.get("DEFENSE", base.regime_cap.DEFENSE)),
        RECOVERY=float(rc_raw.get("RECOVERY", base.regime_cap.RECOVERY)),
        recovery_stage_bars=int(
            rc_raw.get("recovery_stage_bars", base.regime_cap.recovery_stage_bars)
        ),
        recovery_stage_caps=tuple(
            float(v) for v in rc_raw.get(
                "recovery_stage_caps", base.regime_cap.recovery_stage_caps
            )
        ),
    )
    rebalance = replace(
        base.rebalance,
        threshold=float(rb_raw.get("threshold", base.rebalance.threshold)),
        max_trade_weight_per_bar=float(
            rb_raw.get("max_trade_weight_per_bar",
                       base.rebalance.max_trade_weight_per_bar)
        ),
        defense_buy_allowed=_as_bool(
            rb_raw.get("defense_buy_allowed"), base.rebalance.defense_buy_allowed
        ),
    )
    costs = replace(
        base.costs,
        fee_bps_per_side=float(co_raw.get("fee_bps_per_side", base.costs.fee_bps_per_side)),
        slippage_bps=float(co_raw.get("slippage_bps", base.costs.slippage_bps)),
    )
    to_raw = raw.get("trend_overlay") or {}
    trend_overlay = replace(
        base.trend_overlay,
        enabled=_as_bool(to_raw.get("enabled"), base.trend_overlay.enabled),
        boost_gain=float(to_raw.get("boost_gain", base.trend_overlay.boost_gain)),
        max_boost=float(to_raw.get("max_boost", base.trend_overlay.max_boost)),
        apply_regimes=tuple(
            str(r) for r in to_raw.get("apply_regimes",
                                       base.trend_overlay.apply_regimes)
        ),
        floor=float(to_raw.get("floor", base.trend_overlay.floor)),
    )
    return replace(
        base,
        interval=str(raw.get("interval", base.interval)),
        period=str(raw.get("period", base.period)),
        base_cycle=base_cycle,
        profile=profile,
        regime_cap=regime_cap,
        rebalance=rebalance,
        costs=costs,
        trend_overlay=trend_overlay,
        weight_model=str(raw.get("weight_model", base.weight_model)),
        buckets=_buckets_from(raw.get("buckets"), base.buckets),
        exp_max_weight=float(raw.get("exp_max_weight", base.exp_max_weight)),
        exp_k=float(raw.get("exp_k", base.exp_k)),
        train_frac=float(raw.get("train_frac", base.train_frac)),
        validation_frac=float(raw.get("validation_frac", base.validation_frac)),
        synthetic_bars=int(raw.get("synthetic_bars", base.synthetic_bars)),
        seed=int(raw.get("seed", base.seed)),
    )
