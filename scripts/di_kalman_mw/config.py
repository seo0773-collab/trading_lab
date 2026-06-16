"""Configuration for the DI Kalman M/W strategy (plan.txt)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class IndicatorConfig:
    di_len: int = 14
    atr_len: int = 14
    kalman_q: float = 0.01
    kalman_r: float = 1.0


@dataclass(frozen=True)
class ExtremeConfig:
    # plan 5A.1: reversal threshold = reversal_mult * rolling_std(kalman, window)
    reversal_mult: float = 1.0
    reversal_std_window: int = 50


@dataclass(frozen=True)
class PatternConfig:
    # plan 5A.2 strengthening condition (W: P4 > P2, M: P4 < P2)
    strict: bool = False
    # Right/left channel-width ratio band that separates the M/W shape into
    # diverging / parallel / converging. A pattern is "parallel" while the
    # ratio stays within 1 +/- parallel_band (shape label is independent of
    # the classify_pattern gate).
    parallel_band: float = 0.20
    # plan 6/8: allow a "weak" setup where only the directional (W-forming)
    # DI line confirms the pattern and the opposite line merely supplies
    # pressure alignment (>= weak_pressure_min). Disable for strict W&M only.
    allow_weak_setup: bool = True
    weak_pressure_min: float = 0.55


@dataclass(frozen=True)
class SignalConfig:
    entry_variant: str = "p4"  # "p4" | "p5" (plan 8)
    pressure_score_min: float = 0.50
    ma_filter_len: int = 0  # 0 = disabled
    atr_pct_min: float = 0.0  # volatility filter; 0 / inf = disabled
    atr_pct_max: float = float("inf")
    require_positive_ev: bool = True  # plan 7: pressure_adjusted_expected_value > 0
    # plan 6/7: fold the next-extreme continuation probability (extreme_transition)
    # into the expected value. Neutral (0.5) leaves EV unchanged.
    use_transition: bool = True
    transition_min_bucket: int = 20


@dataclass(frozen=True)
class ExitConfig:
    stop_type: str = "atr"  # "atr" | "swing" (plan 9)
    atr_stop_mult: float = 2.0
    swing_lookback: int = 10
    swing_buffer_mult: float = 0.5
    tp_type: str = "fixed_r"  # "fixed_r" | "pressure_rr" | "none" (plan 10)
    rr_target: float = 2.0
    base_rr: float = 1.5
    rr_min: float = 1.0
    rr_max: float = 3.0
    trailing: bool = False
    trail_mult: float = 2.0
    opposite_exit: bool = True
    max_hold_bars: int = 48  # plan 11
    # plan 10: scale the pressure_rr take-profit target by the pattern's
    # continuation probability (>0.5 widens, <0.5 tightens). Only affects
    # tp_type == "pressure_rr".
    continuation_rr: bool = True


@dataclass(frozen=True)
class CostConfig:
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0005
    cost_mult: float = 1.0  # plan 15 Phase 4 cost sensitivity

    @property
    def fee(self) -> float:
        return self.fee_rate * self.cost_mult

    @property
    def slippage(self) -> float:
        return self.slippage_rate * self.cost_mult

    @property
    def round_trip_cost(self) -> float:
        return 2.0 * (self.fee + self.slippage)


@dataclass(frozen=True)
class SplitConfig:
    train_frac: float = 0.60
    validation_frac: float = 0.20  # remainder is test


@dataclass(frozen=True)
class StatsConfig:
    # plan 7A: Pass 1 baseline exit rule and bucket sample minimums
    min_bucket_trades: int = 30
    min_global_trades: int = 10
    baseline_atr_stop_mult: float = 2.0
    baseline_rr_target: float = 2.0
    baseline_max_hold_bars: int = 48


@dataclass(frozen=True)
class SimilarityEvConfig:
    enabled: bool = False
    neighbors: int = 50
    temperature: float = 1.0
    min_neighbors: int = 10
    entry_margin: float = 0.0
    lower_quantile: float = 0.25


@dataclass(frozen=True)
class OnlineConfig:
    enabled: bool = False
    neighbors: int = 50
    temperature: float = 1.0
    min_neighbors: int = 10
    hold_threshold: float = 0.0
    exit_threshold: float = 0.0
    reversal_margin: float = 0.002
    min_confidence: float = 0.25
    confirm_bars: int = 2


@dataclass(frozen=True)
class StrategyConfig:
    direction_mode: str = "both"  # "long" | "short" | "both"
    indicators: IndicatorConfig = field(default_factory=IndicatorConfig)
    extremes: ExtremeConfig = field(default_factory=ExtremeConfig)
    patterns: PatternConfig = field(default_factory=PatternConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    exits: ExitConfig = field(default_factory=ExitConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    stats: StatsConfig = field(default_factory=StatsConfig)
    similarity_ev: SimilarityEvConfig = field(
        default_factory=SimilarityEvConfig
    )
    online: OnlineConfig = field(default_factory=OnlineConfig)


def combo_config(name: str) -> StrategyConfig:
    """Discussion combos A-D from plan 17."""
    name = name.upper()
    if name == "A":
        return StrategyConfig(
            signal=SignalConfig(entry_variant="p4"),
            exits=ExitConfig(
                stop_type="atr", atr_stop_mult=2.0, tp_type="pressure_rr",
                max_hold_bars=48, opposite_exit=True,
            ),
        )
    if name == "B":
        return StrategyConfig(
            signal=SignalConfig(entry_variant="p5"),
            exits=ExitConfig(
                stop_type="atr", atr_stop_mult=2.0, tp_type="fixed_r",
                rr_target=2.0, max_hold_bars=48, opposite_exit=True,
            ),
        )
    if name == "C":
        return StrategyConfig(
            signal=SignalConfig(entry_variant="p5"),
            exits=ExitConfig(
                stop_type="atr", atr_stop_mult=2.0, tp_type="none",
                trailing=True, trail_mult=2.0, max_hold_bars=72,
                opposite_exit=True,
            ),
        )
    if name == "D":
        return StrategyConfig(
            signal=SignalConfig(entry_variant="p4"),
            exits=ExitConfig(
                stop_type="atr", atr_stop_mult=1.5, tp_type="fixed_r",
                rr_target=1.5, max_hold_bars=20, opposite_exit=True,
            ),
        )
    raise ValueError(f"unknown combo: {name}")


# plan 4: train sufficiency rules; passing any one criterion is enough.
TRAIN_SUFFICIENCY_RULES = {
    "1h": {"min_years": 2.0, "min_candles": 10_000, "min_events": 500},
    "4h": {"min_years": 3.0, "min_candles": 5_000, "min_events": 300},
    "1d": {"min_years": 10.0, "min_candles": 1_500, "min_events": None},
}


def timeframe_minutes(timeframe: str) -> float:
    units = {"m": 1.0, "h": 60.0, "d": 1440.0, "w": 10080.0}
    tf = timeframe.strip().lower()
    try:
        return float(tf[:-1]) * units[tf[-1]]
    except (KeyError, ValueError, IndexError):
        raise ValueError(f"cannot parse timeframe: {timeframe!r}") from None


def bars_per_year(timeframe: str) -> float:
    return 525_600.0 / timeframe_minutes(timeframe)


def sufficiency_rule(timeframe: str) -> dict:
    minutes = timeframe_minutes(timeframe)
    if minutes <= 60:
        return TRAIN_SUFFICIENCY_RULES["1h"]
    if minutes < 1440:
        return TRAIN_SUFFICIENCY_RULES["4h"]
    return TRAIN_SUFFICIENCY_RULES["1d"]
