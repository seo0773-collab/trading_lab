from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from trading_lab.paths import ROOT
from trading_lab.strategies.base import StrategyHandler


@dataclass(frozen=True)
class StrategyDefinition:
    strategy_id: str
    version: str
    description: str
    config_path: Path
    enabled: bool
    live_eligible: bool
    handler_factory: str  # "module:Callable" producing a StrategyHandler


_STRATEGIES = {
    "h72-price-v1": StrategyDefinition(
        strategy_id="h72-price-v1",
        version="1",
        description="Adaptive Kalman 72-bar PRICE direction strategy",
        config_path=ROOT / "configs" / "strategies" / "h72_price_v1.json",
        enabled=True,
        live_eligible=False,
        handler_factory="trading_lab.strategies.h72:H72Handler",
    ),
    "di-kalman-mw-v1": StrategyDefinition(
        strategy_id="di-kalman-mw-v1",
        version="1",
        description="+DI/-DI Kalman M/W 패턴 + pressure 우위 전략",
        config_path=ROOT / "configs" / "strategies" / "di_kalman_mw_v1.json",
        enabled=True,
        live_eligible=False,
        handler_factory="trading_lab.strategies.di_kalman_mw:DiKalmanMwHandler",
    ),
    "fin-sensitivity-v1": StrategyDefinition(
        strategy_id="fin-sensitivity-v1",
        version="1",
        description="재무 팩터 변화 민감도(rolling Ridge) 기반 long-only",
        config_path=ROOT / "configs" / "strategies" / "fin_sensitivity_v1.json",
        enabled=True,
        live_eligible=False,
        handler_factory="trading_lab.strategies.fin_sensitivity:FinSensitivityHandler",
    ),
    "profile-sizing-v1": StrategyDefinition(
        strategy_id="profile-sizing-v1",
        version="1",
        description="Cumulative profile percentile 기반 국면별 목표비중 사이징(long-only)",
        config_path=ROOT / "configs" / "strategies" / "profile_sizing_v1.json",
        enabled=True,
        live_eligible=False,
        handler_factory="trading_lab.strategies.profile_sizing:ProfileSizingHandler",
    ),
    "profile-sizing-trend-v1": StrategyDefinition(
        strategy_id="profile-sizing-trend-v1",
        version="1",
        description="profile sizing + 상승추세 가산(NORMAL/RECOVERY 익스포저 상향)",
        config_path=ROOT / "configs" / "strategies" / "profile_sizing_trend_v1.json",
        enabled=True,
        live_eligible=False,
        handler_factory="trading_lab.strategies.profile_sizing:ProfileSizingHandler",
    ),
    "profile-sizing-trend-v2": StrategyDefinition(
        strategy_id="profile-sizing-trend-v2",
        version="2",
        description="profile sizing 추세 floor(상승장 거의 풀투자 + 하락장만 방어)",
        config_path=ROOT / "configs" / "strategies" / "profile_sizing_trend_v2.json",
        enabled=True,
        live_eligible=False,
        handler_factory="trading_lab.strategies.profile_sizing:ProfileSizingHandler",
    ),
    "profile-sizing-exp-v1": StrategyDefinition(
        strategy_id="profile-sizing-exp-v1",
        version="1",
        description="profile sizing 지수형 비중 + cap 상향(구조적 저노출 완화)",
        config_path=ROOT / "configs" / "strategies" / "profile_sizing_exp_v1.json",
        enabled=True,
        live_eligible=False,
        handler_factory="trading_lab.strategies.profile_sizing:ProfileSizingHandler",
    ),
    "yoon1": StrategyDefinition(
        strategy_id="yoon1",
        version="1",
        description="다종목 포트폴리오(구 profile-portfolio-v1): 상위 K개 상승종목 추종 + 개별 방어 합산(현금화)",
        config_path=ROOT / "configs" / "strategies" / "yoon1.json",
        enabled=True,
        live_eligible=False,
        handler_factory="trading_lab.strategies.profile_portfolio:ProfilePortfolioHandler",
    ),
    "yoon1b": StrategyDefinition(
        strategy_id="yoon1b",
        version="1",
        description="yoon1 + 노출 게인 1.25(평상장 풀투자 근접). 방어 유지하며 수익 갭 축소(val→test 검증)",
        config_path=ROOT / "configs" / "strategies" / "yoon1b.json",
        enabled=True,
        live_eligible=False,
        handler_factory="trading_lab.strategies.profile_portfolio:ProfilePortfolioHandler",
    ),
}


def list_strategies() -> list[StrategyDefinition]:
    return list(_STRATEGIES.values())


def get_strategy(strategy_id: str) -> StrategyDefinition:
    try:
        return _STRATEGIES[strategy_id]
    except KeyError as exc:
        raise KeyError(f"unknown strategy: {strategy_id}") from exc


def get_handler(strategy_id: str) -> StrategyHandler:
    import importlib

    module_name, _, factory_name = get_strategy(
        strategy_id
    ).handler_factory.partition(":")
    factory = getattr(importlib.import_module(module_name), factory_name)
    return factory()
