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
    "yoon1h": StrategyDefinition(
        strategy_id="yoon1h",
        version="1",
        description="yoon1b의 percentile 사이징을 POC/VA 매물대 위치로 교체. 종목별 rolling 볼륨 프로파일에서 POC·VAH·VAL을 뽑아 현재가의 VA 대비 연속 위치(VAL→싸다/VAH→비싸다, 밖은 외삽)로 비중 결정. 포트폴리오 엔진·방어 로직은 yoon1b 그대로",
        config_path=ROOT / "configs" / "strategies" / "yoon1h.json",
        enabled=True,
        live_eligible=False,
        handler_factory="trading_lab.strategies.profile_portfolio:ProfilePortfolioHandler",
    ),
    "yoon1i": StrategyDefinition(
        strategy_id="yoon1i",
        version="1",
        description="yoon1b + heatmap2 HVN 지지/저항 기대값 게이트(블렌드). 종목별 인접 지지/저항으로 상방여지/하방위험 비율 EV=(저항−종가)/(저항−지지)를 [g_min,1] 게이트로 점수에 곱해 매수(지지근처)/매도(저항근처) 기대를 반영",
        config_path=ROOT / "configs" / "strategies" / "yoon1i.json",
        enabled=True,
        live_eligible=False,
        handler_factory="trading_lab.strategies.profile_portfolio:ProfilePortfolioHandler",
    ),
    "yoon1e": StrategyDefinition(
        strategy_id="yoon1e",
        version="1",
        description="yoon1b + SPY 200일선 약세 구간 현금 일부를 SPY 숏 헤지로 전환",
        config_path=ROOT / "configs" / "strategies" / "yoon1e.json",
        enabled=True,
        live_eligible=False,
        handler_factory="trading_lab.strategies.profile_portfolio:ProfilePortfolioHandler",
    ),
    "yoon1c": StrategyDefinition(
        strategy_id="yoon1c",
        version="1",
        description="yoon1b + 종목별 섹터 레짐 필터(SPY 단일 대신 SOXX/XLV/XLE 등 자기 섹터 추세로 방어)",
        config_path=ROOT / "configs" / "strategies" / "yoon1c.json",
        enabled=True,
        live_eligible=False,
        handler_factory="trading_lab.strategies.profile_portfolio:ProfilePortfolioHandler",
    ),
    "yoon1d": StrategyDefinition(
        strategy_id="yoon1d",
        version="1",
        description="yoon1b + SPY 일봉 RSI(14) MA 50 레짐 필터(50 미만이면 전체 노출 축소)",
        config_path=ROOT / "configs" / "strategies" / "yoon1d.json",
        enabled=True,
        live_eligible=False,
        handler_factory="trading_lab.strategies.profile_portfolio:ProfilePortfolioHandler",
    ),
    "yoon1f": StrategyDefinition(
        strategy_id="yoon1f",
        version="1",
        description="yoon1b 엔진을 섹터 ETF 11종+채권/금(TLT·GLD) 유니버스에 적용. 방어 신뢰형(생존편향X·저상관): vs SPY 위험조정 우위·MDD 절반, 수익은 양보. gain 절충 스윕 결과 1.0이 Sharpe·Calmar·MDD 최적(gain↑=낙폭만 악화)이라 게인 1.0 확정",
        config_path=ROOT / "configs" / "strategies" / "yoon1f.json",
        enabled=True,
        live_eligible=False,
        handler_factory="trading_lab.strategies.profile_portfolio:ProfilePortfolioHandler",
    ),
    "yoon1j": StrategyDefinition(
        strategy_id="yoon1j",
        version="1",
        description="yoon1f 섹터 엔진 + 연속형 안전자산 슬리브(safe-sleeve): 주식=공격(섹터11)/채권·금=방어 전담으로 역할 분리. 약세장 현금 완충분(spare)을 추세 ON(자기 100MA 위) 안전자산으로 연속 회전(이진 전환X→손실 상한=버퍼). r1.25/cap0.4/ma100. vs 섹터 현금방어: Sharpe·CAGR↑, MDD 동등(추세게이트로 2022 채권하락 회피)",
        config_path=ROOT / "configs" / "strategies" / "yoon1j.json",
        enabled=True,
        live_eligible=False,
        handler_factory="trading_lab.strategies.profile_portfolio:ProfilePortfolioHandler",
    ),
    "yoon1k": StrategyDefinition(
        strategy_id="yoon1k",
        version="1",
        description="계층 포트폴리오: 원/달러 50/50 통화배분 리밸런싱 × 통화별 yoon1j(한국 yoon1j_kr 원화 + 미국 yoon1j 달러→원화환산). 한국·미국 yoon1j 무상관(~0)+환율 다변화로 결합 Sharpe가 개별 슬리브(0.77~0.86)보다 도약(원화기준 2007~ Sharpe 1.15·MDD-14%). 수익은 양보·변동성/낙폭 최소(방어형). base_currency/fx_symbol/sub_strategies config",
        config_path=ROOT / "configs" / "strategies" / "yoon1k.json",
        enabled=True,
        live_eligible=False,
        handler_factory="trading_lab.strategies.yoon1k:Yoon1kHandler",
    ),
    "yoon1g": StrategyDefinition(
        strategy_id="yoon1g",
        version="1",
        description="yoon1f(섹터 방어) + 회복 레버리지 슬리브: '깊은 저가권 회복(RECOVERY) ∧ 시장정상(SPY>200MA)' 동시일 때만 2x 섹터를 태움. holdout Sharpe 1.07·full-cycle MDD-17.6%(=무레버 동일)로 2008 취약성 없이 상방만 보강",
        config_path=ROOT / "configs" / "strategies" / "yoon1g.json",
        enabled=True,
        live_eligible=False,
        handler_factory="trading_lab.strategies.profile_portfolio:ProfilePortfolioHandler",
    ),
    "yoon3": StrategyDefinition(
        strategy_id="yoon3",
        version="1",
        description="yoon1b + 칼만 히스토그램 누적프로파일 모멘텀 게이트(블렌드: 저가권×모멘텀). 종목별 kalHist 백분위를 [g_min,1] 게이트로 점수에 곱해 회복 진입 타이밍/노출 공백 보강",
        config_path=ROOT / "configs" / "strategies" / "yoon3.json",
        enabled=True,
        live_eligible=False,
        handler_factory="trading_lab.strategies.profile_portfolio:ProfilePortfolioHandler",
    ),
    "yoon2": StrategyDefinition(
        strategy_id="yoon2",
        version="1",
        description="단일종목 Kalman MACD 타이밍: 칼만 히스토그램 델타 전환 + 0선 추세필터 long-only(실데이터 검증: 추세종목 B&H 상회·전반 MDD 축소)",
        config_path=ROOT / "configs" / "strategies" / "yoon2.json",
        enabled=True,
        live_eligible=False,
        handler_factory="trading_lab.strategies.yoon2:Yoon2Handler",
    ),
    "heatmap1": StrategyDefinition(
        strategy_id="heatmap1",
        version="1",
        description="단일종목 Volume Profile 신호: 롤링 프로파일 POC/VAH/VAL 기반 밸류에어리어 평균회귀(va_reversion)/돌파(va_breakout). 소스무관(equity=yfinance, crypto=ccxt)",
        config_path=ROOT / "configs" / "strategies" / "heatmap1.json",
        enabled=True,
        live_eligible=False,
        handler_factory="trading_lab.strategies.heatmap1:Heatmap1Handler",
    ),
    "heatmap2": StrategyDefinition(
        strategy_id="heatmap2",
        version="1",
        description="단일종목 고볼륨 노드(HVN) 지지/저항 롱숏: 롤링 볼륨 프로파일의 '색 짙은 구간'(로컬 볼륨 피크)을 추출해 현재가 인접 지지/저항으로 보고 반등(지지→롱/저항→숏)/돌파 매매. 로그스케일·양방향. heatmap1 시뮬엔진 상속",
        config_path=ROOT / "configs" / "strategies" / "heatmap2.json",
        enabled=True,
        live_eligible=False,
        handler_factory="trading_lab.strategies.heatmap2:Heatmap2Handler",
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
