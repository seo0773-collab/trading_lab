from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from trading_lab.paths import ROOT
from trading_lab.strategies.base import StrategyHandler

# 전략 분류(UI 선택 그룹 + 입력 위젯 결정). 핸들러 문자열로 추측하지 않고
# registry에서 명시 선언한다 — 새 핸들러를 만들어도 UI가 깨지지 않도록.
#   single        : 단일자산(종목 1개 선택)
#   portfolio     : 다종목 포트폴리오(유니버스 선택)
#   hierarchical  : 계층 포트폴리오(fund-of-funds; 구성이 config에 고정 → 종목 선택 무의미)
CATEGORY_SINGLE = "single"
CATEGORY_PORTFOLIO = "portfolio"
CATEGORY_HIERARCHICAL = "hierarchical"
STRATEGY_CATEGORIES = (CATEGORY_SINGLE, CATEGORY_PORTFOLIO, CATEGORY_HIERARCHICAL)


@dataclass(frozen=True)
class StrategyDefinition:
    strategy_id: str
    version: str
    description: str
    config_path: Path
    enabled: bool
    live_eligible: bool
    handler_factory: str  # "module:Callable" producing a StrategyHandler
    category: str = CATEGORY_SINGLE


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
    "rolling-profile-expectancy-v1": StrategyDefinition(
        strategy_id="rolling-profile-expectancy-v1",
        version="1",
        description="63봉 rolling multiple profile 근사분포 위 캔들 위치 기대값 기반 long-only 비중",
        config_path=ROOT / "configs" / "strategies" / "rolling_profile_expectancy_v1.json",
        enabled=True,
        live_eligible=False,
        handler_factory="trading_lab.strategies.rolling_profile_expectancy:RollingProfileExpectancyHandler",
    ),
    "yoon_pattern": StrategyDefinition(
        strategy_id="yoon_pattern",
        version="1",
        description="300틱 패턴 투영(유사 패턴 앙상블 직후수익 중앙값) 기대값 사이징 long-only",
        config_path=ROOT / "configs" / "strategies" / "yoon_pattern.json",
        enabled=True,
        live_eligible=False,
        handler_factory="trading_lab.strategies.yoon_pattern:YoonPatternHandler",
    ),
    "yoon_profile": StrategyDefinition(
        strategy_id="yoon_profile",
        version="1",
        description="SMA200 멀티플×기울기 z-각도 expanding percentile 3×3 상태 가중치 사이징 long-only (BTC 일봉 검증: 눌림목 알파·칼 회피·모멘텀 지속 셀)",
        config_path=ROOT / "configs" / "strategies" / "yoon_profile.json",
        enabled=True,
        live_eligible=False,
        handler_factory="trading_lab.strategies.yoon_profile:YoonProfileHandler",
    ),
    "yoon_profile_1": StrategyDefinition(
        strategy_id="yoon_profile_1",
        version="1",
        description="yoon_profile + 떨어지는 칼 셀(0,0) 숏 −0.5 + trail15 스퀴즈 가드(에피소드 최저가 +15% 반등 시 숏 금지). holdout 수익·Sharpe 개선(+23.1%/0.487 vs +15.7%/0.395), 펀딩비용 미반영",
        config_path=ROOT / "configs" / "strategies" / "yoon_profile_1.json",
        enabled=True,
        live_eligible=False,
        handler_factory="trading_lab.strategies.yoon_profile:YoonProfileHandler",
    ),
    "yoon_profile_2": StrategyDefinition(
        strategy_id="yoon_profile_2",
        version="1",
        description="주식형 yoon_profile: 숏 제거, 워밍업 100% 보유, 상태별 85~100% 노출. 저멀티플+하락기울기를 숏이 아닌 점진 매수/보유 구간으로 해석",
        config_path=ROOT / "configs" / "strategies" / "yoon_profile_2.json",
        enabled=True,
        live_eligible=False,
        handler_factory="trading_lab.strategies.yoon_profile:YoonProfileHandler",
    ),
    "yoon_profile_3": StrategyDefinition(
        strategy_id="yoon_profile_3",
        version="1",
        description="yoon_profile + path_tilt: M1A1 셀을 필드 속도(mult_pct 10봉 변화) 부호로 분리(상승 진입 1.0/하강 진입 0.5). BTC holdout 수익·Sharpe·MDD 모두 개선(+18.5%/0.439/−20.2% vs +15.7%/0.391/−21.4%)",
        config_path=ROOT / "configs" / "strategies" / "yoon_profile_3.json",
        enabled=True,
        live_eligible=False,
        handler_factory="trading_lab.strategies.yoon_profile:YoonProfileHandler",
    ),
    "yoon2m_defensive": StrategyDefinition(
        strategy_id="yoon2m_defensive",
        version="1",
        description="yoon2m 구조에 y2m_defensive 3x3 레짐 필드 점수를 적용하고, 종목/일봉별 필드 위치 누적 히스토리를 dashboard에 노출",
        config_path=ROOT / "configs" / "strategies" / "yoon2m_defensive.json",
        enabled=True,
        live_eligible=False,
        handler_factory="trading_lab.strategies.yoon2m_defensive:Yoon2mDefensiveHandler",
        category=CATEGORY_HIERARCHICAL,
    ),
    "yoon1": StrategyDefinition(
        strategy_id="yoon1",
        version="1",
        description="다종목 포트폴리오(구 profile-portfolio-v1): 상위 K개 상승종목 추종 + 개별 방어 합산(현금화)",
        config_path=ROOT / "configs" / "strategies" / "yoon1.json",
        enabled=True,
        live_eligible=False,
        handler_factory="trading_lab.strategies.profile_portfolio:ProfilePortfolioHandler",
        category=CATEGORY_PORTFOLIO,
    ),
    "yoon1b": StrategyDefinition(
        strategy_id="yoon1b",
        version="1",
        description="yoon1 + 노출 게인 1.25(평상장 풀투자 근접). 방어 유지하며 수익 갭 축소(val→test 검증)",
        config_path=ROOT / "configs" / "strategies" / "yoon1b.json",
        enabled=True,
        live_eligible=False,
        handler_factory="trading_lab.strategies.profile_portfolio:ProfilePortfolioHandler",
        category=CATEGORY_PORTFOLIO,
    ),
    "yoon1h": StrategyDefinition(
        strategy_id="yoon1h",
        version="1",
        description="yoon1b의 percentile 사이징을 POC/VA 매물대 위치로 교체. 종목별 rolling 볼륨 프로파일에서 POC·VAH·VAL을 뽑아 현재가의 VA 대비 연속 위치(VAL→싸다/VAH→비싸다, 밖은 외삽)로 비중 결정. 포트폴리오 엔진·방어 로직은 yoon1b 그대로",
        config_path=ROOT / "configs" / "strategies" / "yoon1h.json",
        enabled=True,
        live_eligible=False,
        handler_factory="trading_lab.strategies.profile_portfolio:ProfilePortfolioHandler",
        category=CATEGORY_PORTFOLIO,
    ),
    "yoon1i": StrategyDefinition(
        strategy_id="yoon1i",
        version="1",
        description="yoon1b + heatmap2 HVN 지지/저항 기대값 게이트(블렌드). 종목별 인접 지지/저항으로 상방여지/하방위험 비율 EV=(저항−종가)/(저항−지지)를 [g_min,1] 게이트로 점수에 곱해 매수(지지근처)/매도(저항근처) 기대를 반영",
        config_path=ROOT / "configs" / "strategies" / "yoon1i.json",
        enabled=True,
        live_eligible=False,
        handler_factory="trading_lab.strategies.profile_portfolio:ProfilePortfolioHandler",
        category=CATEGORY_PORTFOLIO,
    ),
    "yoon1e": StrategyDefinition(
        strategy_id="yoon1e",
        version="1",
        description="yoon1b + SPY 200일선 약세 구간 현금 일부를 SPY 숏 헤지로 전환",
        config_path=ROOT / "configs" / "strategies" / "yoon1e.json",
        enabled=True,
        live_eligible=False,
        handler_factory="trading_lab.strategies.profile_portfolio:ProfilePortfolioHandler",
        category=CATEGORY_PORTFOLIO,
    ),
    "yoon1c": StrategyDefinition(
        strategy_id="yoon1c",
        version="1",
        description="yoon1b + 종목별 섹터 레짐 필터(SPY 단일 대신 SOXX/XLV/XLE 등 자기 섹터 추세로 방어)",
        config_path=ROOT / "configs" / "strategies" / "yoon1c.json",
        enabled=True,
        live_eligible=False,
        handler_factory="trading_lab.strategies.profile_portfolio:ProfilePortfolioHandler",
        category=CATEGORY_PORTFOLIO,
    ),
    "yoon1d": StrategyDefinition(
        strategy_id="yoon1d",
        version="1",
        description="yoon1b + SPY 일봉 RSI(14) MA 50 레짐 필터(50 미만이면 전체 노출 축소)",
        config_path=ROOT / "configs" / "strategies" / "yoon1d.json",
        enabled=True,
        live_eligible=False,
        handler_factory="trading_lab.strategies.profile_portfolio:ProfilePortfolioHandler",
        category=CATEGORY_PORTFOLIO,
    ),
    "yoon1f": StrategyDefinition(
        strategy_id="yoon1f",
        version="1",
        description="yoon1b 엔진을 섹터 ETF 11종+채권/금(TLT·GLD) 유니버스에 적용. 방어 신뢰형(생존편향X·저상관): vs SPY 위험조정 우위·MDD 절반, 수익은 양보. gain 절충 스윕 결과 1.0이 Sharpe·Calmar·MDD 최적(gain↑=낙폭만 악화)이라 게인 1.0 확정",
        config_path=ROOT / "configs" / "strategies" / "yoon1f.json",
        enabled=True,
        live_eligible=False,
        handler_factory="trading_lab.strategies.profile_portfolio:ProfilePortfolioHandler",
        category=CATEGORY_PORTFOLIO,
    ),
    "yoon1j": StrategyDefinition(
        strategy_id="yoon1j",
        version="1",
        description="yoon1f 섹터 엔진 + 연속형 안전자산 슬리브(safe-sleeve): 주식=공격(섹터11)/채권·금=방어 전담으로 역할 분리. 약세장 현금 완충분(spare)을 추세 ON(자기 100MA 위) 안전자산으로 연속 회전(이진 전환X→손실 상한=버퍼). r1.25/cap0.4/ma100. vs 섹터 현금방어: Sharpe·CAGR↑, MDD 동등(추세게이트로 2022 채권하락 회피)",
        config_path=ROOT / "configs" / "strategies" / "yoon1j.json",
        enabled=True,
        live_eligible=False,
        handler_factory="trading_lab.strategies.profile_portfolio:ProfilePortfolioHandler",
        category=CATEGORY_PORTFOLIO,
    ),
    "yoon1k": StrategyDefinition(
        strategy_id="yoon1k",
        version="1",
        description="계층 포트폴리오: 원/달러 50/50 통화배분 리밸런싱 × 통화별 yoon1j(한국 yoon1j_kr 원화 + 미국 yoon1j 달러→원화환산). 한국·미국 yoon1j 무상관(~0)+환율 다변화로 결합 Sharpe가 개별 슬리브(0.77~0.86)보다 도약(원화기준 2007~ Sharpe 1.15·MDD-14%). 수익은 양보·변동성/낙폭 최소(방어형). base_currency/fx_symbol/sub_strategies config",
        config_path=ROOT / "configs" / "strategies" / "yoon1k.json",
        enabled=True,
        live_eligible=False,
        handler_factory="trading_lab.strategies.yoon1k:Yoon1kHandler",
        category=CATEGORY_HIERARCHICAL,
    ),
    "yoon1k_levmix": StrategyDefinition(
        strategy_id="yoon1k_levmix",
        version="1",
        description="yoon2 파생 공격형: 2단계 yoon2 코어 60% + QLD(2x Nasdaq100) 40%를 월간 리밸런싱. 50/50 패시브 B&H와 유사한 MDD를 허용하고 상승폭 확대",
        config_path=ROOT / "configs" / "strategies" / "yoon1k_levmix.json",
        enabled=True,
        live_eligible=False,
        handler_factory="trading_lab.strategies.yoon1k_leverage_mix:Yoon1kLeverageMixHandler",
        category=CATEGORY_HIERARCHICAL,
    ),
    "yoon1a": StrategyDefinition(
        strategy_id="yoon1a",
        version="1",
        description="yoon2 파생 공격형: 2단계 yoon2 코어 65% + USD(2x Semiconductors) 35%를 월간 리밸런싱. 50/50 패시브 B&H와 유사한 MDD에서 CAGR 극대화 후보",
        config_path=ROOT / "configs" / "strategies" / "yoon1a.json",
        enabled=True,
        live_eligible=False,
        handler_factory="trading_lab.strategies.yoon1k_leverage_mix:Yoon1kLeverageMixHandler",
        category=CATEGORY_HIERARCHICAL,
    ),
    "yoon2a": StrategyDefinition(
        strategy_id="yoon2a",
        version="1",
        description="yoon2 파생 공격형(생존편향 회피): 2단계 yoon2 코어 74% + SSO(2x S&P500) 26%를 월간 리밸런싱. 섹터/지수 손픽 대신 광의 시장 레버리지를 써 hindsight 편향 제거. 비중은 Strict+Sharpe 재최적화(B&H를 MDD·Sharpe 양 축에서 동시 지배)",
        config_path=ROOT / "configs" / "strategies" / "yoon2a.json",
        enabled=True,
        live_eligible=False,
        handler_factory="trading_lab.strategies.yoon1k_leverage_mix:Yoon1kLeverageMixHandler",
        category=CATEGORY_HIERARCHICAL,
    ),
    "qld_bh_krw": StrategyDefinition(
        strategy_id="qld_bh_krw",
        version="1",
        description="QLD(2x Nasdaq100) 100% buy & hold, USD/KRW 환산. yoon1k 계열 50/50 패시브 벤치와 비교하는 고수익·고낙폭 기준선",
        config_path=ROOT / "configs" / "strategies" / "qld_bh_krw.json",
        enabled=True,
        live_eligible=False,
        handler_factory="trading_lab.strategies.yoon1k_leverage_mix:Yoon1kLeverageMixHandler",
        category=CATEGORY_HIERARCHICAL,
    ),
    "yoon1g": StrategyDefinition(
        strategy_id="yoon1g",
        version="1",
        description="yoon1f(섹터 방어) + 회복 레버리지 슬리브: '깊은 저가권 회복(RECOVERY) ∧ 시장정상(SPY>200MA)' 동시일 때만 2x 섹터를 태움. holdout Sharpe 1.07·full-cycle MDD-17.6%(=무레버 동일)로 2008 취약성 없이 상방만 보강",
        config_path=ROOT / "configs" / "strategies" / "yoon1g.json",
        enabled=True,
        live_eligible=False,
        handler_factory="trading_lab.strategies.profile_portfolio:ProfilePortfolioHandler",
        category=CATEGORY_PORTFOLIO,
    ),
    "yoon3": StrategyDefinition(
        strategy_id="yoon3",
        version="1",
        description="yoon1b + 칼만 히스토그램 누적프로파일 모멘텀 게이트(블렌드: 저가권×모멘텀). 종목별 kalHist 백분위를 [g_min,1] 게이트로 점수에 곱해 회복 진입 타이밍/노출 공백 보강",
        config_path=ROOT / "configs" / "strategies" / "yoon3.json",
        enabled=True,
        live_eligible=False,
        handler_factory="trading_lab.strategies.profile_portfolio:ProfilePortfolioHandler",
        category=CATEGORY_PORTFOLIO,
    ),
    "yoon2": StrategyDefinition(
        strategy_id="yoon2",
        version="2",
        description="2단계 계층 전략: 미국 저가형 yoon1j_lp(USD) + 한국 yoon1j_kr(KRW)를 원화 기준 50/50 월간 리밸런싱",
        config_path=ROOT / "configs" / "strategies" / "yoon2.json",
        enabled=True,
        live_eligible=False,
        handler_factory="trading_lab.strategies.yoon1k:Yoon1kHandler",
        category=CATEGORY_HIERARCHICAL,
    ),
    "yoon2m": StrategyDefinition(
        strategy_id="yoon2m",
        version="1",
        description="yoon2 저예산(mini) 파생: 두 슬리브 top_k=3으로 집중(보유 6종=US3+KR3, SSO 없음) → 200만원 정수주 자동매수 적합. 백테스트 2007~2026 CAGR 15.30%/MDD -18.06%/Sharpe 1.157로 원본 yoon2를 수익·Sharpe에서 개선하며 방어 유지(패시브 -31%의 절반대). 부트스트랩 P(MDD<벤치)=95.2%·롤링WF 17/18로 낙폭방어 통계 확정",
        config_path=ROOT / "configs" / "strategies" / "yoon2m.json",
        enabled=True,
        live_eligible=False,
        handler_factory="trading_lab.strategies.yoon1k:Yoon1kHandler",
        category=CATEGORY_HIERARCHICAL,
    ),
    "yoon2_final": StrategyDefinition(
        strategy_id="yoon2_final",
        version="1",
        description="yoon2 계열 최종 확정형(2026-07-05): yoon2m 검증 완료 상태의 동결 스냅샷 — US yoon1j_lp(VGT)+KR yoon1j_kr 50/50, top_k 3/3, 월간, 안전슬리브 유지, SSO 없음. CAGR 15.49%/MDD -17.89%/Sharpe 1.169, 부트스트랩 P(MDD<벤치)=95.6%. 안전슬리브 ablation·리밸런스 4축·시장 오버레이 2종 기각·GFC 대리검증 완료된 방어형. config 동결(실험은 파생으로만), 승격 게이트=페이퍼 전진검증+KIS 실수수료 확정",
        config_path=ROOT / "configs" / "strategies" / "yoon2_final.json",
        enabled=True,
        live_eligible=False,
        handler_factory="trading_lab.strategies.yoon1k:Yoon1kHandler",
        category=CATEGORY_HIERARCHICAL,
    ),
    "yoon2c": StrategyDefinition(
        strategy_id="yoon2c",
        version="1",
        description="yoon2 통화배분 업그레이드: World Bank 버핏지수 변화율 revert 비중을 k=1.25로 50/50 주변에 scaled 적용. MDD를 yoon2와 거의 유지하면서 CAGR/총수익 개선 후보",
        config_path=ROOT / "configs" / "strategies" / "yoon2c.json",
        enabled=True,
        live_eligible=False,
        handler_factory="trading_lab.strategies.yoon1k:Yoon1kHandler",
        category=CATEGORY_HIERARCHICAL,
    ),
    "yoon4": StrategyDefinition(
        strategy_id="yoon4",
        version="1",
        description="Global Capital Flow Regime Engine: 글로벌 자금흐름(CRI)으로 투자 국가를 동적 선택하는 다국가 ETF 로테이션. yoon2(미국/한국 50:50 고정)를 일반화 — 모멘텀/상대강도/신고가/거래량/변동성/가속도 Z-score 통합 CRI 상위 N개 국가 비중배분 + 글로벌 Risk-On/Off 레짐 게이트(VIX·DXY·SPY 200MA)로 총노출 조절 + 안전자산(TLT/GLD) 회전. 전 종목 미국 상장 → KIS 해외주식 거래 가능(단일 USD, 원화 환산)",
        config_path=ROOT / "configs" / "strategies" / "yoon4.json",
        enabled=True,
        live_eligible=False,
        handler_factory="trading_lab.strategies.yoon4:Yoon4Handler",
        category=CATEGORY_HIERARCHICAL,
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
