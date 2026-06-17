from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import streamlit as st

from trading_lab.market_catalog import (
    filter_market_options,
    load_market_options,
    option_label,
)
from trading_lab.portfolio_universes import portfolio_universe
from trading_lab.strategies import list_strategies
from trading_lab.ui.config import PERIOD_OPTIONS, TF_OPTIONS, strategy_config_dict


PORTFOLIO_STRATEGY_ID = "yoon1"


@dataclass(frozen=True)
class MarketSelection:
    chart_type: str
    symbol: str
    chart_detail: str
    bars_per_year: int
    synthetic: bool


@dataclass(frozen=True)
class PortfolioSelection:
    market: MarketSelection
    config_overrides: dict[str, Any]


@st.cache_data(ttl=21_600, show_spinner=False)
def cached_market_options(chart_type: str):
    return load_market_options(chart_type)


def render_portfolio_mode_selector() -> str:
    return st.radio(
        "포트폴리오 유형",
        ["단일 포트폴리오", "멀티 포트폴리오"],
        horizontal=True,
        key="portfolio-mode",
    )


def render_strategy_selector(*, include_multi: bool = False) -> str:
    strategies = [item for item in list_strategies() if item.enabled]
    if not include_multi:
        strategies = [
            item for item in strategies
            if item.strategy_id != PORTFOLIO_STRATEGY_ID
        ] or strategies
    return st.selectbox("전략", [item.strategy_id for item in strategies])


def render_strategy_tunables(strategy_id: str) -> dict[str, Any]:
    tunables = strategy_config_dict(strategy_id).get("tunables") or []
    overrides: dict[str, Any] = {}
    if not tunables:
        return overrides
    with st.expander("전략 파라미터", expanded=False):
        for spec in tunables:
            name = spec["name"]
            label = spec.get("label", name)
            kind = spec.get("type", "number")
            key = f"tunable-{strategy_id}-{name}"
            if kind == "select":
                options = spec["options"]
                default = spec.get("default", options[0])
                index = options.index(default) if default in options else 0
                overrides[name] = st.selectbox(
                    label, options, index=index, key=key
                )
            elif kind == "int":
                overrides[name] = int(st.number_input(
                    label,
                    min_value=int(spec["min"]),
                    max_value=int(spec["max"]),
                    value=int(spec["default"]),
                    step=int(spec.get("step", 1)),
                    key=key,
                ))
            else:
                overrides[name] = float(st.number_input(
                    label,
                    min_value=float(spec["min"]),
                    max_value=float(spec["max"]),
                    value=float(spec["default"]),
                    step=float(spec.get("step", 0.01)),
                    format="%.4f",
                    key=key,
                ))
    return overrides


def render_market_selector() -> MarketSelection:
    chart_label = st.selectbox("차트 타입", ["크립토", "주식", "랜덤"])
    chart_type = {"크립토": "crypto", "주식": "stock", "랜덤": "random"}[
        chart_label
    ]
    if chart_type == "random":
        st.info("랜덤 차트는 재현 가능한 합성 데이터를 사용합니다.")
        return MarketSelection(
            chart_type=chart_type,
            symbol="RANDOM",
            chart_detail="랜덤",
            bars_per_year=8760,
            synthetic=True,
        )

    search = st.text_input(
        "종목 검색", placeholder="심볼 또는 종목명 검색 (예: BTC, Bitcoin, SPY)"
    )
    market_options = cached_market_options(chart_type)
    filtered_options = filter_market_options(market_options, search)
    if filtered_options:
        selected_option = st.selectbox(
            "종목 (시가총액 높은 순)",
            filtered_options,
            format_func=option_label,
        )
        symbol = selected_option.symbol
        chart_detail = selected_option.detail
        bars_per_year = selected_option.bars_per_year
    else:
        st.warning("검색 조건에 맞는 종목이 없습니다.")
        symbol = ""
        chart_detail = ""
        bars_per_year = 8760 if chart_type == "crypto" else 1638
    st.caption(
        "시가총액은 Yahoo Finance 조회값으로 정렬하며, 조회 실패 시 내장 순서를 사용합니다."
    )
    return MarketSelection(
        chart_type=chart_type,
        symbol=symbol,
        chart_detail=chart_detail,
        bars_per_year=bars_per_year,
        synthetic=False,
    )


def render_multi_portfolio_selector() -> PortfolioSelection:
    type_label = st.selectbox(
        "종목 타입",
        ["크립토", "주식", "랜덤", "크립토&주식"],
        key="multi-portfolio-asset-type",
    )
    chart_type = {
        "크립토": "crypto",
        "주식": "stock",
        "랜덤": "random",
        "크립토&주식": "mixed",
    }[type_label]
    if chart_type == "random":
        synthetic_symbols = int(st.number_input(
            "합성 종목 수",
            min_value=2,
            max_value=50,
            value=6,
            step=1,
            key="multi-synthetic-symbols",
        ))
        st.info("랜덤 멀티 포트폴리오는 재현 가능한 합성 유니버스를 사용합니다.")
        return PortfolioSelection(
            market=MarketSelection(
                chart_type="random",
                symbol="RANDOM_PORTFOLIO",
                chart_detail="랜덤",
                bars_per_year=8760,
                synthetic=True,
            ),
            config_overrides={"synthetic_symbols": synthetic_symbols},
        )

    universe = portfolio_universe(chart_type)
    st.caption(f"{type_label} 유니버스 {len(universe)}개 종목으로 실행합니다.")
    bars_per_year = 8760 if chart_type == "crypto" else 1638
    return PortfolioSelection(
        market=MarketSelection(
            chart_type=chart_type,
            symbol="PORTFOLIO",
            chart_detail=type_label,
            bars_per_year=bars_per_year,
            synthetic=False,
        ),
        config_overrides={"universe": universe},
    )


def render_data_controls(
    strategy_id: str,
    base_config: dict[str, Any],
    selection: MarketSelection,
) -> dict[str, Any]:
    default_tf = str(base_config.get("interval", "1d"))
    default_period = str(base_config.get("period", "max"))
    tf_choices = list(dict.fromkeys([default_tf, *TF_OPTIONS]))
    period_choices = list(dict.fromkeys([default_period, *PERIOD_OPTIONS]))

    st.subheader("데이터 설정")
    tf_col, period_col = st.columns(2)
    interval = tf_col.selectbox(
        "타임프레임 (TF)",
        tf_choices,
        index=tf_choices.index(default_tf),
        key=f"tf-{strategy_id}",
    )
    overrides: dict[str, Any] = {"interval": interval}
    if selection.chart_type == "random":
        period_col.caption("합성 차트는 기간 대신 합성 봉 수를 사용합니다.")
    else:
        period = period_col.selectbox(
            "데이터 기간",
            period_choices,
            index=period_choices.index(default_period),
            key=f"period-{strategy_id}",
        )
        overrides["period"] = period
    st.caption(
        "yfinance는 짧은 TF에서 받을 수 있는 기간이 제한됩니다 "
        "(예: 1h는 약 730일, 1m은 약 7일)."
    )
    return overrides
