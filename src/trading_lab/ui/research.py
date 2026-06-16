"""연구 탭 — DI Kalman M/W 패턴 수집/시각화 (전략 전용, 공통 파이프라인과 분리).

CLAUDE.md 가드레일: 공통 백테스트 결과 렌더러(presentation.py)는 건드리지 않고,
이 전략 전용 연구 화면을 별도 모듈로 격리한다. 여기 로직은 di-kalman-mw 전용이다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from trading_lab.market_catalog import (
    filter_market_options,
    load_market_options,
    option_label,
)
from trading_lab.paths import ROOT
from trading_lab.strategies import get_strategy
from trading_lab.strategies.di_kalman_mw import DiKalmanMwHandler

SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from di_kalman_mw.extreme_transition import build_pattern_dataset  # noqa: E402

RESEARCH_STRATEGY = "di-kalman-mw-v1"
TF_OPTIONS = ["5m", "15m", "30m", "1h", "1d", "1wk", "1mo"]
PERIOD_OPTIONS = ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"]
# 극점 종류별 마커 색 (수집 형태 위에 표시).
PATTERN_COLOR = {"W": "#4cc9f0", "M": "#f72585"}


@st.cache_data(ttl=21_600, show_spinner=False)
def _cached_market_options(chart_type: str):
    return load_market_options(chart_type)


def _config_dict(overrides: dict) -> dict:
    path = get_strategy(RESEARCH_STRATEGY).config_path
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    config.update(overrides)
    return config


@st.cache_data(show_spinner=False)
def _collect(overrides_items: tuple, symbol: str, synthetic: bool, strict: bool):
    """데이터 로드 → M/W 패턴 수집. 동일 입력은 캐시되어 뷰 위젯 조작 시 재계산 없음."""
    config = _config_dict(dict(overrides_items))
    raw = DiKalmanMwHandler().load_data(symbol, config, synthetic=synthetic)
    cfg = DiKalmanMwHandler._strategy_config(config)
    dataset = build_pattern_dataset(raw, cfg, strict)
    return dataset.instances, raw.index


def _shape_figure(
    instances: list,
    pattern: str,
    *,
    line_filter: str,
    normalize: bool,
    include_p5: bool,
    max_overlay: int,
) -> tuple[go.Figure, int]:
    """한 타입(W/M)의 수집 패턴을 정규화 형태로 겹쳐 그린다 + 중앙 형태."""
    selected = [
        x for x in instances
        if x.pattern == pattern
        and (line_filter == "both" or x.line == line_filter)
    ]
    fig = go.Figure()
    if not selected:
        return fig, 0

    shown = selected[:max_overlay]
    curves: list[list[float]] = []
    for x in shown:
        vals = list(x.window_val)
        if include_p5 and x.has_p5:
            vals.append(x.p5_value)
        if normalize:
            anchor = vals[0]
            scale = x.mean_leg if x.mean_leg else 1.0
            ys = [(v - anchor) / scale for v in vals]
        else:
            ys = vals
        curves.append(ys)
        fig.add_trace(go.Scatter(
            x=list(range(1, len(ys) + 1)),
            y=ys,
            mode="lines+markers",
            line={"width": 1, "color": "rgba(150,170,210,0.22)"},
            marker={"size": 4},
            showlegend=False,
            hoverinfo="skip",
        ))

    max_len = max(len(c) for c in curves)
    median = [
        float(np.median([c[k] for c in curves if len(c) > k]))
        for k in range(max_len)
    ]
    fig.add_trace(go.Scatter(
        x=list(range(1, max_len + 1)),
        y=median,
        mode="lines+markers",
        line={"width": 3, "color": PATTERN_COLOR.get(pattern, "#ffcc00")},
        marker={"size": 8},
        name="중앙 형태",
    ))
    suffix = f", 표시 {len(shown)}" if len(shown) < len(selected) else ""
    fig.update_layout(
        template="plotly_dark",
        height=380,
        title=f"{pattern} 패턴 ({len(selected)}건 수집{suffix})",
        xaxis={
            "title": "극점 순서",
            "tickmode": "array",
            "tickvals": list(range(1, max_len + 1)),
            "ticktext": [f"P{k}" for k in range(1, max_len + 1)],
        },
        yaxis_title="P1 기준 정규화 값" if normalize else "Kalman 값",
        showlegend=True,
        legend={"orientation": "h"},
    )
    return fig, len(selected)


def _instances_frame(instances: list, index: pd.Index) -> pd.DataFrame:
    rows = []
    for x in instances:
        rows.append({
            "라인": "+DI" if x.line == "plus" else "-DI",
            "타입": x.pattern,
            "P4 확정 시각": index[x.p4_conf_idx],
            "P5 보유": x.has_p5,
            "정규화 변위": round(x.dv_norm, 3) if x.has_p5 else None,
            "continuation": x.continuation if x.has_p5 else None,
            "leg3_ratio": round(x.features["leg3_ratio"], 3),
            "p3_vs_p1_norm": round(x.features["p3_vs_p1_norm"], 3),
            "평균 레그": round(x.mean_leg, 4),
        })
    return pd.DataFrame(rows)


def render_research_page() -> None:
    st.title("연구 — M/W 패턴 수집")
    st.caption(
        "DI Kalman +DI/-DI 극점(LV0)에서 M/W 패턴을 분류해 타입별로 수집 형태를 "
        "겹쳐 봅니다. 공통 백테스트 파이프라인과 분리된 연구 화면입니다."
    )

    chart_label = st.selectbox("차트 타입", ["크립토", "주식", "합성"], key="rs-charttype")
    chart_type = {"크립토": "crypto", "주식": "stock", "합성": "random"}[chart_label]

    overrides: dict = {}
    if chart_type == "random":
        symbol = "RANDOM"
        synthetic = True
        overrides["synthetic_bars"] = int(st.number_input(
            "합성 봉 수", min_value=1000, max_value=60_000, value=9000,
            step=1000, key="rs-bars",
        ))
    else:
        search = st.text_input(
            "종목 검색",
            placeholder="심볼 또는 종목명 검색 (예: BTC, Bitcoin, SPY)",
            key="rs-search",
        )
        market_options = _cached_market_options(chart_type)
        filtered_options = filter_market_options(market_options, search)
        if filtered_options:
            selected_option = st.selectbox(
                "종목 (시가총액 높은 순)",
                filtered_options,
                format_func=option_label,
                key="rs-symbol",
            )
            symbol = selected_option.symbol
        else:
            st.warning("검색 조건에 맞는 종목이 없습니다.")
            symbol = ""
        synthetic = False
        st.caption(
            "시가총액은 외부 조회값으로 정렬하며, 조회 실패 시 내장 순서를 사용합니다."
        )

    col1, col2 = st.columns(2)
    interval = col1.selectbox(
        "타임프레임 (TF)", TF_OPTIONS, index=TF_OPTIONS.index("1d"), key="rs-tf",
    )
    overrides["interval"] = interval
    if chart_type != "random":
        period = col2.selectbox(
            "데이터 기간", PERIOD_OPTIONS, index=PERIOD_OPTIONS.index("max"),
            key="rs-period",
        )
        overrides["period"] = period
    strict = st.checkbox(
        "strict 분류 (W: P4>P2 / M: P4<P2)", value=False, key="rs-strict"
    )

    if st.button("패턴 수집", type="primary", key="rs-run", disabled=not symbol):
        st.session_state["rs_params"] = (
            tuple(sorted(overrides.items())), symbol, synthetic, strict,
        )

    params = st.session_state.get("rs_params")
    if not params:
        st.info("데이터를 선택하고 '패턴 수집'을 누르면 M/W 패턴을 추출합니다.")
        return

    with st.spinner("DI Kalman 극점 추출 및 M/W 분류 중..."):
        try:
            instances, index = _collect(*params)
        except Exception as exc:  # 데이터 로드 실패 등
            st.error(f"패턴 수집 실패: {type(exc).__name__}: {exc}")
            return

    if not instances:
        st.warning("수집된 M/W 패턴이 없습니다. 기간/TF를 늘리거나 reversal 파라미터를 조정하세요.")
        return

    w = [x for x in instances if x.pattern == "W"]
    m = [x for x in instances if x.pattern == "M"]
    plus = [x for x in instances if x.line == "plus"]
    with_p5 = [x for x in instances if x.has_p5]
    cells = [
        ("총 패턴", str(len(instances))),
        ("W", str(len(w))),
        ("M", str(len(m))),
        ("+DI / -DI", f"{len(plus)} / {len(instances) - len(plus)}"),
        ("P5 보유", str(len(with_p5))),
    ]
    for column, (label, value) in zip(st.columns(len(cells)), cells):
        column.metric(label, value)

    view_cols = st.columns(4)
    line_label = view_cols[0].radio(
        "라인", ["둘 다", "+DI", "-DI"], horizontal=True, key="rs-line",
    )
    line_filter = {"둘 다": "both", "+DI": "plus", "-DI": "minus"}[line_label]
    normalize = view_cols[1].checkbox("정규화 형태", value=True, key="rs-norm")
    include_p5 = view_cols[2].checkbox("P5 포함", value=True, key="rs-p5")
    max_overlay = int(view_cols[3].number_input(
        "최대 겹침 수", min_value=10, max_value=1000, value=120, step=10,
        key="rs-overlay",
    ))

    plot_cols = st.columns(2)
    for column, pattern in zip(plot_cols, ("W", "M")):
        figure, count = _shape_figure(
            instances, pattern, line_filter=line_filter, normalize=normalize,
            include_p5=include_p5, max_overlay=max_overlay,
        )
        with column:
            if count == 0:
                st.info(f"{pattern} 패턴이 없습니다 (현재 라인 필터 기준).")
            else:
                st.plotly_chart(
                    figure, width="stretch",
                    config={"scrollZoom": True, "displaylogo": False},
                )
    st.caption(
        "옅은 선 = 개별 수집 패턴, 굵은 선 = 중앙 형태. 정규화는 각 패턴을 P1 기준으로 "
        "옮기고 평균 레그 진폭으로 나눠 형태를 비교 가능하게 만듭니다."
    )

    with st.expander("수집 인스턴스 표"):
        frame = _instances_frame(instances, index)
        if line_filter != "both":
            tag = "+DI" if line_filter == "plus" else "-DI"
            frame = frame[frame["라인"] == tag]
        st.dataframe(frame, width="stretch", hide_index=True, height=420)
