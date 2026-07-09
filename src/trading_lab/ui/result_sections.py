from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from trading_lab.execution.integer_dca import (
    DEFAULT_KIS_KR_COST,
    DEFAULT_KIS_US_COST,
    IntegerDcaSettings,
    simulate_integer_dca,
    supports_integer_dca,
)
from trading_lab.ui import portfolio_report
from trading_lab.ui.artifact_io import artifact_path, load_json, load_json_frame
from trading_lab.ui.formatting import currency_spec, metric_text, money_text
from trading_lab.ui.presentation import (
    DERIVED_LABELS,
    available_extra_kinds,
    build_account_figure,
    build_bar_figure,
    build_cycle_profile_figure,
    build_field_dot_figure,
    build_heatmap_figure,
    build_heatmap_trace,
    build_price_indicator_figure,
    build_scatter_figure,
    build_stacked_area_figure,
    build_trade_overview,
    build_trade_report,
    field_move_threshold_options,
    forecast_has_cycle_profile,
    forecast_is_portfolio,
    indicator_series,
    resolve_extra_panels,
    visible_trades_until,
)

REPLAY_SPEED_OPTIONS = (1, 2, 5, 10, 20)


@st.cache_data(ttl=900, show_spinner=False)
def _cached_integer_dca_simulation(
    strategy_id: str,
    config: dict[str, Any],
    symbol: str,
    phase_index: tuple[str, ...],
    initial_capital: float,
    monthly_contribution: float,
    cost_kr: float,
    cost_us: float,
    min_kr_order: float,
    min_us_order_usd: float,
) -> dict[str, Any]:
    idx = pd.DatetimeIndex(pd.to_datetime(list(phase_index)))
    return simulate_integer_dca(
        strategy_id,
        config,
        symbol,
        idx,
        IntegerDcaSettings(
            initial_capital=initial_capital,
            monthly_contribution=monthly_contribution,
            cost_kr=cost_kr,
            cost_us=cost_us,
            min_kr_order=min_kr_order,
            min_us_order_usd=min_us_order_usd,
        ),
    )


def _render_replay_controls(n_bars: int, run_id: str) -> int:
    """봉 단위 재생 컨트롤. 반환값은 현재 재생 위치(0-based, 이 인덱스까지 표시).

    session_state에 재생 위치/재생 여부를 유지한다 — 슬라이더를 직접 옮기면 그
    위치로 이동하고 재생은 멈춘다. 재생 중에는 위젯을 그리기 *전에* 다음 위치로
    미리 전진시킨 뒤(so the slider reflects the advanced frame), 차트 렌더 이후
    호출부에서 짧게 sleep하고 st.rerun()해 다음 프레임으로 넘어간다.
    """
    idx_key = f"replay_idx_{run_id}"
    play_key = f"replay_playing_{run_id}"
    speed_key = f"replay_speed_{run_id}"
    last_bar = n_bars - 1
    if idx_key not in st.session_state:
        st.session_state[idx_key] = last_bar
    if play_key not in st.session_state:
        st.session_state[play_key] = False

    if st.session_state[play_key]:
        speed = int(st.session_state.get(speed_key, REPLAY_SPEED_OPTIONS[0]))
        nxt = st.session_state[idx_key] + speed
        if nxt >= last_bar:
            st.session_state[idx_key] = last_bar
            st.session_state[play_key] = False
        else:
            st.session_state[idx_key] = nxt

    st.markdown("**리플레이 (봉 단위 재생)**")
    cols = st.columns([1, 1, 2, 3])
    with cols[0]:
        if st.button("⏮ 처음", key=f"replay_reset_{run_id}"):
            st.session_state[idx_key] = 0
            st.session_state[play_key] = False
    with cols[1]:
        playing = st.session_state[play_key]
        if st.button("⏸ 정지" if playing else "▶ 재생", key=f"replay_toggle_{run_id}"):
            if st.session_state[idx_key] >= last_bar:
                st.session_state[idx_key] = 0
            st.session_state[play_key] = not playing
    with cols[2]:
        st.selectbox(
            "속도(봉/스텝)", REPLAY_SPEED_OPTIONS, key=speed_key,
        )
    idx = st.slider("시점", 0, last_bar, key=idx_key) if last_bar > 0 else 0
    with cols[3]:
        st.caption(f"{idx + 1}/{n_bars}봉")

    return int(st.session_state[idx_key])


def render_backtest_detail(
    run: dict[str, Any],
    config: dict[str, Any],
    initial_capital: float,
    forecast: pd.DataFrame,
    trades: pd.DataFrame,
    equity: pd.Series,
    benchmark_equity: pd.Series | None = None,
) -> None:
    horizon = int(config.get("horizon", 72))
    confidence_quantile = float(config.get("confidence_quantile", 0.85))
    quantile_window = int(config.get("quantile_window", 2000))

    indicators = indicator_series(
        forecast,
        horizon=horizon,
        confidence_quantile=confidence_quantile,
        quantile_window=quantile_window,
    )
    dashboard_config = config.get("dashboard") or {}
    indicator_labels = {
        **DERIVED_LABELS,
        **{
            str(key): str(value)
            for key, value in (dashboard_config.get("indicator_labels") or {}).items()
        },
    }
    default_indicators = [
        name for name in dashboard_config.get("default_indicators", [])
        if name in indicators
    ] or list(indicators)[:4]
    portfolio_view = forecast_is_portfolio(forecast)
    # 포트폴리오형(NAV+구성)은 가격·웨이브폼보다 표준 리포트가 메인이므로 기본
    # 접힘, 단일자산 전략은 기본 펼침 — 그 외 내용(지표선택/리플레이/Cycle
    # profile)은 이 카테고리 익스팬더 안에서 중첩 없이 나란히 배치한다.
    with st.expander("📊 차트 · 보조지표", expanded=not portfolio_view):
        selected_indicators = st.multiselect(
            "표시할 보조지표 (스케일이 비슷한 지표는 같은 패널에 겹쳐 표시)",
            list(indicators),
            default=default_indicators,
            format_func=lambda name: indicator_labels.get(name, name),
            key=f"indicators-{run['run_id']}",
        )
        heatmap_spec, heatmap_frame = _resolve_heatmap_panel(run, dashboard_config)
        heatmap_overlay = None
        # overlay=false 패널(예: 세로축이 절대가격이 아닌 상대위치)은 가격 y축과
        # 정렬되지 않으므로 가격 차트에 겹치지 않고 독립 패널로만 노출한다.
        if heatmap_frame is not None and bool(heatmap_spec.get("overlay", True)):
            show_heatmap = st.checkbox(
                "가격 그래프에 청산 히트맵 겹쳐 표시",
                value=True,
                key=f"heatmap-overlay-{run['run_id']}",
            )
            if show_heatmap:
                heatmap_overlay = build_heatmap_trace(
                    heatmap_frame,
                    x=str(heatmap_spec.get("x") or "time"),
                    column_normalize=bool(heatmap_spec.get("column_normalize", False)),
                    opacity=0.55,
                    showscale=False,
                    overlay=True,
                )
        replay_idx = (
            _render_replay_controls(len(forecast), run["run_id"])
            if len(forecast) >= 2 else None
        )
        series_map = {name: indicators[name] for name in selected_indicators}
        if replay_idx is not None and replay_idx < len(forecast) - 1:
            forecast_view = forecast.iloc[: replay_idx + 1]
            trades_view = visible_trades_until(trades, forecast.index[replay_idx])
            series_map = {
                name: series.reindex(forecast_view.index)
                for name, series in series_map.items()
            }
        else:
            forecast_view, trades_view = forecast, trades
        st.plotly_chart(
            build_price_indicator_figure(
                forecast_view,
                trades_view,
                symbol=run["symbol"],
                horizon=horizon,
                series_map=series_map,
                labels=indicator_labels,
                heatmap_overlay=heatmap_overlay,
            ),
            width="stretch",
            config={"scrollZoom": True, "displaylogo": False},
        )
        if st.session_state.get(f"replay_playing_{run['run_id']}", False):
            time.sleep(0.35)
            st.rerun()
        if forecast_has_cycle_profile(forecast) and bool(
            (dashboard_config.get("cycle_profile") or {}).get("enabled", True)
        ):
            st.markdown("**Cycle multiple · rolling profile anchor**")
            valid = forecast.dropna(
                subset=["cm_open", "cm_high", "cm_low", "cm_close"],
                how="any",
            )
            if not valid.empty:
                anchor_pos = st.slider(
                    "Anchor",
                    min_value=0,
                    max_value=max(0, len(valid) - 1),
                    value=len(valid) - 1,
                    format="%d",
                    key=f"cycle-profile-anchor-{run['run_id']}",
                )
                anchor_time = valid.index[int(anchor_pos)]
                st.caption(pd.Timestamp(anchor_time).strftime("%Y-%m-%d %H:%M"))
                st.plotly_chart(
                    build_cycle_profile_figure(
                        forecast,
                        config,
                        symbol=run["symbol"],
                        anchor_time=anchor_time,
                    ),
                    width="stretch",
                    config={"scrollZoom": True, "displaylogo": False},
                )

    currency_sym, currency_dec, currency_step, _ = currency_spec(
        config.get("base_currency"))
    with st.expander("💰 계좌 · 실행 시뮬레이션", expanded=True):
        invest_mode = st.radio(
            "투자 방식", ["거치식", "적립식"], horizontal=True,
            key=f"invest-mode-{run['run_id']}",
            help="거치식=초기 일시금 전액 투입. 적립식=초기 일시금 + 매월 정액 추가 투입.",
        )
        monthly_contribution = 0.0
        if invest_mode == "적립식":
            monthly_contribution = float(st.number_input(
                f"월 적립액 ({currency_sym})", min_value=0.0,
                value=float(currency_step), step=float(currency_step),
                key=f"monthly-contrib-{run['run_id']}",
            ))
        st.plotly_chart(
            build_account_figure(
                equity,
                initial_capital=initial_capital,
                symbol=run["symbol"],
                benchmark_price=(None if portfolio_view else forecast["close"]),
                benchmark_equity=benchmark_equity,
                monthly_contribution=monthly_contribution,
                currency_symbol=currency_sym,
            ),
            width="stretch",
            config={"scrollZoom": True, "displaylogo": False},
        )
        render_integer_dca_execution_option(
            run,
            config,
            equity,
            initial_capital=initial_capital,
            monthly_contribution=monthly_contribution,
            currency_symbol=currency_sym,
        )

    with st.expander("📈 성과 리포트", expanded=True):
        # 포트폴리오형(NAV 기반) 전략은 가격·웨이브폼이 의미없으므로 표준 포트폴리오
        # 리포트로 대체 보강한다. config dashboard.portfolio_report 로 강제할 수도 있다.
        if portfolio_view or dashboard_config.get("portfolio_report"):
            render_portfolio_report(run, equity, benchmark_equity)

        st.subheader("트레이드 리포트")
        overview = build_trade_overview(trades)
        overview_values = [
            ("롱 횟수", str(overview["long_trades"])),
            ("숏 횟수", str(overview["short_trades"])),
            ("롱 평균 수익률", metric_text(overview["long_avg_return"], percent=True)),
            ("숏 평균 수익률", metric_text(overview["short_avg_return"], percent=True)),
            ("롱 청산율", metric_text(overview["long_close_rate"], percent=True)),
            ("숏 청산율", metric_text(overview["short_close_rate"], percent=True)),
        ]
        for column, (label, value) in zip(st.columns(6), overview_values):
            column.metric(label, value)
        st.caption("청산율은 해당 방향의 진입 거래 중 청산 시각이 기록된 거래 비율입니다.")

        report = build_trade_report(
            trades,
            equity,
            initial_capital=initial_capital,
            horizon=horizon,
            execution=str(config.get("execution", "next_open")),
        )
        if report.empty:
            st.info("선택한 구간에서 체결된 거래가 없습니다.")
        else:
            display = report.rename(columns={
                "trade_number": "번호",
                "symbol": "종목",
                "side": "방향",
                "entry_time": "진입 시각",
                "entry_price": "진입가",
                "exit_time": "청산 시각",
                "exit_price": "청산가",
                "stop_loss_price": "손절가",
                "take_profit_price": "익절가",
                "exit_reason": "청산 사유",
                "net_return_pct": "결과 손익 %",
                "account_value_after": "거래 후 계좌 금액",
                "entry_reason": "진입 근거",
            })
            for column in ("손절가", "익절가"):
                display[column] = display[column].map(
                    lambda value: "미사용" if pd.isna(value) else f"{value:,.4f}"
                )
            display["결과 손익 %"] = display["결과 손익 %"].map(
                lambda value: f"{value:+.2f}%"
            )
            display["거래 후 계좌 금액"] = display["거래 후 계좌 금액"].map(
                lambda value: money_text(value, currency_sym, currency_dec)
            )
            st.dataframe(display, width="stretch", hide_index=True, height=520)
            if report[["stop_loss_price", "take_profit_price"]].isna().all().all():
                st.caption(
                    "이 전략은 고정 손절/익절 주문을 사용하지 않습니다. "
                    "해당 가격은 임의 추정하지 않고 '미사용'으로 표시합니다."
                )

        render_extra_panels(run, dashboard_config)


def render_integer_dca_execution_option(
    run: dict[str, Any],
    config: dict[str, Any],
    equity: pd.Series,
    *,
    initial_capital: float,
    monthly_contribution: float,
    currency_symbol: str,
) -> None:
    if run.get("chart_type") == "random" or not supports_integer_dca(config):
        return
    with st.expander("실거래 정수주 실행 시뮬레이션", expanded=False):
        enabled = st.checkbox(
            "정수주, 월 적립, 수수료를 반영한 실행 계좌를 계산",
            value=False,
            key=f"integer-dca-enabled-{run['run_id']}",
            help=(
                "하위 포트폴리오 목표비중을 종목 단위 주문으로 변환해 월말 리밸런스, "
                "정수주, 최소 주문금액, 편도 비용을 적용합니다."
            ),
        )
        col1, col2, col3 = st.columns(3)
        monthly = col1.number_input(
            f"월 적립액 ({currency_symbol})",
            min_value=0.0,
            value=float(monthly_contribution if monthly_contribution > 0 else 200_000.0),
            step=100_000.0,
            key=f"integer-dca-monthly-{run['run_id']}",
        )
        cost_kr_pct = col2.number_input(
            "KR 편도 비용 (%)",
            min_value=0.0,
            value=DEFAULT_KIS_KR_COST * 100.0,
            step=0.001,
            format="%.4f",
            key=f"integer-dca-cost-kr-{run['run_id']}",
        )
        cost_us_pct = col3.number_input(
            "US 편도 비용 (%)",
            min_value=0.0,
            value=DEFAULT_KIS_US_COST * 100.0,
            step=0.001,
            format="%.4f",
            key=f"integer-dca-cost-us-{run['run_id']}",
        )
        col4, col5 = st.columns(2)
        min_kr = col4.number_input(
            f"KR 최소 주문금액 ({currency_symbol})",
            min_value=0.0,
            value=50_000.0,
            step=10_000.0,
            key=f"integer-dca-min-kr-{run['run_id']}",
        )
        min_us = col5.number_input(
            "US 최소 주문금액 ($)",
            min_value=0.0,
            value=50.0,
            step=10.0,
            key=f"integer-dca-min-us-{run['run_id']}",
        )
        if not enabled:
            st.caption("체크하면 현재 백테스트 기간 기준으로 실거래 실행 계좌를 계산합니다.")
            return
        with st.spinner("정수주 실행 시뮬레이션 계산 중..."):
            result = _cached_integer_dca_simulation(
                str(run["strategy_id"]),
                config,
                str(run["symbol"]),
                tuple(pd.DatetimeIndex(equity.index).astype(str)),
                float(initial_capital),
                float(monthly),
                float(cost_kr_pct) / 100.0,
                float(cost_us_pct) / 100.0,
                float(min_kr),
                float(min_us),
            )
        summary = result["summary"]
        frame = result["frame"]
        render_integer_dca_summary(summary, currency_symbol)
        st.plotly_chart(
            build_integer_dca_figure(frame, currency_symbol),
            width="stretch",
            config={"scrollZoom": True, "displaylogo": False},
        )
        skipped = summary.get("skipped_high_price") or []
        if skipped:
            st.caption(
                "주당가격/현금 제약 빈도 상위: "
                + ", ".join(f"{sym} {count}회" for sym, count in skipped[:8])
            )
        with st.expander("정수주 실행 원본 시계열", expanded=False):
            display = frame.reset_index().rename(columns={"index": "date"})
            st.dataframe(display.sort_values("date", ascending=False), width="stretch", hide_index=True)


def render_integer_dca_summary(summary: dict[str, Any], currency_symbol: str) -> None:
    ideal = summary["ideal"]
    integer = summary["integer"]
    tracking = summary["tracking"]
    values = [
        ("정수주 최종", money_text(integer["final_value"], currency_symbol)),
        ("이론 대비", metric_text(tracking["final_integer_vs_ideal"], percent=True)),
        ("TWR CAGR", metric_text(integer["cagr"], percent=True)),
        ("TWR Sharpe", metric_text(integer["sharpe"])),
        ("계좌 MDD", metric_text(integer["account_mdd"], percent=True)),
        ("평균 재현율", metric_text(integer["avg_target_fit"], percent=True)),
    ]
    for column, (label, value) in zip(st.columns(6), values):
        column.metric(label, value)
    st.caption(
        f"이론 최종 {money_text(ideal['final_value'], currency_symbol)} · "
        f"누적원금 {money_text(summary['invested_principal'], currency_symbol)} · "
        f"일간 수익률 상관 {tracking['daily_return_corr']:.4f} · "
        f"최저 재현율 {integer['min_target_fit']:.2%} · "
        f"평균/최대 현금 {integer['avg_cash_weight']:.2%}/{integer['max_cash_weight']:.2%} · "
        f"총 비용 {money_text(integer['cost_paid'], currency_symbol)}"
    )


def build_integer_dca_figure(frame: pd.DataFrame, currency_symbol: str) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=frame.index,
        y=frame["ideal_account"],
        mode="lines",
        name="이론 계좌",
        line={"color": "#66bb6a", "width": 1.6},
    ))
    figure.add_trace(go.Scatter(
        x=frame.index,
        y=frame["integer_account"],
        mode="lines",
        name="정수주 실행",
        line={"color": "#ffa726", "width": 1.6},
    ))
    figure.add_trace(go.Scatter(
        x=frame.index,
        y=frame["invested_principal"],
        mode="lines",
        name="누적원금",
        line={"color": "#90a4ae", "width": 1.0, "dash": "dot"},
    ))
    figure.update_layout(
        title="이론 계좌 vs 정수주 실행 계좌",
        height=430,
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.12, "x": 0},
        margin={"l": 45, "r": 45, "t": 85, "b": 40},
        template="plotly_dark",
    )
    figure.update_yaxes(title_text="계좌 금액", tickprefix=currency_symbol)
    return figure


def _resolve_heatmap_panel(
    run: dict[str, Any], dashboard_config: dict[str, Any]
) -> tuple[dict[str, Any], pd.DataFrame | None]:
    """heatmap 타입 보조 패널이 있으면 그 스펙과 프레임을 돌려준다(없으면 None).

    전략명을 가리지 않는 범용 탐지: ``dashboard.panels`` 중 ``type == "heatmap"``
    선언이 있고 실제 아티팩트가 존재하면 가격 그래프 오버레이용으로 로드한다.
    """
    available = available_extra_kinds(run)
    for panel in resolve_extra_panels(dashboard_config, available):
        if panel.get("type") != "heatmap":
            continue
        path = artifact_path(run, panel["kind"])
        records = load_json(path) if path and Path(path).exists() else None
        if not records:
            continue
        frame = pd.DataFrame(records)
        if frame.empty:
            continue
        return panel, frame
    return {}, None


def render_portfolio_report(run: dict[str, Any], equity, benchmark_equity) -> None:
    """포트폴리오형 전략(NAV 기반)에 QuantStats 표준 리포트를 노출한다 —
    멀티에셋/계층 전략용 '플랫폼 표준 형식'(지표표 + 누적수익·낙폭·월별
    히트맵·롤링 샤프). equity/benchmark는 StrategyArtifacts가 이미 만든다."""
    st.subheader("포트폴리오 표준 리포트 (QuantStats)")
    if not portfolio_report.is_available():
        st.info(
            "QuantStats 미설치 — `.venv/bin/pip install quantstats` 후 표준 "
            "리포트가 표시됩니다."
        )
        return
    try:
        table = portfolio_report.metrics_table(equity, benchmark_equity)
        st.dataframe(table, width="stretch")
    except Exception as exc:  # noqa: BLE001
        st.warning(f"지표표 생성 실패: {exc}")

    figures = portfolio_report.report_figures(equity, benchmark_equity)
    columns = st.columns(2)
    for i, name in enumerate(portfolio_report.REPORT_FIGURES):
        figure = figures.get(name)
        if figure is None:
            continue
        with columns[i % 2]:
            st.caption(portfolio_report.FIGURE_LABELS.get(name, name))
            st.pyplot(figure, clear_figure=True)


def _panel_scatter(frame: pd.DataFrame, spec: dict[str, Any]) -> Any:
    x, y = spec.get("x"), spec.get("y")
    if x in frame and y in frame:
        return build_scatter_figure(frame, x, y, label=spec["label"])
    return None


def _panel_bar(frame: pd.DataFrame, spec: dict[str, Any]) -> Any:
    x, y = spec.get("x"), spec.get("y")
    if x in frame and y in frame:
        return build_bar_figure(frame, x, y, label=spec["label"])
    return None


def _panel_stacked_area(frame: pd.DataFrame, spec: dict[str, Any]) -> Any:
    return build_stacked_area_figure(
        frame, x=str(spec.get("x") or "time"), label=spec["label"])


def _panel_heatmap(frame: pd.DataFrame, spec: dict[str, Any]) -> Any:
    return build_heatmap_figure(
        frame, x=str(spec.get("x") or "time"), label=spec["label"],
        column_normalize=bool(spec.get("column_normalize", False)),
        yaxis_title=str(spec.get("yaxis_title", "가격")))


def _panel_field_dots(frame: pd.DataFrame, spec: dict[str, Any]) -> Any:
    """상태 필드 점 분포 + 궤적 노브(궤적 on/off, 파동 임계값 선택).

    임계값 선택지는 프레임에 사전 계산된 ``move_dir_<pct>`` 컬럼에서 발견한다
    (없으면 노브 없이 기존 동작). 기본값은 패널 선언의 ``move_threshold``.
    """
    options = field_move_threshold_options(frame)
    show_moves, pct = True, None
    if options or "move_dir" in frame:
        key = f"field-{spec['kind']}"
        toggle_col, slider_col = st.columns([1, 3])
        with toggle_col:
            show_moves = st.toggle(
                "큰 파동 궤적", value=True, key=f"{key}-moves"
            )
        if options and show_moves:
            default_pct = int(round(float(spec.get("move_threshold", 0.0)) * 100))
            if default_pct not in options:
                default_pct = options[len(options) // 2]
            with slider_col:
                pct = st.select_slider(
                    "파동 임계값 % (이 진폭 이상의 상승/하락만 궤적으로)",
                    options=options,
                    value=default_pct,
                    key=f"{key}-threshold",
                )
    return build_field_dot_figure(
        frame, label=spec["label"], show_moves=show_moves,
        move_threshold_pct=pct,
    )


# 패널 type → figure 빌더 레지스트리. 새 시각화를 추가할 때는 presentation.py에
# 빌더를 만들고 여기 한 줄만 등록하면 된다 — render_extra_panels 디스패치 본체는
# 더 이상 수정하지 않는다(Open-Closed). 미등록 타입은 원본 테이블로 폴백한다.
PANEL_RENDERERS: dict[str, Any] = {
    "scatter": _panel_scatter,
    "bar": _panel_bar,
    "stacked_area": _panel_stacked_area,
    "heatmap": _panel_heatmap,
    "field_dots": _panel_field_dots,
}


def _render_panel_figure(spec: dict[str, Any], frame: pd.DataFrame) -> None:
    """spec["type"]을 PANEL_RENDERERS로 디스패치. 미등록 타입이거나 필요한
    컬럼이 없어 빌더가 None을 주면 원본 테이블로 폴백한다(전략명·타입 하드코딩 없음)."""
    renderer = PANEL_RENDERERS.get(str(spec.get("type")))
    figure = renderer(frame, spec) if renderer is not None else None
    if figure is None:
        st.dataframe(frame, width="stretch", hide_index=True)
        return
    st.plotly_chart(figure, width="stretch", config={"displaylogo": False})


def render_extra_panels(run: dict[str, Any], dashboard_config: dict[str, Any]) -> None:
    learning = dashboard_config.get("learning_tab") or {}
    learning_kinds = {
        str(learning[key])
        for key in (
            "summary_kind", "predictions_kind",
            "sensitivity_kind", "events_kind",
        )
        if learning.get(key)
    }
    available = [
        kind for kind in available_extra_kinds(run)
        if kind not in learning_kinds
    ]
    if not available:
        return
    panels = resolve_extra_panels(dashboard_config, available)
    spec_by_kind = {panel["kind"]: panel for panel in panels}
    labels = {panel["kind"]: panel["label"] for panel in panels}

    st.subheader("보조 패널")
    chosen = st.multiselect(
        "표시할 보조 패널 (전략별 추가 자료 - 선택해서 추가/삭제)",
        [panel["kind"] for panel in panels],
        default=[panel["kind"] for panel in panels if panel["default"]],
        format_func=lambda kind: labels.get(kind, kind),
        key=f"panels-{run['run_id']}",
    )
    for kind in chosen:
        spec = spec_by_kind[kind]
        path = artifact_path(run, kind)
        records = load_json(path) if path and Path(path).exists() else None
        if not records:
            continue
        frame = pd.DataFrame(records)
        if frame.empty:
            continue
        st.markdown(f"**{spec['label']}**")
        _render_panel_figure(spec, frame)


def render_learning_result(
    run: dict[str, Any], learning_config: dict[str, Any]
) -> None:
    st.subheader("재무 변화로 학습한 주가 반응")
    st.caption(
        "각 재무 발표 시점에 과거에 이미 실현된 사례만 사용해 모델을 다시 "
        "학습합니다. 실제 수익률은 진입 판단이 아니라 사후 예측력 평가에 사용됩니다."
    )

    summary = load_json_frame(run, learning_config.get("summary_kind"))
    if summary.empty:
        st.info("표본이 부족해 학습 진단값이 생성되지 않았습니다.")
    else:
        for _, row in summary.sort_values("horizon_days").iterrows():
            horizon = int(row["horizon_days"])
            st.markdown(f"**{horizon}일 예측 성능**")
            values = [
                ("평가 표본", f"{int(row['samples'])}건"),
                ("Spearman IC", metric_text(row.get("spearman_ic"))),
                ("평균 절대오차", metric_text(row.get("mae"), percent=True)),
                (
                    "방향 적중률",
                    metric_text(row.get("direction_accuracy"), percent=True),
                ),
                (
                    "평균 예상수익률",
                    metric_text(row.get("mean_predicted_return"), percent=True),
                ),
                (
                    "평균 실제수익률",
                    metric_text(row.get("mean_actual_return"), percent=True),
                ),
            ]
            for column, (label, value) in zip(st.columns(6), values):
                column.metric(label, value)

    predictions = load_json_frame(
        run, learning_config.get("predictions_kind")
    )
    if not predictions.empty:
        st.markdown("**예상수익률과 실제수익률**")
        columns = st.columns(2)
        for column, horizon in zip(columns, (20, 60)):
            x, y = f"pred_ret_{horizon}d", f"ret_{horizon}d"
            if x in predictions and y in predictions:
                column.plotly_chart(
                    build_scatter_figure(
                        predictions, x, y,
                        label=f"{horizon}일 예상 vs 실제",
                    ),
                    width="stretch",
                    config={"displaylogo": False},
                )

    sensitivity = load_json_frame(
        run, learning_config.get("sensitivity_kind")
    )
    if (
        not sensitivity.empty
        and "factor" in sensitivity
        and "sensitivity_mean" in sensitivity
    ):
        st.markdown("**재무 팩터별 평균 민감도**")
        st.plotly_chart(
            build_bar_figure(
                sensitivity,
                "factor",
                "sensitivity_mean",
                label="20일 수익률에 대한 평균 민감도",
            ),
            width="stretch",
            config={"displaylogo": False},
        )
        st.caption(
            "양수는 해당 재무 변화가 클수록 예상수익률이 높아지는 관계, "
            "음수는 반대 관계를 뜻합니다. 인과관계를 증명하는 값은 아닙니다."
        )

    events = load_json_frame(run, learning_config.get("events_kind"))
    if not events.empty:
        st.markdown("**발표 이벤트별 학습 데이터**")
        st.dataframe(events, width="stretch", hide_index=True, height=520)
