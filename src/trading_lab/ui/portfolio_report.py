"""포트폴리오형 전략(NAV 기반)용 표준 리포트 어댑터 — QuantStats 위임.

이 플랫폼은 단일자산 가격예측 백테스터로 출발해 시각화가 OHLC+예측파형에
묶여 있다. yoon1k 같은 멀티에셋/계층 포트폴리오는 ``forecast``가 NAV+구성
시리즈라 가격·웨이브폼 차트가 의미가 없다. 이 모듈은 ``StrategyArtifacts``가
이미 만드는 ``equity``(정규화 NAV)·``benchmark``만 받아, 검증된 QuantStats로
표준 포트폴리오 리포트(누적수익·드로다운·월별 히트맵·롤링 샤프·통계표)를
생성한다 — 자체 구현하지 않고 "이미 있는 것"을 활용한다.

QuantStats/matplotlib 미설치 환경에서도 임포트가 깨지지 않도록 방어한다
(``is_available()`` 로 폴백). 차트는 앱의 다크 미니멀 톤에 맞춰 후처리한다.
"""
from __future__ import annotations

import pandas as pd

_IMPORT_ERROR: Exception | None = None
try:  # 선택적 의존성 — 없으면 UI가 안내만 하고 건너뛴다.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import quantstats as qs
except Exception as exc:  # noqa: BLE001
    plt = None  # type: ignore[assignment]
    qs = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc

# 앱 다크 미니멀 톤(presentation._apply_theme와 동일 팔레트).
_BG = "#0E1117"
_FG = "#C9D1D9"
_GRID = (1, 1, 1, 0.08)

REPORT_FIGURES = ("snapshot", "drawdown", "monthly_heatmap", "rolling_sharpe")
FIGURE_LABELS = {
    "snapshot": "누적수익·드로다운·일별수익 스냅샷",
    "drawdown": "낙폭(Underwater)",
    "monthly_heatmap": "월별 수익률 히트맵",
    "rolling_sharpe": "롤링 샤프(6개월)",
}


def is_available() -> bool:
    """QuantStats·matplotlib가 사용 가능한지."""
    return qs is not None and plt is not None


def import_error() -> str | None:
    return None if _IMPORT_ERROR is None else str(_IMPORT_ERROR)


def _to_returns(equity) -> pd.Series:
    """equity(Series/단일컬럼 DataFrame, 1.0 기준 NAV) → 일별 수익률 시리즈."""
    series = equity.iloc[:, 0] if hasattr(equity, "columns") else equity
    series = pd.Series(series).astype(float).dropna()
    if getattr(series.index, "tz", None) is not None:
        series.index = series.index.tz_localize(None)
    return series.pct_change().dropna()


def metrics_table(equity, benchmark=None) -> pd.DataFrame:
    """QuantStats 표준 지표표(전체 모드). 벤치마크가 있으면 나란히 비교."""
    returns = _to_returns(equity)
    bench = _to_returns(benchmark) if benchmark is not None else None
    table = qs.reports.metrics(
        returns, benchmark=bench, display=False, mode="full")
    return table


def _restyle_dark(fig) -> None:
    """QuantStats matplotlib figure를 앱 다크 톤으로 후처리한다."""
    fig.patch.set_facecolor(_BG)
    for ax in fig.axes:
        ax.set_facecolor(_BG)
        ax.tick_params(colors=_FG, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(_GRID)
        ax.title.set_color(_FG)
        ax.xaxis.label.set_color(_FG)
        ax.yaxis.label.set_color(_FG)
        ax.grid(True, color=_GRID)
        legend = ax.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor(_BG)
            for text in legend.get_texts():
                text.set_color(_FG)


def report_figures(equity, benchmark=None, names=REPORT_FIGURES) -> dict:
    """이름별 matplotlib Figure dict. 개별 실패는 건너뛴다(부분 노출 허용)."""
    returns = _to_returns(equity)
    bench = _to_returns(benchmark) if benchmark is not None else None
    figures: dict[str, object] = {}
    for name in names:
        try:
            plotter = getattr(qs.plots, name)
            kwargs = {"show": False}
            if name in ("snapshot", "drawdown", "rolling_sharpe") and bench is not None:
                # 벤치마크 인자를 받는 플롯에만 전달(시그니처 차이 방어).
                try:
                    fig = plotter(returns, benchmark=bench, **kwargs)
                except TypeError:
                    fig = plotter(returns, **kwargs)
            else:
                fig = plotter(returns, **kwargs)
            _restyle_dark(fig)
            figures[name] = fig
        except Exception:  # noqa: BLE001
            continue
    return figures
