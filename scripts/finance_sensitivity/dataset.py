"""이벤트 패널 조립: 피처 + 사용가능일 + forward 타깃 + 밸류에이션/품질/제외.

finance_plan.txt §3·§4·§8·§22. 분기 발표 이벤트 단위로,
- 피처: d_<factor>(전분기 대비 변화),
- 사용가능일/진입: available_date 다음 봉 시가(next_open),
- 타깃: 진입 기준 forward 20/60일 수익률 + **타깃 실현 시점**(인과 학습용),
- 밸류에이션 z·품질 점수·제외 플래그(모두 과거만 쓰는 expanding)
를 채운 per-event 테이블을 만든다. forward window가 덜 찬 말미 이벤트의 타깃은
NaN으로 두어 학습에서 제외한다(§13 5번 누수 방지).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .availability import AVAILABLE_DATE, PERIOD_END, with_available_date
from .config import FinSensitivityConfig
from .fundamentals import (
    INTERACTION_FEATURES, factor_changes, feature_columns,
)


def _naive_index(ohlcv: pd.DataFrame) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(pd.to_datetime(ohlcv.index))
    return idx.tz_localize(None) if idx.tz is not None else idx


def _expanding_z(series: pd.Series) -> pd.Series:
    """직전까지(현재 제외)의 평균·표준편차로 표준화 — 인과적."""
    mean = series.expanding().mean().shift(1)
    std = series.expanding().std(ddof=0).shift(1)
    return (series - mean) / std.replace(0.0, np.nan)


def build_event_table(
    ohlcv: pd.DataFrame, fundamentals: pd.DataFrame, cfg: FinSensitivityConfig,
    market_close: pd.Series | None = None,
    rates: pd.Series | None = None,
) -> pd.DataFrame:
    """발표 이벤트 패널을 만든다(시간순). 컬럼은 모듈 docstring 참조.

    ``market_close``가 있고 ``cfg.target_excess``면 forward 타깃을 시장 대비
    초과수익(abnormal return)으로 둔다 — 시장 베타·종목 추세 교란을 제거한다.

    ``rates``(일별 미국 금리)가 있고 ``cfg.use_rate_feature``면 각 이벤트
    진입 시점의 금리 수준(z)·변화(``rate_level``/``d_rate``)를 피처로 싣는다.
    금리는 발표 즉시 공개되므로 진입일 시점 값을 그대로 쓴다(누수 없음).
    """
    changes = factor_changes(fundamentals, cfg)
    events = with_available_date(changes, cfg)  # available_date 부여 + 정렬

    index = _naive_index(ohlcv)
    close = pd.Series(np.asarray(ohlcv["close"], dtype=float), index=index)
    open_ = pd.Series(
        np.asarray(ohlcv["open"], dtype=float)
        if "open" in ohlcv else close.to_numpy(),
        index=index,
    )
    use_excess = cfg.target_excess and market_close is not None
    if use_excess:
        mkt = pd.to_numeric(market_close, errors="coerce")
        mkt.index = _naive_index(market_close.to_frame())
        mkt = mkt.reindex(index).ffill()

    rate_series = None
    if cfg.use_rate_feature and rates is not None:
        rate_idx = pd.DatetimeIndex(pd.to_datetime(rates.index))
        if rate_idx.tz is not None:
            rate_idx = rate_idx.tz_localize(None)
        rr = pd.Series(np.asarray(rates, dtype=float), index=rate_idx)
        rate_series = rr.reindex(index).ffill()
    n = len(index)
    h20, h60 = cfg.horizon_20, cfg.horizon_60

    rows = []
    for _, ev in events.iterrows():
        avail = ev[AVAILABLE_DATE]
        # next_open: available_date 이후 첫 봉.
        entry_idx = int(index.searchsorted(avail, side="right"))
        if entry_idx >= n:
            continue
        entry_price = float(open_.iloc[entry_idx])
        entry_time = index[entry_idx]

        def fwd(h: int) -> tuple[float, object]:
            j = entry_idx + h
            if j >= n:
                return (np.nan, pd.NaT)
            # 타깃은 close 기준(entry close→exit close); 초과수익이면 시장 차감.
            stock = float(close.iloc[j]) / float(close.iloc[entry_idx]) - 1.0
            if use_excess and np.isfinite(mkt.iloc[entry_idx]) and mkt.iloc[entry_idx]:
                stock -= float(mkt.iloc[j]) / float(mkt.iloc[entry_idx]) - 1.0
            return (stock, index[j])

        ret20, t20 = fwd(h20)
        ret60, t60 = fwd(h60)

        # 모든 펀더멘털 변화 피처(d_/y_/s_)를 싣는다(밸류·상호작용은 아래에서).
        row = {
            c: ev.get(c, np.nan) for c in ev.index
            if str(c).startswith(("d_", "y_", "s_"))
        }
        if rate_series is not None:
            rate_now = float(rate_series.iloc[entry_idx])
            prev_idx = entry_idx - cfg.rate_change_lookback
            rate_prev = (
                float(rate_series.iloc[prev_idx]) if prev_idx >= 0 else np.nan
            )
            row["rate_level_raw"] = rate_now
            row["d_rate"] = rate_now - rate_prev
        row.update({
            PERIOD_END: ev[PERIOD_END],
            AVAILABLE_DATE: avail,
            "entry_idx": entry_idx,
            "entry_time": entry_time,
            "entry_price": entry_price,
            "ret_20d": ret20, "ret20_time": t20,
            "ret_60d": ret60, "ret60_time": t60,
            "missing_ratio": ev.get("missing_ratio", np.nan),
            "operating_income": ev.get("operating_income", np.nan),
            "operating_cashflow": ev.get("operating_cashflow", np.nan),
            "debt_ratio": ev.get("debt_ratio", np.nan),
            "inventory": ev.get("inventory", np.nan),
            "revenue": ev.get("revenue", np.nan),
            "eps_ttm": ev.get("eps_ttm", np.nan),
            "bvps": ev.get("bvps", np.nan),
            "spr": ev.get("spr", np.nan),
        })
        rows.append(row)

    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table = table.sort_values(AVAILABLE_DATE).reset_index(drop=True)

    # 밸류에이션(진입가 기준) 과거평균 대비 z — expanding(인과).
    per = table["entry_price"] / table["eps_ttm"].replace(0.0, np.nan)
    pbr = table["entry_price"] / table["bvps"].replace(0.0, np.nan)
    psr = table["entry_price"] / table["spr"].replace(0.0, np.nan)
    table["valuation_z"] = (
        _expanding_z(per).fillna(0.0)
        + _expanding_z(pbr).fillna(0.0)
        + _expanding_z(psr).fillna(0.0)
    ) / 3.0

    # 품질 점수: 개선 방향으로 부호 맞춘 핵심 변화의 합(과거 분포로 표준화).
    quality_raw = (
        table.get("d_roe", 0.0).fillna(0.0)
        + table.get("d_operating_cashflow", 0.0).fillna(0.0)
        + table.get("d_net_income", 0.0).fillna(0.0)
        - table.get("d_debt_ratio", 0.0).fillna(0.0)
    )
    table["quality_score"] = _expanding_z(quality_raw).fillna(0.0)

    # 미국 금리 피처: 수준은 과거평균 대비 z(인과), 변화는 그대로. 금리데이터가
    # 없어도(실데이터 로드 실패 등) 컬럼은 0으로 둬 모델 피처 셋과 정합을 지킨다.
    if cfg.use_rate_feature:
        if "rate_level_raw" in table:
            table["rate_level"] = _expanding_z(table["rate_level_raw"]).fillna(0.0)
            table["d_rate"] = table["d_rate"].fillna(0.0)
        else:
            table["rate_level"] = 0.0
            table["d_rate"] = 0.0

    # 밸류에이션 상호작용(§3: 고평가일수록 실적 반응 둔화).
    model_feats = feature_columns(cfg)
    for ix_name, src in INTERACTION_FEATURES.items():
        if ix_name in model_feats and src in table:
            table[ix_name] = table[src] * table["valuation_z"]

    _add_exclusions(table, cfg)
    return table


def _add_exclusions(table: pd.DataFrame, cfg: FinSensitivityConfig) -> None:
    """plan §9 제외 플래그를 과거/현재만으로 계산해 컬럼으로 추가."""
    ex = cfg.exclusions

    op = table["operating_income"]
    loss = (op < 0).astype(int)
    table["excl_operating_loss"] = (
        loss.rolling(ex.operating_loss_streak, min_periods=ex.operating_loss_streak)
        .sum()
        .ge(ex.operating_loss_streak)
        .fillna(False)
    )

    ocf_down = (table["operating_cashflow"].diff() < 0).astype(int)
    table["excl_ocf_decline"] = (
        ocf_down.rolling(ex.ocf_decline_streak, min_periods=ex.ocf_decline_streak)
        .sum()
        .ge(ex.ocf_decline_streak)
        .fillna(False)
    )

    table["excl_debt_jump"] = table["debt_ratio"].diff().gt(ex.debt_ratio_jump).fillna(False)
    table["excl_valuation_overheat"] = table["valuation_z"].gt(ex.valuation_overheat_z)
    table["excl_missing"] = table["missing_ratio"].gt(ex.max_missing_ratio).fillna(True)
    table["excluded"] = table[[
        "excl_operating_loss", "excl_ocf_decline", "excl_debt_jump",
        "excl_valuation_overheat", "excl_missing",
    ]].any(axis=1)
