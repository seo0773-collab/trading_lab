"""진입/청산 규칙 + 종목 점수 + trades 생성 (finance_plan.txt §9·§10·§23).

이벤트 테이블(예측 포함)과 일봉을 받아 long-only 진입/청산을 적용해 거래 목록을
만든다. 이 모듈은 신호 '규칙'까지이며, 거래를 equity/metrics/StrategyArtifacts로
엮어 공통 파이프라인에 태우는 것은 핸들러(백테스트 단계)의 몫이다.

청산 사유: rebalance(다음 발표 시 예측/실제값으로 포트폴리오 재구성) /
signal_flip(갱신된 예상수익률 음전환) / stop_loss(넓은 손절) / end_of_data(마지막
보유 포지션을 데이터 끝까지 보유). **보유기간(max_hold)에 의한 청산은 없다** —
시간이 아니라 다음 재무 발표의 예측·실제값이 청산/재진입을 결정한다.
포지션 중첩은 허용하지 않는다(단일 종목 100%).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .availability import AVAILABLE_DATE
from .config import FinSensitivityConfig

TRADE_COLUMNS = [
    "direction", "entry_time", "entry_price", "exit_time", "exit_price",
    "stop_loss_price", "net_return", "exit_reason", "entry_reason",
    "pred_ret_20d", "pred_ret_60d", "score",
]


def market_filter_series(
    ohlcv: pd.DataFrame, cfg: FinSensitivityConfig,
    market_close: pd.Series | None = None,
) -> pd.Series:
    """진입 허용 구간 bool 시리즈(장기 MA 위 = True). §12 시장 필터.

    시장 지수 종가가 있으면 그것의 MA를, 없고 fallback이면 자기 종가 MA를 쓴다.
    """
    index = pd.DatetimeIndex(pd.to_datetime(ohlcv.index))
    if index.tz is not None:
        index = index.tz_localize(None)
    ref = market_close
    if ref is None:
        if not cfg.market_filter.fallback_self_ma:
            return pd.Series(True, index=index)
        ref = pd.Series(np.asarray(ohlcv["close"], dtype=float), index=index)
    ref = ref.reindex(index).ffill()
    ma = ref.rolling(cfg.market_filter.ma_len, min_periods=1).mean()
    return (ref >= ma).fillna(False)


def score_event(row: pd.Series, cfg: FinSensitivityConfig) -> float:
    """§10 종목 점수: 예상수익률 + 품질 − 밸류에이션 과열. top-n 정렬용(B)."""
    pred20 = float(row.get("pred_ret_20d") or 0.0)
    pred60 = float(row.get("pred_ret_60d") or 0.0)
    quality = float(row.get("quality_score") or 0.0)
    val_z = float(row.get("valuation_z") or 0.0)
    return 0.5 * pred20 + 0.5 * pred60 + 0.1 * quality - 0.1 * max(val_z, 0.0)


def _entry_ok(row: pd.Series, cfg: FinSensitivityConfig) -> bool:
    if bool(row.get("insufficient", True)) or bool(row.get("excluded", False)):
        return False
    pred20 = row.get("pred_ret_20d")
    pred60 = row.get("pred_ret_60d")
    if pd.isna(pred20) or pd.isna(pred60):
        return False
    return float(pred20) > cfg.pred20_min and float(pred60) > cfg.pred60_min


def build_trades(
    table: pd.DataFrame, ohlcv: pd.DataFrame, cfg: FinSensitivityConfig,
    market_close: pd.Series | None = None,
) -> pd.DataFrame:
    """이벤트 테이블 + 일봉 → long-only 거래 목록(중첩 금지)."""
    if table.empty:
        return pd.DataFrame(columns=TRADE_COLUMNS)

    index = pd.DatetimeIndex(pd.to_datetime(ohlcv.index))
    if index.tz is not None:
        index = index.tz_localize(None)
    n = len(index)
    open_ = np.asarray(ohlcv["open"], dtype=float) if "open" in ohlcv \
        else np.asarray(ohlcv["close"], dtype=float)
    low = np.asarray(ohlcv["low"], dtype=float) if "low" in ohlcv \
        else np.asarray(ohlcv["close"], dtype=float)
    close = np.asarray(ohlcv["close"], dtype=float)
    market_ok = market_filter_series(ohlcv, cfg, market_close)
    fee = 2.0 * cfg.fee_bps_per_side / 10_000.0

    events = table.sort_values(AVAILABLE_DATE).reset_index(drop=True)
    entry_indices = events["entry_idx"].astype(int).to_numpy()
    trades: list[dict] = []
    open_until_idx = -1  # 중첩 방지: 이 봉 이전엔 새 진입 불가

    for i in range(len(events)):
        row = events.iloc[i]
        e_idx = int(entry_indices[i])
        if e_idx <= open_until_idx or e_idx >= n:
            continue
        if not bool(market_ok.iloc[e_idx]):
            continue
        if not _entry_ok(row, cfg):
            continue

        entry_price = float(open_[e_idx])
        stop_price = entry_price * (1.0 - cfg.stop_loss_pct)

        # 보유기간(max_hold) 청산 없음 — 다음 발표 때 예측/실제값으로 재구성.
        # 다음 이벤트가 있으면 그 시점에 리밸런싱, 없으면 데이터 끝까지 보유.
        if i + 1 < len(events):
            exit_idx, reason = int(entry_indices[i + 1]), "rebalance"
        else:
            exit_idx, reason = n - 1, "end_of_data"

        # 손절: 진입 다음 봉부터 exit_idx까지 저가가 손절선 이탈.
        stop_hit = np.flatnonzero(low[e_idx + 1: exit_idx + 1] <= stop_price)
        if stop_hit.size:
            exit_idx = e_idx + 1 + int(stop_hit[0])
            reason = "stop_loss"
            exit_price = stop_price
        else:
            # rebalance인데 다음 이벤트 예측이 음전환 → signal_flip.
            if reason == "rebalance" and i + 1 < len(events):
                nxt = events.iloc[i + 1]
                p20, p60 = nxt.get("pred_ret_20d"), nxt.get("pred_ret_60d")
                if (pd.notna(p20) and float(p20) <= 0) or (
                    pd.notna(p60) and float(p60) <= 0
                ):
                    reason = "signal_flip"
            exit_price = float(open_[exit_idx]) if cfg.execution == "next_open" \
                else float(close[exit_idx])

        net = (exit_price / entry_price - 1.0) - fee
        trades.append({
            "direction": 1,
            "entry_time": index[e_idx],
            "entry_price": entry_price,
            "exit_time": index[exit_idx],
            "exit_price": exit_price,
            "stop_loss_price": stop_price,
            "net_return": net,
            "exit_reason": reason,
            "entry_reason": _entry_reason(row),
            "pred_ret_20d": float(row.get("pred_ret_20d") or np.nan),
            "pred_ret_60d": float(row.get("pred_ret_60d") or np.nan),
            "score": score_event(row, cfg),
        })
        open_until_idx = exit_idx

    return pd.DataFrame(trades, columns=TRADE_COLUMNS)


def _entry_reason(row: pd.Series) -> str:
    p20 = float(row.get("pred_ret_20d") or 0.0)
    p60 = float(row.get("pred_ret_60d") or 0.0)
    return (
        f"재무발표 진입 · 예상20 {p20*100:.1f}% · 예상60 {p60*100:.1f}% · "
        f"품질 {float(row.get('quality_score') or 0.0):.2f}"
    )
