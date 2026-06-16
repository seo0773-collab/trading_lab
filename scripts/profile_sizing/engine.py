"""백테스트 엔진 (profile_plan.txt §12).

목표 비중 경로(actual_weight)를 평가자산(equity)으로 환산하고, 같은 구간의
buy & hold(상시 100% 보유) equity를 함께 산출해 비교한다. 비중 변화는 lot 기반
FIFO로 분해해 대시보드 trades 계약(진입/청산/순수익)에 매핑한다.

무누수 규칙: 봉 t에서 결정한 비중 w[t]는 t+1 수익률에 적용한다(weight를 1봉 지연).
거래비용(수수료+슬리피지)은 비중이 바뀐 봉의 회전율에 비례해 차감한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ProfileSizingConfig
from .regime import DEFENSE

_ANN = {"1d": 252.0, "1h": 24 * 365.0, "1wk": 52.0, "1mo": 12.0}


def _ann_factor(interval: str) -> float:
    return _ANN.get(str(interval).lower(), 252.0)


def portfolio_returns(
    close: pd.Series, weight: pd.Series, cfg: ProfileSizingConfig
) -> pd.Series:
    """일별 포트폴리오 수익률 = 지연비중×자산수익률 − 거래비용."""
    ret = close.pct_change().fillna(0.0)
    w_prev = weight.shift(1).fillna(0.0)
    gross = w_prev * ret
    fee_rate = (cfg.costs.fee_bps_per_side + cfg.costs.slippage_bps) / 10_000.0
    turnover = weight.diff().abs().fillna(weight.iloc[0] if len(weight) else 0.0)
    cost = fee_rate * turnover
    return (gross - cost).rename("port_ret")


def equity_from_returns(port_ret: pd.Series) -> pd.Series:
    return (1.0 + port_ret).cumprod().rename("equity")


def buy_hold_equity(close: pd.Series) -> pd.Series:
    ret = close.pct_change().fillna(0.0)
    return (1.0 + ret).cumprod().rename("buy_hold")


def lot_trades(
    close: pd.Series, weight: pd.Series, regime: pd.Series, cfg: ProfileSizingConfig
) -> pd.DataFrame:
    """비중 증감을 FIFO lot으로 분해해 거래 단위(체결)로 환산한다.

    비중 증가 = 신규 lot 매수, 감소 = 가장 오래된 lot부터 부분 청산.
    각 청산분이 하나의 trade(long, net_return=가격수익−양편 비용)이다.
    """
    fee = (cfg.costs.fee_bps_per_side + cfg.costs.slippage_bps) / 10_000.0
    idx = close.index
    px = close.to_numpy(dtype=float)
    w = weight.to_numpy(dtype=float)
    reg = np.asarray(regime, dtype=object)

    open_lots: list[dict] = []  # {entry_time, entry_price, qty, entry_regime}
    rows: list[dict] = []

    def close_qty(qty_to_close: float, t_idx: int, reason: str) -> None:
        remaining = qty_to_close
        while remaining > 1e-9 and open_lots:
            lot = open_lots[0]
            take = min(lot["qty"], remaining)
            gross = px[t_idx] / lot["entry_price"] - 1.0
            net = gross - 2.0 * fee  # 진입+청산 양편 비용
            rows.append({
                "direction": 1,
                "entry_time": lot["entry_time"],
                "entry_price": lot["entry_price"],
                "exit_time": idx[t_idx],
                "exit_price": px[t_idx],
                "weight": take,
                "net_return": net,
                "exit_reason": reason,
                "entry_reason": lot["entry_reason"],
            })
            lot["qty"] -= take
            remaining -= take
            if lot["qty"] <= 1e-9:
                open_lots.pop(0)

    prev = 0.0
    for t in range(len(idx)):
        dw = w[t] - prev
        if dw > 1e-9:
            open_lots.append({
                "entry_time": idx[t],
                "entry_price": px[t],
                "qty": dw,
                "entry_regime": str(reg[t]),
                "entry_reason": f"{str(reg[t])} 진입 · 비중 +{dw:.2f}",
            })
        elif dw < -1e-9:
            reason = "defense_cut" if str(reg[t]) == DEFENSE else "rebalance"
            close_qty(-dw, t, reason)
        prev = w[t]

    # 잔여 보유분은 마지막 봉에서 청산 처리(평가용).
    if open_lots:
        close_qty(sum(l["qty"] for l in open_lots), len(idx) - 1, "end_of_data")

    columns = [
        "direction", "entry_time", "entry_price", "exit_time", "exit_price",
        "weight", "net_return", "exit_reason", "entry_reason",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows)[columns]


def performance(equity: pd.Series, port_ret: pd.Series, interval: str) -> dict:
    """equity 기반 성과지표(목표비중 전략의 권위 지표)."""
    if equity.empty:
        return {"total_return": 0.0, "sharpe": None, "max_drawdown": None,
                "cagr": None, "volatility": None}
    total_return = float(equity.iloc[-1] - 1.0)
    dd = float((equity / equity.cummax() - 1.0).min())
    ann = _ann_factor(interval)
    r = port_ret.to_numpy(dtype=float)
    if r.size > 1 and np.nanstd(r, ddof=0) > 0:
        sharpe = float(np.nanmean(r) / np.nanstd(r, ddof=0) * np.sqrt(ann))
        vol = float(np.nanstd(r, ddof=0) * np.sqrt(ann))
    else:
        sharpe, vol = None, None
    n_years = max(len(equity) / ann, 1e-9)
    cagr = float(equity.iloc[-1] ** (1.0 / n_years) - 1.0) if equity.iloc[-1] > 0 else None
    return {
        "total_return": total_return,
        "sharpe": sharpe,
        "max_drawdown": dd,
        "cagr": cagr,
        "volatility": vol,
    }
