"""명시적 계좌 시뮬레이션 (현금·주식·평가액 추적 + 수익 재투자 토글).

기존 `account_value_series`(정규화 equity × 초기자본)는 복리 결과를 스칼라로 투영할
뿐, 계좌 내부(현금/보유주수)나 "재투자 vs 비재투자" 비교를 보여주지 못한다. 이 모듈은
매 봉 계좌를 실제 금액으로 굴린다.

- **reinvest=True (복리)**: 목표 주식가치 = weight × **현재 총 계좌가치**. 수익이 나면
  계좌가 커지고 다음 매수 규모도 커진다 = 수익 자동 재투자.
- **reinvest=False (비복리)**: 목표 주식가치 = weight × **초기자본 고정**. 실현 수익은
  현금에 쌓이기만 하고 재투자되지 않는다(재투자 효과 비교용 베이스라인).

무누수: 봉 t에서 결정한 weight[t]로 그 봉 종가에 리밸런싱하고, 다음 봉 가격변화가
평가손익에 반영된다(엔진의 weight 1봉 지연과 동일 관행).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ProfileSizingConfig


def simulate_account(
    close: pd.Series,
    weight: pd.Series,
    cfg: ProfileSizingConfig,
    initial_capital: float = 10_000.0,
    *,
    reinvest: bool = True,
) -> pd.DataFrame:
    """일별 계좌 시계열. long-only·무레버리지(목표 주식가치는 계좌가치로 상한)."""
    fee = (cfg.costs.fee_bps_per_side + cfg.costs.slippage_bps) / 10_000.0
    idx = close.index
    px = np.asarray(close, dtype=float)
    w = np.asarray(weight, dtype=float)
    n = len(idx)

    cash = float(initial_capital)
    shares = 0.0
    rows = np.empty((n, 6), dtype=float)  # shares, position_value, cash, account, target, fees
    cum_fees = 0.0

    for t in range(n):
        position_value = shares * px[t]
        account = cash + position_value
        base = account if reinvest else initial_capital
        # 목표 주식가치: 0 ~ 현재 계좌가치(무레버리지) 범위로 제한.
        target = float(np.clip(w[t] * base, 0.0, account))
        trade = target - position_value
        cost = fee * abs(trade)
        cum_fees += cost
        cash -= trade + cost
        shares = target / px[t] if px[t] > 0 else 0.0
        # 비용 차감 후 시가 평가.
        account_after = cash + shares * px[t]
        rows[t] = (shares, shares * px[t], cash, account_after, target, cum_fees)

    out = pd.DataFrame(
        rows, index=idx,
        columns=["shares", "position_value", "cash", "account_value",
                 "target_position_value", "cum_fees"],
    )
    out["invested_ratio"] = np.where(
        out["account_value"] > 0,
        out["position_value"] / out["account_value"], 0.0,
    )
    return out


def account_summary(
    close: pd.Series, weight: pd.Series, cfg: ProfileSizingConfig,
    initial_capital: float = 10_000.0,
) -> pd.DataFrame:
    """재투자 on/off 최종 결과 비교표(수익 재투자 효과 가시화)."""
    rows = []
    for reinvest in (True, False):
        acct = simulate_account(close, weight, cfg, initial_capital, reinvest=reinvest)
        final = float(acct["account_value"].iloc[-1])
        rows.append({
            "mode": "재투자(복리)" if reinvest else "비재투자(고정원금)",
            "final_account_value": round(final, 2),
            "total_return_pct": round((final / initial_capital - 1.0) * 100.0, 2),
            "total_fees": round(float(acct["cum_fees"].iloc[-1]), 2),
            "avg_invested_ratio": round(float(acct["invested_ratio"].mean()), 3),
        })
    return pd.DataFrame(rows)
