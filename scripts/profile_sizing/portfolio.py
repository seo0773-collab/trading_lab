"""다종목 포트폴리오 엔진 (profile-portfolio-v1).

각 종목의 profile-sizing target weight(percentile 저가권↑ + 추세 + regime cap이
녹아 있음)를 **종목 점수 sᵢ**로 재사용한다. 매 리밸런스(기본 월간)마다:

1. 점수 sᵢ > 0 인 종목 중 **상위 K개** 선택(러프한 상승 트렌드 추종).
2. **전체 주식 노출 = mean(top-K 점수)** — 모두 강세면 노출↑(추종), 모두 약세(DEFENSE)면
   노출↓(방어). 개별 종목 방어 로직이 합산되어 포트폴리오 현금 비중을 자동으로 올린다.
3. 그 노출을 점수 비례로 top-K에 배분, 나머지는 현금. 레버리지·공매도 없음.

성과는 포트폴리오 NAV(평가자산)에서 계산하고, 벤치마크는 같은 유니버스의
equal-weight buy & hold(분산 B&H)다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ProfileSizingConfig
from .run import run_pipeline


def compute_universe(
    panels: dict[str, pd.DataFrame], cfg: ProfileSizingConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """종목별 run_pipeline → (점수 패널, 종가 패널). 공통 마스터 인덱스로 정렬.

    점수 = final_target_weight(0~1). 상장 전/​warmup 구간은 점수 0, 가격은 NaN.
    """
    score_cols, price_cols = {}, {}
    for sym, raw in panels.items():
        if raw is None or raw.empty or "close" not in raw:
            continue
        idx = pd.DatetimeIndex(pd.to_datetime(raw.index))
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        out = run_pipeline(raw, cfg)
        fc = out["forecast"]
        score_cols[sym] = pd.Series(fc["final_target_weight"].to_numpy(), index=idx)
        price_cols[sym] = pd.Series(fc["close"].to_numpy(), index=idx)

    if not score_cols:
        return pd.DataFrame(), pd.DataFrame()
    master = sorted(set().union(*[s.index for s in price_cols.values()]))
    master = pd.DatetimeIndex(master)
    scores = pd.DataFrame(
        {s: v.reindex(master) for s, v in score_cols.items()}
    ).fillna(0.0)
    prices = pd.DataFrame(
        {s: v.reindex(master) for s, v in price_cols.items()}
    ).ffill()
    # ffill로도 안 채워지는 선두(상장 전)는 NaN 유지 → 미보유.
    prices = prices.where(
        pd.DataFrame({s: v.reindex(master).notna().cummax()
                      for s, v in price_cols.items()})
    )
    return scores, prices


def rebalance_dates(index: pd.DatetimeIndex, freq: str) -> set:
    """리밸런스 날짜 집합. 각 기간의 마지막 거래일."""
    f = freq.lower()
    if f in ("d", "daily", "1d"):
        return set(index)
    rule = {"w": "W", "weekly": "W", "m": "ME", "monthly": "ME"}.get(f, "ME")
    s = pd.Series(index, index=index)
    try:
        picks = s.groupby(pd.Grouper(freq=rule)).last().dropna()
    except ValueError:  # 옛 pandas: ME 미지원
        picks = s.groupby(pd.Grouper(freq="M")).last().dropna()
    return set(pd.DatetimeIndex(picks.values))


def simulate_portfolio(
    scores: pd.DataFrame,
    prices: pd.DataFrame,
    cfg: ProfileSizingConfig,
    *,
    top_k: int,
    rebal_freq: str,
    initial_capital: float = 10_000.0,
    market_close: pd.Series | None = None,
    market_ma_len: int = 200,
    market_off_scale: float = 0.5,
) -> dict:
    """월간(기본) 리밸런스 다종목 시뮬레이션 → NAV·노출·trades·벤치마크.

    market_close가 주어지면 시장 레짐 필터를 켠다: 시장 지수가 장기 MA(market_ma_len)
    아래면 리밸런스 시 전체 목표 노출을 market_off_scale배로 줄여(전면 약세장 회피)
    낙폭을 추가로 억제한다. MA warmup 구간은 정상으로 본다.
    """
    fee = (cfg.costs.fee_bps_per_side + cfg.costs.slippage_bps) / 10_000.0
    idx = prices.index
    syms = list(prices.columns)
    rebal = rebalance_dates(idx, rebal_freq)

    market_ok = None
    if market_close is not None:
        m = pd.Series(market_close).reindex(idx).ffill()
        ma = m.rolling(market_ma_len, min_periods=market_ma_len).mean()
        market_ok = (m > ma).to_numpy()  # True=정상, NaN비교는 False
        market_nan = ma.isna().to_numpy()  # warmup은 정상 취급
    market_flag = np.ones(len(idx))

    shares = {s: 0.0 for s in syms}
    prev_w = {s: 0.0 for s in syms}
    cash = float(initial_capital)
    holding: dict[str, tuple] = {}
    trades: list[dict] = []

    nav = np.empty(len(idx)); exposure = np.empty(len(idx))
    n_hold = np.empty(len(idx)); cash_ratio = np.empty(len(idx))

    for i, t in enumerate(idx):
        px = prices.loc[t]
        if market_ok is not None and not market_nan[i] and not market_ok[i]:
            market_flag[i] = market_off_scale
        if t in rebal:
            target_w = _target_weights(scores.loc[t], px, top_k)
            if market_flag[i] < 1.0:  # 약세 시장: 전체 노출 축소
                target_w = {s: w * market_flag[i] for s, w in target_w.items()}
            account = cash + sum(
                shares[s] * px[s] for s in syms if pd.notna(px[s])
            )
            if account > 0:
                for s in syms:
                    if pd.isna(px[s]):
                        continue
                    target_val = target_w[s] * account
                    trade_val = target_val - shares[s] * px[s]
                    if abs(trade_val) > 1e-9:
                        cash -= trade_val + fee * abs(trade_val)
                        shares[s] = target_val / px[s]
                    _track_trade(trades, holding, s, prev_w[s], target_w[s],
                                 t, px[s], top_k, fee)
                prev_w = target_w

        pos_val = sum(shares[s] * px[s] for s in syms if pd.notna(px[s]))
        account = cash + pos_val
        nav[i] = account
        exposure[i] = pos_val / account if account > 0 else 0.0
        cash_ratio[i] = cash / account if account > 0 else 1.0
        n_hold[i] = sum(1 for s in syms
                        if pd.notna(px[s]) and shares[s] * px[s] > 1e-6)

    # 마지막 봉에 잔여 보유 청산(평가용).
    last = idx[-1]
    for s, (e_t, e_p) in list(holding.items()):
        if pd.notna(prices.loc[last, s]):
            trades.append(_trade_row(s, e_t, e_p, last,
                                     float(prices.loc[last, s]), top_k, fee,
                                     "end_of_data"))

    nav_s = pd.Series(nav, index=idx, name="nav")
    forecast = pd.DataFrame({
        "close": nav_s,
        "stock_exposure": exposure,
        "cash_ratio": cash_ratio,
        "n_holdings": n_hold,
        "market_ok": market_flag,  # 1=정상, market_off_scale=약세 회피
    }, index=idx)
    bench = benchmark_equal_weight(prices) * initial_capital
    return {
        "nav": nav_s,
        "forecast": forecast,
        "trades": _trades_frame(trades),
        "benchmark": bench,
        "n_symbols": len(syms),
    }


def _target_weights(score_row: pd.Series, px: pd.Series, top_k: int) -> dict:
    """top-K 선택 + 노출=mean(top-K 점수) + 점수 비례 배분."""
    syms = list(score_row.index)
    out = {s: 0.0 for s in syms}
    avail = score_row[(score_row > 0) & px.reindex(score_row.index).notna()]
    if avail.empty:
        return out
    topk = avail.nlargest(min(top_k, len(avail)))
    exposure = float(topk.mean())          # 0~1: 전체 주식 노출(방어 신호)
    ssum = float(topk.sum())
    if ssum <= 0:
        return out
    for s, sc in topk.items():
        out[s] = exposure * (sc / ssum)    # 합 = exposure, 나머지 현금
    return out


def _track_trade(trades, holding, sym, prev_w, new_w, t, price, top_k, fee) -> None:
    if prev_w <= 1e-9 and new_w > 1e-9:
        holding[sym] = (t, price)
    elif prev_w > 1e-9 and new_w <= 1e-9 and sym in holding:
        e_t, e_p = holding.pop(sym)
        trades.append(_trade_row(sym, e_t, e_p, t, price, top_k, fee,
                                 "rebalance_out"))


def _trade_row(sym, e_t, e_p, x_t, x_p, top_k, fee, reason) -> dict:
    return {
        "symbol": sym, "direction": 1,
        "entry_time": e_t, "entry_price": e_p,
        "exit_time": x_t, "exit_price": x_p,
        "net_return": x_p / e_p - 1.0 - 2.0 * fee,
        "exit_reason": reason,
        "entry_reason": f"{sym} top{top_k} 편입",
    }


def _trades_frame(trades: list[dict]) -> pd.DataFrame:
    cols = ["direction", "entry_time", "entry_price", "exit_time", "exit_price",
            "net_return", "exit_reason", "entry_reason", "symbol"]
    if not trades:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(trades)[cols]


def benchmark_equal_weight(prices: pd.DataFrame) -> pd.Series:
    """유니버스 equal-weight buy & hold NAV(1.0 기준). 각 종목 상장 후 균등 편입.

    매 봉 가용 종목의 단순수익률 평균을 누적한다(상장 시점부터 자동 편입).
    """
    rets = prices.pct_change()
    eq = rets.mean(axis=1, skipna=True).fillna(0.0)
    return (1.0 + eq).cumprod().rename("buy_hold")
