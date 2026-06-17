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
    exposure_gain: float = 1.0,
    sector_close: dict | None = None,
    symbol_sector: dict | None = None,
    sector_off_scale: float | None = None,
) -> dict:
    """월간(기본) 리밸런스 다종목 시뮬레이션 → NAV·노출·trades·벤치마크.

    market_close가 주어지면 시장 레짐 필터를 켠다: 시장 지수가 장기 MA(market_ma_len)
    아래면 리밸런스 시 전체 목표 노출을 market_off_scale배로 줄여(전면 약세장 회피)
    낙폭을 추가로 억제한다. MA warmup 구간은 정상으로 본다.

    exposure_gain>1.0이면 전체 주식 노출(=mean top-K 점수)에 게인을 곱한 뒤 1.0으로
    클립한다. 평범한 상승장(점수가 중간)에서 풀투자에 가깝게 끌어올려 수익 갭을 줄이되,
    약세장(점수 낮음)에서는 곱해도 낮게 남아 방어 성격을 유지한다(기본 1.0=불변).

    sector_close(섹터지수→종가)와 symbol_sector(종목→섹터)가 주어지면 **종목별 섹터
    레짐 필터**를 켠다(yoon1c): 한 종목의 섹터 지수가 자기 장기 MA 아래면 그 종목의 목표
    비중만 sector_off_scale배로 줄인다(SPY 단일 필터와 달리 섹터별로 방어). 무누수를 위해
    섹터 신호도 전봉 기준이며 MA warmup은 정상 취급. 둘 중 하나라도 None이면 불변.
    """
    fee = (cfg.costs.fee_bps_per_side + cfg.costs.slippage_bps) / 10_000.0
    idx = prices.index
    syms = list(prices.columns)
    rebal = rebalance_dates(idx, rebal_freq)

    # 무누수: 신호(점수·시장레짐)는 전봉 종가 기준으로 판단하고 당봉 종가에 체결한다.
    sig = scores.shift(1).fillna(0.0)

    market_ok = None
    if market_close is not None:
        m = pd.Series(market_close).reindex(idx).ffill()
        ma = m.rolling(market_ma_len, min_periods=market_ma_len).mean()
        ok = (m > ma).shift(1)       # 전봉 레짐으로 판단(무누수)
        market_ok = ok.to_numpy()    # True=정상, NaN비교는 False
        market_nan = ma.shift(1).isna().to_numpy()  # warmup은 정상 취급
    market_flag = np.ones(len(idx))

    # 종목별 섹터 레짐 필터: 섹터지수가 자기 MA 아래(전봉)면 약세 → 해당 종목만 축소.
    sector_weak: dict | None = None
    soff = market_off_scale if sector_off_scale is None else float(sector_off_scale)
    if sector_close is not None and symbol_sector is not None:
        sector_weak = {}
        for sec, ser in sector_close.items():
            sm = pd.Series(ser).reindex(idx).ffill()
            sma = sm.rolling(market_ma_len, min_periods=market_ma_len).mean()
            below = (sm <= sma).shift(1).to_numpy()      # 약세=True(전봉 기준)
            warm = sma.shift(1).isna().to_numpy()        # warmup은 정상
            sector_weak[sec] = np.where(warm, False, below == True)  # noqa: E712

    shares = {s: 0.0 for s in syms}
    prev_w = {s: 0.0 for s in syms}
    cash = float(initial_capital)
    holding: dict[str, tuple] = {}
    trades: list[dict] = []
    last_target: dict = {}        # 가장 최근 리밸런스의 목표 비중(페이퍼 트레이딩용)
    last_rebal_date = None

    nav = np.empty(len(idx)); exposure = np.empty(len(idx))
    n_hold = np.empty(len(idx)); cash_ratio = np.empty(len(idx))

    for i, t in enumerate(idx):
        px = prices.loc[t]
        if market_ok is not None and not market_nan[i] and not market_ok[i]:
            market_flag[i] = market_off_scale
        if t in rebal:
            target_w = _target_weights(sig.loc[t], px, top_k, exposure_gain)
            if market_flag[i] < 1.0:  # 약세 시장: 전체 노출 축소
                target_w = {s: w * market_flag[i] for s, w in target_w.items()}
            if sector_weak is not None:  # 종목별 섹터 약세: 해당 종목만 축소
                for s in target_w:
                    sec = symbol_sector.get(s)
                    if sec is not None and sector_weak.get(sec) is not None \
                            and sector_weak[sec][i]:
                        target_w[s] *= soff
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
            last_target = {s: w for s, w in target_w.items() if w > 1e-9}
            last_rebal_date = t

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
    return {
        "nav": nav_s,
        "forecast": forecast,
        "trades": _trades_frame(trades),
        # 기본 벤치마크 = 진짜 buy & hold(초기 균등 후 보유). 1.0 기준 → 자본 환산.
        "benchmark": benchmark_buy_hold(prices) * initial_capital,
        "benchmark_ew": benchmark_equal_weight(prices) * initial_capital,
        "n_symbols": len(syms),
        "last_target": last_target,
        "last_rebal_date": last_rebal_date,
    }


def _target_weights(score_row: pd.Series, px: pd.Series, top_k: int,
                    exposure_gain: float = 1.0) -> dict:
    """top-K 선택 + 노출=mean(top-K 점수)*게인(클립 1.0) + 점수 비례 배분."""
    syms = list(score_row.index)
    out = {s: 0.0 for s in syms}
    avail = score_row[(score_row > 0) & px.reindex(score_row.index).notna()]
    if avail.empty:
        return out
    topk = avail.nlargest(min(top_k, len(avail)))
    exposure = min(float(topk.mean()) * exposure_gain, 1.0)  # 0~1: 전체 주식 노출
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
    """**매일** 균등비중으로 재조정하는 지수 NAV(1.0 기준) — 참고용.

    매 봉 가용 종목의 단순수익률 평균을 누적한다. 진짜 buy & hold가 아니라
    상시 리밸런싱 equal-weight라는 점에 유의(승자를 매일 덜어냄).
    """
    rets = prices.pct_change()
    eq = rets.mean(axis=1, skipna=True).fillna(0.0)
    return (1.0 + eq).cumprod().rename("equal_weight_index")


def benchmark_buy_hold(prices: pd.DataFrame) -> pd.Series:
    """진짜 buy & hold NAV(1.0 기준): 각 종목 첫 상장일에 1/N 자본을 넣고 보유.

    상장 전 자본 조각은 현금으로 둔다(성장 없음). 리밸런싱 없음 = "사서 묻어두기".
    """
    n = prices.shape[1]
    if n == 0:
        return pd.Series(dtype=float, name="buy_hold")
    alloc = 1.0 / n
    first_px = prices.apply(
        lambda c: c.dropna().iloc[0] if c.notna().any() else float("nan")
    )
    shares = alloc / first_px
    listed = prices.notna()
    value = prices.mul(shares, axis=1).where(listed, other=alloc)
    return value.sum(axis=1).rename("buy_hold")
