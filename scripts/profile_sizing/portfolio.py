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
from .indicators import moving_average
from .run import run_pipeline


def compute_universe(
    panels: dict[str, pd.DataFrame], cfg: ProfileSizingConfig,
    *, leveraged_symbols: set | None = None,
    leverage_regimes: tuple[str, ...] = ("RECOVERY",),
    market_ok: pd.Series | None = None,
    mom_gate: dict | None = None,
    sr_gate: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """종목별 run_pipeline → (점수 패널, 종가 패널). 공통 마스터 인덱스로 정렬.

    점수 = final_target_weight(0~1). 상장 전/​warmup 구간은 점수 0, 가격은 NaN.

    leveraged_symbols(레버리지 슬리브)에 든 종목은 *자기 국면이 leverage_regimes일
    때만* 점수를 살리고 그 외엔 0으로 죽인다. 컨트래리언 점수는 폭락 중에도 "싸다"고
    레버리지를 사들이므로, "깊은 저가권 + 회복 확인"(=RECOVERY 국면)으로 제한해
    바닥 칼잡기·횡보 decay를 피하고 회복 램프에서만 레버리지를 태운다.

    market_ok(시장필터 ON 불리언 시리즈)가 주어지면, 레버리지 종목 점수를 *시장이
    OFF인 바(SPY<200MA)* 에서 추가로 0으로 죽인다. RECOVERY는 자기 국면 기준이라
    2008류 위기 바닥의 false 반등에 물릴 수 있으므로, 시장 추세까지 ON일 때만
    레버리지를 태워 위기 바닥을 차단한다(additive·기본 off).

    mom_gate(yoon3 모멘텀 게이트 config)가 주어지고 enabled면, 종목별 칼만 히스토그램
    누적프로파일 백분위를 [g_min,1.0] 게이트로 만들어 *그 종목 점수에 곱한다*(블렌드:
    저가권 × 모멘텀). 저가권 점수(컨트래리언)가 가리켜도 모멘텀이 여전히 자기 분포
    하위면 게이트가 비중을 억제해 "떨어지는 칼날"을 피하고, 모멘텀이 올라오면 게이트가
    열려 회복 진입 타이밍을 보강한다. 무누수(과거·현재만 누적)이며 기본 off.
    """
    leveraged_symbols = leveraged_symbols or set()
    use_gate = bool(mom_gate) and bool(mom_gate.get("enabled"))
    use_sr = bool(sr_gate) and bool(sr_gate.get("enabled"))
    score_cols, price_cols = {}, {}
    for sym, raw in panels.items():
        if raw is None or raw.empty or "close" not in raw:
            continue
        idx = pd.DatetimeIndex(pd.to_datetime(raw.index))
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        out = run_pipeline(raw, cfg)
        fc = out["forecast"]
        score = fc["final_target_weight"].to_numpy().copy()
        if use_gate:
            from .momentum import momentum_gate  # noqa: E402
            gate = momentum_gate(
                pd.Series(fc["close"].to_numpy(), index=idx), mom_gate,
            )
            score = score * gate.to_numpy()
        if use_sr:
            from .sr_gate import sr_gate as _sr_gate  # noqa: E402
            ohlc = raw.copy()
            ohlc.index = idx  # tz 제거 인덱스로 정렬(score와 동일 순서)
            score = score * _sr_gate(ohlc, sr_gate).to_numpy()
        if sym in leveraged_symbols:
            reg = np.asarray(out["regime"])
            allowed = np.array([str(r) in leverage_regimes for r in reg])
            score = np.where(allowed, score, 0.0)
        score_cols[sym] = pd.Series(score, index=idx)
        price_cols[sym] = pd.Series(fc["close"].to_numpy(), index=idx)

    if not score_cols:
        return pd.DataFrame(), pd.DataFrame()
    master = sorted(set().union(*[s.index for s in price_cols.values()]))
    master = pd.DatetimeIndex(master)
    scores = pd.DataFrame(
        {s: v.reindex(master) for s, v in score_cols.items()}
    ).fillna(0.0)
    if leveraged_symbols and market_ok is not None:
        mok = pd.Series(market_ok)
        if mok.index.tz is not None:
            mok.index = mok.index.tz_localize(None)
        mok = mok.reindex(master).ffill().fillna(True).astype(bool)
        for col in scores.columns:
            if col in leveraged_symbols:
                scores[col] = scores[col].where(mok.to_numpy(), 0.0)
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
    market_mode: str = "binary",
    market_entry_ma_len: int = 50,
    market_recover_floor: float = 0.85,
    market_kalman_q: float = 0.01,
    market_kalman_r: float = 0.10,
    market_kalman_fast: int = 12,
    market_kalman_slow: int = 26,
    market_z_win: int = 200,
    market_z_scale: float = 1.0,
    exposure_gain: float = 1.0,
    max_exposure: float = 1.0,
    borrow_rate_annual: float = 0.0,
    weight_scheme: str = "score",
    vol_lookback: int = 63,
    volumes: pd.DataFrame | None = None,
    sector_close: dict | None = None,
    symbol_sector: dict | None = None,
    sector_off_scale: float | None = None,
    rsi_close: pd.Series | None = None,
    rsi_len: int = 14,
    rsi_ma_len: int = 14,
    rsi_ma_kind: str = "SMA",
    rsi_threshold: float = 50.0,
    rsi_above_scale: float = 1.0,
    rsi_below_scale: float = 0.5,
    short_hedge_close: pd.Series | None = None,
    short_hedge_symbol: str = "SPY",
    short_hedge_ratio: float = 0.0,
    short_hedge_max: float = 0.0,
    short_hedge_trailing_pct: float | None = None,
    safe_close: dict | None = None,
    safe_ratio: float = 0.0,
    safe_max: float = 0.0,
    safe_ma_len: int = 100,
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

    rsi_close가 주어지면 포트폴리오 레벨 RSI 필터를 켠다: RSI(length)의 MA가 threshold
    이상이면 rsi_above_scale, 미만이면 rsi_below_scale을 전체 목표 노출에 곱한다. 신호는
    전봉 기준이고 warmup은 정상 취급한다.

    short_hedge_close가 주어지면 시장 필터 약세 구간에서 남는 목표 현금 일부를 지수
    숏 헤지로 배정한다. 숏 헤지는 개별 종목 선정 로직과 분리된 방어 오버레이다.
    short_hedge_trailing_pct가 양수면 숏 진입 후 최고 수익률에서 해당 폭만큼 반등할 때
    숏을 즉시 청산하고, 시장 레짐이 정상으로 회복될 때까지 재진입하지 않는다.

    safe_close(안전자산 심볼→종가 dict)가 주어지면 **연속형 안전자산 슬리브**를 켠다
    (yoon1j): 약세장에서 남는 현금 완충분(spare=1−주식목표노출)의 일부를 안전자산
    (TLT/GLD 등)으로 돌린다. 이진 전환이 아니라 버퍼만 회전하므로 손실 상한이 버퍼로
    제한된다. 각 안전자산은 *자기 추세(전봉 종가>MA safe_ma_len)가 ON일 때만* 매수
    (2022류 채권 동반하락 시엔 사지 않고 현금 유지). 슬리브 목표비중 =
    min(safe_max, safe_spare, safe_spare*safe_ratio)을 추세 ON 안전자산에 균등 배분
    (safe_spare 상한으로 무레버 보장: 주식+안전+현금 ≤ 1), 리밸런스 주기에 맞춰
    회전한다. 셋 중 하나라도 0/None이면 비활성.
    """
    fee = (cfg.costs.fee_bps_per_side + cfg.costs.slippage_bps) / 10_000.0
    idx = prices.index
    syms = list(prices.columns)
    rebal = rebalance_dates(idx, rebal_freq)

    # 무누수: 신호(점수·시장레짐)는 전봉 종가 기준으로 판단하고 당봉 종가에 체결한다.
    sig = scores.shift(1).fillna(0.0)

    # 내부 배분용 가중 패널(전봉까지만, shift(1)로 무누수). score 스킴이면 미사용.
    #  - inv_vol: 전봉까지 수익률 표준편차 → 1/σ 비례(risk-parity).
    #  - dollar_volume: 전봉까지 달러거래량(volume×close) 평균 → 비례(거래 쏠림 추종).
    vol_panel = None
    if str(weight_scheme) == "inv_vol":
        vol_panel = (
            prices.pct_change()
            .rolling(vol_lookback, min_periods=max(5, vol_lookback // 3))
            .std()
            .shift(1)
        )
    elif str(weight_scheme) == "dollar_volume" and volumes is not None:
        dv = volumes.reindex(prices.index).reindex(columns=prices.columns) * prices
        vol_panel = (
            dv.rolling(vol_lookback, min_periods=max(5, vol_lookback // 3))
            .mean()
            .shift(1)
        )

    # 시장 레짐 스케일: 1.0=정상, <1.0=약세 축소. mode로 이진/칼만 연속 선택.
    #  - binary(기본): 지수가 장기 MA 아래(전봉)면 off_scale로 계단 축소(원본 동작).
    #  - kalman: 지수 칼만 MACD z-score를 tanh로 [off_scale,1] 연속 매핑 → 절대
    #    빠지지 않고 추세 강도에 비례 축소·반등 시 빠른 재진입. 둘 다 전봉 기준(무누수).
    market_scale = np.ones(len(idx))
    if market_close is not None:
        m = pd.Series(market_close).reindex(idx).ffill()
        if str(market_mode) == "kalman":
            market_scale = _market_kalman_scale(
                m, market_kalman_fast, market_kalman_slow,
                market_kalman_q, market_kalman_r,
                market_z_win, market_z_scale, market_off_scale,
            )
        elif str(market_mode) == "asym":
            # 비대칭 히스테리시스: 디리스크=장기MA(market_ma_len) 하향이탈,
            # 복귀=단기MA(market_entry_ma_len) 상향돌파. "느리게 나가고 빠르게 들어온다"
            # → V자 반등 초기 포착, 방어력(나가는 신호)은 장기MA로 보존.
            ma_x = m.rolling(market_ma_len, min_periods=market_ma_len).mean()
            ma_e = m.rolling(market_entry_ma_len,
                             min_periods=market_entry_ma_len).mean()
            above_x = (m > ma_x).to_numpy()
            above_e = (m > ma_e).to_numpy()
            warm = ma_x.isna().to_numpy()
            st = np.ones(len(idx))
            cur = 1.0
            for i in range(len(idx)):
                if warm[i]:
                    cur = 1.0
                elif cur >= 1.0:                 # ON: 장기MA 이탈 시 OFF
                    if not above_x[i]:
                        cur = market_off_scale
                else:                            # OFF: 단기MA 회복 시 ON
                    if above_e[i]:
                        cur = 1.0
                st[i] = cur
            market_scale = (
                pd.Series(st, index=idx).shift(1).fillna(1.0).to_numpy()
            )
        elif str(market_mode) == "ramp":
            # 연속 복귀: 장기MA 대비 비율로 [off_scale,1] 선형 보간. 바닥서 반등해
            # MA에 근접할수록(ratio→1) 노출 점진 복원 → 이진 계단의 늦은 복귀 완화.
            ma = m.rolling(market_ma_len, min_periods=market_ma_len).mean()
            ratio = m / ma
            fl = float(market_recover_floor)
            sc = market_off_scale + (1.0 - market_off_scale) * (
                (ratio - fl) / (1.0 - fl)
            )
            sc = sc.clip(market_off_scale, 1.0).where(ratio < 1.0, 1.0)
            warm = ma.isna()
            market_scale = (
                sc.where(~warm, 1.0).shift(1).fillna(1.0).to_numpy()
            )
        else:
            ma = m.rolling(market_ma_len, min_periods=market_ma_len).mean()
            below = (m <= ma).shift(1).to_numpy()
            warm = ma.shift(1).isna().to_numpy()
            market_scale = np.where(
                warm, 1.0, np.where(below == True, market_off_scale, 1.0)  # noqa: E712
            )
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

    rsi_scale = np.ones(len(idx))
    rsi_ma_values = np.full(len(idx), np.nan)
    if rsi_close is not None:
        rc = pd.Series(rsi_close).reindex(idx).ffill()
        rsi = _rsi_wilder(rc, rsi_len)
        rma = moving_average(rsi, rsi_ma_len, rsi_ma_kind)
        rsi_ma_values = rma.to_numpy()
        below = (rma < float(rsi_threshold)).shift(1).to_numpy()
        warm = rma.shift(1).isna().to_numpy()
        rsi_scale = np.where(
            warm,
            float(rsi_above_scale),
            np.where(below == True, float(rsi_below_scale), float(rsi_above_scale)),
        )

    hedge_px = None
    hedge_enabled = (
        short_hedge_close is not None
        and float(short_hedge_ratio) > 0.0
        and float(short_hedge_max) > 0.0
    )
    if hedge_enabled:
        hedge_px = pd.Series(short_hedge_close).reindex(idx).ffill()

    # 안전자산 슬리브: 종가 패널 + 추세게이트(전봉 종가>MA, 무누수) 사전계산.
    safe_enabled = (
        safe_close is not None
        and float(safe_ratio) > 0.0
        and float(safe_max) > 0.0
    )
    safe_px_df = None
    safe_gate = None
    safe_syms: list[str] = []
    if safe_enabled:
        cols = {
            str(s): pd.Series(c).reindex(idx).ffill()
            for s, c in safe_close.items()
        }
        safe_px_df = pd.DataFrame(cols)
        sma = safe_px_df.rolling(safe_ma_len, min_periods=safe_ma_len).mean()
        # 전봉 추세 ON(전봉 종가>MA)만 매수. shift(1)의 첫 행 NaN이 bool(NaN)=True로
        # 새던 엣지를 막기 위해 False로 채운다(무누수·warmup은 미보유).
        safe_gate = (safe_px_df > sma).shift(1, fill_value=False)
        safe_syms = list(safe_px_df.columns)
    safe_shares = {s: 0.0 for s in safe_syms}
    prev_safe_w = {s: 0.0 for s in safe_syms}

    shares = {s: 0.0 for s in syms}
    hedge_shares = 0.0
    prev_short_w = 0.0
    hedge_peak_profit = 0.0
    hedge_trailing_locked = False
    hedge_trailing_pct = (
        float(short_hedge_trailing_pct)
        if short_hedge_trailing_pct is not None else 0.0
    )
    prev_w = {s: 0.0 for s in syms}
    cash = float(initial_capital)
    holding: dict[str, tuple] = {}
    hedge_holding: tuple | None = None
    trades: list[dict] = []
    last_target: dict = {}        # 가장 최근 리밸런스의 목표 비중(페이퍼 트레이딩용)
    last_rebal_date = None

    nav = np.empty(len(idx)); exposure = np.empty(len(idx))
    n_hold = np.empty(len(idx)); cash_ratio = np.empty(len(idx))
    short_exposure = np.zeros(len(idx)); net_exposure = np.empty(len(idx))
    safe_exposure = np.zeros(len(idx))
    hedge_target_history = np.zeros(len(idx), dtype=float)
    target_history = np.zeros((len(idx), len(syms)), dtype=float)
    safe_target_history = np.zeros((len(idx), len(safe_syms)), dtype=float)

    # 일봉 가정(연 252봉)으로 마진 차입이자를 봉별 차감. 0이면 불변.
    borrow_per_bar = float(borrow_rate_annual) / 252.0
    for i, t in enumerate(idx):
        px = prices.loc[t]
        if borrow_per_bar > 0.0 and cash < 0.0:
            cash += cash * borrow_per_bar  # cash<0 → 차입잔액에 이자(더 음수로)
        hpx = (
            float(hedge_px.iloc[i])
            if hedge_px is not None and pd.notna(hedge_px.iloc[i])
            else float("nan")
        )
        spx = safe_px_df.loc[t] if safe_enabled else None
        market_flag[i] = market_scale[i]
        if market_flag[i] >= 1.0:
            hedge_trailing_locked = False
        if (
            hedge_trailing_pct > 0.0 and hedge_holding is not None
            and prev_short_w > 1e-9 and np.isfinite(hpx)
        ):
            _, entry_price = hedge_holding
            current_profit = _short_profit(entry_price, hpx)
            hedge_peak_profit = max(hedge_peak_profit, current_profit)
            if (
                hedge_peak_profit > 0.0
                and hedge_peak_profit - current_profit >= hedge_trailing_pct
            ):
                hedge_trade_val = -hedge_shares * hpx
                cash -= hedge_trade_val + fee * abs(hedge_trade_val)
                e_t, e_p = hedge_holding
                trades.append(_short_trade_row(
                    short_hedge_symbol,
                    e_t,
                    e_p,
                    t,
                    hpx,
                    fee,
                    "trailing_take_profit",
                ))
                hedge_shares = 0.0
                prev_short_w = 0.0
                hedge_holding = None
                hedge_peak_profit = 0.0
                hedge_trailing_locked = True
        if t in rebal:
            vol_row = vol_panel.loc[t] if vol_panel is not None else None
            target_w = _target_weights(
                sig.loc[t], px, top_k, exposure_gain,
                scheme=weight_scheme, vol_row=vol_row,
                max_exposure=max_exposure,
            )
            if market_flag[i] < 1.0:  # 약세 시장: 전체 노출 축소
                target_w = {s: w * market_flag[i] for s, w in target_w.items()}
            if rsi_scale[i] != 1.0:  # RSI MA 약세/강세 액션: 전체 노출 조절
                target_w = {s: w * rsi_scale[i] for s, w in target_w.items()}
            if sector_weak is not None:  # 종목별 섹터 약세: 해당 종목만 축소
                for s in target_w:
                    sec = symbol_sector.get(s)
                    if sec is not None and sector_weak.get(sec) is not None \
                            and sector_weak[sec][i]:
                        target_w[s] *= soff
            long_target_exposure = sum(target_w.values())
            spare = max(0.0, 1.0 - long_target_exposure)
            target_short_w = 0.0
            if (
                hedge_enabled and market_flag[i] < 1.0
                and not hedge_trailing_locked and np.isfinite(hpx)
            ):
                target_short_w = min(
                    float(short_hedge_max),
                    spare * float(short_hedge_ratio),
                )
            # 안전자산 슬리브 목표비중(추세 ON 자산에 균등). spare에서 숏 배정분 제외.
            safe_on: list[str] = []
            safe_w_each = 0.0
            if safe_enabled:
                safe_on = [
                    s for s in safe_syms
                    if bool(safe_gate.loc[t, s]) and pd.notna(spx[s])
                ]
                safe_spare = max(0.0, spare - target_short_w)
                # 무레버 불변식: 안전자산 배정은 완충분(safe_spare)을 넘지 못한다.
                # ratio>1에서 spare*ratio가 버퍼를 초과해 암묵적·무비용 레버리지
                # (cash<0)가 생기던 버그를 차단 → 주식+안전+현금 ≤ 1, 손실 상한=버퍼.
                safe_budget = (
                    min(float(safe_max), safe_spare, safe_spare * float(safe_ratio))
                    if safe_on else 0.0
                )
                safe_w_each = safe_budget / len(safe_on) if safe_on else 0.0
            account = cash + sum(
                shares[s] * px[s] for s in syms if pd.notna(px[s])
            )
            if np.isfinite(hpx):
                account += hedge_shares * hpx
            if safe_enabled:
                account += sum(
                    safe_shares[s] * spx[s]
                    for s in safe_syms if pd.notna(spx[s])
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
                if hedge_enabled and np.isfinite(hpx):
                    hedge_target_val = -target_short_w * account
                    hedge_trade_val = hedge_target_val - hedge_shares * hpx
                    if abs(hedge_trade_val) > 1e-9:
                        cash -= hedge_trade_val + fee * abs(hedge_trade_val)
                        hedge_shares = hedge_target_val / hpx
                    hedge_holding = _track_short_trade(
                        trades,
                        hedge_holding,
                        short_hedge_symbol,
                        prev_short_w,
                        target_short_w,
                        t,
                        hpx,
                        fee,
                    )
                    if prev_short_w <= 1e-9 and target_short_w > 1e-9:
                        hedge_peak_profit = 0.0
                    elif target_short_w <= 1e-9:
                        hedge_peak_profit = 0.0
                    prev_short_w = target_short_w
                if safe_enabled:
                    for s in safe_syms:
                        if pd.isna(spx[s]):
                            continue
                        tw = safe_w_each if s in safe_on else 0.0
                        target_val = tw * account
                        trade_val = target_val - safe_shares[s] * spx[s]
                        if abs(trade_val) > 1e-9:
                            cash -= trade_val + fee * abs(trade_val)
                            safe_shares[s] = target_val / spx[s]
                        prev_safe_w[s] = tw
                prev_w = target_w
            last_target = {s: w for s, w in target_w.items() if w > 1e-9}
            last_rebal_date = t

        pos_val = sum(shares[s] * px[s] for s in syms if pd.notna(px[s]))
        hedge_val = hedge_shares * hpx if np.isfinite(hpx) else 0.0
        safe_val = (
            sum(safe_shares[s] * spx[s] for s in safe_syms if pd.notna(spx[s]))
            if safe_enabled else 0.0
        )
        account = cash + pos_val + hedge_val + safe_val
        nav[i] = account
        exposure[i] = pos_val / account if account > 0 else 0.0
        short_exposure[i] = abs(hedge_val) / account if account > 0 else 0.0
        safe_exposure[i] = safe_val / account if account > 0 else 0.0
        net_exposure[i] = exposure[i] - short_exposure[i]
        cash_ratio[i] = cash / account if account > 0 else 1.0
        n_hold[i] = sum(1 for s in syms
                        if pd.notna(px[s]) and shares[s] * px[s] > 1e-6)
        target_history[i, :] = [prev_w.get(s, 0.0) for s in syms]
        hedge_target_history[i] = prev_short_w
        if safe_syms:
            safe_target_history[i, :] = [prev_safe_w.get(s, 0.0) for s in safe_syms]

    # 마지막 봉에 잔여 보유 청산(평가용).
    last = idx[-1]
    for s, (e_t, e_p) in list(holding.items()):
        if pd.notna(prices.loc[last, s]):
            trades.append(_trade_row(s, e_t, e_p, last,
                                     float(prices.loc[last, s]), top_k, fee,
                                     "end_of_data"))
    if hedge_holding is not None and hedge_px is not None:
        last_hpx = hedge_px.reindex([last]).iloc[0]
        if pd.notna(last_hpx):
            e_t, e_p = hedge_holding
            trades.append(_short_trade_row(
                short_hedge_symbol,
                e_t,
                e_p,
                last,
                float(last_hpx),
                fee,
                "end_of_data",
            ))

    nav_s = pd.Series(nav, index=idx, name="nav")
    target_weights = pd.DataFrame(target_history, index=idx, columns=syms)
    short_target_weights = pd.DataFrame(
        {f"{short_hedge_symbol}_SHORT": hedge_target_history}, index=idx,
    )
    safe_target_weights = pd.DataFrame(
        {f"{s}_SAFE": safe_target_history[:, j] for j, s in enumerate(safe_syms)},
        index=idx,
    )
    forecast = pd.DataFrame({
        "close": nav_s,
        "stock_exposure": exposure,
        "short_exposure": short_exposure,
        "net_exposure": net_exposure,
        "cash_ratio": cash_ratio,
        "n_holdings": n_hold,
        "market_ok": market_flag,  # 1=정상, market_off_scale=약세 회피
        "rsi_filter": rsi_scale,
        "rsi_ma": rsi_ma_values,
    }, index=idx)
    if safe_enabled:
        forecast["safe_exposure"] = safe_exposure  # 안전자산 슬리브 비중
    return {
        "nav": nav_s,
        "forecast": forecast,
        "trades": _trades_frame(trades),
        # 기본 벤치마크 = 진짜 buy & hold(초기 균등 후 보유). 1.0 기준 → 자본 환산.
        "benchmark": benchmark_buy_hold(prices) * initial_capital,
        "benchmark_ew": benchmark_equal_weight(prices) * initial_capital,
        "n_symbols": len(syms),
        "target_weights": target_weights,
        "short_target_weights": short_target_weights,
        "safe_target_weights": safe_target_weights,
        "last_target": last_target,
        "last_rebal_date": last_rebal_date,
    }


def _market_kalman_scale(
    close: pd.Series, fast: int, slow: int, q: float, r: float,
    z_win: int, z_scale: float, off_scale: float,
) -> np.ndarray:
    """지수 칼만 MACD z-score → [off_scale, 1.0] 연속 시장 레짐 스케일(전봉, 무누수).

    macd = EMA(fast)−EMA(slow); kal = kalman_1d(macd); z = kal / rolling_std(kal).
    scale = off_scale + (1−off_scale)·½(1+tanh(z/z_scale)). 강한 상승 z≫0 → ~1.0,
    강한 하락 z≪0 → ~off_scale, 추세중립 z≈0 → 중간값. warmup은 정상(1.0).
    """
    from indicators.kalman import kalman_1d  # noqa: E402
    c = close.astype(float)
    macd = (c.ewm(span=fast, adjust=False).mean()
            - c.ewm(span=slow, adjust=False).mean())
    kal = kalman_1d(macd, q, r)
    sd = kal.rolling(z_win, min_periods=z_win).std()
    z = kal / sd.replace(0.0, np.nan)
    scale = off_scale + (1.0 - off_scale) * 0.5 * (1.0 + np.tanh(z / z_scale))
    scale = scale.clip(off_scale, 1.0).shift(1)  # 전봉 신호
    return scale.fillna(1.0).to_numpy()  # warmup/결측 = 정상


def _target_weights(score_row: pd.Series, px: pd.Series, top_k: int,
                    exposure_gain: float = 1.0, *, scheme: str = "score",
                    vol_row: pd.Series | None = None,
                    max_exposure: float = 1.0) -> dict:
    """top-K 선택 + 노출=mean(top-K 점수)*게인(클립 1.0) + top-K 내부 비중 배분.

    선택·노출 레벨은 스킴과 무관(점수 기준). 배분만 분기한다:
      - score(기본): 점수 비례(저가권↑ = 컨트래리언 틸트).
      - inv_vol: 1/변동성 비례(risk-parity). 변동성 결측은 평균치로 대체, 전부
        결측이면 점수 비례로 폴백.
    """
    syms = list(score_row.index)
    out = {s: 0.0 for s in syms}
    avail = score_row[(score_row > 0) & px.reindex(score_row.index).notna()]
    if avail.empty:
        return out
    topk = avail.nlargest(min(top_k, len(avail)))
    # 전체 주식 노출. max_exposure>1.0이면 레버리지(현금 차입) 허용.
    exposure = min(float(topk.mean()) * exposure_gain, max_exposure)

    if scheme == "inv_vol" and vol_row is not None:
        sig_v = pd.Series(vol_row).reindex(topk.index)
        inv = 1.0 / sig_v.where(sig_v > 0.0)        # σ>0만, 0/NaN은 결측
        if inv.notna().any():
            inv = inv.fillna(float(inv[inv.notna()].mean()))  # 결측=평균 비중
            wsum = float(inv.sum())
            if wsum > 0:
                for s in topk.index:
                    out[s] = exposure * (float(inv[s]) / wsum)
                return out
        # 전부 결측(warmup) → 점수 비례로 폴백

    if scheme == "dollar_volume" and vol_row is not None:
        dv = pd.Series(vol_row).reindex(topk.index).where(lambda s: s > 0.0)
        if dv.notna().any():
            dv = dv.fillna(float(dv[dv.notna()].mean()))  # 결측=평균 비중
            wsum = float(dv.sum())
            if wsum > 0:
                for s in topk.index:
                    out[s] = exposure * (float(dv[s]) / wsum)
                return out
        # 전부 결측(warmup) → 점수 비례로 폴백

    ssum = float(topk.sum())
    if ssum <= 0:
        return out
    for s, sc in topk.items():
        out[s] = exposure * (sc / ssum)    # 합 = exposure, 나머지 현금
    return out


def _rsi_wilder(close: pd.Series, length: int) -> pd.Series:
    """Wilder RSI(0~100). 길이 미만 구간은 NaN으로 둔다."""
    c = pd.Series(np.asarray(close, dtype=float), index=close.index)
    delta = c.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / length, adjust=False,
                        min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1.0 / length, adjust=False,
                        min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.where(avg_loss > 0.0, 100.0).where(avg_gain > 0.0, 0.0)
    return rsi.where(avg_gain.notna() & avg_loss.notna())


def _track_trade(trades, holding, sym, prev_w, new_w, t, price, top_k, fee) -> None:
    if prev_w <= 1e-9 and new_w > 1e-9:
        holding[sym] = (t, price)
    elif prev_w > 1e-9 and new_w <= 1e-9 and sym in holding:
        e_t, e_p = holding.pop(sym)
        trades.append(_trade_row(sym, e_t, e_p, t, price, top_k, fee,
                                 "rebalance_out"))


def _track_short_trade(
    trades, holding, sym, prev_w, new_w, t, price, fee,
) -> tuple | None:
    if prev_w <= 1e-9 and new_w > 1e-9:
        return (t, price)
    if prev_w > 1e-9 and new_w <= 1e-9 and holding is not None:
        e_t, e_p = holding
        trades.append(_short_trade_row(sym, e_t, e_p, t, price, fee,
                                       "hedge_off"))
        return None
    return holding


def _trade_row(sym, e_t, e_p, x_t, x_p, top_k, fee, reason) -> dict:
    return {
        "symbol": sym, "direction": 1,
        "entry_time": e_t, "entry_price": e_p,
        "exit_time": x_t, "exit_price": x_p,
        "net_return": x_p / e_p - 1.0 - 2.0 * fee,
        "exit_reason": reason,
        "entry_reason": f"{sym} top{top_k} 편입",
    }


def _short_trade_row(sym, e_t, e_p, x_t, x_p, fee, reason) -> dict:
    return {
        "symbol": sym, "direction": -1,
        "entry_time": e_t, "entry_price": e_p,
        "exit_time": x_t, "exit_price": x_p,
        "net_return": e_p / x_p - 1.0 - 2.0 * fee,
        "exit_reason": reason,
        "entry_reason": f"{sym} 시장 약세 헤지 숏",
    }


def _short_profit(entry_price: float, current_price: float) -> float:
    if current_price <= 0:
        return 0.0
    return float(entry_price / current_price - 1.0)


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
