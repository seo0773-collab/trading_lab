"""final0706 소액계좌(정수주 제약) 전체기간 재현 검증 — 시드 250만원.

목적: "250만원으로 실제 운용하면 final0706의 경향성(성과 프로필)이 유지되는가"를
연속비중(공식 백테스트) vs 정수주 집행(매 리밸런스 integer_execution_plan 적용) 비교로
직접 실측한다. 사이징 상담(₩90만=절대최소, ₩250만=권장최소)에서 이어지는 검증.

방법(재구현 최소화):
  1) 공식 파이프라인(Yoon1kHandler.build_artifacts)을 그대로 호출해 상위 통화틸트·
     신용게이트 목표비중(target_w_yoon1j_lp/kr/__CASH__, 전체 기간)과 연속비중
     equity를 얻는다.
  2) `profile_sizing.portfolio.simulate_portfolio`를 몽키패치해 하위 슬리브(lp/kr)
     호출 시 내부 sim(전체 기간 종목별 target_weights/safe_target_weights)을
     그대로 가로챈다 — 재구현 없이 공식 엔진의 실제 산출물을 얻는 방법.
  3) 상위×하위 비중을 곱해 "전체 포트폴리오 기준 종목별 목표비중" 전체 기간
     시계열을 복원하고, 비중이 바뀌는 날(리밸런스일)마다 `integer_execution_plan`
     (라이브 실행기와 동일 그리디 정수주 알고리즘)으로 실제 정수주 주문을 낸다.
     체결비용은 각 슬리브 config의 fee_bps_per_side+slippage_bps를 매 주문 편도로 부과.
  4) 정수주 NAV(일별 실제가 재평가) vs 공식 연속비중 equity를 CAGR/Sharpe/MDD/추적오차로 비교.

Usage:
    PYTHONPATH=src:scripts python3 scripts/profile_sizing/final0706_small_capital_backtest.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

SEED_KRW = 2_500_000.0

import trading_lab.strategies.profile_portfolio as pp_mod  # noqa: E402
from profile_sizing.engine import performance  # noqa: E402
from profile_sizing.live.execution_layer import integer_execution_plan  # noqa: E402
from profile_sizing.portfolio import rebalance_dates  # noqa: E402
from run_kalman_pipeline import load_yfinance  # noqa: E402
from trading_lab.strategies.yoon1k import Yoon1kHandler  # noqa: E402


def capture_sub_sims():
    """profile_portfolio 모듈에 바인딩된 simulate_portfolio를 몽키패치해
    호출 순서대로 sim dict를 리스트에 적립(모듈 로컬 바인딩이라 여기서 패치해야 함)."""
    captured = []
    orig = pp_mod.simulate_portfolio

    def wrapped(*args, **kwargs):
        sim = orig(*args, **kwargs)
        captured.append(sim)
        return sim

    pp_mod.simulate_portfolio = wrapped
    return captured, orig


def build_official():
    cfg = json.loads((ROOT / "configs" / "strategies" / "final0706.json").read_text("utf-8"))
    captured, orig = capture_sub_sims()
    try:
        h = Yoon1kHandler()
        raw3 = h.load_data("PORTFOLIO", cfg, synthetic=False)
        art = h.build_artifacts(raw3, cfg, symbol="PORTFOLIO", phase="all", bars_per_year=252)
    finally:
        pp_mod.simulate_portfolio = orig
    assert len(captured) == 2, f"슬리브 2개 기대, {len(captured)}개 포착됨"
    sub_ids = [s["config"] for s in cfg["sub_strategies"]]
    lp_sim, kr_sim = captured[sub_ids.index("yoon1j_lp")], captured[sub_ids.index("yoon1j_kr")]
    return cfg, art, lp_sim, kr_sim


def local_weight_frame(sim) -> pd.DataFrame:
    tw = sim["target_weights"].copy()
    stw = sim.get("safe_target_weights")
    if stw is not None and len(stw.columns):
        stw = stw.rename(columns={c: str(c).removesuffix("_SAFE") for c in stw.columns})
        tw = tw.join(stw, how="left").fillna(0.0)
    return tw


def load_prices(symbols: list[str]) -> pd.DataFrame:
    cols = {}
    for s in symbols:
        ser = load_yfinance(s, "1d", "max")["close"]
        idx = pd.DatetimeIndex(pd.to_datetime(ser.index))
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        cols[s] = pd.Series(ser.to_numpy(dtype=float), index=idx)
    return pd.DataFrame(cols)


def main() -> None:
    print("공식 파이프라인(통화틸트+신용게이트+하위 슬리브) 실행 중...", flush=True)
    cfg, art, lp_sim, kr_sim = build_official()

    f = art.forecast
    window = f.index
    top_w = pd.DataFrame({
        "yoon1j_lp": f.get("target_w_yoon1j_lp", 0.0),
        "yoon1j_kr": f.get("target_w_yoon1j_kr", 0.0),
        "__CASH__": f.get("target_w___CASH__", 0.0),
    }).reindex(window).fillna(0.0)
    official_equity = f["close"].reindex(window)

    lp_local = local_weight_frame(lp_sim).reindex(window).ffill().fillna(0.0)
    kr_local = local_weight_frame(kr_sim).reindex(window).ffill().fillna(0.0)

    us_syms = list(lp_local.columns)
    kr_syms = list(kr_local.columns)
    gate = ((cfg.get("currency_allocation") or {}).get("risk_gate")) or {}
    ief_sym = gate.get("safe_asset")  # "IEF" — __CASH__ 배정분이 실제로 담기는 자산

    print("전체 종목·FX 가격 로드 중...", flush=True)
    us_px = load_prices(sorted(set(us_syms + ([ief_sym] if ief_sym else []))))
    kr_px = load_prices(kr_syms)
    fx = load_yfinance("KRW=X", "1d", "max")["close"]
    fx.index = pd.DatetimeIndex(pd.to_datetime(fx.index)).tz_localize(None) if fx.index.tz else pd.DatetimeIndex(fx.index)
    fx = fx.reindex(window).ffill().bfill()

    us_px_w = us_px.reindex(window).ffill()
    kr_px_w = kr_px.reindex(window).ffill()

    # 전체 포트폴리오 기준 종목별 목표비중(원화, 전체 자본=1.0 기준) 복원.
    combined_w = pd.DataFrame(0.0, index=window, columns=us_syms + kr_syms + ([ief_sym] if ief_sym else []))
    for s in us_syms:
        combined_w[s] = top_w["yoon1j_lp"] * lp_local[s]
    for s in kr_syms:
        combined_w[s] = top_w["yoon1j_kr"] * kr_local[s]
    if ief_sym:
        combined_w[ief_sym] = combined_w.get(ief_sym, 0.0) + top_w["__CASH__"]

    # 원화가 KRW 가격 패널(US는 FX 환산).
    px_krw = pd.DataFrame(index=window)
    for s in us_syms + ([ief_sym] if ief_sym and ief_sym not in kr_syms else []):
        if s in us_px_w.columns:
            px_krw[s] = us_px_w[s] * fx
    for s in kr_syms:
        px_krw[s] = kr_px_w[s]

    us_cost = float((json.loads((ROOT / "configs/strategies/yoon1j_lp.json").read_text())["costs"]["fee_bps_per_side"]
                     + json.loads((ROOT / "configs/strategies/yoon1j_lp.json").read_text())["costs"]["slippage_bps"])) / 10_000.0
    kr_cost = float((json.loads((ROOT / "configs/strategies/yoon1j_kr.json").read_text())["costs"]["fee_bps_per_side"]
                     + json.loads((ROOT / "configs/strategies/yoon1j_kr.json").read_text())["costs"]["slippage_bps"])) / 10_000.0
    cost_bps = {s: us_cost for s in us_syms}
    cost_bps.update({s: kr_cost for s in kr_syms})
    if ief_sym:
        cost_bps[ief_sym] = us_cost  # IEF=미국 상장

    # 리밸런스일: 상위 결합 주기(final0706="monthly")를 그대로 따른다. 통화틸트
    # 신호 자체는 매일 갱신되지만(63일 모멘텀), 실제 집행은 월 1회 체크가 현실적
    # (라이브 오케스트레이터도 월간 페이퍼 목표 기준) — 매일 정수주 재조정하면
    # 신호 잡음만큼 회전율이 폭증해 소액계좌 문제와 무관한 비용 왜곡이 생긴다.
    rebal_set = rebalance_dates(window, str(cfg.get("rebalance_freq", "monthly")))
    rebal_dates = pd.DatetimeIndex(sorted(rebal_set)).intersection(window)
    if len(rebal_dates) == 0 or rebal_dates[0] != window[0]:
        rebal_dates = pd.DatetimeIndex([window[0]]).append(rebal_dates)
    print(f"리밸런스 이벤트 {len(rebal_dates)}회 (구간 {window[0].date()}~{window[-1].date()})")

    holdings: dict[str, float] = {}
    cash = SEED_KRW
    nav_hist = pd.Series(index=window, dtype=float)
    total_cost_paid = 0.0
    n_orders = 0
    fit_at_rebal = []

    for i, dt in enumerate(window):
        prices_today = {s: float(px_krw.at[dt, s]) if s in px_krw.columns and pd.notna(px_krw.at[dt, s]) else 0.0
                        for s in combined_w.columns}
        if dt in rebal_dates:
            targets = {s: max(0.0, float(combined_w.at[dt, s])) for s in combined_w.columns}
            plan = integer_execution_plan(targets, holdings, prices_today, cash, min_trade_value=1000.0)
            for o in plan["orders"]:
                notional = o["qty"] * o["price"]
                fee = notional * cost_bps.get(o["symbol"], us_cost)
                if o["side"] == "BUY":
                    cash -= (notional + fee)
                    holdings[o["symbol"]] = holdings.get(o["symbol"], 0.0) + o["qty"]
                else:
                    cash += (notional - fee)
                    holdings[o["symbol"]] = holdings.get(o["symbol"], 0.0) - o["qty"]
                total_cost_paid += fee
                n_orders += 1
            fit_at_rebal.append({"date": str(dt.date()), "target_fit": plan["target_fit"],
                                 "tracking_error_abs": plan["tracking_error_abs"]})
        nav_hist.loc[dt] = cash + sum(holdings.get(s, 0.0) * prices_today.get(s, 0.0) for s in holdings)

    integer_equity = (nav_hist / nav_hist.iloc[0]).rename("equity")
    integer_ret = integer_equity.pct_change().fillna(0.0)
    official_norm = (official_equity / official_equity.iloc[0]).rename("official")
    official_ret = official_norm.pct_change().fillna(0.0)

    perf_int = performance(integer_equity, integer_ret, "1d")
    perf_off = performance(official_norm, official_ret, "1d")
    corr = float(integer_ret.corr(official_ret))
    fit_df = pd.DataFrame(fit_at_rebal)

    print("\n=== 성과 비교 (2011-01 이후 권위 구간과 동일 전체기간, 시드 ₩2,500,000) ===")
    for label, p in (("정수주(₩250만)", perf_int), ("연속비중(공식)", perf_off)):
        print(f"{label:16} CAGR {p['cagr']*100:6.2f}%  Sharpe {p['sharpe']:.3f}  "
              f"MDD {p['max_drawdown']*100:6.2f}%  총수익 {p['total_return']*100:7.1f}%")
    print(f"\n일간수익률 상관계수: {corr:.4f}")
    print(f"리밸런스 재현율(target_fit) 평균 {fit_df['target_fit'].mean()*100:.1f}% "
          f"(최저 {fit_df['target_fit'].min()*100:.1f}% @ {fit_df.loc[fit_df['target_fit'].idxmin(),'date']})")
    print(f"누적 거래비용: ₩{total_cost_paid:,.0f} ({total_cost_paid/SEED_KRW*100:.2f}% of 시드) / 주문 {n_orders}건 / 리밸런스 {len(rebal_dates)}회")
    print(f"최종 NAV: ₩{nav_hist.iloc[-1]:,.0f} (연속비중 동일시드 환산 ₩{SEED_KRW*float(official_norm.iloc[-1]):,.0f})")

    outdir = ROOT / "reports" / "profile_sizing"
    payload = {
        "seed_krw": SEED_KRW,
        "window": [str(window[0].date()), str(window[-1].date())],
        "n_rebalances": int(len(rebal_dates)),
        "n_orders": n_orders,
        "total_cost_paid_krw": round(total_cost_paid, 0),
        "total_cost_pct_of_seed": round(total_cost_paid / SEED_KRW, 4),
        "corr_daily_returns": corr,
        "target_fit_mean": float(fit_df["target_fit"].mean()),
        "target_fit_min": float(fit_df["target_fit"].min()),
        "integer_lot": {k: float(v) if v is not None else None for k, v in perf_int.items()},
        "continuous_official": {k: float(v) if v is not None else None for k, v in perf_off.items()},
        "final_nav_krw": float(nav_hist.iloc[-1]),
        "final_nav_if_continuous_krw": float(SEED_KRW * official_norm.iloc[-1]),
        "fit_by_rebalance": fit_at_rebal,
    }
    (outdir / "final0706_small_capital_backtest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"\nwrote {outdir / 'final0706_small_capital_backtest.json'}")


if __name__ == "__main__":
    main()
