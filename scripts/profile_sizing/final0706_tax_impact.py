"""final0706 양도소득세 반영 세후 비교 (개선계획 B2).

미반영이던 세금 비대칭을 정량화한다:
  - **US 슬리브(미국 상장 ETF)**: 양도차익 22%(지방세 포함), 연 250만원 기본공제,
    연내 손익통산. 월간 top3 회전이라 이익이 계속 실현된다.
  - **KR 슬리브(국내 주식형 ETF)**: 매매차익 비과세.
  - **B&H 벤치(SPY/QQQ 원화환산)**: 미국 상장이므로 과세 대상이지만 보유 중엔
    이연되고 최종 청산 시 1회 과세(기본공제 1년치만 적용 — 보수적).

근사 가정(사전 명시):
  1. US 슬리브의 연간 실현이익 = 그 해 US 슬리브 기여 손익 × 실현비율 1.0
     (**전략에 최대한 불리한 상한** — top3 월간 회전+통화틸트 이동이라 연내 대부분
     실현된다고 가정). 실제로는 hold_buffer로 일부 이월되므로 이보다 완화된다.
  2. US 슬리브 일간 기여 = 계좌평가액 × 목표 USD 비중(전일) × US 슬리브 KRW 수익률.
     (월간 리밸런스 사이의 드리프트는 무시 — forecast의 target_w를 그대로 사용)
  3. 과세는 해당 연도 말에 즉시 차감(실제는 이듬해 5월 납부 — 복리 손해를 약간
     과대평가하는 보수적 방향).
  4. 환차익은 원화 환산 손익에 포함(실제 과세도 원화 기준 — 방향 일치).

시나리오: ① 거치식 1억원, ② 적립식(초기 200만 + 월 20만, dca_lumpsum_eval 관행).
구간: 권위 구간 2011-01~(final0706_data_gap_diagnosis.md).

사용:  PYTHONPATH=src:scripts .venv/bin/python scripts/profile_sizing/final0706_tax_impact.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from profile_sizing.metrics import equity_metrics  # noqa: E402
from run_kalman_pipeline import load_yfinance  # noqa: E402
from trading_lab.strategies.yoon1k import Yoon1kHandler  # noqa: E402

TAX_RATE = 0.22
DEDUCTION = 2_500_000.0
START = pd.Timestamp("2011-01-01")
LUMP_SEED = 100_000_000.0
DCA_SEED = 2_000_000.0
DCA_MONTHLY = 200_000.0


def load_strategy() -> tuple[pd.Series, pd.Series, pd.Series]:
    """(전체 equity, US 슬리브 KRW NAV, 목표 USD 비중) — 권위 구간으로 절단."""
    cfg = json.loads((ROOT / "configs" / "strategies" / "final0706.json").read_text("utf-8"))
    h = Yoon1kHandler()
    raw = h.load_data("PORTFOLIO", cfg, synthetic=False)
    art = h.build_artifacts(raw, cfg, symbol="PORTFOLIO", phase="all", bars_per_year=252)
    f = art.forecast
    eq = art.equity[art.equity.index >= START]
    us_nav = f["sleeve_yoon1j_lp"].reindex(eq.index).ffill()
    us_w = f["target_w_yoon1j_lp"].reindex(eq.index).ffill().fillna(0.5)
    return eq / eq.iloc[0], us_nav / us_nav.iloc[0], us_w


def simulate_strategy_after_tax(
    eq: pd.Series, us_nav: pd.Series, us_w: pd.Series,
    *, seed: float, monthly: float = 0.0,
) -> tuple[pd.Series, float]:
    """연말 과세(US 기여분 실현이익 상한 가정) 반영 계좌 평가액. 반환: (평가액, 총세액)."""
    r_total = eq.pct_change().fillna(0.0)
    r_us = us_nav.pct_change().fillna(0.0)
    w_lag = us_w.shift(1).fillna(us_w.iloc[0])
    value = seed
    values, total_tax = [], 0.0
    year_us_gain = 0.0
    idx = eq.index
    month_seen = {(idx[0].year, idx[0].month)}
    prev_year = idx[0].year
    for i, t in enumerate(idx):
        if i > 0:
            us_pnl = value * float(w_lag.iloc[i]) * float(r_us.iloc[i])
            other_pnl = value * (float(r_total.iloc[i]) - float(w_lag.iloc[i]) * float(r_us.iloc[i]))
            value += us_pnl + other_pnl
            year_us_gain += us_pnl
        if monthly > 0.0 and (t.year, t.month) not in month_seen:
            month_seen.add((t.year, t.month))
            value += monthly
        if t.year != prev_year or i == len(idx) - 1:
            tax = TAX_RATE * max(0.0, year_us_gain - DEDUCTION)
            value -= tax
            total_tax += tax
            year_us_gain = 0.0
            prev_year = t.year
        values.append(value)
    return pd.Series(values, index=idx), total_tax


def simulate_bnh_after_tax(
    nav: pd.Series, *, seed: float, monthly: float = 0.0, taxable: bool,
) -> tuple[pd.Series, float]:
    """B&H: 보유 중 무과세(이연), 마지막 날 일괄 청산 과세(기본공제 1년치)."""
    r = nav.pct_change().fillna(0.0)
    idx = nav.index
    value, contributed = seed, seed
    values = []
    month_seen = {(idx[0].year, idx[0].month)}
    for i, t in enumerate(idx):
        if i > 0:
            value *= 1.0 + float(r.iloc[i])
        if monthly > 0.0 and (t.year, t.month) not in month_seen:
            month_seen.add((t.year, t.month))
            value += monthly
            contributed += monthly
        values.append(value)
    tax = TAX_RATE * max(0.0, (value - contributed) - DEDUCTION) if taxable else 0.0
    values[-1] = value - tax
    return pd.Series(values, index=idx), tax


def main() -> None:
    eq, us_nav, us_w = load_strategy()
    idx = eq.index
    print(f"구간: {idx[0].date()}~{idx[-1].date()} ({len(idx)}봉)\n")

    fx = load_yfinance("KRW=X", "1d", "max")["close"]
    fx.index = pd.DatetimeIndex(pd.to_datetime(fx.index)).tz_localize(None)
    bench: dict[str, tuple[pd.Series, bool]] = {}
    for sym in ("SPY", "QQQ"):
        px = load_yfinance(sym, "1d", "max")["close"]
        px.index = pd.DatetimeIndex(pd.to_datetime(px.index)).tz_localize(None)
        krw = (px * fx.reindex(px.index).ffill()).reindex(idx).ffill()
        bench[f"{sym}_KRW_bnh"] = (krw / krw.iloc[0], True)

    results: dict[str, dict] = {}
    for scen, (seed, monthly) in {
        "거치식 1억": (LUMP_SEED, 0.0),
        "적립식 200만+월20만": (DCA_SEED, DCA_MONTHLY),
    }.items():
        rows = {}
        pre, _ = simulate_strategy_after_tax(eq, us_nav, us_w * 0.0, seed=seed, monthly=monthly)
        post, tax = simulate_strategy_after_tax(eq, us_nav, us_w, seed=seed, monthly=monthly)
        rows["final0706 세전"] = {"value": float(pre.iloc[-1]), "tax": 0.0}
        rows["final0706 세후"] = {"value": float(post.iloc[-1]), "tax": tax}
        for name, (nav, taxable) in bench.items():
            b_pre, _ = simulate_bnh_after_tax(nav, seed=seed, monthly=monthly, taxable=False)
            b_post, b_tax = simulate_bnh_after_tax(nav, seed=seed, monthly=monthly, taxable=taxable)
            rows[f"{name} 세전"] = {"value": float(b_pre.iloc[-1]), "tax": 0.0}
            rows[f"{name} 세후(청산시 일괄)"] = {"value": float(b_post.iloc[-1]), "tax": b_tax}
        # 세후 CAGR(거치식만 의미) 및 드래그
        if monthly == 0.0:
            for k, v in rows.items():
                nav_end = v["value"] / seed
                yrs = (idx[-1] - idx[0]).days / 365.25
                v["cagr"] = nav_end ** (1.0 / yrs) - 1.0
        results[scen] = rows
        print(f"=== {scen} ===")
        for k, v in rows.items():
            cagr = f"  CAGR {v['cagr']*100:6.2f}%" if "cagr" in v else ""
            print(f"  {k:28s} 최종 {v['value']:>16,.0f}원  세액 {v['tax']:>13,.0f}원{cagr}")
        print()

    out = ROOT / "reports" / "profile_sizing" / "final0706_tax_impact.json"
    out.write_text(json.dumps({
        "assumptions": {"tax_rate": TAX_RATE, "deduction": DEDUCTION,
                        "realized_fraction": 1.0, "start": str(START.date())},
        "results": results,
    }, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"JSON: {out}")


if __name__ == "__main__":
    main()
