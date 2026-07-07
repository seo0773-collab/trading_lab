"""final0706 레짐 의존 분해 (개선계획 B3).

질문 두 가지:
  1. 2025~26 폭발적 성과(+52.9%/+41.4%)는 어디서 왔나 — US 슬리브(VGT 테크 랠리)?
     KR 슬리브? 통화틸트/게이트?
  2. 2011~2016 저성과(연 0.2~5.6%)는 왜 — 어느 컴포넌트가 드래그였나?

방법: 공식 파이프라인 forecast의 슬리브 NAV(원화 정규화)와 목표비중으로 일간
수익 기여를 분해한다: r_p ≈ w_us·r_us + w_kr·r_kr + w_cash·r_safe. 연도별로
합산(단리 근사 — 컴포넌트 간 비교 목적이라 충분). US 슬리브 내부(VGT vs 나머지)는
보유 내역이 상위 아티팩트에 없어 시장 맥락(VGT/KODEX200 원화 연수익)과 US 슬리브
수익의 대비로 정성 판단한다(한계 사전 명시).

사용:  PYTHONPATH=src:scripts .venv/bin/python scripts/profile_sizing/final0706_regime_attribution.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from run_kalman_pipeline import load_yfinance  # noqa: E402
from trading_lab.strategies.yoon1k import Yoon1kHandler  # noqa: E402

START = pd.Timestamp("2011-01-01")
PERIODS = {
    "2011~2016 (저성과)": ("2011-01-01", "2016-12-31"),
    "2017~2024": ("2017-01-01", "2024-12-31"),
    "2025~26H1 (랠리)": ("2025-01-01", "2026-12-31"),
}


def krw_nav(sym: str, fx: pd.Series | None, idx: pd.DatetimeIndex) -> pd.Series:
    px = load_yfinance(sym, "1d", "max")["close"]
    px.index = pd.DatetimeIndex(pd.to_datetime(px.index)).tz_localize(None)
    s = (px * fx.reindex(px.index).ffill()) if fx is not None else px
    s = s.reindex(idx).ffill()
    return s / s.dropna().iloc[0]


def main() -> None:
    cfg = json.loads((ROOT / "configs" / "strategies" / "final0706.json").read_text("utf-8"))
    h = Yoon1kHandler()
    raw = h.load_data("PORTFOLIO", cfg, synthetic=False)
    art = h.build_artifacts(raw, cfg, symbol="PORTFOLIO", phase="all", bars_per_year=252)
    f = art.forecast
    eq = art.equity[art.equity.index >= START]
    idx = eq.index

    us = f["sleeve_yoon1j_lp"].reindex(idx).ffill()
    kr = f["sleeve_yoon1j_kr"].reindex(idx).ffill()
    w_us = f["target_w_yoon1j_lp"].reindex(idx).ffill().fillna(0.5).shift(1)
    w_kr = f["target_w_yoon1j_kr"].reindex(idx).ffill().fillna(0.5).shift(1)
    w_cash = f.get("target_w___CASH__")
    w_cash = (w_cash.reindex(idx).ffill().fillna(0.0).shift(1)
              if w_cash is not None else pd.Series(0.0, index=idx))

    fx = load_yfinance("KRW=X", "1d", "max")["close"]
    fx.index = pd.DatetimeIndex(pd.to_datetime(fx.index)).tz_localize(None)
    ief_krw = krw_nav("IEF", fx, idx)

    r_p = eq.pct_change().fillna(0.0)
    contrib = pd.DataFrame({
        "US 슬리브": w_us * us.pct_change(),
        "KR 슬리브": w_kr * kr.pct_change(),
        "게이트(IEF 대피)": w_cash * ief_krw.pct_change(),
        "전략 전체": r_p,
    }, index=idx).fillna(0.0)
    contrib["잔차(리밸런스·상호작용)"] = (
        contrib["전략 전체"] - contrib[["US 슬리브", "KR 슬리브", "게이트(IEF 대피)"]].sum(axis=1))

    # 시장 맥락(원화 기준 연수익)
    ctx = pd.DataFrame({
        "VGT_KRW": krw_nav("VGT", fx, idx), "SPY_KRW": krw_nav("SPY", fx, idx),
        "KODEX200": krw_nav("069500.KS", None, idx),
    }, index=idx)

    yearly = contrib.groupby(contrib.index.year).sum()
    ctx_yearly = ctx.resample("YE").last().pct_change(fill_method=None)
    ctx_yearly.index = ctx_yearly.index.year

    print("=== 연도별 수익 기여(단리 합산, %p) 및 시장 맥락(%) ===")
    tbl = (yearly * 100).round(1)
    tbl[["VGT_KRW", "SPY_KRW", "KODEX200"]] = (ctx_yearly * 100).round(1)
    print(tbl.to_string())

    print("\n=== 구간 합산 기여(%p) ===")
    period_rows = {}
    for label, (a, b) in PERIODS.items():
        w = contrib[(contrib.index >= a) & (contrib.index <= b)]
        gate_days = f.get("target_w___CASH__")
        gd = (gate_days.reindex(w.index).fillna(0.0) > 0).mean() if gate_days is not None else 0.0
        row = (w.sum() * 100).round(1).to_dict()
        row["게이트 발동일 비중(%)"] = round(float(gd) * 100, 1)
        period_rows[label] = row
        print(f"  {label:22s} " + "  ".join(f"{k} {v:+.1f}" for k, v in row.items()))

    out = ROOT / "reports" / "profile_sizing" / "final0706_regime_attribution.json"
    out.write_text(json.dumps({
        "yearly_contrib_pct": {str(k): v for k, v in (yearly * 100).round(2).T.to_dict().items()},
        "context_yearly_pct": {str(k): v for k, v in (ctx_yearly * 100).round(2).T.to_dict().items()},
        "period_contrib": period_rows,
    }, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"\nJSON: {out}")


if __name__ == "__main__":
    main()
