"""yoon2m GFC 커버리지 대리 검증 — 원본 SPDR 유니버스(1998~)로 top_k=3 방어 스택 시험.

yoon2m의 저가 ETF 유니버스는 ~2013 상장 + KR 데이터 갭(2007-02~2009-04)으로
닷컴(2000~02)·GFC(2008~09)가 미검증이다. 구조 대리: 같은 방어 스택(레짐 cap +
SPY MA100 시장필터 + TLT/GLD 안전 슬리브)을 가진 yoon1j 원본 유니버스(SPDR 섹터,
1998-12 상장)에 yoon2m의 top_k=3을 적용해 두 폭락을 통과시킨다.

한계(사전 명시): US 슬리브 단독(KR ETF는 2006+ 상장이라 GFC 커버 불가), 안전자산
TLT는 2002-07·GLD는 2004-11 상장이라 닷컴 구간은 안전 슬리브 부분 가용. 따라서
이 검증은 "yoon2m 수치의 재현"이 아니라 "방어 스택이 -50%급 약세장에서 작동하는가"
의 구조 검증이다.

사용:  PYTHONPATH=src .venv/bin/python scripts/profile_sizing/yoon2m_gfc_proxy.py
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

from profile_sizing.engine import performance  # noqa: E402
from run_kalman_pipeline import load_yfinance  # noqa: E402
from trading_lab.strategies.profile_portfolio import ProfilePortfolioHandler  # noqa: E402

EPISODES = [
    ("2000-03-24", "2002-10-09", "닷컴 붕괴"),
    ("2007-10-09", "2009-03-09", "GFC"),
    ("2011-04-29", "2011-10-03", "2011 급락"),
]


def run_sleeve(top_k: int) -> tuple[pd.Series, pd.DataFrame]:
    cfg = dict(json.loads((ROOT / "configs" / "strategies" / "yoon1j.json").read_text("utf-8")))
    cfg["top_k"] = top_k
    h = ProfilePortfolioHandler()
    raw = h.load_data("SYNTH", cfg, synthetic=False)
    art = h.build_artifacts(raw, cfg, symbol="SYNTH", phase="all", bars_per_year=252)
    return art.equity, art.forecast[["stock_exposure", "safe_exposure"]]


def episode_row(eq: pd.Series, expo: pd.DataFrame, start: str, end: str) -> dict:
    w = eq[(eq.index >= start) & (eq.index <= end)]
    if w.empty:
        return {}
    e = expo.reindex(w.index)
    return {
        "dd": float((w / w.cummax() - 1.0).min()),
        "stock_min": float(e["stock_exposure"].min()),
        "safe_max": float(e["safe_exposure"].max()),
    }


def main() -> None:
    spy = load_yfinance("SPY", "1d", "max")["close"]
    sidx = pd.DatetimeIndex(pd.to_datetime(spy.index))
    if sidx.tz is not None:
        sidx = sidx.tz_localize(None)
    spy = pd.Series(np.asarray(spy, dtype=float), index=sidx)

    results = {}
    for k in (3, 11):
        eq, expo = run_sleeve(k)
        results[k] = (eq, expo)

    print("=== GFC 대리 검증 (yoon1j 원본 SPDR 유니버스, US 슬리브 단독) ===\n")
    rows = []
    for start, end, label in EPISODES:
        s = spy[(spy.index >= start) & (spy.index <= end)]
        spy_dd = float((s / s.cummax() - 1.0).min())
        row = {"episode": label, "spy_dd": spy_dd}
        for k, (eq, expo) in results.items():
            r = episode_row(eq, expo, start, end)
            row[f"dd_top{k}"] = r.get("dd")
            row[f"stock_min_top{k}"] = r.get("stock_min")
            row[f"safe_max_top{k}"] = r.get("safe_max")
        rows.append(row)
    df = pd.DataFrame(rows).set_index("episode")
    pct = lambda x: ("    n/a" if pd.isna(x) else f"{x*100:7.1f}%")  # noqa: E731
    print(df.to_string(formatters={c: pct for c in df.columns}))

    print("\n=== 전 구간(1999~) 성과 ===")
    perf_rows = {}
    for k, (eq, _) in results.items():
        e = eq / eq.iloc[0]
        p = performance(e, e.pct_change().fillna(0.0), "1d")
        perf_rows[f"top_k={k}"] = p
        print(f"  top_k={k:2d}  {eq.index[0].date()}~  CAGR {p['cagr']*100:6.2f}%  "
              f"MDD {p['max_drawdown']*100:7.2f}%  Sharpe {p['sharpe']:.3f}")
    sw = spy[spy.index >= results[3][0].index[0]]
    e = sw / sw.iloc[0]
    p = performance(e, e.pct_change().fillna(0.0), "1d")
    perf_rows["SPY"] = p
    print(f"  SPY       동일기간   CAGR {p['cagr']*100:6.2f}%  "
          f"MDD {p['max_drawdown']*100:7.2f}%  Sharpe {p['sharpe']:.3f}")

    out = ROOT / "reports" / "profile_sizing" / "yoon2m_gfc_proxy.json"
    out.write_text(json.dumps({
        "episodes": rows, "full_period": perf_rows,
    }, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"\nJSON: {out}")


if __name__ == "__main__":
    main()
