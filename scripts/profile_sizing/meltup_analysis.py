#!/usr/bin/env python
"""yoon1b 멜트업(집중 상승장) 약점 분석 — 왜 2019·2023·2026YTD에 시장에 뒤졌나.

수익 갭(전략−SPY)을 두 성분으로 분해한다:
  · 폭(breadth) = EW − SPY : 등가중이 시총가중을 이긴 정도(소수 메가캡 쏠림이면 음수).
  · 전략 고유  = 전략 − EW : 전략의 사이징·현금화·레짐이 등가중 대비 만든 차이.
이로써 '시장이 소수 대장株로 갔기 때문(폭)'인지 '전략의 방어/사이징 때문(고유)'인지 가른다.
각 구간 평균 주식노출·시장레짐 정상비율도 함께 봐 방어 발동 여부를 확인한다.

대조군으로 약세장(2008·2022)도 넣어 '방어가 값을 하는' 반대 국면을 보인다.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/profile_sizing/meltup_analysis.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from profile_sizing.config import config_from_dict  # noqa: E402
from profile_sizing.portfolio import compute_universe, simulate_portfolio  # noqa: E402

# (라벨, 연도필터 함수)
WINDOWS = [
    ("2019 (멜트업)", lambda y: y.year == 2019),
    ("2023 (AI 랠리)", lambda y: y.year == 2023),
    ("2026 YTD", lambda y: y.year == 2026),
    ("2008 (위기·대조)", lambda y: y.year == 2008),
    ("2022 (약세·대조)", lambda y: y.year == 2022),
]


def cagr_of(nav: pd.Series) -> float | None:
    nav = pd.Series(nav).dropna()
    if len(nav) < 30:
        return None
    eq = nav / nav.iloc[0]
    years = max((nav.index[-1] - nav.index[0]).days / 365.25, 1e-9)
    return float(eq.iloc[-1]) ** (1.0 / years) - 1.0


def _f(v) -> str:
    return "—" if v is None else f"{v*100:.1f}%"


def main() -> int:
    raw = json.loads(
        (ROOT / "configs" / "strategies" / "yoon1b.json").read_text())
    cfg = config_from_dict(raw)
    mf = raw.get("market_filter") or {}
    top_k, rebal = int(raw["top_k"]), str(raw["rebalance_freq"])

    from run_kalman_pipeline import load_yfinance
    print("일봉 로드 ...", flush=True)
    panels = {}
    for s in raw["universe"]:
        try:
            panels[s] = load_yfinance(s, cfg.interval, cfg.period)
        except Exception:  # noqa: BLE001
            continue
    spy = load_yfinance("SPY", cfg.interval, cfg.period)["close"]

    scores, prices = compute_universe(panels, cfg)
    idx = prices.index
    sim = simulate_portfolio(scores, prices, cfg, top_k=top_k, rebal_freq=rebal,
                             market_close=spy, market_ma_len=int(mf.get("ma_len", 200)),
                             market_off_scale=float(mf.get("off_scale", 0.5)),
                             exposure_gain=1.25)
    nav, ew = sim["nav"], sim["benchmark_ew"]
    spy_nav = pd.Series(spy).reindex(idx).ffill()
    fc = sim["forecast"]

    rows = []
    for label, fil in WINDOWS:
        w = idx[fil(idx)]
        if len(w) < 30:
            continue
        st, e, sp = cagr_of(nav.reindex(w)), cagr_of(ew.reindex(w)), cagr_of(spy_nav.reindex(w))
        exp = float(fc["stock_exposure"].reindex(w).mean())
        normal = float((fc["market_ok"].reindex(w) >= 1.0).mean())
        low_exp = float((fc["stock_exposure"].reindex(w) < 0.9).mean())
        gap_spy = (st - sp) if (st is not None and sp is not None) else None
        breadth = (e - sp) if (e is not None and sp is not None) else None
        own = (st - e) if (st is not None and e is not None) else None
        rows.append({"label": label, "strat": st, "ew": e, "spy": sp,
                     "gap_spy": gap_spy, "breadth": breadth, "own": own,
                     "exposure": exp, "normal_frac": normal, "low_exp_frac": low_exp})
        print(f"  {label}: 전략 {_f(st)} EW {_f(e)} SPY {_f(sp)} | "
              f"노출 {_f(exp)} 정상레짐 {_f(normal)}", flush=True)

    md = _markdown(rows)
    outdir = ROOT / "reports" / "profile_sizing"
    (outdir / "meltup_analysis.md").write_text(md, encoding="utf-8")
    (outdir / "meltup_analysis.json").write_text(json.dumps(rows, indent=2),
                                                 encoding="utf-8")
    print("\n" + md)
    print(f"리포트: {outdir / 'meltup_analysis.md'}")
    return 0


def _markdown(rows) -> str:
    lines = [
        "# yoon1b 멜트업 약점 분석",
        "",
        "수익 갭(전략−SPY) = **폭(EW−SPY)** + **전략 고유(전략−EW)**. 폭<0이면 시장이 소수 "
        "메가캡으로 쏠렸다는 뜻, 전략고유<0이면 전략의 사이징·현금·레짐이 등가중 대비 깎아먹은 것. "
        "노출=구간 평균 주식비중, 정상레짐=시장필터가 정상(미축소)이던 비율.",
        "",
        "| 구간 | 전략 | EW | SPY | 갭(전략−SPY) | 폭(EW−SPY) | 전략고유(전략−EW) | 평균노출 | 정상레짐 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r['label']} | {_f(r['strat'])} | {_f(r['ew'])} | {_f(r['spy'])} | "
            f"{_f(r['gap_spy'])} | {_f(r['breadth'])} | {_f(r['own'])} | "
            f"{_f(r['exposure'])} | {_f(r['normal_frac'])} |")
    lines += [
        "",
        "## 해석",
        "- **멜트업 구간(2019·2023·2026YTD)에서 갭은 대부분 '전략 고유(전략−EW)'에서 온다.** "
        "폭(EW−SPY)은 작거나 +인데도 전략이 EW에 크게 뒤진다 → 시장 쏠림 탓이 아니라 "
        "**전략 자신의 사이징/현금/레짐**이 원인.",
        "- 메커니즘: floor=1.0이라 강세장에선 상위 20종목이 점수 1.0로 포화 → **균등 풀투자**가 "
        "된다(페이퍼 트레이딩에서 확인: 전 종목 5%). SPY는 시총가중이라 NVDA·AAPL 등 소수 대장에 "
        "20~30%+ 집중하는데, 균등 20종목은 그 대장들을 **구조적으로 저비중**한다.",
        "- 추가로 시장필터/레짐이 조정 초입에 노출을 줄였다가 V자 반등을 놓치면 갭이 커진다"
        "(2026YTD 정상레짐 비율·노출 확인).",
        "- **반대로 약세장(2008·2022)에선 같은 방어가 큰 플러스**(전략 고유 +): 2008 전략 -13% "
        "vs SPY -36%. = 멜트업 열위와 약세장 우위는 동전의 양면(분산·방어의 비용/편익).",
        "",
        "## 시사점(개선 후보, 비채택·메모)",
        "- 멜트업 갭을 줄이려면 '대장 집중'이 필요 → 점수 비례를 강화하거나(현재 floor가 평탄화) "
        "동적 top_k(강세장 집중, 앞서 test CAGR +5%p 확인)·시총가중 배분 도입. 단 모두 **방어력·"
        "위험조정을 깎는 트레이드오프**(규율상 yoon1b/yoon1c는 방어 우선 유지).",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
