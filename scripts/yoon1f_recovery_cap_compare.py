"""yoon1f RECOVERY cap 램프 개선 — 회복 노출 복원 속도 보강.

진단: 시장필터(이제 100MA)를 빠르게 한 뒤 남은 회복 지연 축은 RECOVERY cap 램프.
현재 recovery_stage_bars=21·caps=[.5/.6,.7/.8,1.0] → 회복 진입 후 풀노출까지 ~2개월.
회복 확인 종목의 cap을 더 빨리/높게 풀면 바닥 반등을 더 탈 수 있다.

⚠️ 메모리: yoon1b(메가캡)에선 RECOVERY 가속이 열위였음 → yoon1f(섹터)서 재검증.
yoon1f(100MA·무레버)로 램프 순효과 격리. val 선정 → test 견고성.

실행:
    PYTHONPATH=src:scripts .venv/bin/python scripts/yoon1f_recovery_cap_compare.py
"""
from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd

from trading_lab.paths import ROOT
from profile_sizing.config import config_from_dict
from profile_sizing.portfolio import compute_universe, simulate_portfolio
from run_kalman_pipeline import load_yfinance

CORE = ["XLK", "XLF", "XLV", "XLE", "XLI", "XLP", "XLY", "XLU",
        "XLB", "XLRE", "TLT", "GLD"]
START = "2007-02-01"


def perf(nav: pd.Series):
    nav = nav.dropna(); eq = nav / nav.iloc[0]; r = eq.pct_change().dropna()
    yrs = max((nav.index[-1] - nav.index[0]).days / 365.25, 1e-9)
    sh = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0.0
    return eq.iloc[-1] ** (1 / yrs) - 1, sh, (eq / eq.cummax() - 1).min()


def main() -> None:
    base = json.loads((ROOT / "configs/strategies/yoon1f.json").read_text("utf-8"))
    base["universe"] = CORE
    panels = {s: load_yfinance(s, "1d", "max") for s in CORE}
    panels = {s: d for s, d in panels.items() if d is not None and not d.empty}
    spy = load_yfinance("SPY", "1d", "max")["close"]; spy.index = pd.to_datetime(spy.index)
    mkw = dict(top_k=12, rebal_freq="monthly", market_close=spy,
               market_off_scale=0.5, exposure_gain=1.25,
               market_mode="binary", market_ma_len=100)

    # (recovery_stage_bars, recovery_stage_caps)
    V = {
        "기존 21/[.6,.8,1]":   (21, [0.6, 0.8, 1.0]),
        "가속 10/[.6,.8,1]":   (10, [0.6, 0.8, 1.0]),
        "시작상향 21/[.8,.9,1]": (21, [0.8, 0.9, 1.0]),
        "빠른상향 10/[.8,1,1]":  (10, [0.8, 1.0, 1.0]),
        "즉시 [1,1,1]":         (21, [1.0, 1.0, 1.0]),
    }
    navs = {}
    for name, (bars, caps) in V.items():
        b = copy.deepcopy(base)
        b["regime_cap"]["recovery_stage_bars"] = bars
        b["regime_cap"]["recovery_stage_caps"] = caps
        cfg = config_from_dict(b)
        scores, prices = compute_universe(panels, cfg)
        sim = simulate_portfolio(scores, prices, cfg, **mkw)
        nav = pd.Series(sim["nav"]); nav.index = pd.to_datetime(nav.index)
        navs[name] = nav[nav.index >= START]
    spy_nav = spy[spy.index >= START]

    idx = navs["기존 21/[.6,.8,1]"].index
    n = len(idx); WIN = {"validation": idx[int(n*0.6):int(n*0.8)], "test": idx[int(n*0.8):]}

    lines = ["# yoon1f RECOVERY cap 램프 개선\n"]
    lines.append(
        "시장필터 100MA 확정 후 남은 회복 지연 축(RECOVERY cap 단계 램프) 개선. "
        "yoon1f(섹터12·무레버·100MA)로 격리. **val 선정→test 견고성.**\n"
    )
    for ph in ("validation", "test"):
        w = WIN[ph]
        lines.append(f"\n## phase={ph}\n")
        lines.append("| 램프 변형 | CAGR | Sharpe | MDD |")
        lines.append("| --- | ---: | ---: | ---: |")
        for name, nav in navs.items():
            cg, sh, md = perf(nav.reindex(w).dropna())
            lines.append(f"| {name} | {cg*100:+.1f}% | {sh:.3f} | {md*100:.1f}% |")
        cg, sh, md = perf(spy_nav.reindex(w).dropna())
        lines.append(f"| SPY | {cg*100:+.1f}% | {sh:.3f} | {md*100:.1f}% |")

    report = "\n".join(lines) + "\n"
    print(report)
    out = ROOT / "reports" / "profile_sizing" / "yoon1f_recovery_cap.md"
    out.write_text(report, encoding="utf-8")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
