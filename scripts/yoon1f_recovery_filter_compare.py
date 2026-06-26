"""yoon1f 회복 포착 보완 — 시장필터 복귀 신호 3종 비교.

진단(reports): yoon1f/g는 하락은 잘 피하나 바닥 회복을 못 탄다. 원인=시장필터
(SPY 200MA)가 항상 "바닥+14~40% 오른 뒤"에야 켜져 V자 반등 초기를 통째로 버림.

처방 3종(엔진 simulate_portfolio market_mode/파라미터, additive·기본 binary 불변):
  · binary-L   : 기존 이진. market_ma_len을 50~200 스윕(대칭 length 최적화).
  · asym       : 비대칭 히스테리시스 — 디리스크=200MA 이탈, 복귀=50MA 돌파.
  · ramp       : 연속 복귀 — 200MA 대비 비율로 off_scale→1.0 점진 복원.

레버리지 변수를 빼고 처방의 순효과만 보려 yoon1f(섹터12, 무레버)로 격리. 전체기간
점수·NAV 산출 후 2007~ 구간을 train/val/test로 분할. **val 선정 → test 견고성 확인.**

실행:
    PYTHONPATH=src:scripts .venv/bin/python scripts/yoon1f_recovery_filter_compare.py
"""
from __future__ import annotations

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


def perf(nav: pd.Series) -> tuple[float, float, float]:
    nav = nav.dropna()
    eq = nav / nav.iloc[0]
    r = eq.pct_change().dropna()
    yrs = max((nav.index[-1] - nav.index[0]).days / 365.25, 1e-9)
    sh = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0.0
    return eq.iloc[-1] ** (1 / yrs) - 1, sh, (eq / eq.cummax() - 1).min()


def main() -> None:
    base = json.loads((ROOT / "configs/strategies/yoon1f.json").read_text("utf-8"))
    base["universe"] = CORE
    cfg = config_from_dict(base)
    panels = {s: load_yfinance(s, "1d", "max") for s in CORE}
    panels = {s: d for s, d in panels.items() if d is not None and not d.empty}
    scores, prices = compute_universe(panels, cfg)
    spy = load_yfinance("SPY", "1d", "max")["close"]
    spy.index = pd.to_datetime(spy.index)

    mkw = dict(top_k=12, rebal_freq="monthly", market_close=spy,
               market_off_scale=0.5, exposure_gain=1.25)

    variants = {
        "binary 200 (기존)": dict(market_mode="binary", market_ma_len=200),
        "binary 150": dict(market_mode="binary", market_ma_len=150),
        "binary 100": dict(market_mode="binary", market_ma_len=100),
        "binary 50": dict(market_mode="binary", market_ma_len=50),
        "asym 200/50": dict(market_mode="asym", market_ma_len=200,
                            market_entry_ma_len=50),
        "asym 200/100": dict(market_mode="asym", market_ma_len=200,
                             market_entry_ma_len=100),
        "ramp 200 fl.85": dict(market_mode="ramp", market_ma_len=200,
                               market_recover_floor=0.85),
        "ramp 200 fl.90": dict(market_mode="ramp", market_ma_len=200,
                               market_recover_floor=0.90),
    }

    navs = {}
    for name, kw in variants.items():
        sim = simulate_portfolio(scores, prices, cfg, **mkw, **kw)
        nav = pd.Series(sim["nav"]); nav.index = pd.to_datetime(nav.index)
        navs[name] = nav[nav.index >= START]
    spy_nav = spy[spy.index >= START]

    idx = navs["binary 200 (기존)"].index
    n = len(idx); t_end = int(n * 0.6); v_end = int(n * 0.8)
    WIN = {"validation": idx[t_end:v_end], "test": idx[v_end:]}

    lines = ["# yoon1f 회복 포착 보완 — 시장필터 복귀신호 3종\n"]
    lines.append(
        "하락 방어는 유지하되 바닥 회복을 빨리 타도록 시장필터 복귀 신호를 보강. "
        "yoon1f(섹터12·무레버)로 처방 순효과 격리. **val 선정→test 견고성 확인.**\n"
    )
    for ph in ("validation", "test"):
        w = WIN[ph]
        lines.append(f"\n## phase={ph}\n")
        lines.append("| 변형 | CAGR | Sharpe | MDD |")
        lines.append("| --- | ---: | ---: | ---: |")
        for name, nav in navs.items():
            cg, sh, md = perf(nav.reindex(w).dropna())
            lines.append(f"| {name} | {cg*100:+.1f}% | {sh:.3f} | {md*100:.1f}% |")
        cg, sh, md = perf(spy_nav.reindex(w).dropna())
        lines.append(f"| SPY | {cg*100:+.1f}% | {sh:.3f} | {md*100:.1f}% |")

    # 코로나 바닥(2020-03-23) 후 노출 복귀 속도 — 기존 vs asym vs ramp
    lines.append("\n## 코로나 바닥(2020-03-23) 후 시장스케일 복귀 (1=풀노출)\n")
    b = pd.Timestamp("2020-03-23")
    pos = idx.searchsorted(b)
    chk = [idx[pos + d] for d in (0, 21, 42, 63) if pos + d < n]
    lines.append("| 변형 | 바닥 | +1M | +2M | +3M |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for name in ("binary 200 (기존)", "asym 200/50", "ramp 200 fl.85"):
        sim = simulate_portfolio(scores, prices, cfg, **mkw, **variants[name])
        flag = pd.Series(sim["forecast"]["market_ok"].to_numpy(),
                         index=pd.to_datetime(sim["forecast"].index))
        vals = [flag.reindex([d]).iloc[0] for d in chk]
        lines.append(f"| {name} | " + " | ".join(f"{v:.2f}" for v in vals) + " |")

    report = "\n".join(lines) + "\n"
    print(report)
    out = ROOT / "reports" / "profile_sizing" / "yoon1f_recovery_filter.md"
    out.write_text(report, encoding="utf-8")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
