"""yoon1g 레버리지 슬리브 × 빠른 복귀신호 — 회복 상방 증폭 검증.

가설: yoon1g 회복 레버리지 슬리브가 거의 발동 못 한 건 게이트(require_market_on)가
느린 SPY 200MA를 기다렸기 때문. 시장필터를 빠른 복귀신호(100MA / asym 200·50)로
바꾸면 슬리브가 회복 초기에 실제 발동 → 상방 증폭. yoon1f(100MA)를 베이스로.

레버리지 효과 격리: 무레버 100 vs 슬리브(bin100) vs 슬리브(asym) vs 슬리브(bin200, 구).
val 선정 → test 견고성 + 연도별 2x 점유율(슬리브 실제 발동 확인).

실행:
    PYTHONPATH=src:scripts .venv/bin/python scripts/yoon1g_fast_recovery_sleeve.py
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
TWO = ["ROM", "UYG", "RXL", "DIG", "UXI", "UGE", "UCC", "UPW", "UYM", "URE"]
START = "2007-02-01"


def perf(nav: pd.Series):
    nav = nav.dropna(); eq = nav / nav.iloc[0]; r = eq.pct_change().dropna()
    yrs = max((nav.index[-1] - nav.index[0]).days / 365.25, 1e-9)
    sh = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0.0
    return eq.iloc[-1] ** (1 / yrs) - 1, sh, (eq / eq.cummax() - 1).min()


def mkt_ok(spy: pd.Series, mode: str, ma_len: int, entry: int = 50) -> pd.Series:
    """슬리브 게이트용 시장 ON 불리언(전봉, 무누수). 노출 필터와 동일 규칙."""
    m = pd.Series(spy)
    if mode == "asym":
        ma_x = m.rolling(ma_len, min_periods=ma_len).mean()
        ma_e = m.rolling(entry, min_periods=entry).mean()
        ax = (m > ma_x).to_numpy(); ae = (m > ma_e).to_numpy()
        warm = ma_x.isna().to_numpy()
        st = np.zeros(len(m), dtype=bool); cur = True
        for i in range(len(m)):
            if warm[i]:
                cur = True
            elif cur:
                cur = bool(ax[i])
            else:
                cur = True if ae[i] else False
            st[i] = cur
        ok = pd.Series(st, index=m.index)
    else:
        ok = m > m.rolling(ma_len, min_periods=ma_len).mean()
    return ok.shift(1).fillna(True)


def main() -> None:
    base = json.loads((ROOT / "configs/strategies/yoon1f.json").read_text("utf-8"))
    cfg = config_from_dict(base)
    panels = {s: load_yfinance(s, "1d", "max") for s in CORE + TWO}
    panels = {s: d for s, d in panels.items() if d is not None and not d.empty}
    spy = load_yfinance("SPY", "1d", "max")["close"]; spy.index = pd.to_datetime(spy.index)
    mkw = dict(top_k=12, rebal_freq="monthly", market_close=spy,
               market_off_scale=0.5, exposure_gain=1.25)

    # 변형: (universe, sleeve여부, mode, ma_len, entry)
    V = {
        "무레버 (bin100)":      (CORE, False, "binary", 100, 50),
        "슬리브 (bin100)=yoon1g": (CORE + TWO, True, "binary", 100, 50),
        "슬리브 (asym200/50)":   (CORE + TWO, True, "asym", 200, 50),
        "슬리브 (bin200, 구)":    (CORE + TWO, True, "binary", 200, 50),
    }
    navs, waves = {}, {}
    for name, (uni, sleeve, mode, ma_len, entry) in V.items():
        ps = {s: panels[s] for s in uni if s in panels}
        ok = mkt_ok(spy, mode, ma_len, entry) if sleeve else None
        scores, prices = compute_universe(
            ps, cfg, leveraged_symbols=set(TWO) if sleeve else None,
            leverage_regimes=("RECOVERY",), market_ok=ok,
        )
        sim = simulate_portfolio(scores, prices, cfg, **mkw, market_mode=mode,
                                 market_ma_len=ma_len, market_entry_ma_len=entry)
        nav = pd.Series(sim["nav"]); nav.index = pd.to_datetime(nav.index)
        navs[name] = nav[nav.index >= START]
        waves[name] = (prices, sim)
    spy_nav = spy[spy.index >= START]

    idx = navs["무레버 (bin100)"].index
    n = len(idx); WIN = {"validation": idx[int(n*0.6):int(n*0.8)], "test": idx[int(n*0.8):]}

    lines = ["# yoon1g 슬리브 × 빠른 복귀신호 — 회복 상방 증폭 검증\n"]
    lines.append(
        "yoon1f(100MA) 베이스. 슬리브 게이트(require_market_on)가 빠른 복귀신호를 따르게 "
        "해 회복 초기 발동을 노림. **val 선정→test 견고성.**\n"
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

    # 연도별 2x 점유율(슬리브 발동) — 슬리브 변형들
    lines.append("\n## 연도별 2x 점유율(주식 중, 슬리브 발동 확인)\n")
    for name in ("슬리브 (bin100)=yoon1g", "슬리브 (asym200/50)", "슬리브 (bin200, 구)"):
        prices, sim = waves[name]
        tw = sim["target_weights"]  # DataFrame? — fallback: forecast exposure 불가, weights 사용
        tw = pd.DataFrame(tw) if not isinstance(tw, pd.DataFrame) else tw
        tw.index = pd.to_datetime(tw.index)
        tw = tw[tw.index >= START]
        two = [s for s in TWO if s in tw.columns]
        one = [s for s in CORE if s in tw.columns]
        yr = tw.index.year
        g = {}
        for y in sorted(set(yr)):
            mask = yr == y
            stk = tw.loc[mask, two + one].to_numpy().sum()
            g[y] = (tw.loc[mask, two].to_numpy().sum() / stk) if stk > 0 else 0.0
        share = " ".join(f"{y%100:02d}:{v*100:.0f}" for y, v in g.items())
        lines.append(f"- **{name}**: `{share}` (%)")

    report = "\n".join(lines) + "\n"
    print(report)
    out = ROOT / "reports" / "profile_sizing" / "yoon1g_fast_recovery_sleeve.md"
    out.write_text(report, encoding="utf-8")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
