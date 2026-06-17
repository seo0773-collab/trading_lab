#!/usr/bin/env python
"""yoon1b/yoon1c 워크포워드 검증 — 엣지가 단일 test 구간의 운인지, 여러 구간에서
반복되는지 확인.

이 전략은 프로덕션에 '학습 파라미터'가 없다(config 고정, 무누수). 따라서 모든 구간이
사실상 out-of-sample이고, 워크포워드의 핵심 질문은 **"여러 독립 구간에서 벤치마크 우위가
일관되게 반복되는가"**다. 두 방식으로 본다:

  (1) 연도별(비중복) — 독립성이 깨끗. 각 연도에서 전략 vs SPY/EW 승패를 센다.
  (2) 롤링 3년(중복, 1년 스텝) — 구간 선택 민감도를 줄인 추세 확인.

전 일봉 히스토리에서 NAV를 한 번 만들고, 구간별로 잘라 재정규화해 측정한다. CAGR은
경과시간 기반, Sharpe는 일봉 252 연율화, MDD는 구간 내 재정규화 equity 기준.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/profile_sizing/walk_forward.py
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
from trading_lab.portfolio_universes import SECTOR_INDEX  # noqa: E402

START_YEAR = 2000   # 유니버스·벤치마크가 충분히 갖춰지는 시점부터
ROLL_LEN = 3        # 롤링 윈도 길이(년)


def perf(nav_w: pd.Series) -> dict | None:
    nav_w = pd.Series(nav_w).dropna()
    if len(nav_w) < 60:
        return None
    eq = nav_w / nav_w.iloc[0]
    ret = eq.pct_change().dropna()
    years = max((nav_w.index[-1] - nav_w.index[0]).days / 365.25, 1e-9)
    cagr = float(eq.iloc[-1]) ** (1.0 / years) - 1.0
    sharpe = (float(ret.mean()) / float(ret.std()) * np.sqrt(252)
              if ret.std() > 0 else None)
    mdd = float((eq / eq.cummax() - 1.0).min())
    return {"cagr": cagr, "sharpe": sharpe, "mdd": mdd}


def _f(v, pct=False) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v*100:.1f}%" if pct else f"{v:.3f}"


def main() -> int:
    base_raw = json.loads(
        (ROOT / "configs" / "strategies" / "yoon1b.json").read_text())
    cfg = config_from_dict(base_raw)
    universe = list(base_raw["universe"])
    mf = base_raw.get("market_filter") or {}
    top_k = int(base_raw["top_k"])
    rebal = str(base_raw["rebalance_freq"])

    from run_kalman_pipeline import load_yfinance
    print(f"일봉 로드: {len(universe)}종목 + SPY + 섹터 ETF ...", flush=True)
    panels = {}
    for s in universe:
        try:
            panels[s] = load_yfinance(s, cfg.interval, cfg.period)
        except Exception:  # noqa: BLE001
            continue
    spy = load_yfinance("SPY", cfg.interval, cfg.period)["close"]
    sect = {}
    for tk in sorted(set(SECTOR_INDEX.values())):
        try:
            sect[tk] = load_yfinance(tk, cfg.interval, cfg.period)["close"]
        except Exception:  # noqa: BLE001
            continue

    print("compute_universe (1회) ...", flush=True)
    scores, prices = compute_universe(panels, cfg)
    idx = prices.index
    mkw = dict(market_close=spy, market_ma_len=int(mf.get("ma_len", 200)),
               market_off_scale=float(mf.get("off_scale", 0.5)))

    sims = {
        "yoon1b": simulate_portfolio(scores, prices, cfg, top_k=top_k,
                                     rebal_freq=rebal, exposure_gain=1.25, **mkw),
        "yoon1c": simulate_portfolio(scores, prices, cfg, top_k=top_k,
                                     rebal_freq=rebal, exposure_gain=1.25,
                                     sector_close=sect, symbol_sector=SECTOR_INDEX,
                                     sector_off_scale=0.5, **mkw),
    }
    navs = {k: v["nav"] for k, v in sims.items()}
    ew = sims["yoon1b"]["benchmark_ew"]
    spy_nav = pd.Series(spy).reindex(idx).ffill()

    end_year = int(idx[-1].year)
    years = list(range(START_YEAR, end_year + 1))

    # (1) 연도별
    yearly = []
    for y in years:
        w = idx[idx.year == y]
        if len(w) < 60:
            continue
        row = {"year": y, "bars": len(w)}
        b = {k: perf(navs[k].reindex(w)) for k in navs}
        pe, ps = perf(ew.reindex(w)), perf(spy_nav.reindex(w))
        row["spy"], row["ew"] = ps, pe
        row.update({k: b[k] for k in navs})
        yearly.append(row)

    # (2) 롤링 3년
    rolling = []
    for y in range(START_YEAR, end_year - ROLL_LEN + 2):
        w = idx[(idx.year >= y) & (idx.year <= y + ROLL_LEN - 1)]
        if len(w) < 250:
            continue
        b = {k: perf(navs[k].reindex(w)) for k in navs}
        rolling.append({"start": y, "end": y + ROLL_LEN - 1,
                        "spy": perf(spy_nav.reindex(w)),
                        "ew": perf(ew.reindex(w)), **b})

    md = _markdown(yearly, rolling)
    outdir = ROOT / "reports" / "profile_sizing"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "walk_forward.md").write_text(md, encoding="utf-8")
    print("\n" + md)
    print(f"리포트: {outdir / 'walk_forward.md'}")
    return 0


def _winrate(rows, variant, bench, metric):
    """variant가 bench보다 metric에서 높은 구간 수 / 유효 구간 수."""
    n = w = 0
    for r in rows:
        a, b = r.get(variant), r.get(bench)
        if not a or not b or a.get(metric) is None or b.get(metric) is None:
            continue
        n += 1
        if a[metric] > b[metric]:
            w += 1
    return w, n


def _markdown(yearly, rolling) -> str:
    lines = [
        "# yoon1b/yoon1c 워크포워드 검증",
        "",
        "전략에 학습 파라미터가 없어 모든 구간이 out-of-sample. **여러 독립 구간에서 "
        "벤치마크 우위가 반복되는지**가 핵심. 주 벤치=SPY(시총가중 시장), 보조=EW 지수.",
        "",
        "## (1) 연도별 (비중복 = 독립성 깨끗)",
        "",
        "| 연도 | yoon1b CAGR | yoon1c CAGR | SPY CAGR | EW CAGR | "
        "yoon1b Sharpe | SPY Sharpe | b−SPY✓ | b−EW✓ |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--: | :--: |",
    ]
    for r in yearly:
        b, c, ps, pe = r.get("yoon1b"), r.get("yoon1c"), r.get("spy"), r.get("ew")
        bs = "✓" if (b and ps and b["cagr"] is not None and b["cagr"] > ps["cagr"]) else "·"
        be = "✓" if (b and pe and b["cagr"] is not None and b["cagr"] > pe["cagr"]) else "·"
        lines.append(
            f"| {r['year']} | {_f(b['cagr'] if b else None, True)} | "
            f"{_f(c['cagr'] if c else None, True)} | {_f(ps['cagr'] if ps else None, True)} | "
            f"{_f(pe['cagr'] if pe else None, True)} | {_f(b['sharpe'] if b else None)} | "
            f"{_f(ps['sharpe'] if ps else None)} | {bs} | {be} |")

    wb_spy = _winrate(yearly, "yoon1b", "spy", "cagr")
    wb_ew = _winrate(yearly, "yoon1b", "ew", "cagr")
    wc_spy = _winrate(yearly, "yoon1c", "spy", "cagr")
    sb_spy = _winrate(yearly, "yoon1b", "spy", "sharpe")
    lines += [
        "",
        "### 연도별 승률 (CAGR 기준)",
        f"- yoon1b > SPY: **{wb_spy[0]}/{wb_spy[1]}년** "
        f"({100*wb_spy[0]/max(wb_spy[1],1):.0f}%)",
        f"- yoon1b > EW : {wb_ew[0]}/{wb_ew[1]}년 "
        f"({100*wb_ew[0]/max(wb_ew[1],1):.0f}%)",
        f"- yoon1c > SPY: {wc_spy[0]}/{wc_spy[1]}년",
        f"- (Sharpe) yoon1b > SPY: {sb_spy[0]}/{sb_spy[1]}년",
        "",
        f"## (2) 롤링 {ROLL_LEN}년 (1년 스텝, 중복)",
        "",
        "| 구간 | yoon1b CAGR | yoon1b Sharpe | yoon1c CAGR | yoon1c Sharpe | "
        "SPY CAGR | SPY Sharpe | EW Sharpe |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rolling:
        b, c, ps, pe = r.get("yoon1b"), r.get("yoon1c"), r.get("spy"), r.get("ew")
        lines.append(
            f"| {r['start']}–{r['end']} | {_f(b['cagr'] if b else None, True)} | "
            f"{_f(b['sharpe'] if b else None)} | {_f(c['cagr'] if c else None, True)} | "
            f"{_f(c['sharpe'] if c else None)} | {_f(ps['cagr'] if ps else None, True)} | "
            f"{_f(ps['sharpe'] if ps else None)} | {_f(pe['sharpe'] if pe else None)} |")

    rb_spy = _winrate(rolling, "yoon1b", "spy", "sharpe")
    rb_spy_c = _winrate(rolling, "yoon1b", "spy", "cagr")
    rc_spy = _winrate(rolling, "yoon1c", "spy", "sharpe")
    lines += [
        "",
        f"### 롤링 {ROLL_LEN}년 승률",
        f"- (Sharpe) yoon1b > SPY: **{rb_spy[0]}/{rb_spy[1]}구간** "
        f"({100*rb_spy[0]/max(rb_spy[1],1):.0f}%)",
        f"- (CAGR)   yoon1b > SPY: {rb_spy_c[0]}/{rb_spy_c[1]}구간",
        f"- (Sharpe) yoon1c > SPY: {rc_spy[0]}/{rc_spy[1]}구간",
        "",
        "*판정: 우위가 다수 독립 구간에서 반복되면 단일 test의 운이 아님(엣지 견고). "
        "특정 연도(예: 약세장)에서의 거동도 함께 볼 것.*",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
