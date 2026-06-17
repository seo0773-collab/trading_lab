#!/usr/bin/env python
"""페이퍼 트레이딩 저널 채점 — 누적 스냅샷의 실현 성과를 실제 가격으로 재구성해
SPY 및 백테스트 기대치와 비교한다(실제 vs 백테스트 괴리 측정 → live 최종판정 근거).

paper_trade.py가 적립한 var/paper_trading/<전략>_journal.jsonl을 읽어:
  · 연속 스냅샷 [t_i, t_{i+1}] 구간마다 그때 들고 있던 목표비중을 실제 가격수익으로
    실현 → 페이퍼 누적 NAV. (현금은 0% 수익 가정)
  · 같은 live 구간 SPY 수익과 비교, 백테스트 equity와의 괴리(추적오차) 보고.
전진 기록이라 스냅샷이 ≥2개(최소 한 번의 리밸런스 경과) 쌓여야 채점된다. 그 전에는
참고용으로 '백테스트 기준선(최근 N개월)'을 보여줘 기대 경로를 제시한다.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/profile_sizing/paper_review.py --strategy yoon1b
"""
from __future__ import annotations

import argparse
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


def _load(universe, cfg, want_spy, want_sect):
    from run_kalman_pipeline import load_yfinance
    panels = {}
    for s in universe:
        try:
            d = load_yfinance(s, cfg.interval, cfg.period)
            if d is not None and not d.empty:
                panels[s] = d
        except Exception:  # noqa: BLE001
            continue
    spy = None
    if want_spy:
        try:
            spy = load_yfinance("SPY", cfg.interval, cfg.period)["close"]
        except Exception:  # noqa: BLE001
            spy = None
    sect = {}
    if want_sect:
        for tk in sorted(set(SECTOR_INDEX.values())):
            try:
                sect[tk] = load_yfinance(tk, cfg.interval, cfg.period)["close"]
            except Exception:  # noqa: BLE001
                continue
    return panels, spy, (sect or None)


def _price_at(prices: pd.DataFrame, sym: str, day: pd.Timestamp) -> float | None:
    if sym not in prices.columns:
        return None
    s = prices[sym].dropna()
    s = s[s.index <= day]
    return float(s.iloc[-1]) if len(s) else None


def _ret(v: float | None, pct=True) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v*100:+.2f}%" if pct else f"{v:.3f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strategy", default="yoon1b")
    ap.add_argument("--reference-months", type=int, default=6,
                    help="저널이 비었을 때 보여줄 백테스트 기준선 기간(개월)")
    args = ap.parse_args(argv)

    raw = json.loads(
        (ROOT / "configs" / "strategies" / f"{args.strategy}.json").read_text())
    cfg = config_from_dict(raw)

    jpath = ROOT / "var" / "paper_trading" / f"{args.strategy}_journal.jsonl"
    snaps = []
    if jpath.exists():
        snaps = [json.loads(ln) for ln in jpath.read_text().splitlines() if ln.strip()]

    print(f"[{args.strategy}] 저널 스냅샷 {len(snaps)}개. 데이터 로드 ...", flush=True)
    panels, spy, _ = _load(list(raw["universe"]), cfg, want_spy=True, want_sect=False)
    _, prices = compute_universe(panels, cfg)
    spy_s = pd.Series(spy).dropna() if spy is not None else None

    print(_review_journal(snaps, prices, spy_s))
    print(_reference(raw, cfg, panels, spy_s, args.reference_months))
    return 0


def _spy_ret(spy_s, d0, d1) -> float | None:
    if spy_s is None:
        return None
    a = spy_s[spy_s.index <= d0]
    b = spy_s[spy_s.index <= d1]
    if not len(a) or not len(b):
        return None
    return float(b.iloc[-1]) / float(a.iloc[-1]) - 1.0


def _review_journal(snaps, prices, spy_s) -> str:
    lines = ["", "=" * 64, " 페이퍼 트레이딩 실현 채점 (전진 기록)", "=" * 64]
    if len(snaps) < 2:
        lines += [
            f" 스냅샷 {len(snaps)}개 — 채점하려면 ≥2개(최소 1회 리밸런스 경과) 필요.",
            " paper_trade.py를 주기적으로(월말 또는 주기적) 실행해 적립하세요.",
            " 누적되면 구간별 실현수익·SPY 대비·백테스트 괴리가 여기 표로 나옵니다.",
        ]
        return "\n".join(lines)

    paper_nav = 1.0
    rows = []
    for a, b in zip(snaps[:-1], snaps[1:]):
        d0 = pd.Timestamp(a["data_through"]); d1 = pd.Timestamp(b["data_through"])
        r = 0.0
        for s, w in a["targets"].items():
            p0, p1 = _price_at(prices, s, d0), _price_at(prices, s, d1)
            if p0 and p1:
                r += w * (p1 / p0 - 1.0)
        paper_nav *= (1.0 + r)
        rows.append((d0.date(), d1.date(), r, _spy_ret(spy_s, d0, d1), paper_nav))
    lines += [
        f" 구간 {len(rows)}개 ({rows[0][0]} ~ {rows[-1][1]})",
        f" {'시작':<12}{'종료':<12}{'페이퍼':>10}{'SPY':>10}{'누적NAV':>10}",
    ]
    for d0, d1, r, sp, nav in rows:
        lines.append(f" {str(d0):<12}{str(d1):<12}{_ret(r):>10}{_ret(sp):>10}{nav:>10.3f}")
    tot = paper_nav - 1.0
    sp_tot = _spy_ret(spy_s, pd.Timestamp(snaps[0]["data_through"]),
                      pd.Timestamp(snaps[-1]["data_through"]))
    lines += ["",
              f" 누적 페이퍼 {_ret(tot)}  vs  SPY {_ret(sp_tot)}  "
              f"(초과 {_ret((tot - sp_tot) if sp_tot is not None else None)})"]
    return "\n".join(lines)


def _reference(raw, cfg, panels, spy_s, months) -> str:
    """백테스트 기준선: 최근 N개월 전략 실현경로(기대치). live 채점의 비교 기준."""
    mf = raw.get("market_filter") or {}
    scores, prices = compute_universe(panels, cfg)
    sim = simulate_portfolio(
        scores, prices, cfg,
        top_k=int(raw["top_k"]), rebal_freq=str(raw["rebalance_freq"]),
        market_close=spy_s, market_ma_len=int(mf.get("ma_len", 200)),
        market_off_scale=float(mf.get("off_scale", 0.5)),
        exposure_gain=float(raw.get("exposure_gain", 1.0)))
    nav = sim["nav"]
    end = nav.index[-1]
    start = end - pd.DateOffset(months=months)
    w = nav[nav.index >= start]
    strat_r = float(w.iloc[-1] / w.iloc[0] - 1.0)
    sp_r = _spy_ret(spy_s, w.index[0], w.index[-1])
    lines = ["", "-" * 64,
             f" 참고: 백테스트 기준선 (최근 {months}개월, {w.index[0].date()}~{end.date()})",
             f"   전략 {_ret(strat_r)}  vs  SPY {_ret(sp_r)}  "
             f"(초과 {_ret((strat_r - sp_r) if sp_r is not None else None)})",
             "   → live 페이퍼 누적이 이 기대 경로와 비슷하게 가면 정상, 크게 벌어지면",
             "     체결·데이터 괴리 점검 필요."]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
