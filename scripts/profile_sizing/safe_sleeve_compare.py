#!/usr/bin/env python
"""연속형 안전자산 블렌드(safe-sleeve) 방어 검증 — yoon1f 현금 방어 대비.

가설: yoon1f는 약세장에서 노출을 줄이고 남는 만큼을 '현금(수익 0)'으로 둔다.
그 현금 완충분(buffer)의 일부를, **레짐 방어 강도에 비례해 연속적으로** 안전자산
(TLT/GLD)으로 돌리면(=이진 전환이 아니라 버퍼만 회전) 위험조정 성과가 개선되는가?

앞선 '공격섹터 ↔ 안전자산 이진 전환'은 -31.5% MDD로 현금 방어(-15.6%)에 패배했다.
이 프로토타입은 두 가지를 다르게 한다:
  (1) 버퍼만 회전 — 주식 북은 그대로, 현금 완충분의 일부만 안전자산으로(손실 상한 = 버퍼).
  (2) 추세 게이트 — 안전자산도 자기 200MA 위일 때만 매수(2022처럼 채권 동반하락 시 현금 유지).

근사 방식(엔진 무수정): 현금은 수익 0이므로, 버퍼 b를 안전자산으로 옮기면 일일수익률에
b×(안전자산수익)이 더해질 뿐이다. 기준 NAV(섹터 11종, 현금방어)에 오버레이만 얹어 비교.
기준 유니버스는 TLT/GLD 오염을 피해 섹터 11종만 사용. 참고로 실제 yoon1f(채권/금 포함)도 병기.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/profile_sizing/safe_sleeve_compare.py
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

from profile_sizing.engine import performance  # noqa: E402
from profile_sizing.config import config_from_dict  # noqa: E402
from profile_sizing.portfolio import rebalance_dates  # noqa: E402
from trading_lab.strategies import get_handler  # noqa: E402

SECTORS = ["XLK", "XLF", "XLV", "XLE", "XLI", "XLP", "XLY", "XLU", "XLB", "XLRE", "XLC"]
SAFE = ["TLT", "GLD"]
FEE = (5.0 + 5.0) / 10_000.0  # yoon1f costs: fee+slippage per side


def _f(v, pct=False) -> str:
    if v is None:
        return "—"
    return f"{v*100:.1f}%" if pct else f"{v:.3f}"


def _load_close(handler, cfg, panels, symbol):
    return handler._load_close(symbol, cfg, panels)


def run_base(strategy_id: str, universe: list[str]):
    """핸들러로 기준 전략을 돌려 (phase별 forecast/nav, cfg, handler, panels) 반환."""
    handler = get_handler(strategy_id)
    cfg_dict = json.loads(
        (ROOT / "configs" / "strategies" / f"{strategy_id}.json").read_text())
    cfg_dict = dict(cfg_dict)
    cfg_dict["universe"] = universe
    cfg = config_from_dict(cfg_dict)
    raw = handler.load_data("PORTFOLIO", cfg_dict, synthetic=False)
    return handler, cfg, cfg_dict, raw


def safe_returns(handler, cfg, panels, safe_ma: int):
    """안전자산 일일수익률 + 추세게이트(전봉 종가 > MA, 무누수)."""
    cols = {}
    for s in SAFE:
        c = _load_close(handler, cfg, panels, s)
        if c is None:
            continue
        c = pd.Series(c)
        idx = pd.DatetimeIndex(pd.to_datetime(c.index))
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        c.index = idx
        cols[s] = c
    if not cols:
        raise RuntimeError("안전자산 로드 실패")
    px = pd.DataFrame(cols).sort_index()
    ret = px.pct_change()
    ma = px.rolling(safe_ma, min_periods=safe_ma).mean()
    gate = (px > ma).shift(1)              # 전봉 추세 ON만 매수
    gate = (gate == True).astype(float)    # noqa: E712 — NaN→0, 무경고
    return ret, gate


def overlay_metrics(nav, cash_ratio, ret, gate, window, interval,
                    safe_ratio, safe_max, *, monthly: bool, costs: bool):
    """기준 NAV에 safe-sleeve 오버레이를 얹어 phase 윈도우 성과 계산.

    monthly=True면 버퍼 비중 s를 월말에만 갱신해 다음 월말까지 고정 보유(현실).
    costs=True면 월별 회전(|Δs| + 안전자산 게이트 변화)에 yoon1f 거래비용을 부과.
    """
    base_ret = nav.pct_change().fillna(0.0)
    idx = nav.index
    sret = ret.reindex(idx).fillna(0.0)
    sgate = gate.reindex(idx).fillna(0.0)
    n_on = sgate.sum(axis=1)
    # 게이트 통과 안전자산 동일가중 수익(없으면 0=현금 유지)
    safe_blend = (sret * sgate).sum(axis=1) / n_on.where(n_on > 0, np.nan)
    safe_blend = safe_blend.fillna(0.0)
    # 목표 버퍼 s = min(ratio*전봉 현금비중, cap), 게이트 전무면 0.
    s_target = (safe_ratio * cash_ratio.shift(1)).clip(upper=safe_max)
    s_target = s_target.where(n_on.shift(0) > 0, 0.0).fillna(0.0)
    if monthly:
        rebal = rebalance_dates(idx, "monthly")
        mask = pd.Series([t in rebal for t in idx], index=idx)
        s_held = s_target.where(mask).ffill().fillna(0.0)
    else:
        s_held = s_target
    over_ret = base_ret + s_held * safe_blend
    if costs:
        turn = s_held.diff().abs().fillna(s_held.iloc[0] if len(s_held) else 0.0)
        over_ret = over_ret - FEE * turn
    over_nav = (1.0 + over_ret).cumprod()
    w = [t for t in window if t in over_nav.index]
    eq = over_nav.reindex(w)
    eq = eq / eq.iloc[0]
    r = eq.pct_change().fillna(0.0)
    perf = performance(eq, r, interval)
    avg_safe = float(s_held.reindex(w).mean())
    return perf, avg_safe


RATIOS = (0.5, 0.75, 1.0, 1.25)
CAPS = (0.15, 0.2, 0.3, 0.4, 0.5)
MAS = (100, 150, 200)


def main() -> int:
    rows = []
    # ----- 기준: 섹터 11종 현금방어 (TLT/GLD 미포함) -----
    handler, cfg, cfg_dict, raw = run_base("yoon1f", SECTORS)
    panels = handler._from_wide(raw)
    # MA별 안전자산 수익/게이트 사전계산
    safe_by_ma = {ma: safe_returns(handler, cfg, panels, ma) for ma in MAS}

    arts = {}
    for phase in ("all", "test"):
        art = handler.build_artifacts(raw, cfg_dict, symbol="PORTFOLIO",
                                      phase=phase, bars_per_year=252)
        arts[phase] = art
        nav_full = art.equity
        bperf = performance(nav_full, nav_full.pct_change().fillna(0.0), cfg.interval)
        rows.append({"variant": "섹터11_현금방어(기준)", "phase": phase,
                     "ratio": None, "cap": None, "ma": None,
                     "cagr": bperf["cagr"], "mdd": bperf["max_drawdown"],
                     "sharpe": bperf["sharpe"], "avg_safe": 0.0})

    # ----- safe-sleeve 그리드 (월간 리밸런스 + 거래비용 = 현실 모델) -----
    for ma in MAS:
        ret, gate = safe_by_ma[ma]
        for sr in RATIOS:
            for sm in CAPS:
                rec = {"ratio": sr, "cap": sm, "ma": ma}
                for phase in ("all", "test"):
                    art = arts[phase]
                    fc = art.forecast
                    nav_idx = pd.Series(art.equity.values, index=fc.index)
                    perf, avg_safe = overlay_metrics(
                        nav_idx, fc["cash_ratio"], ret, gate, fc.index,
                        cfg.interval, sr, sm, monthly=True, costs=True)
                    rec[phase] = {"cagr": perf["cagr"], "mdd": perf["max_drawdown"],
                                  "sharpe": perf["sharpe"], "avg_safe": avg_safe}
                rows.append({"variant": f"safe r{sr}/cap{sm}/ma{ma}",
                             "ratio": sr, "cap": sm, "ma": ma,
                             "cagr": rec["all"]["cagr"], "mdd": rec["all"]["mdd"],
                             "sharpe": rec["all"]["sharpe"],
                             "avg_safe": rec["all"]["avg_safe"],
                             "t_cagr": rec["test"]["cagr"], "t_mdd": rec["test"]["mdd"],
                             "t_sharpe": rec["test"]["sharpe"]})

    # 참고: 실제 yoon1f(TLT/GLD 유니버스 포함)
    h2 = get_handler("yoon1f")
    cfg2 = json.loads((ROOT / "configs" / "strategies" / "yoon1f.json").read_text())
    raw2 = h2.load_data("PORTFOLIO", cfg2, synthetic=False)
    ref = {}
    for phase in ("all", "test"):
        m = h2.build_artifacts(raw2, cfg2, symbol="PORTFOLIO", phase=phase,
                               bars_per_year=252).metrics
        ref[phase] = (m["cagr"], m["max_drawdown"], m["sharpe"],
                      m.get("buy_hold_cagr"), m.get("buy_hold_max_drawdown"),
                      m.get("buy_hold_sharpe"))

    # 상위(test Sharpe 기준) 12개만 출력
    grid = [r for r in rows if r.get("t_sharpe") is not None]
    grid.sort(key=lambda r: r["t_sharpe"], reverse=True)
    base_all = next(r for r in rows if r["variant"].startswith("섹터11") and r["sharpe"])
    base_test = [r for r in rows if r["variant"].startswith("섹터11")][1]

    print("\n[현실 모델: 월간 리밸런스 + 거래비용 10bps/side]")
    print(f"{'config':22s} | {'CAGR':>6s} {'MDD':>7s} {'Shrp':>5s} {'안전':>5s} "
          f"| {'t.CAGR':>6s} {'t.MDD':>7s} {'t.Shrp':>6s}")
    print("-" * 78)
    print(f"{'섹터11 현금방어(기준)':22s} | {_f(base_all['cagr'],1):>6s} "
          f"{_f(base_all['mdd'],1):>7s} {_f(base_all['sharpe']):>5s} {'0%':>5s} "
          f"| {_f(base_test['cagr'],1):>6s} {_f(base_test['mdd'],1):>7s} "
          f"{_f(base_test['sharpe']):>6s}")
    for r in grid[:12]:
        cfgname = f"r{r['ratio']}/cap{r['cap']}/ma{r['ma']}"
        print(f"{cfgname:22s} | {_f(r['cagr'],1):>6s} {_f(r['mdd'],1):>7s} "
              f"{_f(r['sharpe']):>5s} {_f(r['avg_safe'],1):>5s} | "
              f"{_f(r['t_cagr'],1):>6s} {_f(r['t_mdd'],1):>7s} {_f(r['t_sharpe']):>6s}")
    print(f"{'yoon1f(채권/금)':22s} | {_f(ref['all'][0],1):>6s} {_f(ref['all'][1],1):>7s} "
          f"{_f(ref['all'][2]):>5s} {'—':>5s} | {_f(ref['test'][0],1):>6s} "
          f"{_f(ref['test'][1],1):>7s} {_f(ref['test'][2]):>6s}")
    print(f"\n[SPY] all CAGR {_f(ref['all'][3],1)} MDD {_f(ref['all'][4],1)} "
          f"Sharpe {_f(ref['all'][5])}")

    outdir = ROOT / "reports" / "profile_sizing"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "safe_sleeve_compare.json").write_text(
        json.dumps(rows, indent=2, default=str), encoding="utf-8")
    print(f"\n리포트: {outdir / 'safe_sleeve_compare.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
