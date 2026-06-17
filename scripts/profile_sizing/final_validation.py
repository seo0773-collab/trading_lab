#!/usr/bin/env python
"""yoon1 1순위 최종 검증 — 거래비용 민감도 + holdout(test) 최종 개봉.

지금까지 파라미터 '선택'은 validation에서만 했고 test는 손대지 않았다. 이 스크립트는
파라미터를 동결(top_k=20·monthly·floor=1.0·시장필터 ON)한 채:

  (1) 거래비용 민감도 — 비용(수수료+슬리피지)을 1x/2x/3x로 올려도 위험조정 우위가
      살아남는지. 점수(compute_universe)는 비용과 무관하므로 1회만 계산하고 비용
      배수만 바꿔 재시뮬한다.
  (2) holdout 최종 개봉 — 한 번도 탐색에 안 쓴 test 구간 성과를 정식 보고하고
      공정 벤치마크(EW 지수)·참고 벤치마크(진짜 B&H) 대비 판정한다.

벤치마크에는 비용을 부과하지 않는다(상시보유라 회전 거의 없음 → 전략에 보수적).

Usage:
    PYTHONPATH=src .venv/bin/python scripts/profile_sizing/final_validation.py
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from profile_sizing.config import config_from_dict  # noqa: E402
from profile_sizing.engine import performance  # noqa: E402
from profile_sizing.portfolio import compute_universe, simulate_portfolio  # noqa: E402
from profile_sizing.run import slice_window  # noqa: E402

COST_MULTS = [1.0, 2.0, 3.0]


def _perf(series_nav: pd.Series, window, interval: str) -> dict:
    nav = series_nav.reindex(window)
    eq = nav / nav.dropna().iloc[0]
    ret = nav.pct_change().fillna(0.0)
    p = performance(eq, ret, interval)
    return {"cagr": p["cagr"], "sharpe": p["sharpe"], "mdd": p["max_drawdown"]}


def _bench(series: pd.Series, window, interval: str) -> dict:
    b = series.reindex(window).ffill().bfill()
    eq = b / b.dropna().iloc[0]
    ret = b.pct_change().fillna(0.0)
    p = performance(eq, ret, interval)
    return {"cagr": p["cagr"], "sharpe": p["sharpe"], "mdd": p["max_drawdown"]}


def _row(sim, window, interval, mult) -> dict:
    strat = _perf(sim["nav"], window, interval)
    ew = _bench(sim["benchmark_ew"], window, interval)
    bh = _bench(sim["benchmark"], window, interval)
    n_trades = int(len(sim["trades"]))
    return {
        "cost_mult": mult,
        "n_trades": n_trades,
        "cagr": strat["cagr"], "mdd": strat["mdd"], "sharpe": strat["sharpe"],
        "ew_cagr": ew["cagr"], "ew_mdd": ew["mdd"], "ew_sharpe": ew["sharpe"],
        "bh_cagr": bh["cagr"], "bh_mdd": bh["mdd"], "bh_sharpe": bh["sharpe"],
        "sharpe_vs_ew": (strat["sharpe"] - ew["sharpe"])
        if (strat["sharpe"] is not None and ew["sharpe"] is not None) else None,
    }


def _f(v, pct=False) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v*100:.1f}%" if pct else f"{v:.3f}"


def main() -> int:
    base_raw = json.loads(
        (ROOT / "configs" / "strategies" / "yoon1.json").read_text()
    )
    base_cfg = config_from_dict(base_raw)
    universe = list(base_raw["universe"])
    mf = base_raw.get("market_filter") or {}
    top_k = int(base_raw["top_k"])
    rebal = str(base_raw["rebalance_freq"])

    from run_kalman_pipeline import load_yfinance
    print(f"데이터 로드: {len(universe)}종목 + SPY ...", flush=True)
    panels = {}
    for s in universe:
        try:
            panels[s] = load_yfinance(s, base_cfg.interval, base_cfg.period)
        except Exception:  # noqa: BLE001
            continue
    try:
        spy = load_yfinance(mf.get("symbol", "SPY"), base_cfg.interval,
                            base_cfg.period)["close"]
    except Exception:  # noqa: BLE001
        spy = None

    print("compute_universe (점수=비용 무관, 1회) ...", flush=True)
    scores, prices = compute_universe(panels, base_cfg)
    idx = prices.index

    # 비용 배수별 재시뮬. 점수/가격 고정, 엔진 fee만 변경.
    all_rows, test_rows = [], []
    base_fee = base_cfg.costs.fee_bps_per_side
    base_slp = base_cfg.costs.slippage_bps
    win_all = slice_window(idx, "all", base_cfg)
    win_test = slice_window(idx, "test", base_cfg)
    for mult in COST_MULTS:
        cfg = replace(base_cfg, costs=replace(
            base_cfg.costs, fee_bps_per_side=base_fee * mult,
            slippage_bps=base_slp * mult))
        sim = simulate_portfolio(
            scores, prices, cfg, top_k=top_k, rebal_freq=rebal,
            market_close=spy, market_ma_len=int(mf.get("ma_len", 200)),
            market_off_scale=float(mf.get("off_scale", 0.5)))
        all_rows.append(_row(sim, win_all, cfg.interval, mult))
        test_rows.append(_row(sim, win_test, cfg.interval, mult))
        print(f"  cost x{mult:g} | all Sharpe {_f(all_rows[-1]['sharpe'])} "
              f"(EW {_f(all_rows[-1]['ew_sharpe'])}) | "
              f"test Sharpe {_f(test_rows[-1]['sharpe'])} "
              f"(EW {_f(test_rows[-1]['ew_sharpe'])})", flush=True)

    one_per_side = base_fee + base_slp  # 편도 bp(기본)
    md = _markdown(all_rows, test_rows, one_per_side, win_test, win_all)
    outdir = ROOT / "reports" / "profile_sizing"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "final_validation.md").write_text(md, encoding="utf-8")
    (outdir / "final_validation.json").write_text(
        json.dumps({"all": all_rows, "test": test_rows,
                    "base_bp_per_side": one_per_side,
                    "test_start": str(win_test[0]), "test_end": str(win_test[-1])},
                   indent=2),
        encoding="utf-8")
    print("\n" + md)
    print(f"리포트: {outdir / 'final_validation.md'}")
    return 0


def _cost_table(rows, base_bp) -> list[str]:
    lines = [
        "| 비용 | 편도 bp | 거래수 | 전략 CAGR | 전략 MDD | 전략 Sharpe | "
        "EW Sharpe | 전략−EW Sharpe |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| x{r['cost_mult']:g} | {base_bp * r['cost_mult']:.0f} | "
            f"{r['n_trades']} | {_f(r['cagr'], True)} | {_f(r['mdd'], True)} | "
            f"{_f(r['sharpe'])} | {_f(r['ew_sharpe'])} | "
            f"**{_f(r['sharpe_vs_ew'])}** |")
    return lines


def _markdown(all_rows, test_rows, base_bp, win_test, win_all) -> str:
    base = next(r for r in all_rows if r["cost_mult"] == 1.0)
    x2 = next(r for r in all_rows if r["cost_mult"] == 2.0)
    survives = (x2["sharpe_vs_ew"] is not None and x2["sharpe_vs_ew"] > 0)
    test1 = next(r for r in test_rows if r["cost_mult"] == 1.0)
    test2 = next(r for r in test_rows if r["cost_mult"] == 2.0)
    lines = [
        "# yoon1 최종 검증 — 거래비용 민감도 + holdout 개봉",
        "",
        "파라미터 동결: top_k=20 · monthly · trend floor=1.0 · 시장필터 ON(SPY/200MA). "
        f"기본 비용 = 편도 {base_bp:.0f}bp(수수료+슬리피지). 벤치마크엔 비용 미부과.",
        "",
        "## 1) 거래비용 민감도 (phase=all)",
        "",
        f"비용을 2배(편도 {base_bp*2:.0f}bp)로 올려도 위험조정 우위(전략−EW Sharpe) "
        + ("**유지됨** ✅" if survives else "**소멸** ⚠️"),
        ". 월간 리밸런스라 회전이 낮아 비용 민감도가 작다.",
        "",
        *_cost_table(all_rows, base_bp),
        "",
        f"- 비용 2배 시 전략 Sharpe {_f(base['sharpe'])} → {_f(x2['sharpe'])} "
        f"(CAGR {_f(base['cagr'], True)} → {_f(x2['cagr'], True)})",
        "",
        "## 2) holdout(test) 최종 개봉",
        "",
        f"한 번도 파라미터 탐색에 쓰지 않은 test 구간({str(win_test[0])[:10]} ~ "
        f"{str(win_test[-1])[:10]}, {len(win_test)}봉) 성과. 공정 벤치마크=EW 지수, "
        "진짜 B&H는 cash-drag 편향 참고용.",
        "",
        "| 비용 | 전략 CAGR | 전략 MDD | 전략 Sharpe | EW CAGR | EW MDD | EW Sharpe | "
        "전략−EW Sharpe | (참고)진짜B&H Sharpe |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in test_rows:
        lines.append(
            f"| x{r['cost_mult']:g} | {_f(r['cagr'], True)} | {_f(r['mdd'], True)} | "
            f"{_f(r['sharpe'])} | {_f(r['ew_cagr'], True)} | {_f(r['ew_mdd'], True)} | "
            f"{_f(r['ew_sharpe'])} | **{_f(r['sharpe_vs_ew'])}** | "
            f"{_f(r['bh_sharpe'])} |")
    test_ok = (test1["sharpe_vs_ew"] is not None and test1["sharpe_vs_ew"] > 0)
    test_dd = (test1["mdd"] is not None and test2["mdd"] is not None)
    lines += [
        "",
        "### 판정",
        f"- 위험조정(test, 기본비용): 전략 Sharpe {_f(test1['sharpe'])} vs "
        f"EW {_f(test1['ew_sharpe'])} → "
        + ("전략 우위 ✅" if test_ok else "EW 우위 ⚠️"),
        f"- 낙폭(test): 전략 MDD {_f(test1['mdd'], True)} vs EW {_f(test1['ew_mdd'], True)}"
        + (" → 전략이 낙폭 방어" if (test_dd and test1['mdd'] > test1['ew_mdd']) else ""),
        f"- 비용 2배에도 test Sharpe {_f(test2['sharpe'])} (EW {_f(test2['ew_sharpe'])}) "
        + ("유지" if (test2['sharpe_vs_ew'] is not None and test2['sharpe_vs_ew'] > 0)
           else "열위"),
        "",
        "*결론은 본문 표 수치로 판단할 것. 이 파일은 자동 생성됨.*",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
