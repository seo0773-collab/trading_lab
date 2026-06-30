"""Phase 0 연구 게이트: yoon2 코어의 고정 50/50 vs EWY/SPY 상대강도 동적 틸트.

전략/registry/config를 건드리지 않고 임시 계산한다. yoon2 핸들러를 재사용해
KRW-base 슬리브 NAV(미국 yoon1j_lp / 한국 yoon1j_kr)를 얻은 뒤, 월간 리밸런싱에서
목표비중을 (1) 고정 50/50, (2) EWY/SPY 상대모멘텀 틸트(T1), (3) 자기 슬리브 NAV
상대모멘텀 틸트(T2, basis 제거)로 바꿔 성과를 비교한다.

판정: train(0.6)에서 lookback/k를 고르고 validation(0.2)에서 재확인.
       train·validation 둘 다에서 Sharpe와 MDD를 동시에 개선할 때만 통과.

사용:  PYTHONPATH=src python scripts/profile_sizing/yoon2b_core_tilt_compare.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
for p in (ROOT / "src", SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from profile_sizing.engine import performance  # noqa: E402
from profile_sizing.portfolio import rebalance_dates  # noqa: E402
from run_kalman_pipeline import load_yfinance  # noqa: E402
from trading_lab.strategies.yoon1k import Yoon1kHandler  # noqa: E402

INTERVAL = "1d"
BAND = (0.30, 0.70)          # w_kr 허용 밴드
REBAL = "monthly"
TRAIN_FRAC, VAL_FRAC = 0.6, 0.2
GRID_L = [126, 252]          # 룩백(6M, 12M)
GRID_K = [0.10, 0.20, 0.30]  # 민감도
Z_WINDOW = 252               # 모멘텀 z-score 윈도


def load_core_sleeves() -> tuple[pd.Series, pd.Series]:
    """yoon2 코어의 KRW-base 슬리브 NAV(미국, 한국)를 반환."""
    cfg = json.loads((ROOT / "configs" / "strategies" / "yoon2.json").read_text("utf-8"))
    h = Yoon1kHandler()
    raw = h.load_data("SYNTH", cfg, synthetic=False)
    art = h.build_artifacts(raw, cfg, symbol="SYNTH", phase="all", bars_per_year=252)
    fc = art.forecast
    us = fc["sleeve_yoon1j_lp"].dropna()
    kr = fc["sleeve_yoon1j_kr"].dropna()
    idx = us.index.intersection(kr.index).sort_values()
    return us.reindex(idx), kr.reindex(idx)


def zscore_momentum(ratio: pd.Series, lookback: int) -> pd.Series:
    """ratio의 lookback 모멘텀을 롤링 z-score로. PIT(과거만)."""
    mom = ratio / ratio.shift(lookback) - 1.0
    mu = mom.rolling(Z_WINDOW, min_periods=Z_WINDOW // 2).mean()
    sd = mom.rolling(Z_WINDOW, min_periods=Z_WINDOW // 2).std(ddof=0)
    return ((mom - mu) / sd.replace(0.0, np.nan))


def target_wkr(signal_z: pd.Series, k: float) -> pd.Series:
    """z-score → 한국 비중 목표. 50% ± tanh 틸트, 밴드 클립. 1일 시프트(룩어헤드 방지)."""
    raw = 0.5 + k * np.tanh(signal_z)
    return raw.clip(*BAND).shift(1)


def combine_dynamic(us: pd.Series, kr: pd.Series, wkr: pd.Series | None) -> pd.Series:
    """월간 리밸런싱 동적 비중 결합 → 총 NAV(1.0 기준).
    wkr=None이면 고정 50/50. wkr는 리밸런스일에 적용할 한국 목표비중."""
    idx = us.index
    reb = rebalance_dates(idx, REBAL)
    r_us = us.pct_change().fillna(0.0).to_numpy()
    r_kr = kr.pct_change().fillna(0.0).to_numpy()
    w = 0.5 if wkr is None else wkr.reindex(idx)
    v_us, v_kr = 0.5, 0.5
    out = np.empty(len(idx))
    for i, t in enumerate(idx):
        v_us *= 1.0 + r_us[i]
        v_kr *= 1.0 + r_kr[i]
        out[i] = v_us + v_kr
        if t in reb:
            tot = v_us + v_kr
            tw = 0.5 if wkr is None else w.iloc[i]
            if not np.isfinite(tw):
                tw = 0.5
            v_kr = tot * tw
            v_us = tot * (1.0 - tw)
    return pd.Series(out, index=idx)


def slice_phase(idx: pd.DatetimeIndex, phase: str) -> pd.DatetimeIndex:
    n = len(idx)
    te, ve = int(n * TRAIN_FRAC), int(n * (TRAIN_FRAC + VAL_FRAC))
    return {"train": idx[:te], "validation": idx[te:ve], "test": idx[ve:], "all": idx}[phase]


def perf_on(nav: pd.Series, window: pd.DatetimeIndex) -> dict:
    eq = nav.reindex(window).dropna()
    if eq.empty:
        return {}
    eq = eq / eq.iloc[0]
    return performance(eq, eq.pct_change().fillna(0.0), INTERVAL)


def fmt(p: dict) -> str:
    if not p:
        return "  (no data)"
    return (f"CAGR {p['cagr']*100:6.2f}% | MDD {p['max_drawdown']*100:7.2f}% | "
            f"Sharpe {p['sharpe']:.3f} | TR {p['total_return']*100:8.1f}%")


def main() -> None:
    print("코어 슬리브 로딩(yoon2: yoon1j_lp/yoon1j_kr, KRW-base)...")
    us, kr = load_core_sleeves()
    idx = us.index
    print(f"  기간 {idx[0].date()}~{idx[-1].date()}  {len(idx)}봉")

    # 신호 ratio: T1=EWY/SPY(둘 다 USD, FX중립이나 EWY는 원화효과 내포), T2=kr/us NAV
    print("EWY/SPY 로딩...")
    ewy = load_yfinance("EWY", INTERVAL, "max")["close"].reindex(idx).ffill()
    spy = load_yfinance("SPY", INTERVAL, "max")["close"].reindex(idx).ffill()
    ratio_t1 = (ewy / spy).dropna()
    ratio_t2 = (kr / us).dropna()

    base_nav = combine_dynamic(us, kr, None)
    phases = ["train", "validation", "test", "all"]

    def eval_variant(ratio: pd.Series, l: int, k: float) -> pd.Series:
        z = zscore_momentum(ratio.reindex(idx).ffill(), l)
        wkr = target_wkr(z, k)
        return combine_dynamic(us, kr, wkr)

    print("\n=== 베이스(고정 50/50) ===")
    base_perf = {ph: perf_on(base_nav, slice_phase(idx, ph)) for ph in phases}
    for ph in phases:
        print(f"  {ph:11s} {fmt(base_perf[ph])}")

    rows = []
    for name, ratio in (("T1_ewy_spy", ratio_t1), ("T2_sleeve_nav", ratio_t2)):
        print(f"\n=== {name}: train 그리드 탐색 (Sharpe↑ & MDD≤base) ===")
        tr_win = slice_phase(idx, "train")
        base_tr = base_perf["train"]
        best = None
        for l in GRID_L:
            for k in GRID_K:
                nav = eval_variant(ratio, l, k)
                p = perf_on(nav, tr_win)
                if not p or p.get("sharpe") is None:
                    continue
                ok = (p["sharpe"] > base_tr["sharpe"]
                      and p["max_drawdown"] >= base_tr["max_drawdown"])
                flag = "OK" if ok else "  "
                print(f"  L={l:3d} k={k:.2f} [{flag}] {fmt(p)}")
                score = p["sharpe"] if ok else -1e9
                if best is None or score > best[0]:
                    best = (score, l, k, ok)
        if best is None:
            print("  유효 결과 없음")
            continue
        _, l, k, ok = best
        nav = eval_variant(ratio, l, k)
        print(f"\n  >> train 선택: L={l}, k={k} (게이트 {'통과' if ok else '미통과'})")
        for ph in phases:
            p = perf_on(nav, slice_phase(idx, ph))
            print(f"     {ph:11s} {fmt(p)}")
            bp = base_perf[ph]
            rows.append({
                "variant": name, "L": l, "k": k, "phase": ph,
                "cagr": p.get("cagr"), "mdd": p.get("max_drawdown"),
                "sharpe": p.get("sharpe"), "total_return": p.get("total_return"),
                "base_cagr": bp.get("cagr"), "base_mdd": bp.get("max_drawdown"),
                "base_sharpe": bp.get("sharpe"),
            })

    # 베이스도 csv에
    for ph in phases:
        bp = base_perf[ph]
        rows.append({"variant": "base_5050", "L": None, "k": None, "phase": ph,
                     "cagr": bp.get("cagr"), "mdd": bp.get("max_drawdown"),
                     "sharpe": bp.get("sharpe"), "total_return": bp.get("total_return"),
                     "base_cagr": bp.get("cagr"), "base_mdd": bp.get("max_drawdown"),
                     "base_sharpe": bp.get("sharpe")})

    out = ROOT / "reports" / "profile_sizing" / "yoon2b_core_tilt_compare.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nCSV: {out}")


if __name__ == "__main__":
    main()
