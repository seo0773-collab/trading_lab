"""yoon_pattern L×M 스윕 + MDD 축소 오버레이 실측.

1부 — "최적 몇 틱": 패턴 길이 L{150,300,450,600} × 매치 수 M{20,50,100} (H=21 고정,
      유사도 행렬은 L당 1회 계산해 M을 한 패스로 파생). 랭킹은 validation Sharpe
      기준(홀드아웃 오염 방지), test는 최종 후보만 보고.

2부 — 기대값이 약할 때 MDD 축소: 최선 조합의 비중 위에 무누수 오버레이를 겹쳐 비교.
      (a) 시장필터: SPY<200MA(전봉)면 비중 ×0.5 — yoon1 계열 검증 메커니즘
      (b) 변동성 타게팅: 비중 × clip(목표 15% / 실현 21봉 연변동성, 0~1) (전봉)
      (c) a+b 결합

사용:  PYTHONPATH=src python scripts/profile_sizing/yoon_pattern_sweep.py
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

from profile_sizing.config import config_from_dict  # noqa: E402
from profile_sizing.engine import performance, portfolio_returns  # noqa: E402
from profile_sizing.pattern_projection import pattern_matrix  # noqa: E402
from profile_sizing.rolling_expectancy import (  # noqa: E402
    expectancy_weights, walk_forward_expectancy,
)
from profile_sizing.run import rebased_equity, slice_window  # noqa: E402
from run_kalman_pipeline import load_yfinance  # noqa: E402

SYMBOL = "SPY"
LS = [150, 300, 450, 600]
MS = [20, 50, 100]
H = 21
STEP = 5
BLOCK = 512


def analog_scores_multi(logp: np.ndarray, L: int, ms: list[int]) -> dict[int, np.ndarray]:
    """유사도 1회 계산으로 여러 M의 앙상블 점수를 동시 산출 (pattern_projection과 동일 규약)."""
    n = len(logp)
    G = pattern_matrix(logp, L, STEP)
    valid = ~np.isnan(G).any(axis=1)
    k = G.shape[1]
    Gz = np.where(valid[:, None], G, 0.0).astype(np.float32)
    fwd = np.full(n, np.nan)
    fwd[: n - H] = logp[H:] - logp[: n - H]
    m_max = max(ms)
    gap = max(H, 5)
    topn = m_max * 4
    out = {m: np.full(n, np.nan) for m in ms}
    for b0 in range(0, n, BLOCK):
        b1 = min(b0 + BLOCK, n)
        S = Gz[b0:b1] @ Gz.T / k
        for t in range(b0, b1):
            if not valid[t]:
                continue
            smax = t - H
            if smax < L:
                continue
            row = S[t - b0, : smax + 1].copy()
            row[~valid[: smax + 1]] = -9.9
            if smax + 1 <= topn:
                cand = np.argsort(row)[::-1]
            else:
                part = np.argpartition(row, -topn)[-topn:]
                cand = part[np.argsort(row[part])[::-1]]
            picked: list[int] = []
            for s in cand:
                if row[s] <= -9.0:
                    break
                if all(abs(int(s) - q) >= gap for q in picked):
                    picked.append(int(s))
                    if len(picked) >= m_max:
                        break
            for m in ms:
                if len(picked) >= max(10, m // 5):
                    out[m][t] = float(np.median(fwd[picked[:m]]))
    return out


def phases_perf(port_ret: pd.Series, idx: pd.DatetimeIndex, cfg) -> dict[str, dict]:
    res = {}
    for ph in ("all", "validation", "test"):
        win = slice_window(idx, ph, cfg)
        eq = rebased_equity(port_ret, win)
        p = performance(eq, eq.pct_change().fillna(0.0), "1d")
        res[ph] = {k: p[k] for k in ("cagr", "sharpe", "max_drawdown")}
    return res


def main() -> None:
    cfgd = json.loads((ROOT / "configs" / "strategies" / "yoon_pattern.json").read_text("utf-8"))
    cfg = config_from_dict(cfgd)
    raw = load_yfinance(SYMBOL, "1d", "max").dropna(subset=["close"])
    close = pd.Series(np.asarray(raw["close"], float), index=raw.index)
    idx = pd.DatetimeIndex(raw.index)
    logp = np.log(close.to_numpy())

    # ---- 1부: L×M 스윕 ----
    results = []
    weights_cache = {}
    for L in LS:
        multi = analog_scores_multi(logp, L, MS)
        for M in MS:
            feats = pd.DataFrame({"profile_score": multi[M]}, index=raw.index)
            exp, _ = walk_forward_expectancy(feats, close, cfg)
            w = expectancy_weights(exp["expectancy_bps"], cfg)["actual_weight"]
            pr = portfolio_returns(close, w, cfg)
            perf = phases_perf(pr, idx, cfg)
            weights_cache[(L, M)] = w
            results.append({"L": L, "M": M, "avg_w": float(w.mean()), **{
                f"{ph}_{k}": v for ph, d in perf.items() for k, v in d.items()}})
            r = results[-1]
            print(f"L={L:3d} M={M:3d}  all {r['all_cagr']*100:5.2f}%/{r['all_sharpe']:.3f}/"
                  f"{r['all_max_drawdown']*100:6.1f}%  val {r['validation_cagr']*100:5.2f}%/"
                  f"{r['validation_sharpe']:.3f}/{r['validation_max_drawdown']*100:6.1f}%  "
                  f"test {r['test_cagr']*100:5.2f}%/{r['test_sharpe']:.3f}/"
                  f"{r['test_max_drawdown']*100:6.1f}%  비중 {r['avg_w']*100:4.1f}%")

    best = max(results, key=lambda r: r["validation_sharpe"])
    L, M = best["L"], best["M"]
    print(f"\n>>> validation Sharpe 최선: L={L} M={M}")

    # ---- 2부: MDD 축소 오버레이 (최선 조합) ----
    w = weights_cache[(L, M)]
    ma200 = close.rolling(200, min_periods=200).mean()
    mf_off = (close <= ma200).shift(1).fillna(False)          # 전봉 신호(무누수)
    rv = close.pct_change().rolling(21).std().mul(np.sqrt(252)).shift(1)
    vt = (0.15 / rv).clip(upper=1.0).fillna(1.0)
    overlays = {
        "오버레이 없음": w,
        "(a) 시장필터 SPY<200MA→×0.5": w.where(~mf_off, w * 0.5),
        "(b) 변동성 타게팅 15%": w * vt,
        "(c) a+b 결합": w.where(~mf_off, w * 0.5) * vt,
    }
    print()
    ov_rows = []
    for name, wo in overlays.items():
        pr = portfolio_returns(close, wo, cfg)
        perf = phases_perf(pr, idx, cfg)
        ov_rows.append({"overlay": name, **{
            f"{ph}_{k}": v for ph, d in perf.items() for k, v in d.items()}})
        r = ov_rows[-1]
        print(f"{name:28s} all {r['all_cagr']*100:5.2f}%/{r['all_sharpe']:.3f}/"
              f"{r['all_max_drawdown']*100:6.1f}%  val {r['validation_cagr']*100:5.2f}%/"
              f"{r['validation_sharpe']:.3f}/{r['validation_max_drawdown']*100:6.1f}%  "
              f"test {r['test_cagr']*100:5.2f}%/{r['test_sharpe']:.3f}/"
              f"{r['test_max_drawdown']*100:6.1f}%")

    out = ROOT / "reports" / "profile_sizing" / "yoon_pattern_sweep.json"
    out.write_text(json.dumps({"symbol": SYMBOL, "h": H, "step": STEP,
                               "sweep": results, "best": {"L": L, "M": M},
                               "overlays": ov_rows},
                              ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"\nJSON: {out}")


if __name__ == "__main__":
    main()
