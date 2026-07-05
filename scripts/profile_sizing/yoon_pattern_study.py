"""yoon_pattern 가설 이벤트 스터디: 300틱 패턴 투영의 예상수익 점수 IC 비교.

가설: 직전 L틱(기본 300 일봉) 패턴을 미래에 투영해 "투영이 가리키는 H틱 예상수익"을
점수로 쓰면 선행수익률 기대값이 있다. 투영 두 방식을 무누수로 만들어 비교한다.

  (a) analog — 역사적 유사 패턴 앙상블: 현재 L틱 창(z-정규화 로그가격, step 서브샘플)과
      과거 창들의 피어슨 유사도 top-M(중복 방지 간격 gap)을 찾아, 그 창들 '직후 H틱'
      실현수익의 중앙값을 점수로 사용. 후보는 s+H ≤ t (직후 경로가 t 이전에 완결)만
      허용해 무누수.
  (b) self — 자기 패턴 연장: 직전 L틱 패턴이 반복된다고 가정, 점수 = 패턴 첫 H틱
      구간의 수익 = logp(t−L+H) − logp(t−L). (주기 L 반복 가정의 문자적 구현)

판정: score와 t→t+H 선행수익의 IC(스피어먼·비중복·1/3구간)·5분위. IC가 유의한
방식/파라미터로 yoon_pattern 전략을 확정한다.

사용:  PYTHONPATH=src python scripts/profile_sizing/yoon_pattern_study.py
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

from run_kalman_pipeline import load_yfinance  # noqa: E402

SYMBOLS = ["SPY", "QQQ"]
LS = [300, 150]
HS = [5, 21]
STEP = 5          # 패턴 서브샘플 간격(속도)
M = 50            # 앙상블 매치 수
BLOCK = 512       # 유사도 행렬 블록 크기(메모리)


def pattern_matrix(logp: np.ndarray, L: int, step: int) -> np.ndarray:
    """행 t = [t−L+step, ..., t] 서브샘플 로그가격의 z-정규화 벡터 (미달 행 NaN)."""
    n = len(logp)
    k = L // step
    offs = np.arange(k) * step - (k - 1) * step   # ... -2*step, -step, 0
    G = np.full((n, k), np.nan, dtype=np.float32)
    for j, o in enumerate(offs):
        lo = -o
        G[lo:, j] = logp[: n + o] if o < 0 else logp
    mu = np.nanmean(G, axis=1, keepdims=True)
    sd = np.nanstd(G, axis=1, keepdims=True)
    G = (G - mu) / np.where(sd > 1e-12, sd, np.nan)
    G[np.isnan(G)] = np.nan
    return G


def analog_scores(logp: np.ndarray, L: int, H: int) -> np.ndarray:
    """무누수 k-NN 앙상블 점수: top-M 유사 창 직후 H틱 수익 중앙값."""
    n = len(logp)
    G = pattern_matrix(logp, L, STEP)
    valid = ~np.isnan(G).any(axis=1)
    k = G.shape[1]
    Gz = np.where(valid[:, None], G, 0.0).astype(np.float32)
    fwd = np.full(n, np.nan)
    fwd[: n - H] = logp[H:] - logp[: n - H]       # s 직후 H틱 로그수익
    scores = np.full(n, np.nan)
    gap = max(H, 5)
    topn = M * 4                                   # 간격 필터 여유분
    for b0 in range(0, n, BLOCK):
        b1 = min(b0 + BLOCK, n)
        S = Gz[b0:b1] @ Gz.T / k                   # (blk × n) 피어슨 유사도
        for t in range(b0, b1):
            if not valid[t]:
                continue
            smax = t - H                           # s+H ≤ t (무누수)
            if smax < L:
                continue
            row = S[t - b0, :smax + 1].copy()
            row[~valid[:smax + 1]] = -9.9
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
                    if len(picked) >= M:
                        break
            if len(picked) >= 10:
                scores[t] = float(np.median(fwd[picked]))
    return scores


def self_scores(logp: np.ndarray, L: int, H: int) -> np.ndarray:
    """자기 패턴 연장 점수: 직전 L틱 패턴의 첫 H틱 수익(주기 L 반복 가정)."""
    n = len(logp)
    s = np.full(n, np.nan)
    s[L:] = logp[H : n - L + H] - logp[: n - L]
    return s


def evaluate(score: np.ndarray, fwd_ret: np.ndarray, H: int) -> dict:
    d = pd.DataFrame({"s": score, "f": fwd_ret}).dropna()
    if len(d) < 500:
        return {"n": int(len(d))}
    nol = d.iloc[::H]
    thirds = np.array_split(d.index, 3)
    q = pd.qcut(d["s"], 5, labels=False, duplicates="drop")
    return {
        "n": int(len(d)),
        "ic_spearman": float(d["s"].corr(d["f"], method="spearman")),
        "ic_nonoverlap": float(nol["s"].corr(nol["f"], method="spearman")),
        "ic_thirds": [float(d.loc[ix, "s"].corr(d.loc[ix, "f"], method="spearman"))
                      for ix in thirds],
        "quintile_bps": [float(d["f"][q == i].mean() * 1e4) for i in range(5)],
    }


def main() -> None:
    out = {"step": STEP, "m": M, "results": []}
    for sym in SYMBOLS:
        close = load_yfinance(sym, "1d", "max")["close"].dropna()
        logp = np.log(close.to_numpy(dtype=np.float64))
        n = len(logp)
        print(f"\n===== {sym} · {close.index[0].date()}~{close.index[-1].date()} ({n}봉) =====")
        for L in LS:
            for H in HS:
                fwd_future = np.full(n, np.nan)
                fwd_future[: n - H] = logp[H:] - logp[: n - H]
                res_a = evaluate(analog_scores(logp, L, H), fwd_future, H)
                res_s = evaluate(self_scores(logp, L, H), fwd_future, H)
                for name, r in (("analog", res_a), ("self", res_s)):
                    if "ic_spearman" not in r:
                        print(f"[L={L} H={H}] {name:6s} 표본 부족")
                        continue
                    print(f"[L={L:3d} H={H:2d}] {name:6s} IC {r['ic_spearman']:+.4f} · "
                          f"비중복 {r['ic_nonoverlap']:+.4f} · "
                          f"1/3 {['%+.3f' % x for x in r['ic_thirds']]} · "
                          f"5분위(bps) {['%.0f' % x for x in r['quintile_bps']]}")
                    out["results"].append({"symbol": sym, "L": L, "H": H,
                                           "method": name, **r})

    path = ROOT / "reports" / "profile_sizing" / "yoon_pattern_study.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nJSON: {path}")


if __name__ == "__main__":
    main()
