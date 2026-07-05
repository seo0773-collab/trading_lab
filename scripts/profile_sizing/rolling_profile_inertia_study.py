"""롤링 프로파일 '신규 질량 관성' 이벤트 스터디.

가설: 최근 들어온 질량(거래량 가중 멀티플)의 드리프트에 관성이 있어, 향후 7일
신규 질량의 위치를 랜덤워크("현재 가격 부근에 쌓인다")보다 잘 예측하고(판정 A),
그 드리프트가 7일 선행수익률 기대값을 갖는다(판정 B).

정의 (멀티플 공간, 전략과 동일: m = close / SMA200):
  - c_k(t)      : 최근 k봉 신규 질량 중심 = 거래량 가중 m 평균
  - v_k(t)      : 드리프트 속도 = (c_k(t) - c_k(t-k)) / k   [멀티플/일]
  - c_fut(t)    : 실현된 미래 신규 질량 중심 = t+1..t+7 거래량 가중 m 평균
  - 관성 예측    : m(t) + 4·v_k  (등속 드리프트 시 7봉 중심의 기대 위치)
  - RW 예측     : m(t)

판정 A: MAE(관성) < MAE(RW) 승률·개선율, 방향 적중률.
판정 B: IC = corr(v_k, 7일 선행수익률) — Pearson/Spearman, 비중복 샘플, 구간별,
        5분위 평균 선행수익률.

사용:  PYTHONPATH=src python scripts/profile_sizing/rolling_profile_inertia_study.py
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

SMA_LEN = 200
HORIZON = 7
KS = [5, 10, 21]
SYMBOLS = ["SPY", "QQQ"]


def vw_mean(x: pd.Series, w: pd.Series, k: int) -> pd.Series:
    """k봉 거래량 가중 평균 (거래량 0/결측이면 단순평균으로 폴백)."""
    num = (x * w).rolling(k).sum()
    den = w.rolling(k).sum()
    out = num / den
    return out.where(den > 0, x.rolling(k).mean())


def study(symbol: str) -> dict:
    df = load_yfinance(symbol, "1d", "max").dropna(subset=["close"])
    close, vol = df["close"], df["volume"].fillna(0.0)
    sma = close.rolling(SMA_LEN).mean()
    m = (close / sma).rename("m")

    # 실현된 미래 신규 질량 중심(t+1..t+HORIZON) — 평가용, 신호에는 미사용
    c_fut = vw_mean(m, vol, HORIZON).shift(-HORIZON)
    fwd_ret = close.shift(-HORIZON) / close - 1.0

    res: dict = {"symbol": symbol, "bars": int(m.notna().sum()),
                 "period": f"{df.index[0].date()}~{df.index[-1].date()}", "ks": {}}
    for k in KS:
        ck = vw_mean(m, vol, k)
        v = (ck - ck.shift(k)) / k
        pred_in = m + 4.0 * v            # 등속 드리프트: 다음 7봉 중심 기대 위치
        pred_rw = m

        d = pd.DataFrame({"m": m, "v": v, "c_fut": c_fut, "fwd": fwd_ret,
                          "in": pred_in, "rw": pred_rw}).dropna()
        # ---- 판정 A: 예측력 ----
        ae_in = (d["c_fut"] - d["in"]).abs()
        ae_rw = (d["c_fut"] - d["rw"]).abs()
        move = d["c_fut"] - d["m"]
        strong = d["v"].abs() > d["v"].abs().median()
        a = {
            "mae_inertia": float(ae_in.mean()),
            "mae_rw": float(ae_rw.mean()),
            "mae_improve_pct": float(1 - ae_in.mean() / ae_rw.mean()) * 100,
            "win_rate": float((ae_in < ae_rw).mean()),
            "dir_hit": float((np.sign(move) == np.sign(d["v"])).mean()),
            "dir_hit_strong": float((np.sign(move[strong]) == np.sign(d["v"][strong])).mean()),
        }
        # ---- 판정 B: 경제성 ----
        nol = d.iloc[::HORIZON]          # 비중복 샘플
        thirds = np.array_split(d.index, 3)
        b = {
            "ic_pearson": float(d["v"].corr(d["fwd"])),
            "ic_spearman": float(d["v"].corr(d["fwd"], method="spearman")),
            "ic_spearman_nonoverlap": float(nol["v"].corr(nol["fwd"], method="spearman")),
            "ic_spearman_thirds": [
                float(d.loc[ix, "v"].corr(d.loc[ix, "fwd"], method="spearman"))
                for ix in thirds],
        }
        q = pd.qcut(d["v"], 5, labels=False, duplicates="drop")
        b["quintile_fwd_bps"] = [float(d["fwd"][q == i].mean() * 1e4) for i in range(5)]
        b["quintile_n"] = [int((q == i).sum()) for i in range(5)]
        res["ks"][k] = {"n": int(len(d)), "A": a, "B": b}
    return res


def main() -> None:
    out = {"sma_len": SMA_LEN, "horizon": HORIZON, "results": []}
    for sym in SYMBOLS:
        r = study(sym)
        out["results"].append(r)
        print(f"\n===== {sym} · {r['period']} =====")
        for k, rr in r["ks"].items():
            a, b = rr["A"], rr["B"]
            print(f"[k={k:2d}] n={rr['n']}")
            print(f"  A. MAE 관성 {a['mae_inertia']:.5f} vs RW {a['mae_rw']:.5f} "
                  f"(개선 {a['mae_improve_pct']:+.1f}%, 승률 {a['win_rate']*100:.1f}%) · "
                  f"방향적중 {a['dir_hit']*100:.1f}% (강신호 {a['dir_hit_strong']*100:.1f}%)")
            print(f"  B. IC(스피어먼) {b['ic_spearman']:+.4f} · 비중복 {b['ic_spearman_nonoverlap']:+.4f} · "
                  f"1/3구간 {['%+.3f' % x for x in b['ic_spearman_thirds']]}")
            print(f"     5분위 7일 선행수익(bps): {['%.0f' % x for x in b['quintile_fwd_bps']]}")

    path = ROOT / "reports" / "profile_sizing" / "rolling_profile_inertia_study.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nJSON: {path}")


if __name__ == "__main__":
    main()
