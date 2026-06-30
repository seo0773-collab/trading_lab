"""yoon2a 워크포워드: 벤치 대비 우위가 여러 독립 구간에서 반복되는가(일반화).

부트스트랩(yoon2a_bootstrap.py)은 '본 데이터 안에서 우위가 운인가'를 봤다. 워크포워드는
한 발 더 나가 '구간을 갈라도(=서로 다른 시기) 우위가 반복되는가'를 본다. yoon2a는
프로덕션 학습 파라미터가 없어(config 고정·무누수) 모든 구간이 사실상 OOS이므로,
핵심 질문은 **"각 구간에서 50/50 벤치를 이기는 비율"**이다.

  (1) 연도별(비중복) — 독립성 깨끗. 각 연도 CAGR/MDD/Sharpe vs 벤치 승패.
  (2) 롤링 3년(1년 스텝, 중복) — 구간 선택 민감도 완화 추세.

전 일봉 NAV를 한 번 만들고(yoon2a_bootstrap.build_series 재사용) 구간별로 잘라
재정규화해 측정. CAGR=경과시간 기반, Sharpe=일봉 252 연율화, MDD=구간 재정규화 기준.
(walk_forward.py 의 perf 규약과 동일)

사용:  PYTHONPATH=src python scripts/profile_sizing/yoon2a_walkforward.py
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

from profile_sizing.yoon2a_bootstrap import build_series  # noqa: E402

ROLL_LEN = 3  # 롤링 윈도 길이(년)


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
    return f"{v*100:6.1f}%" if pct else f"{v:6.3f}"


def _mark(a, b, shallower=False) -> str:
    """a(yoon2a)가 b(벤치)를 이기면 ✓. shallower=True 면 덜 음수(MDD 방어)가 승."""
    if a is None or b is None:
        return " "
    win = (a > b) if not shallower else (a > b)  # MDD는 음수라 a>b == 덜 깊음
    return "✓" if win else "·"


def windows(idx: pd.DatetimeIndex):
    years = sorted({d.year for d in idx})
    yearly = [(y, idx[(idx.year == y)]) for y in years]
    rolling = []
    for s in range(years[0], years[-1] - ROLL_LEN + 2):
        e = s + ROLL_LEN - 1
        w = idx[(idx.year >= s) & (idx.year <= e)]
        rolling.append((f"{s}-{e}", w))
    return yearly, rolling


def run(nav, bench, idx, label_pairs, title) -> tuple[list, dict]:
    print(f"\n=== {title} ===")
    print(f"{'구간':>9} | {'yoon2a CAGR/MDD/Shp':>26} | {'벤치 CAGR/MDD/Shp':>26} | win C/M/S")
    rows, wins = [], {"cagr": 0, "mdd": 0, "sharpe": 0, "n": 0}
    for lab, w in label_pairs:
        pa, pb = perf(nav.reindex(w)), perf(bench.reindex(w))
        if pa is None or pb is None:
            continue
        wc = _mark(pa["cagr"], pb["cagr"])
        wm = _mark(pa["mdd"], pb["mdd"], shallower=True)
        ws = _mark(pa["sharpe"], pb["sharpe"])
        wins["cagr"] += wc == "✓"; wins["mdd"] += wm == "✓"
        wins["sharpe"] += ws == "✓"; wins["n"] += 1
        print(f"{lab:>9} | {_f(pa['cagr'],1)} {_f(pa['mdd'],1)} {_f(pa['sharpe'])} | "
              f"{_f(pb['cagr'],1)} {_f(pb['mdd'],1)} {_f(pb['sharpe'])} |  {wc} {wm} {ws}")
        rows.append({"window": lab, "yoon2a": pa, "bench": pb})
    n = wins["n"]
    print(f"\n  승률(yoon2a가 벤치 우위):  CAGR {wins['cagr']}/{n}  "
          f"MDD방어 {wins['mdd']}/{n}  Sharpe {wins['sharpe']}/{n}")
    return rows, wins


def main() -> None:
    nav, bench, idx = build_series()
    print(f"기간 {idx[0].date()}~{idx[-1].date()} · {len(idx)}봉 · 롤링 {ROLL_LEN}년")
    yearly, rolling = windows(idx)
    yr_rows, yr_win = run(nav, bench, idx, yearly, "연도별 (비중복)")
    rl_rows, rl_win = run(nav, bench, idx, rolling, f"롤링 {ROLL_LEN}년 (1년 스텝)")

    out = ROOT / "reports" / "profile_sizing" / "yoon2a_walkforward.json"
    out.write_text(json.dumps({
        "period": f"{idx[0].date()}~{idx[-1].date()}", "roll_len": ROLL_LEN,
        "yearly": {"rows": yr_rows, "winrate": yr_win},
        "rolling": {"rows": rl_rows, "winrate": rl_win},
    }, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"\nJSON: {out}")


if __name__ == "__main__":
    main()
