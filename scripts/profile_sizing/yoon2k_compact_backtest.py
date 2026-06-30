"""yoon2k(kompact) 견적 백테스트: 200만원 정수주 자동매수용 yoon2a 파생.

문제: yoon2a는 17종목(위성 1~3%)이라 200만원 정수주로는 10종목이 0주가 되고 현금
45%가 놀아 전략이 깨진다. 원인은 top_k(보유종목수)가 커서 종목당 비중이 잘게 쪼개지는 것.

설계: yoon2a 구조(US 0.37 / KR 0.37 / SSO 0.26, 월간 리밸런스)는 유지하되, 두
yoon1j 슬리브의 top_k를 줄여 **종목당 비중을 키운다**. 추세게이트 방어(market_filter +
safe_sleeve + regime_cap)는 그대로라 yoon2의 낙폭방어 메커니즘을 보존한다. 여기서
top_k 1/2/3을 yoon2a(기본)·패시브벤치와 비교해 어느 농도가 방어를 유지하며 정수주
적합한지 본다.

사용:  PYTHONPATH=src python scripts/profile_sizing/yoon2k_compact_backtest.py
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

from profile_sizing.engine import performance  # noqa: E402
from profile_sizing.portfolio import rebalance_dates  # noqa: E402
from trading_lab.strategies.profile_portfolio import ProfilePortfolioHandler  # noqa: E402
from profile_sizing.yoon2a_bootstrap import build_series, to_base  # noqa: E402
from run_kalman_pipeline import load_yfinance  # noqa: E402

# yoon2a 평탄화 비중 (US 코어 / KR 코어 / SSO)
W_US, W_KR, W_SSO = 0.37, 0.37, 0.26


def sleeve_equity(cfg_id: str, top_k: int) -> pd.Series:
    cfg = dict(json.loads((ROOT / "configs" / "strategies" / f"{cfg_id}.json").read_text("utf-8")))
    cfg["top_k"] = top_k
    h = ProfilePortfolioHandler()
    raw = h.load_data("SYNTH", cfg, synthetic=False)
    art = h.build_artifacts(raw, cfg, symbol="SYNTH", phase="all", bars_per_year=252)
    return art.equity


def combine_w(navs: list[pd.Series], weights: list[float], idx, freq="monthly") -> pd.Series:
    reb = rebalance_dates(idx, freq)
    rets = [n.reindex(idx).ffill().pct_change().fillna(0.0).to_numpy() for n in navs]
    v = list(weights)
    out = np.empty(len(idx))
    for i, t in enumerate(idx):
        for j in range(len(v)):
            v[j] *= 1.0 + rets[j][i]
        tot = sum(v)
        out[i] = tot
        if t in reb:
            v = [tot * w for w in weights]
    return pd.Series(out, index=idx)


def metr(eq: pd.Series) -> dict:
    eq = (eq / eq.iloc[0]).dropna()
    p = performance(eq, eq.pct_change().fillna(0.0), "1d")
    return {"cagr": p["cagr"], "mdd": p["max_drawdown"], "sharpe": p["sharpe"],
            "tot": (eq.iloc[-1] - 1.0)}


def main() -> None:
    # 공통 자원: yoon2a(기본)·패시브벤치(50/50)·SSO(원화환산)
    yoon2a, bench, base_idx = build_series()
    sso_close = load_yfinance("SSO", "1d", "max")["close"].dropna()
    fx = load_yfinance("KRW=X", "1d", "max")["close"]

    # 슬리브(top_k 변형) 사전 계산
    variants = {1: None, 2: None, 3: None}
    sleeves = {}
    for k in variants:
        sleeves[k] = (sleeve_equity("yoon1j_lp", k), sleeve_equity("yoon1j_kr", k))

    # 모든 변형 + 기준이 공유할 공통 인덱스
    idx = base_idx
    for lp, kr in sleeves.values():
        idx = idx.intersection(lp.index).intersection(kr.index)
    idx = idx.intersection(sso_close.index).sort_values()

    sso = to_base((sso_close.reindex(idx).ffill()).pipe(lambda s: s / s.iloc[0]), fx)
    rows = []

    def add(name, eq):
        m = metr(eq.reindex(idx))
        rows.append((name, m))

    add("패시브 50/50 벤치", bench)
    add("yoon2a (기본 top_k 11/5)", yoon2a)
    for k, (lp, kr) in sleeves.items():
        us_krw = to_base((lp.reindex(idx).ffill()).pipe(lambda s: s / s.iloc[0]), fx)
        kr_n = (kr.reindex(idx).ffill()); kr_n = kr_n / kr_n.iloc[0]
        nav = combine_w([us_krw, kr_n, sso], [W_US, W_KR, W_SSO], idx)
        add(f"yoon2k top_k={k} (+SSO)", nav)

    print(f"기간 {idx[0].date()}~{idx[-1].date()} · {len(idx)}봉\n")
    print(f"{'전략':28s}{'CAGR':>8}{'MDD':>9}{'Sharpe':>8}{'총수익':>10}")
    for name, m in rows:
        print(f"{name:28s}{m['cagr']*100:7.2f}%{m['mdd']*100:8.2f}%{m['sharpe']:8.3f}{m['tot']*100:9.1f}%")

    out = ROOT / "reports" / "profile_sizing" / "yoon2k_compact_backtest.json"
    out.write_text(json.dumps(
        {"period": f"{idx[0].date()}~{idx[-1].date()}", "bars": len(idx),
         "weights": {"US": W_US, "KR": W_KR, "SSO": W_SSO},
         "rows": [{"name": n, **m} for n, m in rows]},
        ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"\nJSON: {out}")


if __name__ == "__main__":
    main()
