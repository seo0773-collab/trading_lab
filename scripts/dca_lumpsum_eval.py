"""일시투입 vs 적립식 평가 비교.

var/runs/<run>/equity.parquet(전략 NAV)와 benchmark.parquet(벤치마크 NAV)에
동일한 납입 스케줄(초기 2,000,000원 + 매월 첫 거래일 200,000원)을 적용해
전략/벤치마크의 평가가 투자 방식에 따라 달라지는지 본다.

이상적 근사: 정수주·수수료·적립금과 리밸런스의 상호작용은 반영하지 않는다.
사용법: PYTHONPATH=src .venv/bin/python scripts/dca_lumpsum_eval.py
출력: reports/dca_lumpsum_eval.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

RUNS = [
    ("yoon2m", "421_yoon2m_크립토_PORTFOLIO_260317"),
    ("yoon2m_defensive", "420_yoon2m_defensive_크립토_PORTFOLIO_260317"),
    ("yoon4", "392_yoon4_주식_GLOBAL_262816"),
    ("yoon_profile(BTC)", "422_yoon_profile_크립토_BTC_260400"),
    ("yoon_profile_1(BTC)", "423_yoon_profile_1_크립토_BTC_260400"),
    ("yoon_profile_2(BTC)", "424_yoon_profile_2_크립토_BTC_260400"),
    ("yoon_pattern(SPY)", "425_yoon_pattern_주식_SPY_260400"),
    ("rolling-profile-expectancy-v1(SPY)", "426_rolling-profile-expectancy-v1_주식_SPY_260400"),
    ("yoon_profile_1(SPY)", "427_yoon_profile_1_주식_SPY_260401"),
    ("yoon_profile_1(NVDA)", "428_yoon_profile_1_주식_NVDA_260401"),
    ("yoon_profile_1(QQQ)", "429_yoon_profile_1_주식_QQQ_260401"),
]

INITIAL = 2_000_000.0
MONTHLY = 200_000.0


def dca_path(nav: pd.Series) -> tuple[pd.Series, pd.Series]:
    """월초 적립식 계좌가치·누적원금 경로."""
    units = INITIAL / nav.iloc[0]
    principal = INITIAL
    firsts = set(nav.groupby([nav.index.year, nav.index.month]).head(1).index)
    values, principals = [], []
    for t, price in nav.items():
        if t in firsts and t != nav.index[0]:
            units += MONTHLY / price
            principal += MONTHLY
        values.append(units * price)
        principals.append(principal)
    return pd.Series(values, index=nav.index), pd.Series(principals, index=nav.index)


def max_drawdown(series: pd.Series) -> float:
    return float((series / series.cummax() - 1.0).min())


def main() -> None:
    rows = []
    for name, run in RUNS:
        path = ROOT / "var" / "runs" / run
        metrics = json.loads((path / "metrics.json").read_text())
        eq = pd.read_parquet(path / "equity.parquet").iloc[:, 0].dropna()
        bm = pd.read_parquet(path / "benchmark.parquet").iloc[:, 0].dropna()
        idx = eq.index.intersection(bm.index)
        eq = eq.loc[idx] / eq.loc[idx].iloc[0]
        bm = bm.loc[idx] / bm.loc[idx].iloc[0]

        acct_s, principal = dca_path(eq)
        acct_b, _ = dca_path(bm)
        dca_s = acct_s.iloc[-1] / principal.iloc[-1] - 1.0
        dca_b = acct_b.iloc[-1] / principal.iloc[-1] - 1.0

        rows.append({
            "strategy": name,
            "run": run,
            "start": idx[0].date(),
            "end": idx[-1].date(),
            "lump_strategy": eq.iloc[-1] - 1.0,
            "lump_bench": bm.iloc[-1] - 1.0,
            "lump_ratio": eq.iloc[-1] / bm.iloc[-1],
            "lump_lead_days": float((eq > bm).mean()),
            "dca_strategy": dca_s,
            "dca_bench": dca_b,
            "dca_ratio": acct_s.iloc[-1] / acct_b.iloc[-1],
            "dca_lead_days": float((acct_s > acct_b).mean()),
            "nav_mdd_strategy": max_drawdown(eq),
            "nav_mdd_bench": max_drawdown(bm),
            "dca_acct_mdd_strategy": max_drawdown(acct_s),
            "dca_acct_mdd_bench": max_drawdown(acct_b),
            "lump_ret_over_mdd_strategy": (eq.iloc[-1] - 1.0) / abs(max_drawdown(eq)),
            "lump_ret_over_mdd_bench": (bm.iloc[-1] - 1.0) / abs(max_drawdown(bm)),
            "dca_ret_over_mdd_strategy": dca_s / abs(max_drawdown(acct_s)),
            "dca_ret_over_mdd_bench": dca_b / abs(max_drawdown(acct_b)),
            "sharpe": metrics.get("sharpe"),
            "trades": metrics.get("trades"),
        })

    df = pd.DataFrame(rows)
    out = ROOT / "reports" / "dca_lumpsum_eval.csv"
    df.to_csv(out, index=False)
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(df[["strategy", "start", "end", "lump_ratio", "dca_ratio",
                  "lump_lead_days", "dca_lead_days",
                  "dca_ret_over_mdd_strategy", "dca_ret_over_mdd_bench"]]
              .to_string(index=False))
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
