"""조건부 적립(하락 시점에만 투입) vs 정기 적립 vs 벤치.

앞선 검증(final0706_dca_vs_benchmark.py)에서 "정기 적립식은 이미 낙폭방어 효과를
공짜로 제공해서 final0706의 능동방어 가치를 깎는다"를 확인했다. 그렇다면 "적립은
정기적으로 하되, 실제 매수는 낙폭 구간에만 몰아서 한다"는 조건부 방식이면 어떨까 —
정기 매수분을 매달 즉시 태우지 않고 무이자 현금 버퍼에 쌓아두다가, 자산이 **자기
직전 고점 대비 threshold% 이상 하락한 달에만** 버퍼 전액을 투입한다. 트리거가 한동안
안 뜨면 버퍼가 계속 쌓인다(장기 미투입 리스크는 감수).

총 투입원금은 정기 적립과 완전히 동일(같은 월 예산, 타이밍만 재배치)하므로
total_return을 그대로 공정 비교할 수 있다.

Usage:
    PYTHONPATH=src:scripts python3 scripts/profile_sizing/final0706_conditional_dca_study.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from trading_lab.strategies.yoon1k import Yoon1kHandler  # noqa: E402
from trading_lab.strategies.yoon1k_leverage_mix import Yoon1kLeverageMixHandler  # noqa: E402

CONFIGS_DIR = ROOT / "configs" / "strategies"
INITIAL = 2_000_000.0
MONTHLY = 200_000.0
DD_THRESHOLDS = (0.05, 0.10, 0.15)  # 고점 대비 낙폭 트리거 후보


def plain_dca_path(nav: pd.Series) -> tuple[pd.Series, pd.Series]:
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


def conditional_dca_path(nav: pd.Series, dd_threshold: float) -> tuple[pd.Series, pd.Series, int]:
    """월 예산은 현금버퍼에 쌓고, 자기 고점 대비 낙폭이 threshold를 넘는 달에만
    버퍼 전액을 투입. 총 원금은 plain DCA와 동일(타이밍만 재배치)."""
    units = INITIAL / nav.iloc[0]
    principal = INITIAL
    buffer = 0.0
    peak = nav.iloc[0]
    firsts = set(nav.groupby([nav.index.year, nav.index.month]).head(1).index)
    values, principals = [], []
    n_deploys = 0
    for t, price in nav.items():
        peak = max(peak, price)
        dd = price / peak - 1.0
        if t in firsts and t != nav.index[0]:
            buffer += MONTHLY
            principal += MONTHLY
        if dd <= -dd_threshold and buffer > 0.0:
            units += buffer / price
            buffer = 0.0
            n_deploys += 1
        values.append(units * price + buffer)  # 버퍼는 현금(수익률 0%)으로 평가
        principals.append(principal)
    return pd.Series(values, index=nav.index), pd.Series(principals, index=nav.index), n_deploys


def max_drawdown(series: pd.Series) -> float:
    return float((series / series.cummax() - 1.0).min())


def main() -> None:
    final0706_cfg = json.loads((CONFIGS_DIR / "final0706.json").read_text("utf-8"))
    final0706_qld_cfg = json.loads((CONFIGS_DIR / "final0706_qld.json").read_text("utf-8"))
    k_handler = Yoon1kHandler()
    lev_handler = Yoon1kLeverageMixHandler()

    print("데이터 로딩 중...", flush=True)
    raw = k_handler.load_data("PORTFOLIO", final0706_cfg, synthetic=False)
    art = k_handler.build_artifacts(
        raw, final0706_cfg, symbol="PORTFOLIO", phase="all", bars_per_year=252)
    nav_strategy = art.equity
    nav_bench = art.benchmark

    raw_qld = lev_handler.load_data("PORTFOLIO", final0706_qld_cfg, synthetic=False)
    art_qld = lev_handler.build_artifacts(
        raw_qld, final0706_qld_cfg, symbol="PORTFOLIO", phase="all", bars_per_year=252)
    nav_qld = art_qld.equity

    idx = nav_strategy.index
    n = len(idx)
    train_frac = float(final0706_cfg.get("train_frac", 0.6))
    validation_frac = float(final0706_cfg.get("validation_frac", 0.2))
    v_end = int(n * (train_frac + validation_frac))
    phases = {
        "all": idx,
        "test": idx[v_end:],
        "recent_3y": idx[idx >= (idx[-1] - pd.Timedelta(days=365 * 3))],
    }

    candidates = [
        ("final0706", nav_strategy),
        ("final0706_qld", nav_qld),
        ("벤치(50/50)", nav_bench),
    ]

    rows = []
    for phase, window in phases.items():
        for label, nav in candidates:
            s = nav.reindex(window).ffill().dropna()
            if len(s) < 60:
                continue
            s = s / s.iloc[0]

            acct_p, principal = plain_dca_path(s)
            ret_p = float(acct_p.iloc[-1] / principal.iloc[-1] - 1.0)
            rows.append({
                "phase": phase, "candidate": label, "schedule": "정기(매달 즉시투입)",
                "total_return": ret_p, "account_mdd": max_drawdown(acct_p), "n_deploys": None,
            })

            for th in DD_THRESHOLDS:
                acct_c, principal_c, n_dep = conditional_dca_path(s, th)
                ret_c = float(acct_c.iloc[-1] / principal_c.iloc[-1] - 1.0)
                rows.append({
                    "phase": phase, "candidate": label,
                    "schedule": f"조건부(고점대비 -{int(th*100)}% 낙폭 시 투입)",
                    "total_return": ret_c, "account_mdd": max_drawdown(acct_c),
                    "n_deploys": n_dep,
                })

    out_json = ROOT / "reports" / "profile_sizing" / "final0706_conditional_dca_study.json"
    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    def pct(v):
        return f"{v * 100:.2f}%"

    lines = [
        "# 조건부 적립(낙폭 트리거) vs 정기 적립 vs 벤치",
        "",
        "- 정기: 매달 첫 거래일 200,000원 즉시 투입(기존 방식).",
        "- 조건부: 월 예산을 무이자 현금버퍼에 쌓다가, 자산이 **자기 고점 대비 -X% 낙폭**한 "
        "달에만 버퍼 전액 투입. 총 투입원금은 정기와 완전히 동일(타이밍만 재배치).",
        "- 초기 2,000,000원 + 매월 200,000원 예산, 정수주·수수료 미반영 근사.",
        "",
    ]
    for phase in phases:
        prows = [r for r in rows if r["phase"] == phase]
        if not prows:
            continue
        lines.append(f"## {phase}")
        lines.append("")
        lines.append("| candidate | schedule | 총수익률 | 계좌 MDD | 투입횟수 |")
        lines.append("| --- | --- | ---: | ---: | ---: |")
        for r in prows:
            dep = "-" if r["n_deploys"] is None else str(r["n_deploys"])
            lines.append(
                f"| {r['candidate']} | {r['schedule']} | {pct(r['total_return'])} | "
                f"{pct(r['account_mdd'])} | {dep} |"
            )
        lines.append("")

    out_md = ROOT / "reports" / "profile_sizing" / "final0706_conditional_dca_study.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"저장: {out_json}")
    print(f"저장: {out_md}")


if __name__ == "__main__":
    main()
