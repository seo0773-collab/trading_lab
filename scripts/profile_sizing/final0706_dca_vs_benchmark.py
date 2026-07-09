"""final0706·final0706_qld 적립식(DCA) vs 벤치마크(50/50 SPY원화+KODEX200) 적립식 비교.

`final0706_variant_dca_compare.py`는 "final0706을 거치식으로 넣을 때 vs 적립식으로
넣을 때"를 비교했는데, 이건 같은 원금이 시장에 노출된 총 기간이 다른 두 계좌를
비교하는 셈이라(거치식은 첫날부터 전액이 19.5년 복리, 적립식은 나중 납입분일수록
짧게 복리) 총수익률 격차가 크게 나는 게 당연하다 — 자금 조달 방식을 못 고르는
실사용자(매달 적립할 돈만 있는 사람)에게는 의미 있는 비교가 아니다.

진짜 의미 있는 질문: **매달 적립할 돈을 final0706(또는 레버리지 얹은
final0706_qld)에 넣는 게 그냥 벤치마크(패시브 50/50 SPY원화+KODEX200)에 적립하는
것보다 나은가?** — 같은 조달방식(적립식), 같은 기간, 같은 원금 스케줄로 두
"투자 대상"만 바꿔서 비교한다(yoon_profile/yoon1f DCA 분석과 같은 방법론).
final0706_qld는 core+QLD(2x Nasdaq100) 블렌드라 변동성이 커서, 적립식 특유의
"저가 매수 효과"가 레버리지와 만나 어떻게 작용하는지가 관전 포인트.

Usage:
    PYTHONPATH=src:scripts python3 scripts/profile_sizing/final0706_dca_vs_benchmark.py
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
    final0706_cfg = json.loads((CONFIGS_DIR / "final0706.json").read_text("utf-8"))
    final0706_qld_cfg = json.loads((CONFIGS_DIR / "final0706_qld.json").read_text("utf-8"))
    k_handler = Yoon1kHandler()
    lev_handler = Yoon1kLeverageMixHandler()

    print("final0706 로드 중...", flush=True)
    raw = k_handler.load_data("PORTFOLIO", final0706_cfg, synthetic=False)
    art = k_handler.build_artifacts(
        raw, final0706_cfg, symbol="PORTFOLIO", phase="all", bars_per_year=252)
    nav_strategy = art.equity
    nav_bench = art.benchmark  # 50/50 SPY원화+KODEX200, 1.0 기준 정규화

    print("final0706_qld 로드 중...", flush=True)
    raw_qld = lev_handler.load_data("PORTFOLIO", final0706_qld_cfg, synthetic=False)
    art_qld = lev_handler.build_artifacts(
        raw_qld, final0706_qld_cfg, symbol="PORTFOLIO", phase="all", bars_per_year=252)
    nav_qld = art_qld.equity
    nav_bench_qld = art_qld.benchmark

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
        ("final0706_qld(core63.5%+QLD36.5%)", nav_qld),
        ("벤치(50/50 SPY원화+KODEX200)", nav_bench),
    ]
    assert nav_bench_qld is not None  # 동일 50/50 벤치, 참고용 sanity check만

    rows = []
    for phase, window in phases.items():
        for label, nav in candidates:
            s = nav.reindex(window).ffill().dropna()
            if len(s) < 60:
                continue
            s = s / s.iloc[0]
            acct, principal = dca_path(s)
            dca_ret = float(acct.iloc[-1] / principal.iloc[-1] - 1.0)
            months = (window[-1] - window[0]).days / 30.44
            annualized = (1.0 + dca_ret) ** (12.0 / months) - 1.0 if months > 0 else float("nan")
            rows.append({
                "phase": phase, "candidate": label,
                "months": round(months, 1),
                "total_contributed": float(principal.iloc[-1]),
                "dca_final_value": float(acct.iloc[-1]),
                "dca_total_return": dca_ret,
                "dca_annualized_approx": annualized,
                "dca_account_mdd": max_drawdown(acct),
            })

    out_json = ROOT / "reports" / "profile_sizing" / "final0706_dca_vs_benchmark.json"
    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    def pct(v):
        return f"{v * 100:.2f}%"

    def won(v):
        return f"{v:,.0f}원"

    lines = [
        "# final0706·final0706_qld 적립식(DCA) vs 벤치마크 적립식 비교",
        "",
        "- 가정: 초기 2,000,000원 + 매월 첫 거래일 200,000원 적립(정수주·수수료 미반영 "
        "이상적 근사). final0706/final0706_qld와 벤치마크(50/50 SPY원화+KODEX200 패시브)에 "
        "**같은 조달방식·같은 기간·같은 원금 스케줄**로 각각 적립했을 때를 비교한다 — "
        "\"거치식 vs 적립식\"이 아니라 \"같은 적립식 조달을 어디에 넣을까\"의 답.",
        "",
    ]
    for phase in phases:
        prows = [r for r in rows if r["phase"] == phase]
        if not prows:
            continue
        lines.append(f"## {phase} (약 {prows[0]['months']}개월)")
        lines.append("")
        lines.append("| candidate | 총 납입원금 | 적립식 평가액 | 총수익률 | 연환산(근사) | 계좌 MDD |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for r in prows:
            lines.append(
                f"| {r['candidate']} | {won(r['total_contributed'])} | {won(r['dca_final_value'])} | "
                f"{pct(r['dca_total_return'])} | {pct(r['dca_annualized_approx'])} | {pct(r['dca_account_mdd'])} |"
            )
        lines.append("")

    out_md = ROOT / "reports" / "profile_sizing" / "final0706_dca_vs_benchmark.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"저장: {out_json}")
    print(f"저장: {out_md}")


if __name__ == "__main__":
    main()
