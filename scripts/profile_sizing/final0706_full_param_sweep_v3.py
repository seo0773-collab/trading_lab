"""final0706(일봉 원본) — cycle_len × rebalance 전면 스윕 v3.

v2(final0706_full_param_sweep.py)에 두 가지를 더한다:
  1. **일 2% 복리 벤치마크**: (1.02)^n (n=해당 phase 봉수) — 사용자 요청. 연 환산
     ~14,600%로 완전히 비현실적인 벤치마크이며 "이길 수 있는" 기준이 아니라
     스케일 참고용(전략 CAGR/이 벤치마크 CAGR 비율 = 사실상 0에 수렴, 그 자체가
     "복리의 힘이 이론적으로 얼마나 큰지" 대비 전략 수익의 초라함을 보여주는 척도).
  2. **과최적화 판단용 train/test 분할**: 각 조합을 phase="all"과 phase="test"
     (train_frac=0.6/validation_frac=0.2 이후 holdout, config 그대로) 둘 다 계산해,
     "전체 구간 1등"이 holdout에서도 상위권을 유지하는지 확인한다 — 유지 못 하면
     과최적화 신호.

raw 데이터 로딩은 그리드 전체에서 동일(유니버스/interval/period 불변)하므로 한 번만
로드해 재사용한다(72회 반복 로드 낭비 제거).

Usage:
    PYTHONPATH=src:scripts python3 scripts/profile_sizing/final0706_full_param_sweep_v3.py
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_lab.strategies.yoon1k import Yoon1kHandler  # noqa: E402
from metrics import equity_metrics  # noqa: E402 (스크립트 디렉토리 flat import)

ROOT = Path(__file__).resolve().parents[2]
TRADING_DAYS_PER_YEAR = 252.0
CYCLE_GRID = (50, 75, 100, 125, 150, 200, 250, 300, 400)
REBAL_GRID = ("monthly", 10, 15, 21, 30, 42, 63, 84)
DAILY_COMPOUND_RATE = 0.02

SCALE_KEYS = [
    ("profile", "rolling_window", 126),
    ("market_filter", "ma_len", 100),
    ("safe_sleeve", "ma_len", 100),
    ("regime_cap", "recovery_stage_bars", 21),
]


def perf_from_equity(equity: pd.Series, ann: float) -> dict[str, float | None]:
    """공통 유틸 위임 — CAGR은 달력 기준 연환산(개선계획 A1, metrics.py 참고)."""
    return equity_metrics(equity, ann=ann)


def daily_compound_benchmark(n_bars: int) -> dict[str, float]:
    total_return = (1.0 + DAILY_COMPOUND_RATE) ** n_bars - 1.0
    cagr = (1.0 + DAILY_COMPOUND_RATE) ** TRADING_DAYS_PER_YEAR - 1.0
    return {"total_return": total_return, "cagr": cagr}


def rebal_label(rb) -> str:
    return "monthly" if rb == "monthly" else f"every_{rb}"


def build_sub_overrides(sub_entry: dict[str, Any], cycle_len: int, rebal_freq: str) -> dict[str, Any]:
    full = Yoon1kHandler._sub_config(sub_entry["config"], sub_entry.get("overrides"))
    full["rebalance_freq"] = rebal_freq
    full["base_cycle"] = {"type": "SMA", "length": cycle_len, "scale": 1.0}
    ratio = cycle_len / 200.0
    for group, key, default in SCALE_KEYS:
        full[group][key] = max(2, int(round(default * ratio)))
    return full


def build_config(base_cfg: dict[str, Any], cycle_len: int, rebal) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    rebal_freq = rebal_label(rebal)
    cfg["rebalance_freq"] = rebal_freq
    ratio = cycle_len / 200.0
    alloc = cfg["currency_allocation"]
    alloc["lookback"] = max(2, int(round(63 * ratio)))
    gate = alloc["risk_gate"]
    gate["lookback"] = max(2, int(round(10 * ratio)))
    gate["sma_lookback"] = max(2, int(round(20 * ratio)))
    for sub in cfg["sub_strategies"]:
        sub["overrides"] = build_sub_overrides(sub, cycle_len, rebal_freq)
    return cfg


def main() -> None:
    base_cfg = json.loads((ROOT / "configs" / "strategies" / "final0706.json").read_text("utf-8"))
    handler = Yoon1kHandler()

    print("raw 데이터 로드 중 (그리드 전체 공용, 1회만)...", flush=True)
    raw = handler.load_data("PORTFOLIO", base_cfg, synthetic=False)
    print(f"로드 완료: {raw.shape}", flush=True)

    rows = []
    for cyc in CYCLE_GRID:
        for rb in REBAL_GRID:
            cfg = build_config(base_cfg, cyc, rb)
            label = rebal_label(rb)
            try:
                art_all = handler.build_artifacts(raw, cfg, symbol="PORTFOLIO", phase="all", bars_per_year=252)
                art_test = handler.build_artifacts(raw, cfg, symbol="PORTFOLIO", phase="test", bars_per_year=252)
            except Exception as e:  # noqa: BLE001
                print(f"cycle={cyc:<4} rebal={label:<10} 실패: {type(e).__name__}: {e}")
                continue
            perf_all = perf_from_equity(art_all.equity, TRADING_DAYS_PER_YEAR)
            perf_test = perf_from_equity(art_test.equity, TRADING_DAYS_PER_YEAR)
            bench_all = daily_compound_benchmark(perf_all["n"])
            rows.append({
                "cycle_len": cyc, "rebal": label,
                "sharpe_all": perf_all["sharpe"], "cagr_all": perf_all["cagr"], "mdd_all": perf_all["max_drawdown"],
                "sharpe_test": perf_test["sharpe"], "cagr_test": perf_test["cagr"], "mdd_test": perf_test["max_drawdown"],
                "cagr_all_vs_2pct_daily_ratio": (perf_all["cagr"] / bench_all["cagr"]) if perf_all["cagr"] else None,
            })
            print(f"cycle={cyc:<4} rebal={label:<10} Sharpe(all)={perf_all['sharpe']:.3f} "
                  f"Sharpe(test)={perf_test['sharpe']:.3f} CAGR(all)={perf_all['cagr']*100:.2f}% "
                  f"vs2%daily={rows[-1]['cagr_all_vs_2pct_daily_ratio']*100:.4f}%")

    df = pd.DataFrame(rows)
    df["rank_all"] = df["sharpe_all"].rank(ascending=False)
    df["rank_test"] = df["sharpe_test"].rank(ascending=False)
    df["rank_diff"] = (df["rank_all"] - df["rank_test"]).abs()

    baseline_row = df[(df["cycle_len"] == 200) & (df["rebal"] == "monthly")].iloc[0]
    best_all = df.loc[df["sharpe_all"].idxmax()]
    best_test = df.loc[df["sharpe_test"].idxmax()]

    outdir = ROOT / "reports" / "profile_sizing"
    out_json = outdir / "final0706_full_param_sweep_v3.json"
    out_json.write_text(json.dumps({
        "daily_compound_rate": DAILY_COMPOUND_RATE,
        "rows": df.to_dict(orient="records"),
        "baseline": baseline_row.to_dict(),
        "best_all": best_all.to_dict(),
        "best_test": best_test.to_dict(),
    }, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    n_bars_test = int(perf_from_equity(art_test.equity, TRADING_DAYS_PER_YEAR)["n"])
    bench_test_cagr = daily_compound_benchmark(n_bars_test)["cagr"]

    pivot_all = df.pivot(index="cycle_len", columns="rebal", values="sharpe_all")[[rebal_label(r) for r in REBAL_GRID]]
    pivot_test = df.pivot(index="cycle_len", columns="rebal", values="sharpe_test")[[rebal_label(r) for r in REBAL_GRID]]

    lines = [
        "# final0706(일봉 원본) — cycle_len × 리밸런스 전면 스윕 v3 (2%/일 복리 벤치마크 + 과최적화 판단)",
        "",
        f"- 베이스라인(cycle=200, monthly): Sharpe(all)={baseline_row['sharpe_all']:.3f}, "
        f"Sharpe(test/holdout)={baseline_row['sharpe_test']:.3f}, CAGR(all)={baseline_row['cagr_all']*100:.2f}%",
        f"- **2%/일 복리 벤치마크**: 연환산 CAGR {bench_test_cagr*100:,.0f}% — 완전히 비현실적인 참고선."
        " 어떤 조합도 이 근처에도 못 감(전략 CAGR/이 벤치마크 CAGR 비율은 사실상 0%) — 이건 '이길 대상'이 아니라"
        " 복리 상한선 대비 실제 전략 수익의 스케일을 보여주는 잣대일 뿐이다.",
        f"- **과최적화 체크**: 전체구간(all) 1등과 holdout(test) 1등이 다른가? "
        f"all 1등=cycle={int(best_all['cycle_len'])}/{best_all['rebal']}(Sharpe {best_all['sharpe_all']:.3f}), "
        f"test 1등=cycle={int(best_test['cycle_len'])}/{best_test['rebal']}(Sharpe {best_test['sharpe_test']:.3f})",
        "",
        "## Sharpe(all) 그리드",
        "",
    ]
    header = "| cycle_len \\ rebal |" + "|".join(f" {c} " for c in pivot_all.columns) + "|"
    sep = "| ---: |" + "|".join(" ---: " for _ in pivot_all.columns) + "|"
    lines += [header, sep]
    for idx_, row in pivot_all.iterrows():
        cells = ["-" if pd.isna(v) else f"{v:.2f}" for v in row]
        marker = " **(현재확정)**" if idx_ == 200 else ""
        lines.append(f"| {idx_}{marker} |" + "|".join(f" {c} " for c in cells) + "|")
    lines += ["", "## Sharpe(test/holdout) 그리드", ""]
    lines += [header, sep]
    for idx_, row in pivot_test.iterrows():
        cells = ["-" if pd.isna(v) else f"{v:.2f}" for v in row]
        marker = " **(현재확정)**" if idx_ == 200 else ""
        lines.append(f"| {idx_}{marker} |" + "|".join(f" {c} " for c in cells) + "|")

    lines += [
        "",
        "## 과최적화 판단",
        "",
        f"- 베이스라인(200,monthly): all순위 {int(baseline_row['rank_all'])}위, test순위 {int(baseline_row['rank_test'])}위"
        f" (순위차 {int(baseline_row['rank_diff'])})",
        f"- all 1등 조합의 test 순위: {int(df.loc[df['sharpe_all'].idxmax(), 'rank_test'])}위",
        f"- 순위차(|all순위-test순위|) 평균: {df['rank_diff'].mean():.1f}, 중앙값: {df['rank_diff'].median():.1f}, "
        f"최대: {int(df['rank_diff'].max())}",
        "",
        "## 파일",
        "",
        f"- JSON: `{out_json.relative_to(ROOT)}`",
    ]
    out_md = outdir / "final0706_full_param_sweep_v3.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nMD: {out_md}")
    print(f"\n베이스라인 all순위={int(baseline_row['rank_all'])} test순위={int(baseline_row['rank_test'])}")
    print(f"all 1등: cycle={int(best_all['cycle_len'])}/{best_all['rebal']} (test순위 {int(df.loc[df['sharpe_all'].idxmax(),'rank_test'])})")
    print(f"test 1등: cycle={int(best_test['cycle_len'])}/{best_test['rebal']}")
    print(f"순위차 평균={df['rank_diff'].mean():.1f} 중앙값={df['rank_diff'].median():.1f} 최대={int(df['rank_diff'].max())}")


if __name__ == "__main__":
    main()
