"""bb_martingale_meanreversion.py 연장 — 추세 가속화 레짐에서는 신규 진입/물타기/플립을
쉬고(기존 포지션은 손대지 않음), 레짐이 풀리면 재개하는 게이트를 얹는다.

레짐 정의(ADX 수준+기울기, "추세가 실제로 붙는 중"에 가장 직결되는 지표):
    bad = ADX(14) > ADX_THRESHOLD  AND  ADX(14)가 ADX_SLOPE_LOOKBACK봉 전보다 상승 중
    (레벨만 보면 "이미 강한 추세"에서도 계속 거래하게 되고, 기울기만 보면 약한 추세에서
    미세한 등락에도 게이트가 켜졌다 꺼졌다 하므로 두 조건을 함께 요구한다.)

게이트 동작: bad 구간에서는 부트스트랩 최초진입/물타기/플립(청산+반대진입)을 전부 정지한다.
기존 포지션은 그대로 마크투마켓만 계속하고, 레짐이 good으로 돌아오면 그 시점의 가격이 여전히
반대 밴드를 넘어서 있으면 그제서야 플립이 실행된다("신규만 정지, 기존 포지션은 유지").

같은 length/mult/offset(20, 2.5, 0)의 48개 마틴게일 조합 그리드를 레짐 게이트 유무로
비교한다 — bb_martingale_sweep.json(게이트 없음, 이미 계산됨)을 베이스라인으로 재사용.

Usage:
    .venv/bin/python3 scripts/bb_martingale_regime_gate.py
출력:
    reports/bb_martingale_regime_sweep.json
    reports/bb_martingale_regime_sweep.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "profile_sizing"))
from bb_crossing_sweep import compute_adx  # noqa: E402
from bb_martingale_meanreversion import (  # noqa: E402
    ADD_STEP_GRID, MAX_LEVELS_GRID, MULT_GRID, OUT_DIR, load_universe, run_combo,
)

ADX_THRESHOLD = 25.0
ADX_SLOPE_LOOKBACK = 10

BASELINE_JSON = OUT_DIR / "bb_martingale_sweep.json"


def regime_fn(df: pd.DataFrame, full_index: pd.DatetimeIndex) -> np.ndarray:
    adx = compute_adx(df)
    bad = (adx > ADX_THRESHOLD) & (adx > adx.shift(ADX_SLOPE_LOOKBACK))
    ok = (~bad.fillna(False)).reindex(full_index, fill_value=True)
    return ok.to_numpy()


def main() -> None:
    print("1) 유니버스 로드...", flush=True)
    universe = load_universe()

    if not BASELINE_JSON.exists():
        raise SystemExit(f"베이스라인 {BASELINE_JSON}이 없습니다. "
                          "bb_martingale_meanreversion.py를 먼저 실행하세요.")
    baseline = pd.DataFrame(json.loads(BASELINE_JSON.read_text("utf-8"))["grid"])

    total = len(MULT_GRID) * len(ADD_STEP_GRID) * len(MAX_LEVELS_GRID)
    print(f"2) 레짐게이트(ADX>{ADX_THRESHOLD:.0f} & 상승중) 적용 그리드 재실행: "
          f"{total}개 조합 x {len(universe)}종목", flush=True)
    rows = []
    done = 0
    for mult in MULT_GRID:
        for add_step in ADD_STEP_GRID:
            for max_levels in MAX_LEVELS_GRID:
                rows.append(run_combo(universe, add_step, mult, max_levels, regime_fn=regime_fn))
                done += 1
                if done % 12 == 0 or done == total:
                    print(f"   {done}/{total} 완료", flush=True)

    gated = pd.DataFrame(rows)
    out_json = OUT_DIR / "bb_martingale_regime_sweep.json"
    out_json.write_text(json.dumps({
        "adx_threshold": ADX_THRESHOLD, "adx_slope_lookback": ADX_SLOPE_LOOKBACK,
        "grid": gated.to_dict(orient="records"),
    }, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    key_cols = ["add_step", "mult", "max_levels"]
    merged = gated.merge(baseline, on=key_cols, suffixes=("_gated", "_base"))

    survivors = merged[(merged["train_ruin_rate_gated"] == 0) &
                        (merged["holdout_ruin_rate_gated"] == 0)]
    ranked = survivors.sort_values("train_median_sharpe_gated", ascending=False)
    top = ranked.head(10)

    base_ruin = baseline[["train_ruin_rate", "holdout_ruin_rate"]].mean()
    gated_ruin = gated[["train_ruin_rate", "holdout_ruin_rate"]].mean()

    lines = [
        "# 볼린저밴드 마틴게일 — 추세가속 레짐 게이트 적용 비교",
        "",
        f"- 레짐: ADX(14) > {ADX_THRESHOLD:.0f} 이고 {ADX_SLOPE_LOOKBACK}봉 전보다 상승 중이면 "
        "'추세가속'으로 보고 신규진입/물타기/플립을 정지(기존 포지션은 유지, 마크투마켓만 계속).",
        "- 나머지 조건(BB length/mult/offset, 비용, base_lot_fraction, 파산 처리)은 "
        "bb_martingale_sweep.py와 동일 — 48개 조합 그리드를 게이트 유무로 비교.",
        "",
        "## 파산율: 게이트 없음 vs 있음 (그리드 전체 평균)",
        "",
        "| 구간 | 게이트 없음 | 게이트 있음 |",
        "| --- | ---: | ---: |",
        f"| train | {base_ruin['train_ruin_rate']*100:.1f}% | "
        f"{gated_ruin['train_ruin_rate']*100:.1f}% |",
        f"| holdout | {base_ruin['holdout_ruin_rate']*100:.1f}% | "
        f"{gated_ruin['holdout_ruin_rate']*100:.1f}% |",
        "",
    ]

    if len(survivors):
        lines += [
            "## 파산 0%(게이트 적용 기준) 조합 중 train Sharpe 상위 — 게이트 유무 나란히 비교",
            "",
            "| mult | add_step | max_levels | Sharpe(무게이트→게이트) train | "
            "MDD(무게이트→게이트) train | Sharpe holdout | MDD holdout |",
            "| ---: | ---: | ---: | :---: | :---: | :---: | :---: |",
        ]
        for _, r in top.iterrows():
            lines.append(
                f"| {r['mult']} | {r['add_step']*100:.0f}% | {int(r['max_levels'])} | "
                f"{r['train_median_sharpe_base']:.3f} → {r['train_median_sharpe_gated']:.3f} | "
                f"{r['train_median_mdd_base']*100:.1f}% → {r['train_median_mdd_gated']*100:.1f}% | "
                f"{r['holdout_median_sharpe_base']:.3f} → {r['holdout_median_sharpe_gated']:.3f} | "
                f"{r['holdout_median_mdd_base']*100:.1f}% → {r['holdout_median_mdd_gated']*100:.1f}% |"
            )
        lines.append("")
        best = top.iloc[0]
        lines += [
            "## 최적 조합 (게이트 적용, 파산 0% 중 train Sharpe 기준)",
            "",
            f"- mult={best['mult']}, add_step={best['add_step']*100:.0f}%, "
            f"max_levels={int(best['max_levels'])}",
            f"- train Sharpe {best['train_median_sharpe_base']:.3f} → "
            f"{best['train_median_sharpe_gated']:.3f} (게이트 적용 후)",
            f"- holdout Sharpe {best['holdout_median_sharpe_base']:.3f} → "
            f"{best['holdout_median_sharpe_gated']:.3f}",
            f"- train total_return {best['train_median_total_return_base']*100:.2f}% → "
            f"{best['train_median_total_return_gated']*100:.2f}%",
            f"- holdout total_return {best['holdout_median_total_return_base']*100:.2f}% → "
            f"{best['holdout_median_total_return_gated']*100:.2f}%",
            "",
        ]
    else:
        lines += ["## 게이트 적용 후에도 파산 0%인 조합이 없습니다.", ""]

    lines += [
        "## 주의",
        "",
        "- ADX_THRESHOLD/ADX_SLOPE_LOOKBACK은 스윕하지 않고 고정값(25 / 10봉)만 시도했다 — "
        "레짐 파라미터 자체를 튜닝하면 결과가 달라질 수 있다.",
        "- 게이트는 신규 행위만 막을 뿐 기존 포지션의 마크투마켓 손익은 그대로 반영된다 — "
        "추세가속 초입에 이미 물려 있었다면 게이트가 켜져도 그 포지션의 손실은 계속 쌓인다.",
        "",
        "## 파일",
        "",
        f"- 게이트 적용 전체 그리드: `{out_json.relative_to(ROOT)}`",
        f"- 게이트 없음 베이스라인: `{BASELINE_JSON.relative_to(ROOT)}`",
    ]
    out_md = OUT_DIR / "bb_martingale_regime_sweep.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nMD: {out_md}")


if __name__ == "__main__":
    main()
