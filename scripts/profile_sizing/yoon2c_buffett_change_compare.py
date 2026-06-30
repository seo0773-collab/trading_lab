"""yoon2c 후보: 한국/미국 버핏지수 변화율 기반 원/달러 비중 검증.

yoon2 코어의 KRW-base 슬리브 NAV는 그대로 두고, 월간 통화 리밸런싱 비중만
버핏지수 레벨이 아니라 전년 대비 변화율로 바꿔 비교한다.

변화율은 연간 World Bank 버핏지수의 growth factor(B_t / B_{t-1})를 쓴다.
단순 변화율은 음수가 가능해 비중 정규화에 부적합하므로, 항상 양수인 growth factor를
비중 공식에 넣는다.

무누수 처리:
  연도 Y 값과 전년 대비 변화율은 다음 해 1월 1일부터 사용 가능하다고 본다.

사용:
  PYTHONPATH=src python3 scripts/profile_sizing/yoon2c_buffett_change_compare.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from profile_sizing.yoon2b_core_tilt_compare import (  # noqa: E402
    combine_dynamic,
    fmt,
    load_core_sleeves,
    perf_on,
    slice_phase,
)
from profile_sizing.yoon2c_buffett_ratio_compare import (  # noqa: E402
    OUT_DIR,
    WB_INDICATOR,
    fetch_world_bank_buffett,
)


def change_weights(idx: pd.DatetimeIndex) -> pd.DataFrame:
    usa = fetch_world_bank_buffett("USA")
    kor = fetch_world_bank_buffett("KOR")
    annual = pd.concat({"us_buffett": usa, "kr_buffett": kor}, axis=1).sort_index()
    annual["source_year"] = annual.index.year - 1
    annual["us_growth"] = annual["us_buffett"] / annual["us_buffett"].shift(1)
    annual["kr_growth"] = annual["kr_buffett"] / annual["kr_buffett"].shift(1)
    annual["us_yoy"] = annual["us_growth"] - 1.0
    annual["kr_yoy"] = annual["kr_growth"] - 1.0
    annual["direct_change_wkr"] = annual["kr_growth"] / (annual["kr_growth"] + annual["us_growth"])
    annual["revert_change_wkr"] = annual["us_growth"] / (annual["kr_growth"] + annual["us_growth"])

    daily = annual.reindex(idx.union(annual.index)).ffill().reindex(idx).ffill()
    daily = daily.dropna(subset=["direct_change_wkr", "revert_change_wkr"])
    if daily.empty:
        raise ValueError("No overlapping Buffett change-rate data with yoon2 sleeves")
    return daily


def metric_rows(nav: pd.Series, base_perf: dict[str, dict], variant: str, idx: pd.DatetimeIndex) -> list[dict]:
    out = []
    for phase in ("train", "validation", "test", "all"):
        p = perf_on(nav, slice_phase(idx, phase))
        bp = base_perf[phase]
        out.append({
            "variant": variant,
            "phase": phase,
            "cagr": p.get("cagr"),
            "mdd": p.get("max_drawdown"),
            "sharpe": p.get("sharpe"),
            "volatility": p.get("volatility"),
            "total_return": p.get("total_return"),
            "base_cagr": bp.get("cagr"),
            "base_mdd": bp.get("max_drawdown"),
            "base_sharpe": bp.get("sharpe"),
        })
    return out


def pct(x: float | None) -> str:
    return "" if x is None or not np.isfinite(x) else f"{x * 100:.2f}%"


def num(x: float | None) -> str:
    return "" if x is None or not np.isfinite(x) else f"{x:.3f}"


def plot(navs: dict[str, pd.Series], weights: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    for label, nav in navs.items():
        axes[0].plot(nav.index, nav / nav.iloc[0], label=label)
    axes[0].set_title("yoon2 vs Buffett change-rate currency allocation")
    axes[0].set_ylabel("NAV, start=1")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="upper left")

    axes[1].plot(weights.index, weights["direct_change_wkr"] * 100.0, label="direct change KRW weight")
    axes[1].plot(weights.index, weights["revert_change_wkr"] * 100.0, label="revert change KRW weight")
    axes[1].axhline(50.0, color="black", lw=1, alpha=0.4)
    axes[1].set_ylabel("KRW weight (%)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_report(result: pd.DataFrame, weights: pd.DataFrame, idx: pd.DatetimeIndex, csv_path: Path, weight_path: Path, png_path: Path) -> Path:
    def row_for(variant: str, phase: str) -> pd.Series:
        return result[(result["variant"] == variant) & (result["phase"] == phase)].iloc[0]

    table = [
        "| 변형 | phase | CAGR | MDD | Sharpe | Vol | TotalRet |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    variants = [
        ("base_5050", "**base 50/50**"),
        ("yoon2c_direct_change", "direct change"),
        ("yoon2c_revert_change", "revert change"),
    ]
    for variant, label in variants:
        for i, phase in enumerate(("train", "validation", "test", "all")):
            r = row_for(variant, phase)
            table.append(
                f"| {label if i == 0 else ''} | {phase} | {pct(r['cagr'])} | "
                f"{pct(r['mdd'])} | {num(r['sharpe'])} | {pct(r['volatility'])} | {pct(r['total_return'])} |"
            )

    base_all = row_for("base_5050", "all")
    direct_all = row_for("yoon2c_direct_change", "all")
    revert_all = row_for("yoon2c_revert_change", "all")
    direct_val = row_for("yoon2c_direct_change", "validation")
    revert_val = row_for("yoon2c_revert_change", "validation")
    base_val = row_for("base_5050", "validation")
    recent = weights.iloc[-1]

    text = f"""# yoon2c 후보: 버핏지수 변화율 기반 통화 조절

yoon2의 미국/한국 슬리브는 유지하고, 월간 통화 리밸런싱 비중만 버핏지수 레벨이 아니라
전년 대비 변화율로 바꿔 검증했다.

- 기간: {idx[0].date()} ~ {idx[-1].date()} ({len(idx)}봉)
- 버핏지수: World Bank `{WB_INDICATOR}` = `Stock market capitalization to GDP (%)`
- 변화율: `growth = Buffett_t / Buffett_(t-1)`; 단순 YoY 변화율은 `growth - 1`
- direct change: `w_KRW = KR_growth / (KR_growth + US_growth)`
- revert change: `w_KRW = US_growth / (KR_growth + US_growth)`; 버핏지수가 더 빨리 오르는 시장을 줄이는 반대 해석
- 무누수 처리: 관측연도 Y의 변화율은 다음 해 1월 1일부터 사용. 2025년 이후는 2024년 확정치 forward-fill.

## 판정: 변화율 기준도 yoon2 50/50 대체 실패

direct change는 train Sharpe와 MDD가 좋아지지만 validation Sharpe가 낮고 전구간 총수익도 줄어든다.
revert change는 전구간 CAGR/총수익은 올라가지만 Sharpe가 낮아지고 test 낙폭이 깊어진다.
둘 다 안정적인 위험조정 개선으로 보기 어렵다.

{chr(10).join(table)}

## 해석

- direct all: CAGR {pct(base_all['cagr'])} → {pct(direct_all['cagr'])}, Sharpe {num(base_all['sharpe'])} → {num(direct_all['sharpe'])}, MDD {pct(base_all['mdd'])} → {pct(direct_all['mdd'])}.
- revert all: CAGR {pct(base_all['cagr'])} → {pct(revert_all['cagr'])}, Sharpe {num(base_all['sharpe'])} → {num(revert_all['sharpe'])}, MDD {pct(base_all['mdd'])} → {pct(revert_all['mdd'])}.
- validation만 보면 revert change Sharpe {num(revert_val['sharpe'])}가 기준 {num(base_val['sharpe'])}보다 약간 높지만, 전구간 Sharpe가 낮고 test MDD가 깊어 채택 근거가 약하다.
- 최신 확정 변화율 기준({int(recent['source_year'])}년 관측치): Korea YoY {recent['kr_yoy'] * 100:.1f}%, US YoY {recent['us_yoy'] * 100:.1f}%, direct KRW 비중 {recent['direct_change_wkr'] * 100:.1f}%, revert KRW 비중 {recent['revert_change_wkr'] * 100:.1f}%.

## 결론

버핏지수의 레벨 비율보다 변화율 기준은 비중이 50% 근처에서 움직여 더 완만하지만,
성과 개선은 나오지 않았다. 현재 기준에서는 yoon2의 고정 50/50 통화 리밸런싱을 유지하는 쪽이 낫다.

## 파일

- 스크립트: `scripts/profile_sizing/yoon2c_buffett_change_compare.py`
- 결과 CSV: `{csv_path.relative_to(ROOT)}`
- 변화율/비중 CSV: `{weight_path.relative_to(ROOT)}`
- 그래프 PNG: `{png_path.relative_to(ROOT)}`
"""
    out = OUT_DIR / "yoon2c_buffett_change_compare.md"
    out.write_text(text, encoding="utf-8")
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("코어 슬리브 로딩(yoon2: yoon1j_lp/yoon1j_kr, KRW-base)...")
    us, kr = load_core_sleeves()
    idx = us.index
    print(f"  기간 {idx[0].date()}~{idx[-1].date()}  {len(idx)}봉")

    print("World Bank 버핏지수 변화율 로딩...")
    weights = change_weights(idx)
    idx = idx.intersection(weights.index)
    us = us.reindex(idx)
    kr = kr.reindex(idx)
    weights = weights.reindex(idx).ffill()

    base_nav = combine_dynamic(us, kr, None)
    direct_nav = combine_dynamic(us, kr, weights["direct_change_wkr"])
    revert_nav = combine_dynamic(us, kr, weights["revert_change_wkr"])

    phases = ("train", "validation", "test", "all")
    base_perf = {phase: perf_on(base_nav, slice_phase(idx, phase)) for phase in phases}
    rows = []
    rows.extend(metric_rows(base_nav, base_perf, "base_5050", idx))
    rows.extend(metric_rows(direct_nav, base_perf, "yoon2c_direct_change", idx))
    rows.extend(metric_rows(revert_nav, base_perf, "yoon2c_revert_change", idx))
    result = pd.DataFrame(rows)

    print("\n=== 베이스(고정 50/50) ===")
    for phase in phases:
        print(f"  {phase:11s} {fmt(base_perf[phase])}")
    print("\n=== yoon2c direct change: KRW = KR_growth/(KR_growth+US_growth) ===")
    for phase in phases:
        print(f"  {phase:11s} {fmt(perf_on(direct_nav, slice_phase(idx, phase)))}")
    print("\n=== yoon2c revert change: KRW = US_growth/(KR_growth+US_growth) ===")
    for phase in phases:
        print(f"  {phase:11s} {fmt(perf_on(revert_nav, slice_phase(idx, phase)))}")

    csv_path = OUT_DIR / "yoon2c_buffett_change_compare.csv"
    weight_path = OUT_DIR / "yoon2c_buffett_change_weights.csv"
    png_path = OUT_DIR / "yoon2c_buffett_change_compare.png"
    result.to_csv(csv_path, index=False)
    weights.to_csv(weight_path, index_label="date")
    plot(
        {
            "base 50/50": base_nav,
            "direct change": direct_nav,
            "revert change": revert_nav,
        },
        weights,
        png_path,
    )
    report_path = write_report(result, weights, idx, csv_path, weight_path, png_path)

    print(f"\nCSV: {csv_path}")
    print(f"Weights: {weight_path}")
    print(f"PNG: {png_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
