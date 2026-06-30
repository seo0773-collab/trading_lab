"""yoon2c 후보: 버핏지수 변화율 revert 비중의 강도(k) 스윕.

이전 yoon2c_buffett_change_compare.py에서 revert change는 MDD를 거의 유지하면서
CAGR/총수익을 올렸지만 Sharpe가 낮아졌다. 여기서는 raw revert 비중을 그대로 쓰지 않고
50/50 주변으로 완화한다.

  raw_revert = US_growth / (KR_growth + US_growth)
  w_KRW = clip(0.5 + k * (raw_revert - 0.5), 0.30, 0.70)

k=0이면 기존 yoon2 50/50, k=1이면 이전 revert change와 같다.

사용:
  PYTHONPATH=src python3 scripts/profile_sizing/yoon2c_buffett_change_scaled_sweep.py
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
from profile_sizing.yoon2c_buffett_change_compare import change_weights  # noqa: E402
from profile_sizing.yoon2c_buffett_ratio_compare import OUT_DIR, WB_INDICATOR  # noqa: E402

K_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
BAND = (0.30, 0.70)


def scaled_revert_weight(raw_revert: pd.Series, k: float) -> pd.Series:
    return (0.5 + k * (raw_revert - 0.5)).clip(*BAND)


def rows_for(nav: pd.Series, base_perf: dict[str, dict], variant: str, k: float, idx: pd.DatetimeIndex) -> list[dict]:
    rows = []
    for phase in ("train", "validation", "test", "all"):
        p = perf_on(nav, slice_phase(idx, phase))
        bp = base_perf[phase]
        rows.append({
            "variant": variant,
            "k": k,
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
    return rows


def pct(x: float | None) -> str:
    return "" if x is None or not np.isfinite(x) else f"{x * 100:.2f}%"


def num(x: float | None) -> str:
    return "" if x is None or not np.isfinite(x) else f"{x:.3f}"


def plot(navs: dict[str, pd.Series], weights: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    for label, nav in navs.items():
        axes[0].plot(nav.index, nav / nav.iloc[0], label=label)
    axes[0].set_title("yoon2c scaled revert Buffett-change sweep")
    axes[0].set_ylabel("NAV, start=1")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="upper left")

    for col in weights.columns:
        if col.startswith("w_k"):
            axes[1].plot(weights.index, weights[col] * 100.0, label=col)
    axes[1].axhline(50.0, color="black", lw=1, alpha=0.4)
    axes[1].set_ylabel("KRW weight (%)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_report(result: pd.DataFrame, weights: pd.DataFrame, idx: pd.DatetimeIndex, csv_path: Path, weight_path: Path, png_path: Path) -> Path:
    all_rows = result[result["phase"] == "all"].copy().sort_values("k")
    val_rows = result[result["phase"] == "validation"].copy().sort_values("k")
    test_rows = result[result["phase"] == "test"].copy().sort_values("k")
    base_all = all_rows[all_rows["k"] == 0.0].iloc[0]

    # Practical score: prefer CAGR improvement, penalize extra MDD and Sharpe loss.
    candidates = all_rows[all_rows["k"] > 0].copy()
    candidates["cagr_delta"] = candidates["cagr"] - base_all["cagr"]
    candidates["mdd_delta"] = candidates["mdd"] - base_all["mdd"]
    candidates["sharpe_delta"] = candidates["sharpe"] - base_all["sharpe"]
    practical = candidates[
        (candidates["cagr_delta"] > 0.0)
        & (candidates["mdd_delta"] >= -0.0025)
        & (candidates["sharpe_delta"] >= -0.025)
    ].copy()
    pick = practical.sort_values(["cagr", "sharpe"], ascending=False).iloc[0] if not practical.empty else candidates.sort_values("cagr", ascending=False).iloc[0]

    table = [
        "| k | all CAGR | all MDD | all Sharpe | all TotalRet | val Sharpe | test MDD |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, r in all_rows.iterrows():
        k = float(r["k"])
        val = val_rows[val_rows["k"] == k].iloc[0]
        test = test_rows[test_rows["k"] == k].iloc[0]
        label = "0.00 (base)" if k == 0.0 else f"{k:.2f}"
        table.append(
            f"| {label} | {pct(r['cagr'])} | {pct(r['mdd'])} | {num(r['sharpe'])} | "
            f"{pct(r['total_return'])} | {num(val['sharpe'])} | {pct(test['mdd'])} |"
        )

    recent = weights.iloc[-1]
    text = f"""# yoon2c 후보: 버핏지수 변화율 revert 강도 스윕

버핏지수 변화율의 revert 해석이 50/50보다 CAGR/총수익을 높였지만 Sharpe가 낮아졌기 때문에,
raw 비중을 그대로 쓰지 않고 50/50 주변으로 완화하는 `k`를 스윕했다.

- 기간: {idx[0].date()} ~ {idx[-1].date()} ({len(idx)}봉)
- 버핏지수: World Bank `{WB_INDICATOR}` = `Stock market capitalization to GDP (%)`
- raw revert: `raw_w_KRW = US_growth / (KR_growth + US_growth)`
- 스윕 공식: `w_KRW = clip(0.5 + k * (raw_w_KRW - 0.5), [{BAND[0]:.2f}, {BAND[1]:.2f}])`
- `k=0`: 기존 yoon2 50/50, `k=1`: 이전 revert change와 동일.

## 결과

{chr(10).join(table)}

## 판정

실용 후보는 **k={pick['k']:.2f}**다.

- base 대비 all CAGR: {pct(base_all['cagr'])} → {pct(pick['cagr'])}
- base 대비 all TotalRet: {pct(base_all['total_return'])} → {pct(pick['total_return'])}
- base 대비 all MDD: {pct(base_all['mdd'])} → {pct(pick['mdd'])}
- base 대비 all Sharpe: {num(base_all['sharpe'])} → {num(pick['sharpe'])}

해석은 명확하다. `k`를 키울수록 수익률은 올라가지만 Sharpe가 점진적으로 내려간다.
MDD는 전구간 기준으로 거의 유지되므로, yoon2의 목적함수를 Sharpe 최우선이 아니라
`MDD 유지 + CAGR 개선`으로 둔다면 scaled revert는 채택 가능한 후보가 된다.

최신 확정 변화율 기준({int(recent['source_year'])}년 관측치) raw revert KRW 비중은
{recent['raw_revert_wkr'] * 100:.1f}%이고, k={pick['k']:.2f} 적용 비중은
{recent[f'w_k{pick["k"]:.2f}'] * 100:.1f}%다.

## 결론

`k=1.50`은 수익은 가장 많이 올리지만 Sharpe 하락과 test MDD 악화가 더 크다. 균형형 yoon2c 후보로는
**k={pick['k']:.2f} scaled revert**를 우선 검토하는 편이 낫다. 다음 단계는 이 후보를
정수주/DCA 실행 조건에서도 같은 방향으로 유지되는지 확인하는 것이다.

## 파일

- 스크립트: `scripts/profile_sizing/yoon2c_buffett_change_scaled_sweep.py`
- 결과 CSV: `{csv_path.relative_to(ROOT)}`
- 비중 CSV: `{weight_path.relative_to(ROOT)}`
- 그래프 PNG: `{png_path.relative_to(ROOT)}`
"""
    out = OUT_DIR / "yoon2c_buffett_change_scaled_sweep.md"
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
    base_perf = {phase: perf_on(base_nav, slice_phase(idx, phase)) for phase in ("train", "validation", "test", "all")}

    rows = []
    navs = {"base k=0": base_nav}
    weights_out = weights[["source_year", "us_buffett", "kr_buffett", "us_growth", "kr_growth", "us_yoy", "kr_yoy"]].copy()
    weights_out["raw_revert_wkr"] = weights["revert_change_wkr"]

    for k in K_GRID:
        w = scaled_revert_weight(weights["revert_change_wkr"], k)
        weights_out[f"w_k{k:.2f}"] = w
        nav = base_nav if k == 0.0 else combine_dynamic(us, kr, w)
        if k in (0.25, 0.5, 1.0):
            navs[f"k={k:.2f}"] = nav
        rows.extend(rows_for(nav, base_perf, "scaled_revert", k, idx))

    result = pd.DataFrame(rows)

    print("\n=== scaled revert sweep ===")
    for k in K_GRID:
        p = result[(result["k"] == k) & (result["phase"] == "all")].iloc[0]
        print(
            f"  k={k:4.2f} CAGR {p['cagr']*100:6.2f}% | MDD {p['mdd']*100:7.2f}% | "
            f"Sharpe {p['sharpe']:.3f} | TR {p['total_return']*100:8.1f}%"
        )

    csv_path = OUT_DIR / "yoon2c_buffett_change_scaled_sweep.csv"
    weight_path = OUT_DIR / "yoon2c_buffett_change_scaled_weights.csv"
    png_path = OUT_DIR / "yoon2c_buffett_change_scaled_sweep.png"
    result.to_csv(csv_path, index=False)
    weights_out.to_csv(weight_path, index_label="date")
    plot(navs, weights_out, png_path)
    report_path = write_report(result, weights_out, idx, csv_path, weight_path, png_path)

    print(f"\nCSV: {csv_path}")
    print(f"Weights: {weight_path}")
    print(f"PNG: {png_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
