"""yoon2c 후보: 한국/미국 버핏지수 비율 기반 원/달러 비중 검증.

전략/registry/config는 건드리지 않고 임시 계산한다. yoon2 코어의 KRW-base
슬리브 NAV(미국 yoon1j_lp / 한국 yoon1j_kr)를 재사용해, 월간 리밸런싱 목표비중을
고정 50/50 대신 World Bank 버핏지수 비율로 바꿔 성과를 비교한다.

버핏지수 = World Bank CM.MKT.LCAP.GD.ZS
           Stock market capitalization to GDP (%)

무누수 처리:
  연간 observation year Y 값은 다음 해 1월 1일부터 사용 가능하다고 본다.
  예: 2024 값은 2025-01-01 이후 월간 리밸런싱부터 사용.

사용:
  PYTHONPATH=src python3 scripts/profile_sizing/yoon2c_buffett_ratio_compare.py
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
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

WB_INDICATOR = "CM.MKT.LCAP.GD.ZS"
WB_URL = "https://api.worldbank.org/v2/country/{country}/indicator/{indicator}?format=json&per_page=1000"
OUT_DIR = ROOT / "reports" / "profile_sizing"


def fetch_world_bank_buffett(country: str) -> pd.Series:
    """World Bank 버핏지수 연간값을 Series(year-end effective source)로 반환."""
    url = WB_URL.format(country=country, indicator=WB_INDICATOR)
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                payload = json.load(resp)
            rows = payload[1] if len(payload) > 1 else []
            vals = {
                int(row["date"]): float(row["value"])
                for row in rows
                if row.get("value") is not None
            }
            if not vals:
                raise ValueError(f"World Bank returned no values for {country}")
            idx = pd.to_datetime([f"{year + 1}-01-01" for year in vals])
            return pd.Series(list(vals.values()), index=idx, name=country).sort_index()
        except Exception as exc:  # pragma: no cover - network retry guard
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"World Bank fetch failed for {country}: {last_error}") from last_error


def buffett_weights(idx: pd.DatetimeIndex) -> pd.DataFrame:
    """일간 인덱스에 맞춘 직접비율/역비율 KRW 목표비중."""
    usa = fetch_world_bank_buffett("USA")
    kor = fetch_world_bank_buffett("KOR")
    raw = pd.concat({"us_buffett": usa, "kr_buffett": kor}, axis=1).sort_index()
    raw["source_year"] = raw.index.year - 1
    daily = raw.reindex(idx.union(raw.index)).ffill().reindex(idx).ffill()
    daily = daily.dropna()
    if daily.empty:
        raise ValueError("No overlapping Buffett indicator data with yoon2 sleeves")

    kr_b = daily["kr_buffett"]
    us_b = daily["us_buffett"]
    direct_wkr = kr_b / (kr_b + us_b)
    inverse_wkr = (1.0 / kr_b) / ((1.0 / kr_b) + (1.0 / us_b))
    # Algebraically inverse_wkr == us_b / (kr_b + us_b). Keep formula explicit.
    out = daily.copy()
    out["ratio_kr_us"] = kr_b / us_b
    out["direct_wkr"] = direct_wkr
    out["inverse_value_wkr"] = inverse_wkr
    return out


def phase_rows(nav: pd.Series, base_perf: dict[str, dict], variant: str, idx: pd.DatetimeIndex) -> list[dict]:
    rows = []
    for phase in ("train", "validation", "test", "all"):
        p = perf_on(nav, slice_phase(idx, phase))
        bp = base_perf[phase]
        rows.append({
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
    return rows


def pct(x: float | None) -> str:
    return "" if x is None or not np.isfinite(x) else f"{x * 100:.2f}%"


def num(x: float | None) -> str:
    return "" if x is None or not np.isfinite(x) else f"{x:.3f}"


def write_report(
    rows: pd.DataFrame,
    weights: pd.DataFrame,
    navs: dict[str, pd.Series],
    idx: pd.DatetimeIndex,
    csv_path: Path,
    weight_path: Path,
    png_path: Path,
) -> Path:
    def row_for(variant: str, phase: str) -> pd.Series:
        return rows[(rows["variant"] == variant) & (rows["phase"] == phase)].iloc[0]

    table = [
        "| 변형 | phase | CAGR | MDD | Sharpe | Vol | TotalRet |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    variants = [
        ("base_5050", "**base 50/50**"),
        ("yoon2c_direct_buffett", "yoon2c direct"),
        ("yoon2c_inverse_value", "inverse value"),
    ]
    for variant, label in variants:
        for j, phase in enumerate(("train", "validation", "test", "all")):
            r = row_for(variant, phase)
            table.append(
                f"| {label if j == 0 else ''} | {phase} | {pct(r['cagr'])} | "
                f"{pct(r['mdd'])} | {num(r['sharpe'])} | {pct(r['volatility'])} | {pct(r['total_return'])} |"
            )

    all_base = row_for("base_5050", "all")
    all_direct = row_for("yoon2c_direct_buffett", "all")
    all_inverse = row_for("yoon2c_inverse_value", "all")
    val_base = row_for("base_5050", "validation")
    val_direct = row_for("yoon2c_direct_buffett", "validation")
    val_inverse = row_for("yoon2c_inverse_value", "validation")

    w = weights
    recent = w.dropna().iloc[-1]
    latest_source_year = int(recent["source_year"])
    report = f"""# yoon2c 후보: 한국/미국 버핏지수 비율 통화 조절

yoon2의 미국 슬리브(`yoon1j_lp`)와 한국 슬리브(`yoon1j_kr`)는 그대로 두고,
월간 통화 리밸런싱 비중만 고정 50/50에서 World Bank 버핏지수 비율로 바꿔 비교했다.

- 기간: {idx[0].date()} ~ {idx[-1].date()} ({len(idx)}봉)
- 버핏지수: World Bank `{WB_INDICATOR}` = `Stock market capitalization to GDP (%)`
- 무누수 처리: 관측연도 Y 값은 다음 해 1월 1일부터 사용. 2025년 이후 구간은 최신 확정치 forward-fill.
- direct: `w_KRW = KoreaBuffett / (KoreaBuffett + USBuffett)`
- inverse value: `w_KRW = (1/KoreaBuffett) / (1/KoreaBuffett + 1/USBuffett)`; 밸류에이션이 낮은 시장을 더 사는 민감도 비교.

## 판정: yoon2c direct는 보류

직접비율 방식은 train 구간 CAGR과 MDD는 개선되지만 validation과 전구간 Sharpe가 기준 50/50보다 낮다.
가치투자식 역비율도 test 구간 수익률은 좋아 보이나 train·validation Sharpe와 MDD가 기준보다 나빠 연구 게이트를 통과하지 못한다.

{chr(10).join(table)}

## 해석

- direct all: CAGR {pct(all_base['cagr'])} → {pct(all_direct['cagr'])}, Sharpe {num(all_base['sharpe'])} → {num(all_direct['sharpe'])}, MDD {pct(all_base['mdd'])} → {pct(all_direct['mdd'])}.
- direct validation: Sharpe {num(val_base['sharpe'])} → {num(val_direct['sharpe'])}, MDD {pct(val_base['mdd'])} → {pct(val_direct['mdd'])}. 기준보다 위험효율이 낮다.
- inverse value all: CAGR {pct(all_inverse['cagr'])}, Sharpe {num(all_inverse['sharpe'])}; 전구간은 좋아도 validation Sharpe {num(val_inverse['sharpe'])}로 기준 {num(val_base['sharpe'])}보다 낮다.
- 최신 확정 버핏지수 기준({latest_source_year}년 관측치): Korea {recent['kr_buffett']:.2f}%, US {recent['us_buffett']:.2f}%, direct KRW 비중 {recent['direct_wkr'] * 100:.1f}%, inverse KRW 비중 {recent['inverse_value_wkr'] * 100:.1f}%.

## 결론

현재 데이터와 게이트 기준에서는 yoon2의 단순 50/50 통화 리밸런싱을 대체할 만큼
버핏지수 비율 신호가 안정적이지 않다. yoon2c를 만든다면 direct 비율보다는
inverse value 쪽이 후보지만, validation 통과 전까지는 실전 전략 등록을 보류하는 게 맞다.

## 파일

- 스크립트: `scripts/profile_sizing/yoon2c_buffett_ratio_compare.py`
- 결과 CSV: `{csv_path.relative_to(ROOT)}`
- 버핏지수/비중 CSV: `{weight_path.relative_to(ROOT)}`
- 그래프 PNG: `{png_path.relative_to(ROOT)}`
"""
    out = OUT_DIR / "yoon2c_buffett_ratio_compare.md"
    out.write_text(report, encoding="utf-8")
    return out


def plot(navs: dict[str, pd.Series], weights: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    for label, nav in navs.items():
        axes[0].plot(nav.index, nav / nav.iloc[0], label=label)
    axes[0].set_title("yoon2 vs Buffett-ratio currency allocation")
    axes[0].set_ylabel("NAV, start=1")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="upper left")

    axes[1].plot(weights.index, weights["direct_wkr"] * 100.0, label="direct KRW weight")
    axes[1].plot(weights.index, weights["inverse_value_wkr"] * 100.0, label="inverse KRW weight")
    axes[1].axhline(50.0, color="black", lw=1, alpha=0.4)
    axes[1].set_ylabel("KRW weight (%)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("코어 슬리브 로딩(yoon2: yoon1j_lp/yoon1j_kr, KRW-base)...")
    us, kr = load_core_sleeves()
    idx = us.index
    print(f"  기간 {idx[0].date()}~{idx[-1].date()}  {len(idx)}봉")

    print("World Bank 버핏지수 로딩...")
    weights = buffett_weights(idx)
    aligned_idx = idx.intersection(weights.index)
    us = us.reindex(aligned_idx)
    kr = kr.reindex(aligned_idx)
    weights = weights.reindex(aligned_idx).ffill()
    idx = aligned_idx

    base_nav = combine_dynamic(us, kr, None)
    direct_nav = combine_dynamic(us, kr, weights["direct_wkr"])
    inverse_nav = combine_dynamic(us, kr, weights["inverse_value_wkr"])

    phases = ("train", "validation", "test", "all")
    base_perf = {phase: perf_on(base_nav, slice_phase(idx, phase)) for phase in phases}
    rows = []
    rows.extend(phase_rows(base_nav, base_perf, "base_5050", idx))
    rows.extend(phase_rows(direct_nav, base_perf, "yoon2c_direct_buffett", idx))
    rows.extend(phase_rows(inverse_nav, base_perf, "yoon2c_inverse_value", idx))
    result = pd.DataFrame(rows)

    print("\n=== 베이스(고정 50/50) ===")
    for phase in phases:
        print(f"  {phase:11s} {fmt(base_perf[phase])}")
    print("\n=== yoon2c direct: KRW = Korea/(Korea+US) ===")
    for phase in phases:
        print(f"  {phase:11s} {fmt(perf_on(direct_nav, slice_phase(idx, phase)))}")
    print("\n=== inverse value: KRW = (1/Korea)/((1/Korea)+(1/US)) ===")
    for phase in phases:
        print(f"  {phase:11s} {fmt(perf_on(inverse_nav, slice_phase(idx, phase)))}")

    csv_path = OUT_DIR / "yoon2c_buffett_ratio_compare.csv"
    weight_path = OUT_DIR / "yoon2c_buffett_ratio_weights.csv"
    png_path = OUT_DIR / "yoon2c_buffett_ratio_compare.png"
    result.to_csv(csv_path, index=False)
    weights.to_csv(weight_path, index_label="date")
    plot(
        {
            "base 50/50": base_nav,
            "yoon2c direct": direct_nav,
            "inverse value": inverse_nav,
        },
        weights,
        png_path,
    )
    report_path = write_report(result, weights, {
        "base 50/50": base_nav,
        "yoon2c direct": direct_nav,
        "inverse value": inverse_nav,
    }, idx, csv_path, weight_path, png_path)

    print(f"\nCSV: {csv_path}")
    print(f"Weights: {weight_path}")
    print(f"PNG: {png_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
