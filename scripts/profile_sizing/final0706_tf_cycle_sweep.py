"""final0706 US 슬리브 percentile 버킷 — TF별 base_cycle(SMA) 길이 스윕.

`final0706_tf_robustness_literal.py`에서 cycle=200봉 고정으로는 5m/15m에서 신호가
사라지고 4h 근방에서만 흔적이 보였다. horizon(5/20/60봉)은 고정한 채 **SMA 길이만
스윕**해서, "몇 봉이어야" 신호가 나타나는지 — 그리고 그게 TF와 무관하게 특정
**wall-clock 기간**(개월 단위)에 수렴하는지를 직접 본다.

Usage:
    PYTHONPATH=src:scripts python3 scripts/profile_sizing/final0706_tf_cycle_sweep.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from final0706_tf_robustness_full import (  # noqa: E402
    DAILY_HORIZONS, TFS, TRADING_MIN_PER_DAY, build_frame, load_raw, load_universe,
    pooled_nw,
)
from profile_sizing.config import Profile  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CYCLE_CANDIDATES = (20, 50, 100, 200, 400, 800, 1600, 3200)


def top_bottom_at(frames, cycle_len, horizons):
    row = {}
    for h in horizons:
        mean_top, se_top, n_top = pooled_nw(frames, 0, 20, f"fwd_{h}", lag=h)
        mean_bot, se_bot, n_bot = pooled_nw(frames, 80, 100, f"fwd_{h}", lag=h)
        delta = mean_top - mean_bot
        se = float(np.sqrt(se_top ** 2 + se_bot ** 2))
        t = delta / se if se > 0 else float("nan")
        row[h] = {"delta_pct": delta * 100, "t": t, "n_top": n_top, "n_bot": n_bot}
    return row


def main() -> None:
    universe, sub_cfg = load_universe()
    profile_cfg = Profile(
        min_mult=float(sub_cfg["profile"]["min_mult"]), max_mult=float(sub_cfg["profile"]["max_mult"]),
        bin_count=int(sub_cfg["profile"]["bin_count"]), rolling_window=int(sub_cfg["profile"]["rolling_window"]),
        weight_mode=str(sub_cfg["profile"]["weight_mode"]), accumulation_mode=str(sub_cfg["profile"]["accumulation_mode"]),
        percentile_value=float(sub_cfg["profile"]["percentile_value"]),
    )

    bar_minutes_map = dict(TFS)
    all_results = {}
    md_sections = []
    for tf_label, bar_minutes in TFS:
        print(f"\n=== {tf_label} ===", flush=True)
        raws = {}
        for sym in universe:
            try:
                raws[sym] = load_raw(sym, tf_label)
            except Exception as e:  # noqa: BLE001
                print(f"  {sym} 로드 실패: {e}")
        if not raws:
            continue
        min_bars = min(len(r) for r in raws.values())
        max_cycle = max(min_bars // 3, 20)
        lengths = [c for c in CYCLE_CANDIDATES if c <= max_cycle]
        if not lengths:
            lengths = [max(20, max_cycle)]
        print(f"  min_bars={min_bars}, sweep lengths={lengths}")

        rows = []
        for cyc in lengths:
            frames = {sym: build_frame(raw, cyc, profile_cfg, DAILY_HORIZONS) for sym, raw in raws.items()}
            tb = top_bottom_at(frames, cyc, DAILY_HORIZONS)
            days_equiv = cyc * bar_minutes / TRADING_MIN_PER_DAY
            rows.append({"cycle_len": cyc, "days_equiv": days_equiv, "top_bottom": tb})
            h60 = tb[60]
            print(f"    cycle={cyc:>5} bars (~{days_equiv:6.1f}거래일): "
                  f"h5 delta={tb[5]['delta_pct']:+.2f}%p(t={tb[5]['t']:+.2f})  "
                  f"h20 delta={tb[20]['delta_pct']:+.2f}%p(t={tb[20]['t']:+.2f})  "
                  f"h60 delta={h60['delta_pct']:+.2f}%p(t={h60['t']:+.2f})")

        all_results[tf_label] = {"bar_minutes": bar_minutes, "min_bars": min_bars, "rows": rows}
        md_sections.append(f"## {tf_label}")
        md_sections.append("")
        md_sections.append("| cycle_len(봉) | ≈거래일 | h=5 delta(t) | h=20 delta(t) | h=60 delta(t) |")
        md_sections.append("| ---: | ---: | ---: | ---: | ---: |")
        for r in rows:
            tb = r["top_bottom"]
            md_sections.append(
                f"| {r['cycle_len']} | {r['days_equiv']:.1f} | "
                f"{tb[5]['delta_pct']:+.2f}%p({tb[5]['t']:+.2f}) | "
                f"{tb[20]['delta_pct']:+.2f}%p({tb[20]['t']:+.2f}) | "
                f"{tb[60]['delta_pct']:+.2f}%p({tb[60]['t']:+.2f}) |"
            )
        md_sections.append("")

    outdir = ROOT / "reports" / "profile_sizing"
    out_json = outdir / "final0706_tf_cycle_sweep.json"
    out_json.write_text(json.dumps(all_results, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    lines = [
        "# final0706 US 슬리브 percentile 버킷 — TF별 base_cycle(SMA) 길이 스윕",
        "",
        f"- 유니버스: {', '.join(universe)}",
        "- horizon(5/20/60봉)은 고정, base_cycle.length(SMA)만 스윕. delta = top(B[0,20)) − bottom(B[80,100]) forward return, pooled NW.",
        "- '≈거래일' = cycle_len × 봉당분 / 390(하루 거래분) — 그 SMA가 실제로 나타내는 wall-clock 기간.",
        "",
    ] + md_sections + ["## 파일", "", f"- JSON: `{out_json.relative_to(ROOT)}`"]
    out_md = outdir / "final0706_tf_cycle_sweep.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nMD: {out_md}")


if __name__ == "__main__":
    main()
