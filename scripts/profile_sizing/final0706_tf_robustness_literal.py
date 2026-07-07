"""final0706 US 슬리브 percentile 버킷 — TF 무관 '리터럴' 파라미터 검증.

이전 스크립트(`final0706_tf_robustness_full.py`)는 일봉 파라미터를 wall-clock
기간이 같아지도록 봉-수 비율로 스케일하거나(불가능하면 압축) 했다. 하지만 60일치
5분봉(≈4,680봉)만 해도 final0706 일봉 스터디 전체(19년, ≈4,800봉)와 맞먹는
**봉 개수**를 갖는다 — "타임프레임을 낮추는" 목적 자체가 더 촘촘한 샘플을 얻는
것이므로, 굳이 wall-clock 기간을 맞출 필요 없이 **원본 파라미터(base_cycle=200봉,
horizon=5/20/60봉)를 그대로**(스케일 없이) 각 TF의 봉 시퀀스에 먹여서 같은
컨트래리언 경향성이 재현되는지 보는 게 더 직접적인 검증이다.

Usage:
    PYTHONPATH=src:scripts python3 scripts/profile_sizing/final0706_tf_robustness_literal.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from final0706_tf_robustness_full import (  # noqa: E402
    BUCKET_ORDER, DAILY_CYCLE_LEN, DAILY_HORIZONS, TFS,
    build_frame, load_raw, load_universe, run_scale, top_bottom_delta,
)
from profile_sizing.config import Profile  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    universe, sub_cfg = load_universe()
    profile_cfg = Profile(
        min_mult=float(sub_cfg["profile"]["min_mult"]), max_mult=float(sub_cfg["profile"]["max_mult"]),
        bin_count=int(sub_cfg["profile"]["bin_count"]), rolling_window=int(sub_cfg["profile"]["rolling_window"]),
        weight_mode=str(sub_cfg["profile"]["weight_mode"]), accumulation_mode=str(sub_cfg["profile"]["accumulation_mode"]),
        percentile_value=float(sub_cfg["profile"]["percentile_value"]),
    )

    all_results = {}
    md_sections = []
    for tf_label, _bar_minutes in TFS:
        print(f"\n=== {tf_label} (리터럴 cycle={DAILY_CYCLE_LEN}, horizons={DAILY_HORIZONS}) ===", flush=True)
        raws = {}
        for sym in universe:
            try:
                raws[sym] = load_raw(sym, tf_label)
            except Exception as e:  # noqa: BLE001
                print(f"  {sym} 로드 실패: {e}")
        if not raws:
            continue
        min_bars = min(len(r) for r in raws.values())
        frames = {sym: build_frame(raw, DAILY_CYCLE_LEN, profile_cfg, DAILY_HORIZONS) for sym, raw in raws.items()}
        stats = run_scale(frames, DAILY_HORIZONS)
        delta = top_bottom_delta(stats, DAILY_HORIZONS)
        print(f"  min_bars={min_bars}")
        print(stats.to_string(index=False))
        print(delta)

        all_results[tf_label] = {
            "min_bars": min_bars, "cycle_len": DAILY_CYCLE_LEN, "horizons": DAILY_HORIZONS,
            "stats": stats.to_dict(orient="records"), "top_bottom_delta": delta,
        }
        md_sections.append(f"## {tf_label} (min_bars={min_bars})")
        md_sections.append("")
        md_sections.append("| bucket(weight) | horizon(봉) | n | mean fwd return | t |")
        md_sections.append("| --- | ---: | ---: | ---: | ---: |")
        for _, r in stats.iterrows():
            md_sections.append(f"| {r['bucket']}({r['bucket_weight']}) | {r['horizon']} | {r['n']} | {r['mean']*100:.2f}% | {r['t_stat']:.2f} |")
        md_sections.append("")
        md_sections.append("**top(B[0,20)) − bottom(B[80,100]) 스프레드**")
        md_sections.append("")
        for h, v in delta.items():
            md_sections.append(f"- {h}: {v['top_minus_bottom_pct']:.2f}%p (t={v['t']:.2f}), 단조감소 {v['monotonic_steps_ok']}")
        md_sections.append("")

    outdir = ROOT / "reports" / "profile_sizing"
    out_json = outdir / "final0706_tf_robustness_literal.json"
    out_json.write_text(json.dumps(all_results, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    lines = [
        "# final0706 US 슬리브 percentile 버킷 — TF 무관 리터럴 파라미터 검증",
        "",
        f"- 유니버스: {', '.join(universe)}",
        f"- **스케일 없음**: 모든 TF에 동일한 base_cycle.length={DAILY_CYCLE_LEN}봉, horizon={DAILY_HORIZONS}봉을 그대로 사용"
        " (wall-clock 기간을 맞추지 않음 — 60일치 5분봉만 해도 일봉 19년 스터디와 맞먹는 봉 개수라 스케일이 불필요하다는 관찰에 근거).",
        "",
    ] + md_sections + ["## 파일", "", f"- JSON: `{out_json.relative_to(ROOT)}`"]
    out_md = outdir / "final0706_tf_robustness_literal.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nMD: {out_md}")


if __name__ == "__main__":
    main()
