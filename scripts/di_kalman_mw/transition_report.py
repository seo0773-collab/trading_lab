#!/usr/bin/env python
"""다음 극점(P5) 값 변위 확률구조 리포트 (연구 스크립트; plan 16 standalone).

LV0 극점 → M/W 분류 → 기하 특징 버킷별 P5 정규화 변위 분포를 train에서
적합하고 validation에서 평가해, 버킷 분포표와 조건부 vs 무조건부 성능을
출력·저장한다. 공통 대시보드 파이프라인은 건드리지 않는다.

Usage:
    python scripts/di_kalman_mw/transition_report.py --synthetic --combo A
    python scripts/di_kalman_mw/transition_report.py \
        --data data/raw/BTCUSDT_4h.parquet --symbol BTCUSDT --timeframe 4h
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from di_kalman_mw import run as runner  # noqa: E402
from di_kalman_mw.config import combo_config  # noqa: E402
from di_kalman_mw.extreme_transition import (  # noqa: E402
    build_pattern_dataset,
    build_transition_stats,
    evaluate_transition,
)


def _instances(df, cfg, strict):
    """+DI / -DI Kalman 극점에서 패턴 전이 인스턴스를 만들고 split을 태깅한다."""
    dataset = build_pattern_dataset(df, cfg, strict)
    instances = dataset.instances
    labels = dataset.labels
    # 결정 시점(P4 확정 인덱스)으로 split을 가른다.
    splits = {id(x): str(labels[x.p4_conf_idx]) for x in instances}
    return instances, splits


def _instances_frame(df, instances, splits) -> pd.DataFrame:
    rows = []
    for x in instances:
        rows.append({
            "split": splits[id(x)],
            "line": x.line,
            "pattern": x.pattern,
            "shape": x.shape,
            "width_ratio": x.width_ratio,
            "p4_time": df.index[x.p4_conf_idx],
            "p4_conf_idx": x.p4_conf_idx,
            "p5_conf_idx": x.p5_conf_idx,
            "has_p5": x.has_p5,
            "mean_leg": x.mean_leg,
            "leg3_ratio": x.features["leg3_ratio"],
            "p3_vs_p1_norm": x.features["p3_vs_p1_norm"],
            "dv_norm": x.dv_norm,
            "p5_vs_p3_norm": x.p5_vs_p3_norm,
            "continuation": x.continuation,
        })
    return pd.DataFrame(rows)


def _bucket_table(stats: dict) -> pd.DataFrame:
    rows = []
    for scope, table in (
        ("bucket", stats["buckets"]),
        ("pattern_shape", stats.get("by_pattern_shape", {})),
        ("pattern", stats["by_pattern"]),
    ):
        for key, s in table.items():
            if s is None:
                continue
            rows.append({"scope": scope, "key": key, **s})
    g = stats.get("global")
    if g:
        rows.append({"scope": "global", "key": "global", **g})
    return pd.DataFrame(rows)


def _shape_breakdown(
    train: list, val: list, stats: dict
) -> pd.DataFrame:
    """plan 17: does the diverging/parallel/converging label actually split
    the continuation rate and the conditional skill? One row per shape."""
    from di_kalman_mw.extreme_transition import evaluate_transition

    rows = []
    for shape in ("diverging", "parallel", "converging"):
        tr = [x for x in train if x.shape == shape and x.has_p5
              and np.isfinite(x.dv_norm)]
        cont = (
            float(np.mean([x.continuation for x in tr])) if tr else float("nan")
        )
        ev = evaluate_transition(stats, [x for x in val if x.shape == shape])
        rows.append({
            "shape": shape,
            "train_n": len(tr),
            "train_p_continuation": cont,
            "val_n": ev["n_evaluated"],
            "val_continuation_rate": ev["realized_continuation_rate"],
            "mae_reduction_pct": ev["mae_reduction_pct"],
            "continuation_brier": ev["continuation_brier"],
        })
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="OHLCV parquet/csv path")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--synthetic-bars", type=int, default=9000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--symbol", default="SYNTH")
    parser.add_argument("--timeframe", default="4h")
    parser.add_argument("--combo", default="A", help="plan 17 combo A|B|C|D")
    parser.add_argument("--strict", action="store_true", help="P4>P2(W)/P4<P2(M)")
    parser.add_argument(
        "--outdir", type=Path,
        default=Path(__file__).resolve().parents[2] / "reports" / "di_kalman_mw",
    )
    args = parser.parse_args(argv)

    if args.synthetic:
        df = runner.make_synthetic_ohlcv(
            args.synthetic_bars, args.seed, args.timeframe
        )
        symbol = "SYNTH" if args.symbol == "SYNTH" else args.symbol
    elif args.data:
        df = runner.load_data(args.data)
        symbol = args.symbol
    else:
        parser.error("--data or --synthetic is required")
        return 2

    cfg = combo_config(args.combo)
    instances, splits = _instances(df, cfg, args.strict)
    train = [x for x in instances if splits[id(x)] == "train"]
    val = [x for x in instances if splits[id(x)] == "validation"]

    stats = build_transition_stats(train)
    evaluation = evaluate_transition(stats, val)

    args.outdir.mkdir(parents=True, exist_ok=True)
    prefix = args.outdir / f"{symbol}_{args.timeframe}_transition"
    frame = _instances_frame(df, instances, splits)
    frame.to_csv(f"{prefix}_instances.csv", index=False)
    table = _bucket_table(stats)
    table.to_csv(f"{prefix}_buckets.csv", index=False)
    shape_table = _shape_breakdown(train, val, stats)
    shape_table.to_csv(f"{prefix}_shapes.csv", index=False)
    with open(f"{prefix}_stats.json", "w", encoding="utf-8") as fh:
        json.dump(
            {"stats": stats, "evaluation": evaluation,
             "shape_breakdown": shape_table.to_dict("records")},
            fh, indent=2, default=str,
        )

    n_pat = sum(1 for x in instances if x.has_p5)
    print(
        f"[transition] {symbol} {args.timeframe} combo={args.combo.upper()} "
        f"bars={len(df)} patterns(with P5)={n_pat} "
        f"train={stats['n_train']} val={evaluation['n_evaluated']}"
    )
    print("  버킷 분포 (정규화 변위 dv_norm):")
    if table.empty:
        print("    (표본 없음)")
    else:
        for _, r in table.iterrows():
            cont = r["p_continuation"]
            cont_s = f"{cont:.2f}" if np.isfinite(cont) else "n/a"
            print(
                f"    {r['scope']:>7} {r['key']:<16} n={int(r['n']):>4} "
                f"median={r['median']:+.3f} [q10={r['q10']:+.3f}, "
                f"q90={r['q90']:+.3f}] P(cont)={cont_s}"
            )
    print("  평가 (validation, 조건부 vs 무조건부):")
    print(
        f"    n={evaluation['n_evaluated']} "
        f"cond_MAE={evaluation['conditional_mae']:.3f} "
        f"global_MAE={evaluation['global_mae']:.3f} "
        f"개선={evaluation['mae_reduction_pct']:.1f}% "
        f"커버리지[10,90]={evaluation['coverage_10_90']:.2f} "
        f"continuation_Brier={evaluation['continuation_brier']:.3f}"
    )
    print("  shape별 분해 (확산/유지/수렴이 continuation을 가르는가):")
    for _, r in shape_table.iterrows():
        def _f(v):
            return f"{v:.2f}" if np.isfinite(v) else "n/a"
        print(
            f"    {r['shape']:>10} train_n={int(r['train_n']):>4} "
            f"P(cont|train)={_f(r['train_p_continuation'])} "
            f"val_n={int(r['val_n']):>4} "
            f"val_cont={_f(r['val_continuation_rate'])} "
            f"MAE개선={_f(r['mae_reduction_pct'])}%"
        )
    print(f"  outputs -> {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
