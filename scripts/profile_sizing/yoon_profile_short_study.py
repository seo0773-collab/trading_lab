"""yoon_profile 숏 포지션 확장 검증.

스터디에서 음수였던 셀 — (0,0) 떨어지는 칼, (2,1) 고멀티플 감속 — 에 음수 비중을
허용했을 때 long-only 대비 성과가 개선되는지 변형별로 비교한다. 등록된 전략은
건드리지 않고 이 스크립트 안에서만 음수 비중을 허용한다.

비용: 기존과 동일(회전율 × 왕복 20bp). 펀딩/차입 비용은 미반영(현물 숏 불가 →
실거래는 선물, 펀딩은 역사적으로 롱이 숏에게 지불하는 쪽이 우세라 숏에 유리한
편향 — 즉 여기서 숏이 안 되면 실전에서도 안 된다고 보긴 어렵지만, 여기서 되면
실전 검증이 더 필요).

Usage:
    PYTHONPATH=scripts .venv/bin/python scripts/profile_sizing/yoon_profile_short_study.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from profile_sizing.cycle_state import compute_cycle_state
from profile_sizing.engine import performance
from profile_sizing.yoon_profile import DEFAULT_STATE_WEIGHTS

ROOT = Path(__file__).resolve().parents[2]
FEE_ROUNDTRIP = 20.0 / 10_000.0  # 왕복 20bp (편도 5+5bp × 2)
REBAL_THRESHOLD = 0.05
MAX_STEP = 0.25

# 변형: (0,0)=떨어지는 칼, (2,1)=고멀티플 감속 셀에 숏 허용.
VARIANTS: dict[str, dict[tuple[int, int], float]] = {
    "base_long_only": {},
    "knife_short_-0.5": {(0, 0): -0.5},
    "knife_short_-1.0": {(0, 0): -1.0},
    "decel_short_-0.3": {(2, 1): -0.3},
    "knife-0.5_decel-0.3": {(0, 0): -0.5, (2, 1): -0.3},
}

WINDOWS = {
    "all": (None, None),
    "validation": ("frac", (0.6, 0.8)),
    "test": ("frac", (0.8, 1.0)),
    "bear_2022": ("date", ("2022-01-01", "2022-12-31")),
    "bull_2024~": ("date", ("2024-01-01", None)),
}


def make_weights(overrides: dict[tuple[int, int], float]) -> np.ndarray:
    w = np.asarray(DEFAULT_STATE_WEIGHTS, dtype=float).copy()
    for (mb, ab), v in overrides.items():
        w[mb, ab] = v
    return w


def target_series(state: pd.DataFrame, w: np.ndarray) -> pd.Series:
    mb = state["mult_bin"].to_numpy()
    ab = state["angle_bin"].to_numpy()
    valid = (mb >= 0) & (ab >= 0)
    out = np.zeros(len(state), dtype=float)
    out[valid] = w[mb[valid], ab[valid]]
    return pd.Series(np.clip(out, -1.0, 1.0), index=state.index)


def gradual(target: pd.Series) -> pd.Series:
    actual = np.zeros(len(target), dtype=float)
    prev = 0.0
    for i, tgt in enumerate(np.asarray(target, dtype=float)):
        diff = tgt - prev
        if abs(diff) < REBAL_THRESHOLD:
            new = prev
        else:
            new = prev + float(np.clip(diff, -MAX_STEP, MAX_STEP))
        actual[i] = float(np.clip(new, -1.0, 1.0))
        prev = actual[i]
    return pd.Series(actual, index=target.index)


def port_returns(close: pd.Series, weight: pd.Series) -> pd.Series:
    ret = close.pct_change().fillna(0.0)
    gross = weight.shift(1).fillna(0.0) * ret
    turnover = weight.diff().abs().fillna(abs(weight.iloc[0]))
    return gross - (FEE_ROUNDTRIP / 2.0) * turnover


def slice_ret(port_ret: pd.Series, spec) -> pd.Series:
    kind_bounds = WINDOWS[spec]
    if kind_bounds == (None, None):
        return port_ret
    kind, bounds = kind_bounds
    if kind == "frac":
        lo, hi = bounds
        n = len(port_ret)
        return port_ret.iloc[int(n * lo):int(n * hi)]
    start, end = bounds
    seg = port_ret
    if start is not None:
        seg = seg[seg.index >= start]
    if end is not None:
        seg = seg[seg.index <= end]
    return seg


def window_perf(port_ret: pd.Series) -> dict:
    seg = port_ret.copy()
    if len(seg):
        seg.iloc[0] = 0.0
    equity = (1.0 + seg).cumprod()
    return performance(equity, seg, "1d")


def main() -> int:
    raw = pd.read_parquet(ROOT / "data" / "raw" / "BTCUSDT_1d.parquet")
    close = pd.Series(np.asarray(raw["close"], dtype=float), index=raw.index)
    state = compute_cycle_state(close)

    rows = []
    detail = {}
    for name, overrides in VARIANTS.items():
        w = make_weights(overrides)
        actual = gradual(target_series(state, w))
        pr = port_returns(close, actual)
        short_bars = int((actual < 0).sum())
        short_ret = float(pr[actual.shift(1).fillna(0.0) < 0].sum())
        for win in WINDOWS:
            perf = window_perf(slice_ret(pr, win))
            rows.append({
                "variant": name, "window": win,
                "total_return": perf["total_return"], "sharpe": perf["sharpe"],
                "max_drawdown": perf["max_drawdown"], "cagr": perf["cagr"],
            })
        detail[name] = {
            "short_bars": short_bars,
            "short_bars_share": round(short_bars / len(actual), 4),
            "short_side_cum_return_simple": round(short_ret, 4),
        }

    table = pd.DataFrame(rows)
    out = {
        "variants": {k: {"overrides": {f"{mb},{ab}": v for (mb, ab), v in ov.items()},
                         **detail[k]}
                     for k, ov in VARIANTS.items()},
        "results": table.to_dict(orient="records"),
    }
    out_path = ROOT / "reports" / "profile_sizing" / "yoon_profile_short_study.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float),
                        encoding="utf-8")
    print(f"wrote {out_path}\n")

    for win in WINDOWS:
        sub = table[table["window"] == win].copy()
        sub["ret%"] = (sub["total_return"] * 100).round(1)
        sub["mdd%"] = (sub["max_drawdown"] * 100).round(1)
        sub["sharpe"] = sub["sharpe"].round(3)
        print(f"=== {win} ===")
        print(sub[["variant", "ret%", "sharpe", "mdd%"]].to_string(index=False))
        print()
    print("숏 노출 상세:", json.dumps(detail, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
