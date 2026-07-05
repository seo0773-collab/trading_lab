"""yoon_profile 숏 스퀴즈 방어장치 검증.

yoon_profile_short_study의 결론: (0,0) 칼 숏 −0.5는 방향은 맞으나 상태 판정
재료(SMA200 기울기)가 느려 급반등(스퀴즈)에 청산이 늦다. 여기서는 숏 상태에서만
작동하는 3종 가드를 얹어 재검증한다:

  ma20      — 종가 > SMA20이면 숏 목표를 0으로 (빠른 추세 오버라이드)
  trailN    — 숏 에피소드 중 최저 종가 대비 +N% 반등 시 에피소드 끝까지 숏 금지
  fast_exit — 숏 축소 방향은 봉당 스텝 제한 없이 즉시 (진입은 0.25 유지)

가드는 전부 인과적(당일 종가·과거 데이터만 사용, 비중은 다음 봉 적용).

Usage:
    PYTHONPATH=scripts .venv/bin/python scripts/profile_sizing/yoon_profile_squeeze_guard_study.py
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
FEE_ROUNDTRIP = 20.0 / 10_000.0
REBAL_THRESHOLD = 0.05
MAX_STEP = 0.25
SHORT_CELL = (0, 0)
SHORT_WEIGHT = -0.5

VARIANTS = {
    "base_long_only": None,
    "knife-0.5_naked": {},
    "knife-0.5_ma20": {"ma_guard": 20},
    "knife-0.5_trail10": {"trail_pct": 0.10},
    "knife-0.5_trail15": {"trail_pct": 0.15},
    "knife-0.5_ma20_fastexit": {"ma_guard": 20, "fast_exit": True},
    "knife-0.5_trail15_fastexit": {"trail_pct": 0.15, "fast_exit": True},
}

WINDOWS = {
    "all": (None, None),
    "validation": ("frac", (0.6, 0.8)),
    "test": ("frac", (0.8, 1.0)),
    "bear_2022": ("date", ("2022-01-01", "2022-12-31")),
    "bull_2024~": ("date", ("2024-01-01", None)),
}


def raw_targets(state: pd.DataFrame, short_cfg: dict | None) -> np.ndarray:
    """가드 적용 전 목표 비중. short_cfg=None이면 long-only 기본 행렬."""
    w = np.asarray(DEFAULT_STATE_WEIGHTS, dtype=float).copy()
    if short_cfg is not None:
        w[SHORT_CELL] = SHORT_WEIGHT
    mb = state["mult_bin"].to_numpy()
    ab = state["angle_bin"].to_numpy()
    valid = (mb >= 0) & (ab >= 0)
    out = np.zeros(len(state), dtype=float)
    out[valid] = w[mb[valid], ab[valid]]
    return out


def apply_guards(
    target: np.ndarray, close: pd.Series, short_cfg: dict | None
) -> tuple[np.ndarray, dict]:
    """숏 목표에만 가드 적용. 반환: (가드 적용 목표, 가드 발동 통계)."""
    if not short_cfg:
        return target, {"guard_bars": 0}
    px = close.to_numpy(dtype=float)
    ma = (
        close.rolling(int(short_cfg["ma_guard"])).mean().to_numpy()
        if "ma_guard" in short_cfg else None
    )
    trail = float(short_cfg.get("trail_pct", 0.0))

    out = target.copy()
    guard_bars = 0
    episode_min = np.nan
    trail_tripped = False
    for i in range(len(out)):
        if target[i] >= 0.0:          # 숏 상태 아님 → 에피소드 리셋
            episode_min = np.nan
            trail_tripped = False
            continue
        episode_min = px[i] if np.isnan(episode_min) else min(episode_min, px[i])
        guarded = False
        if ma is not None and np.isfinite(ma[i]) and px[i] > ma[i]:
            guarded = True
        if trail > 0.0:
            if px[i] > episode_min * (1.0 + trail):
                trail_tripped = True
            if trail_tripped:
                guarded = True
        if guarded:
            out[i] = 0.0
            guard_bars += 1
    return out, {"guard_bars": guard_bars}


def gradual(target: np.ndarray, index: pd.Index, fast_exit: bool) -> pd.Series:
    actual = np.zeros(len(target), dtype=float)
    prev = 0.0
    for i, tgt in enumerate(target):
        diff = tgt - prev
        if abs(diff) < REBAL_THRESHOLD:
            new = prev
        else:
            # fast_exit: 숏 축소(음수 → 0 방향)는 스텝 제한 없이 즉시.
            closing_short = prev < 0.0 and diff > 0.0
            if fast_exit and closing_short:
                step = min(diff, -prev) if tgt > 0.0 else diff
                new = prev + step
            else:
                new = prev + float(np.clip(diff, -MAX_STEP, MAX_STEP))
        actual[i] = float(np.clip(new, -1.0, 1.0))
        prev = actual[i]
    return pd.Series(actual, index=index)


def port_returns(close: pd.Series, weight: pd.Series) -> pd.Series:
    ret = close.pct_change().fillna(0.0)
    gross = weight.shift(1).fillna(0.0) * ret
    turnover = weight.diff().abs().fillna(abs(weight.iloc[0]))
    return gross - (FEE_ROUNDTRIP / 2.0) * turnover


def slice_ret(port_ret: pd.Series, win: str) -> pd.Series:
    kind_bounds = WINDOWS[win]
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
    rets = {}
    for name, short_cfg in VARIANTS.items():
        tgt = raw_targets(state, short_cfg)
        tgt, stats = apply_guards(tgt, close, short_cfg)
        fast_exit = bool(short_cfg.get("fast_exit", False)) if short_cfg else False
        actual = gradual(tgt, close.index, fast_exit)
        pr = port_returns(close, actual)
        rets[name] = pr
        detail[name] = {
            **stats,
            "short_bars": int((actual < 0).sum()),
            "turnover_sum": round(float(actual.diff().abs().sum()), 1),
        }
        for win in WINDOWS:
            perf = window_perf(slice_ret(pr, win))
            rows.append({"variant": name, "window": win,
                         "total_return": perf["total_return"],
                         "sharpe": perf["sharpe"],
                         "max_drawdown": perf["max_drawdown"]})

    table = pd.DataFrame(rows)
    # 연도별 기여(무방비 숏 대비 가드가 바꾼 것 + long-only 대비)
    yearly = {}
    base = rets["base_long_only"]
    for name in VARIANTS:
        if name == "base_long_only":
            continue
        diff = (rets[name] - base)
        yearly[name] = (diff.groupby(diff.index.year).sum() * 100).round(1).to_dict()

    out = {
        "short_cell": list(SHORT_CELL), "short_weight": SHORT_WEIGHT,
        "detail": detail,
        "results": table.to_dict(orient="records"),
        "yearly_diff_vs_long_only_pp": yearly,
    }
    out_path = ROOT / "reports" / "profile_sizing" / "yoon_profile_squeeze_guard_study.json"
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
    print("가드/회전 상세:")
    for k, v in detail.items():
        print(f"  {k}: {v}")
    print("\nlong-only 대비 연도별 기여(pp):")
    for k, v in yearly.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
