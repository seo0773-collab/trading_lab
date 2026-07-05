"""Adaptive 3x3 field profile backtest.

초기 calibration 기간으로 종목별 3x3 필드 fingerprint를 만들고, 9개 셀 가중치를
직접 최적화하지 않고 사전 정의된 프로파일 중 하나를 선택한다. 선택 후 OOS 구간은
고정 프로파일로만 운용한다.

Usage:
    PYTHONPATH=src:scripts .venv/bin/python \
        scripts/profile_sizing/adaptive_field_profile_backtest.py
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from profile_sizing.config import config_from_dict  # noqa: E402
from profile_sizing.cycle_state import (  # noqa: E402
    compute_cycle_state,
    forward_returns,
    newey_west_mean_test,
)
from profile_sizing.engine import (  # noqa: E402
    buy_hold_equity,
    equity_from_returns,
    performance,
    portfolio_returns,
)
from run_kalman_pipeline import load_yfinance  # noqa: E402


ETF_SYMBOLS = (
    "SPY", "QQQ", "GLD",
    "XLK", "XLV", "XLF", "XLY", "XLC", "XLI", "XLP", "XLE", "XLU", "XLB", "XLRE",
)
CRYPTO_FILES = {
    "BTC": "BTCUSDT_1d.parquet",
    "ETH": "ETHUSDT_1d.parquet",
    "SOL": "SOLUSDT_1d.parquet",
}
HORIZON = 60
CALIB_BARS = 756
MIN_OOS_BARS = 252


@dataclass(frozen=True)
class Profile:
    name: str
    weights: tuple[tuple[float, ...], ...]


PROFILES: dict[str, Profile] = {
    "drift_soft": Profile(
        "drift_soft",
        (
            (0.85, 1.00, 1.00),
            (0.95, 1.00, 0.90),
            (0.85, 0.90, 1.00),
        ),
    ),
    "balanced": Profile(
        "balanced",
        (
            (0.50, 1.00, 0.90),
            (0.70, 0.90, 0.80),
            (0.50, 0.80, 1.00),
        ),
    ),
    "crypto_trend": Profile(
        "crypto_trend",
        (
            (0.00, 1.00, 0.70),
            (0.35, 0.85, 0.70),
            (0.00, 0.60, 1.00),
        ),
    ),
    "falling_knife_defense": Profile(
        "falling_knife_defense",
        (
            (0.00, 0.80, 0.50),
            (0.25, 0.75, 0.50),
            (0.00, 0.40, 0.85),
        ),
    ),
}
BASE_CRYPTO = Profile(
    "crypto_base",
    (
        (0.00, 1.00, 1.00),
        (0.50, 0.80, 0.30),
        (0.00, 0.30, 1.00),
    ),
)


def load_symbol(symbol: str, kind: str) -> pd.DataFrame:
    if kind == "crypto":
        return pd.read_parquet(ROOT / "data" / "raw" / CRYPTO_FILES[symbol])
    return load_yfinance(symbol, "1d", "max")


def cell_stats(close: pd.Series) -> pd.DataFrame:
    state = compute_cycle_state(close)
    fwd = forward_returns(close, (HORIZON,))
    df = state.join(fwd)
    valid = df[(df["mult_bin"] >= 0) & (df["angle_bin"] >= 0)].copy()
    rows: list[dict[str, Any]] = []
    for mb in range(3):
        for ab in range(3):
            mask = (valid["mult_bin"] == mb) & (valid["angle_bin"] == ab)
            vals = valid.loc[mask, f"fwd_{HORIZON}"].to_numpy(dtype=float)
            mean, se, t_stat = newey_west_mean_test(vals, lag=HORIZON)
            rows.append({
                "cell": f"M{mb}A{ab}",
                "mult_bin": mb,
                "angle_bin": ab,
                "n": int(np.isfinite(vals).sum()),
                "share": float(mask.mean()) if len(valid) else np.nan,
                "fwd60": mean,
                "t60": t_stat,
            })
    return pd.DataFrame(rows)


def select_profile(stats: pd.DataFrame) -> tuple[str, str]:
    by = stats.set_index("cell")
    m0a0 = float(by.loc["M0A0", "fwd60"])
    m0a1 = float(by.loc["M0A1", "fwd60"])
    m1a1 = float(by.loc["M1A1", "fwd60"])
    m2a2 = float(by.loc["M2A2", "fwd60"])
    broad_positive = int((stats["fwd60"] > 0.0).sum())

    if m0a0 > 0.0 and broad_positive >= 7:
        return (
            "drift_soft",
            f"M0A0 양수({m0a0:+.1%}) + 대부분 셀 양수({broad_positive}/9)",
        )
    if m0a0 < 0.0 and m2a2 > max(m0a1, m1a1, 0.0):
        return (
            "crypto_trend",
            f"M0A0 음수({m0a0:+.1%}) + M2A2 모멘텀 우위({m2a2:+.1%})",
        )
    if m0a0 < 0.0:
        return (
            "falling_knife_defense",
            f"M0A0 음수({m0a0:+.1%})이나 M2A2 우위 약함",
        )
    return (
        "balanced",
        f"혼합형: M0A0 {m0a0:+.1%}, M1A1 {m1a1:+.1%}, M2A2 {m2a2:+.1%}",
    )


def calibrated_tilt_profile(stats: pd.DataFrame, kind: str) -> Profile:
    """Baseline weights plus small per-symbol calibration tilts.

    이 변형은 discrete selector보다 더 보수적이다. ETF는 0.85~1.00 범위 안에서만
    ±4%p 틸트하고, crypto는 기존 yoon_profile식 prior에서 ±15%p만 움직인다.
    """
    base = PROFILES["drift_soft"].weights if kind == "etf" else BASE_CRYPTO.weights
    prior = np.asarray(base, dtype=float)
    values = np.full((3, 3), np.nan, dtype=float)
    for row in stats.itertuples(index=False):
        if int(row.n) >= 30 and np.isfinite(float(row.fwd60)):
            values[int(row.mult_bin), int(row.angle_bin)] = float(row.fwd60)

    med = float(np.nanmedian(values))
    scale = float(np.nanmedian(np.abs(values - med)) * 1.4826)
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 0.05 if kind == "etf" else 0.10

    amp = 0.04 if kind == "etf" else 0.15
    tilt = amp * np.tanh((values - med) / scale)
    weights = np.where(np.isfinite(values), prior + tilt, prior)
    if kind == "etf":
        weights = np.clip(weights, 0.85, 1.00)
    else:
        # crypto prior의 핵심 방어 셀은 calibration 노이즈로 크게 풀지 않는다.
        weights[0, 0] = min(weights[0, 0], 0.10)
        weights[2, 0] = min(weights[2, 0], 0.10)
        weights = np.clip(weights, 0.0, 1.0)
    return Profile("calib_tilt", tuple(tuple(float(x) for x in row) for row in weights))


def target_from_weights(state: pd.DataFrame, weights: tuple[tuple[float, ...], ...]) -> pd.Series:
    w = np.asarray(weights, dtype=float)
    mb = state["mult_bin"].to_numpy(dtype=int)
    ab = state["angle_bin"].to_numpy(dtype=int)
    valid = (mb >= 0) & (ab >= 0)
    out = np.zeros(len(state), dtype=float)
    out[valid] = w[mb[valid], ab[valid]]
    return pd.Series(np.clip(out, 0.0, 1.0), index=state.index, name="target_weight")


def gradual(target: pd.Series, threshold: float, step_cap: float) -> pd.Series:
    actual = np.zeros(len(target), dtype=float)
    prev = 0.0
    for i, tgt in enumerate(target.to_numpy(dtype=float)):
        diff = tgt - prev
        if abs(diff) < threshold:
            new = prev
        else:
            new = prev + float(np.clip(diff, -step_cap, step_cap))
        actual[i] = float(np.clip(new, 0.0, 1.0))
        prev = actual[i]
    return pd.Series(actual, index=target.index, name="actual_weight")


def run_profile(
    close: pd.Series,
    state: pd.DataFrame,
    weights: tuple[tuple[float, ...], ...],
    oos_start: pd.Timestamp,
    *,
    step_cap: float,
) -> dict[str, Any]:
    oos_close = close.loc[oos_start:]
    oos_state = state.loc[oos_start:]
    target = target_from_weights(oos_state, weights)
    actual = gradual(target, threshold=0.05, step_cap=step_cap)
    cfg = config_from_dict({"interval": "1d"})
    port_ret = portfolio_returns(oos_close, actual, cfg)
    equity = equity_from_returns(port_ret)
    perf = performance(equity, port_ret, "1d")
    perf["avg_exposure"] = float(actual.mean())
    return perf


def run_bnh(close: pd.Series, oos_start: pd.Timestamp) -> dict[str, Any]:
    oos_close = close.loc[oos_start:]
    bnh = buy_hold_equity(oos_close)
    ret = oos_close.pct_change().fillna(0.0)
    perf = performance(bnh, ret, "1d")
    perf["avg_exposure"] = 1.0
    return perf


def analyze_symbol(symbol: str, kind: str) -> tuple[dict[str, Any], pd.DataFrame]:
    raw = load_symbol(symbol, kind).dropna(subset=["close"])
    close = pd.Series(np.asarray(raw["close"], dtype=float), index=pd.to_datetime(raw.index))
    state = compute_cycle_state(close)
    valid_idx = state.index[(state["mult_bin"] >= 0) & (state["angle_bin"] >= 0)]
    if len(valid_idx) < CALIB_BARS + MIN_OOS_BARS:
        raise RuntimeError(
            f"{symbol}: valid bars {len(valid_idx)} < {CALIB_BARS + MIN_OOS_BARS}"
        )

    calib_eval_end = valid_idx[CALIB_BARS - HORIZON - 1]
    calib_stats = cell_stats(close.loc[:calib_eval_end])
    profile_name, reason = select_profile(calib_stats)
    selected = PROFILES[profile_name]
    tilted = calibrated_tilt_profile(calib_stats, kind)
    oos_start = valid_idx[CALIB_BARS]
    step_cap = 1.0 if kind == "etf" else 0.25

    baseline = PROFILES["drift_soft"] if kind == "etf" else BASE_CRYPTO
    variants = {
        "adaptive_profile": selected,
        "calib_tilt": tilted,
        "class_baseline": baseline,
    }
    rows: list[dict[str, Any]] = []
    for variant, profile in variants.items():
        perf = run_profile(close, state, profile.weights, oos_start, step_cap=step_cap)
        rows.append({
            "symbol": symbol,
            "kind": kind,
            "variant": variant,
            "profile": profile.name,
            "selection_reason": reason if variant == "adaptive_profile" else "",
            "oos_start": str(pd.Timestamp(oos_start).date()),
            "oos_bars": int(len(close.loc[oos_start:])),
            **perf,
        })
    rows.append({
        "symbol": symbol,
        "kind": kind,
        "variant": "bnh",
        "profile": "buy_hold",
        "selection_reason": "",
        "oos_start": str(pd.Timestamp(oos_start).date()),
        "oos_bars": int(len(close.loc[oos_start:])),
        **run_bnh(close, oos_start),
    })
    detail = calib_stats.assign(symbol=symbol, kind=kind, selected_profile=profile_name)
    return rows, detail


def fmt_pct(x: Any) -> str:
    if x is None or not np.isfinite(float(x)):
        return "n/a"
    return f"{float(x) * 100:+.1f}%"


def fmt_num(x: Any) -> str:
    if x is None or not np.isfinite(float(x)):
        return "n/a"
    return f"{float(x):.3f}"


def write_report(results: pd.DataFrame, details: pd.DataFrame, out: Path) -> None:
    lines = [
        "# Adaptive 3x3 field profile backtest",
        "",
        f"- Calibration: first {CALIB_BARS} valid 3x3 bars after indicator warmup",
        f"- Calibration stats exclude the last {HORIZON} bars to avoid forward-return leakage",
        "- OOS: fixed selected profile, no refit",
        "- Costs: fee 5bp + slippage 5bp per turnover, same engine convention as yoon_profile",
        "",
        "## Selection",
        "",
        "| symbol | kind | selected | reason | OOS start |",
        "| --- | --- | --- | --- | --- |",
    ]
    sel = results[results["variant"] == "adaptive_profile"]
    for r in sel.itertuples(index=False):
        lines.append(
            f"| {r.symbol} | {r.kind} | {r.profile} | {r.selection_reason} | {r.oos_start} |"
        )

    lines += [
        "",
        "## OOS performance",
        "",
        "| symbol | variant | profile | total | CAGR | Sharpe | MDD | avg exp |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in results.itertuples(index=False):
        lines.append(
            f"| {r.symbol} | {r.variant} | {r.profile} | "
            f"{fmt_pct(r.total_return)} | {fmt_pct(r.cagr)} | {fmt_num(r.sharpe)} | "
            f"{fmt_pct(r.max_drawdown)} | {fmt_pct(r.avg_exposure)} |"
        )

    piv = results.pivot_table(
        index=["kind", "symbol"],
        columns="variant",
        values=["cagr", "sharpe", "max_drawdown"],
        aggfunc="first",
    )
    lines += [
        "",
        "## Verdict table",
        "",
        "| symbol | variant | vs class baseline | vs B&H | verdict |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for (kind, symbol), _ in piv.iterrows():
        sub = results[(results["kind"] == kind) & (results["symbol"] == symbol)]
        c = sub[sub["variant"] == "class_baseline"].iloc[0]
        b = sub[sub["variant"] == "bnh"].iloc[0]
        for variant in ("adaptive_profile", "calib_tilt"):
            a = sub[sub["variant"] == variant].iloc[0]
            d_cagr_c = float(a["cagr"] - c["cagr"])
            d_sh_c = float(a["sharpe"] - c["sharpe"])
            d_cagr_b = float(a["cagr"] - b["cagr"])
            d_sh_b = float(a["sharpe"] - b["sharpe"])
            if d_sh_c > 0 and d_cagr_c >= -0.002:
                verdict = "baseline 대비 개선"
            elif d_sh_c < 0 and d_cagr_c < 0:
                verdict = "baseline 열위"
            else:
                verdict = "혼재"
            lines.append(
                f"| {symbol} | {variant} | CAGR {d_cagr_c * 100:+.2f}p / "
                f"Sharpe {d_sh_c:+.3f} | CAGR {d_cagr_b * 100:+.2f}p / "
                f"Sharpe {d_sh_b:+.3f} | {verdict} |"
            )

    lines += [
        "",
        "## Notes",
        "",
        "- This is a first-pass initial-calibration design, not a production strategy.",
        "- The profile selector is rule-based and predeclared in the script; it does not grid-search weights.",
        "- ETF baseline is `drift_soft`; crypto baseline is the existing yoon_profile-style `crypto_base`.",
        "- A stronger next test is annual walk-forward reclassification with hysteresis.",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    details.to_csv(out.with_name("adaptive_field_profile_calibration_cells.csv"), index=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir", type=Path, default=ROOT / "reports" / "profile_sizing"
    )
    args = parser.parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    details: list[pd.DataFrame] = []
    for sym in ETF_SYMBOLS:
        r, d = analyze_symbol(sym, "etf")
        rows.extend(r)
        details.append(d)
    for sym in CRYPTO_FILES:
        r, d = analyze_symbol(sym, "crypto")
        rows.extend(r)
        details.append(d)

    results = pd.DataFrame(rows)
    detail_df = pd.concat(details, ignore_index=True)
    results_path = args.outdir / "adaptive_field_profile_backtest.csv"
    report_path = args.outdir / "adaptive_field_profile_backtest.md"
    results.to_csv(results_path, index=False)
    write_report(results, detail_df, report_path)
    print(f"wrote {results_path}")
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
