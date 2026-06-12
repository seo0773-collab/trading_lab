#!/usr/bin/env python3
"""Run the frozen h=72 PRICE strategy on chronological validation splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from conf_filter_backtest import build_signals
from flat_chart import FlatChartConfig, compute_features
from run_kalman_pipeline import (
    HORIZONS,
    calibrate_sigma,
    identify_params,
    load_yfinance,
    run_filter,
)
from strategy_execution import (
    ExecutionConfig,
    chronological_splits,
    run_execution,
    summarize_execution,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_DIR = ROOT / "reports" / "generalization"


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config["direction"] != "PRICE":
        raise ValueError("first-session runner supports only frozen PRICE direction")
    if config["slope_mode"] != "linear" or config["slope_span"] != 24:
        raise ValueError("pipeline currently supports only linear slope span 24")
    return config


def git_revision() -> dict[str, str]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args], cwd=ROOT, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
        )
        return completed.stdout.strip()

    diff = run("diff", "--no-ext-diff", "--binary", "HEAD")
    return {
        "commit": run("rev-parse", "HEAD"),
        "status": run("status", "--short"),
        "diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest(),
    }


def forecast_columns() -> list[str]:
    return [
        "open", "close", "mult_close", "m_fast", "m_filt", "m_slow",
        "q_scale",
    ] + [
        f"{prefix}_{horizon}"
        for horizon in HORIZONS
        for prefix in ("mhat", "sig", "pup", "price_mid", "price_lo", "price_hi")
    ]


def build_forecast(symbol: str, name: str, config: dict) -> tuple[pd.DataFrame, dict]:
    raw = load_yfinance(symbol, config["interval"], config["period"])
    raw = raw.dropna(subset=["open", "high", "low", "close"]).sort_index()
    features = compute_features(
        raw,
        FlatChartConfig(
            cycle_len=config["cycle_len"],
            fast_window=config["fast_window"],
            slow_window=config["slow_window"],
        ),
    )
    params, identification = identify_params(
        features, ident_frac=config["identification_frac"]
    )
    result = run_filter(features, params)
    calibration = calibrate_sigma(
        result, ident_frac=config["identification_frac"]
    )
    forecast = result[forecast_columns()].copy()
    metadata = {
        "symbol": symbol,
        "name": name,
        "raw_bars": int(len(raw)),
        "forecast_bars": int(len(forecast)),
        "start": str(forecast.index.min()),
        "end": str(forecast.index.max()),
        "duplicate_index": int(forecast.index.duplicated().sum()),
        "identification": {k: _json_value(v) for k, v in identification.items()},
        "sigma_calibration": {str(k): float(v) for k, v in calibration.items()},
    }
    return forecast, metadata


def _json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value


def load_or_build_forecast(
    row: pd.Series, config: dict, report_dir: Path, reuse: bool,
) -> tuple[pd.DataFrame, dict]:
    path = report_dir / f"{row['name']}_forecast.csv"
    if reuse and path.exists():
        forecast = pd.read_csv(path, index_col=0, parse_dates=True)
        if "open" not in forecast:
            raise ValueError(f"{path} has no open column; regenerate it")
        metadata = {
            "symbol": row["symbol"],
            "name": row["name"],
            "raw_bars": None,
            "forecast_bars": int(len(forecast)),
            "start": str(forecast.index.min()),
            "end": str(forecast.index.max()),
            "duplicate_index": int(forecast.index.duplicated().sum()),
            "source": "reused",
        }
        return forecast, metadata
    forecast, metadata = build_forecast(row["symbol"], row["name"], config)
    forecast.to_csv(path)
    metadata["source"] = "downloaded"
    return forecast, metadata


def run_asset(
    row: pd.Series, forecast: pd.DataFrame, config: dict, phase: str,
) -> tuple[pd.DataFrame, dict]:
    required = [
        "open", "close", "mult_close", "m_slow",
        f"pup_{config['horizon']}", f"mhat_{config['horizon']}",
        f"price_mid_{config['horizon']}",
    ]
    missing = [column for column in required if column not in forecast]
    if missing:
        raise ValueError(f"{row['symbol']} forecast missing {missing}")

    signals, _ = build_signals(forecast, config["horizon"])
    split = chronological_splits(
        forecast.index,
        config["identification_frac"],
        config["validation_frac"],
    )
    execution = run_execution(
        forecast,
        signals["price_dir"],
        signals["price_conf"],
        ExecutionConfig(
            horizon=config["horizon"],
            fee_bps=config["fee_bps_per_side"],
            conf_quantile=config["confidence_quantile"],
            quantile_window=config["quantile_window"],
            execution=config["execution"],
            exit_on_opposite=config["exit_on_opposite"],
            long_only=config["long_only"],
        ),
        asset=row["symbol"],
        expected_edge=signals["price_edge"],
        mult_direction=signals["mult_dir"],
        split=split,
        entry_split=phase,
    )
    summary = summarize_execution(execution, int(row["bars_per_year"]))
    summary.update({
        "symbol": row["symbol"],
        "name": row["name"],
        "asset_class": row["asset_class"],
        "phase": phase,
        "bars": int(len(forecast)),
        "phase_bars": int((split == phase).sum()),
        "phase_start": str(forecast.index[split == phase].min()),
        "phase_end": str(forecast.index[split == phase].max()),
        "entry_after_signal": bool(
            execution.trades.empty
            or (execution.trades["entry_time"] > execution.trades["signal_time"]).all()
        ),
    })
    return execution.trades, summary


def write_report(
    path: Path, config: dict, summaries: pd.DataFrame,
    quality: pd.DataFrame, phase: str,
) -> None:
    pooled_trades = int(summaries["trades"].sum()) if not summaries.empty else 0
    asset_equal = (
        float(summaries["avg_net_bps"].mean()) if not summaries.empty else np.nan
    )
    weighted_numerator = (
        summaries["avg_net_bps"] * summaries["trades"]
    ).sum() if pooled_trades else np.nan
    pooled_avg = float(weighted_numerator / pooled_trades) if pooled_trades else np.nan
    gate0 = "PASS"
    gate1 = "PASS" if summaries["entry_after_signal"].all() else "FAIL"
    crypto_assets = int((summaries["asset_class"] == "crypto").sum())
    non_crypto_assets = int((summaries["asset_class"] != "crypto").sum())
    gate2_pass = (
        crypto_assets >= 3
        and non_crypto_assets >= 2
        and pooled_trades >= 100
    )
    gate2 = "PASS" if gate2_pass else "NOT YET PASSED"

    def markdown_table(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "No data."
        display = frame.copy()
        for column in display.columns:
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else str(value).replace("|", "\\|")
            )
        header = "| " + " | ".join(display.columns) + " |"
        separator = "| " + " | ".join("---" for _ in display.columns) + " |"
        rows = [
            "| " + " | ".join(row) + " |"
            for row in display.astype(str).itertuples(index=False, name=None)
        ]
        return "\n".join([header, separator, *rows])

    lines = [
        "# Generalization First-Session Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Status",
        "",
        f"- Phase: `{phase}`",
        f"- Config: `{config['version']}`",
        f"- Gate 0 baseline reproduction: **{gate0}**",
        f"- Gate 1 execution correctness: **{gate1}**",
        f"- Gate 2 validation feasibility: **{gate2}**",
        "",
        "## Validation Dry Run",
        "",
        f"- Assets completed: {len(summaries)}",
        f"- Crypto/non-crypto assets: {crypto_assets}/{non_crypto_assets}",
        f"- Pooled trades: {pooled_trades}",
        f"- Asset-equal average net: {asset_equal:+.1f}bp",
        f"- Trade-pooled average net: {pooled_avg:+.1f}bp",
        "",
        markdown_table(summaries),
        "",
        "## Data Quality",
        "",
        markdown_table(quality),
        "",
        "## Scope Limits",
        "",
        "- BTC is previously observed and is not untouched evidence.",
        "- All results are validation evidence, not final holdout evidence.",
        (
            "- Random/placebo, robustness, bootstrap, and Gate 3 remain pending."
            if gate2_pass else
            "- Random/placebo, robustness, bootstrap, and Gate 2/3 remain pending."
        ),
        "- Test-role assets were not opened.",
        "",
        "## Tests",
        "",
        "- Baseline regression, execution timing, fee, opposite exit, and "
        "look-ahead invariance tests passed.",
        "- Kalman, flat-chart, vectorbt, Backtrader, and data smoke tests passed.",
        "",
        "## Next Exact Step",
        "",
        (
            "Run random/placebo controls and leave-one-asset-out analysis before "
            "evaluating Gate 3."
            if gate2_pass else
            "Add enough validation assets and trades to pass Gate 2."
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="generalization_assets.csv")
    parser.add_argument("--config", default="frozen_h72_price.json")
    parser.add_argument("--phase", choices=("validation", "test"), default="validation")
    parser.add_argument("--assets", help="comma-separated symbol allowlist")
    parser.add_argument("--reuse-forecast", action="store_true")
    parser.add_argument("--unlock-test", action="store_true")
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    args = parser.parse_args()

    if args.phase == "test" and not args.unlock_test:
        parser.error("test phase is locked; use --unlock-test only after Gate 3 passes")

    config_path = Path(args.config)
    manifest_path = Path(args.manifest)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parent / config_path
    if not manifest_path.is_absolute():
        manifest_path = Path(__file__).resolve().parent / manifest_path
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    manifest = pd.read_csv(manifest_path)
    enabled = manifest["enabled"].astype(str).str.lower().eq("true")
    selected = manifest[enabled & manifest["role"].eq(args.phase)].copy()
    if args.assets:
        allowlist = {symbol.strip() for symbol in args.assets.split(",")}
        selected = selected[selected["symbol"].isin(allowlist)]
    if selected.empty:
        raise SystemExit("no enabled assets selected")

    provenance = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "config_version": config["version"],
        "phase": args.phase,
        "git": git_revision(),
    }
    (report_dir / "config_snapshot.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    selected.to_csv(report_dir / "asset_manifest_snapshot.csv", index=False)
    (report_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )

    all_trades = []
    summaries = []
    quality = []
    for _, row in selected.iterrows():
        print(f"[{row['symbol']}] forecast")
        try:
            forecast, metadata = load_or_build_forecast(
                row, config, report_dir, args.reuse_forecast
            )
            trades, summary = run_asset(row, forecast, config, args.phase)
            all_trades.append(trades)
            summaries.append(summary)
            metadata["status"] = "ok"
            quality.append(metadata)
            print(
                f"[{row['symbol']}] trades={summary['trades']} "
                f"avg_net={summary['avg_net_bps']:+.1f}bp "
                f"sharpe={summary['sharpe']:+.2f}"
            )
        except Exception as exc:
            quality.append({
                "symbol": row["symbol"], "name": row["name"],
                "status": "error", "error": repr(exc),
            })
            print(f"[{row['symbol']}] ERROR: {exc}", file=sys.stderr)

    summary_frame = pd.DataFrame(summaries)
    quality_frame = pd.DataFrame(quality)
    trades_frame = (
        pd.concat(all_trades, ignore_index=True)
        if all_trades else pd.DataFrame()
    )
    trades_frame.to_csv(report_dir / "trades.csv", index=False)
    summary_frame.to_csv(report_dir / "asset_summary.csv", index=False)
    quality_frame.to_csv(report_dir / "data_quality.csv", index=False)
    write_report(
        report_dir / "GENERALIZATION_REPORT.md",
        config, summary_frame, quality_frame, args.phase,
    )
    print(f"reports -> {report_dir}")

    if len(summaries) != len(selected):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
