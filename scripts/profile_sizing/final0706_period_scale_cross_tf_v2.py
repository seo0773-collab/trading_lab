"""v1(final0706_period_scale_cross_tf.py) + 리밸런싱 주기도 같은 비율로 스케일.

v1은 base_cycle/rolling_window/ma_len류만 스케일하고 rebalance_freq="monthly"는
달력 개념이라 그대로 뒀다. 이번엔 그것도 같은 ratio로 줄인다: "monthly"≈21거래일을
기준으로 rebal_every_bars = round(21 × ratio) 봉마다 리밸런싱하도록 바꾼다.

`profile_sizing.portfolio.rebalance_dates`는 달력 문자열만 받으므로, "every_N"
포맷을 추가로 해석하는 패치를 씌운다(원본 동작은 그대로 두고 새 포맷만 추가).

Usage:
    PYTHONPATH=src:scripts python3 scripts/profile_sizing/final0706_period_scale_cross_tf_v2.py
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import profile_sizing.portfolio as portfolio_mod  # noqa: E402
from run_kalman_pipeline import load_yfinance  # noqa: E402
from trading_lab.strategies.profile_portfolio import ProfilePortfolioHandler  # noqa: E402
from trading_lab.strategies.yoon1k import Yoon1kHandler  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TRADING_MIN_PER_DAY = 390.0
TRADING_DAYS_PER_YEAR = 252.0
MONTHLY_BARS_DAILY = 21  # "monthly" ≈ 21거래일(일봉 기준)
SCALE_KEYS = [
    ("base_cycle", "length", 200),
    ("profile", "rolling_window", 126),
    ("market_filter", "ma_len", 100),
    ("safe_sleeve", "ma_len", 100),
    ("regime_cap", "recovery_stage_bars", 21),
]
TFS = [("1m", 1), ("5m", 5), ("15m", 15), ("30m", 30), ("1h", 60), ("4h", 240)]
RATIOS = (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)
EXTRA_SYMBOLS = ["SPY", "TLT", "IAU"]

_orig_rebalance_dates = portfolio_mod.rebalance_dates


def _patched_rebalance_dates(index: pd.DatetimeIndex, freq: str) -> set:
    if isinstance(freq, str) and freq.startswith("every_"):
        n = max(1, int(freq.split("_", 1)[1]))
        return set(index[::n])
    return _orig_rebalance_dates(index, freq)


portfolio_mod.rebalance_dates = _patched_rebalance_dates


def load_raw(sym: str, tf_label: str) -> pd.DataFrame:
    if tf_label == "4h":
        raw_1h = load_yfinance(sym, "1h", "max")
        agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
        if "volume" in raw_1h:
            agg["volume"] = "sum"
        return raw_1h.resample("4h", origin="start").agg(agg).dropna(subset=["close"])
    return load_yfinance(sym, tf_label, "max")


def us_sleeve_config() -> dict[str, Any]:
    final0706_cfg = json.loads((ROOT / "configs" / "strategies" / "final0706.json").read_text("utf-8"))
    us_sub = next(s for s in final0706_cfg["sub_strategies"] if s["currency"] == "USD")
    return Yoon1kHandler._sub_config(us_sub["config"], us_sub.get("overrides"))


def scale_config(base_cfg: dict[str, Any], ratio: float, *, interval: str) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    cfg["interval"] = interval
    for group, key, default in SCALE_KEYS:
        cur = int(cfg.get(group, {}).get(key, default))
        cfg[group][key] = max(2, int(round(cur * ratio)))
    rebal_bars = max(1, int(round(MONTHLY_BARS_DAILY * ratio)))
    cfg["rebalance_freq"] = f"every_{rebal_bars}"
    cfg["_rebal_bars"] = rebal_bars  # 리포트용 메타(엔진은 무시)
    return cfg


def bars_per_year(bar_minutes: float) -> float:
    return (TRADING_MIN_PER_DAY / bar_minutes) * TRADING_DAYS_PER_YEAR


def perf_from_equity(equity: pd.Series, ann: float) -> dict[str, float | None]:
    eq = equity.dropna()
    if len(eq) < 10:
        return {"sharpe": None, "cagr": None, "max_drawdown": None, "total_return": None}
    ret = eq.pct_change().fillna(0.0).to_numpy()
    sd = np.nanstd(ret, ddof=0)
    sharpe = float(np.nanmean(ret) / sd * np.sqrt(ann)) if sd > 0 else None
    n_years = max(len(eq) / ann, 1e-9)
    cagr = float(eq.iloc[-1] ** (1.0 / n_years) - 1.0) if eq.iloc[-1] > 0 else None
    mdd = float((eq / eq.cummax() - 1.0).min())
    return {"sharpe": sharpe, "cagr": cagr, "max_drawdown": mdd,
           "total_return": float(eq.iloc[-1] - 1.0)}


def run_variant(panels: dict[str, pd.DataFrame], cfg: dict[str, Any]) -> pd.Series | None:
    handler = ProfilePortfolioHandler()
    wide = handler._to_wide(panels)
    try:
        art = handler.build_artifacts(wide, cfg, symbol="PORTFOLIO", phase="all", bars_per_year=252)
    except Exception as e:  # noqa: BLE001
        print(f"    실패: {type(e).__name__}: {e}")
        return None
    return art.equity


def main() -> None:
    base_cfg = us_sleeve_config()
    universe = list(base_cfg["universe"])
    all_symbols = universe + EXTRA_SYMBOLS

    print("일봉(무스케일) 전체 기간 로드 중...", flush=True)
    daily_panels_full = {sym: load_yfinance(sym, "1d", "max") for sym in all_symbols}

    all_results = {}
    md_sections = []
    for tf_label, bar_minutes in TFS:
        print(f"\n=== {tf_label} ===", flush=True)
        panels = {}
        for sym in all_symbols:
            try:
                panels[sym] = load_raw(sym, tf_label)
            except Exception as e:  # noqa: BLE001
                print(f"  {sym} 로드 실패: {e}")
        min_bars = min(len(panels[s]) for s in universe if s in panels)
        span_start = min(panels[s].index.min() for s in universe if s in panels)
        span_end = max(panels[s].index.max() for s in universe if s in panels)
        print(f"  span={span_start}~{span_end}, min_bars(유니버스)={min_bars}")

        daily_panels_window = {}
        for sym, df in daily_panels_full.items():
            d = df.loc[(df.index >= span_start.normalize() - pd.Timedelta(days=250)) & (df.index <= span_end)]
            daily_panels_window[sym] = d
        ref_cfg = copy.deepcopy(base_cfg)
        ref_cfg["interval"] = "1d"
        ref_equity = run_variant(daily_panels_window, ref_cfg)
        ref_equity_win = ref_equity.loc[ref_equity.index >= span_start.normalize()] if ref_equity is not None else None
        ref_perf_win = perf_from_equity(ref_equity_win, TRADING_DAYS_PER_YEAR) if ref_equity_win is not None else {}
        print(f"  기준(일봉, 해당 구간, monthly 리밸런싱 유지) Sharpe={ref_perf_win.get('sharpe')}")

        ann = bars_per_year(bar_minutes)
        max_bars = int(min_bars // 3)
        rows = []
        for r in RATIOS:
            cyc_len = max(2, int(round(200 * r)))
            if cyc_len > max_bars:
                print(f"  ratio={r}: cycle_len={cyc_len} > 가용한도 {max_bars} — 스킵")
                continue
            scaled = scale_config(base_cfg, r, interval=tf_label)
            eq = run_variant(panels, scaled)
            perf = perf_from_equity(eq, ann) if eq is not None else {}
            days_equiv = cyc_len * bar_minutes / TRADING_MIN_PER_DAY
            rebal_days_equiv = scaled["_rebal_bars"] * bar_minutes / TRADING_MIN_PER_DAY
            rows.append({"ratio": r, "cycle_len": cyc_len, "days_equiv": days_equiv,
                        "rebal_bars": scaled["_rebal_bars"], "rebal_days_equiv": rebal_days_equiv, **perf})
            print(f"  ratio={r:<5} cycle_len={cyc_len:<5}(~{days_equiv:6.1f}일) "
                  f"rebal_every={scaled['_rebal_bars']:<4}봉(~{rebal_days_equiv:4.1f}일) "
                  f"Sharpe={perf.get('sharpe')} CAGR={perf.get('cagr')} MDD={perf.get('max_drawdown')}")

        all_results[tf_label] = {
            "bar_minutes": bar_minutes, "min_bars": min_bars,
            "span": [str(span_start), str(span_end)],
            "reference_daily_same_window": ref_perf_win, "rows": rows,
        }

        ref_sharpe = ref_perf_win.get("sharpe")
        md_sections.append(f"## {tf_label}")
        md_sections.append("")
        md_sections.append(f"- 커버 구간: `{span_start}~{span_end}` (유니버스 최소 유효봉수 {min_bars})")
        md_sections.append(f"- 기준(일봉·무스케일·monthly 리밸런싱·동일구간) Sharpe: **{ref_sharpe:.3f}**" if ref_sharpe is not None else "- 기준 Sharpe: 계산불가")
        md_sections.append("")
        md_sections.append("| ratio | cycle_len(봉,≈일) | 리밸런싱 주기(봉,≈일) | Sharpe | CAGR | MDD |")
        md_sections.append("| ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in rows:
            sh = row.get("sharpe")
            cg = row.get("cagr")
            mv = row.get("max_drawdown")
            sh_str = "-" if sh is None else f"{sh:.3f}" + (f" ({sh/ref_sharpe*100:.0f}%)" if ref_sharpe else "")
            cg_str = "-" if cg is None else f"{cg*100:.2f}%"
            mdd_str = "-" if mv is None else f"{mv*100:.2f}%"
            md_sections.append(
                f"| {row['ratio']} | {row['cycle_len']}(~{row['days_equiv']:.1f}일) | "
                f"{row['rebal_bars']}(~{row['rebal_days_equiv']:.1f}일) | {sh_str} | {cg_str} | {mdd_str} |"
            )
        md_sections.append("")

    outdir = ROOT / "reports" / "profile_sizing"
    out_json = outdir / "final0706_period_scale_cross_tf_v2.json"
    out_json.write_text(json.dumps(all_results, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    lines = [
        "# final0706 US 슬리브 — TF(1m~4h) × 기간요소+리밸런싱주기 축소비율 크로스 (v2)",
        "",
        f"- 유니버스: {', '.join(universe)} + market_filter/safe_sleeve용 {', '.join(EXTRA_SYMBOLS)}",
        "- v1과 차이: rebalance_freq도 같은 ratio로 스케일(monthly≈21거래일 → round(21×ratio)봉마다 리밸런싱).",
        "- 기준은 v1과 동일하게 일봉·무스케일·monthly 리밸런싱(그 TF가 덮는 달력 구간).",
        "",
    ] + md_sections + ["## 파일", "", f"- JSON: `{out_json.relative_to(ROOT)}`"]
    out_md = outdir / "final0706_period_scale_cross_tf_v2.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nMD: {out_md}")


if __name__ == "__main__":
    main()
