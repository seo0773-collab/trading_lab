"""final0706 US 슬리브 — TF(1m~4h) × 기간요소 축소비율 크로스, Sharpe 유지 지점 탐색.

final0706 US 슬리브(yoon1j_lp, top_k=3/hold_buffer=2)의 "기간요소"(길이 파라미터)를
전부 같은 비율로 축소하며 실제 전략 파이프라인(ProfilePortfolioHandler, market_filter·
safe_sleeve·regime_cap·top_k 선택 전부 포함)을 TF별로 실행해 Sharpe를 직접 비교한다.

스케일 대상(같은 비율 r을 곱함, 반올림):
    base_cycle.length (200), profile.rolling_window (126),
    market_filter.ma_len (100), safe_sleeve.ma_len (100),
    regime_cap.recovery_stage_bars (21)
스케일 대상 아님(그대로 유지):
    rebalance_freq="monthly"(달력 개념), hold_buffer=2(리밸런스 이벤트 수 단위),
    profile.bin_count/min_mult/max_mult(가격배수 축, 시간축 아님), costs(수수료).

기준(reference) Sharpe: 그 TF가 실제로 덮는 달력 구간만 잘라낸 **일봉·무스케일**
final0706 US 슬리브 성과 — TF 효과와 레짐(그 기간이 상승장이었는지 등) 효과를
분리하기 위한 대조군.

Sharpe/CAGR은 engine.performance()의 _ann_factor(1d/1h/1wk/1mo만 인식)에 기대지
않고, 각 TF의 실제 하루 거래시간(6.5h) 기준 연간 봉수로 직접 재계산한다.

Usage:
    PYTHONPATH=src:scripts python3 scripts/profile_sizing/final0706_period_scale_cross_tf.py
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from run_kalman_pipeline import load_yfinance  # noqa: E402
from trading_lab.strategies.profile_portfolio import ProfilePortfolioHandler  # noqa: E402
from trading_lab.strategies.yoon1k import Yoon1kHandler  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TRADING_MIN_PER_DAY = 390.0
TRADING_DAYS_PER_YEAR = 252.0
SCALE_KEYS = [
    ("base_cycle", "length", 200),
    ("profile", "rolling_window", 126),
    ("market_filter", "ma_len", 100),
    ("safe_sleeve", "ma_len", 100),
    ("regime_cap", "recovery_stage_bars", 21),
]
TFS = [("1m", 1), ("5m", 5), ("15m", 15), ("30m", 30), ("1h", 60), ("4h", 240)]
RATIOS = (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)
EXTRA_SYMBOLS = ["SPY", "TLT", "IAU"]  # market_filter + safe_sleeve


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

        # 대조군: 그 TF가 덮는 달력 구간만 잘라낸 일봉·무스케일 성과.
        daily_panels_window = {}
        for sym, df in daily_panels_full.items():
            d = df.loc[(df.index >= span_start.normalize() - pd.Timedelta(days=250)) & (df.index <= span_end)]
            daily_panels_window[sym] = d
        ref_cfg = copy.deepcopy(base_cfg)
        ref_cfg["interval"] = "1d"
        ref_equity = run_variant(daily_panels_window, ref_cfg)
        ref_perf = perf_from_equity(ref_equity, TRADING_DAYS_PER_YEAR) if ref_equity is not None else {}
        ref_equity_win = ref_equity.loc[ref_equity.index >= span_start.normalize()] if ref_equity is not None else None
        ref_perf_win = perf_from_equity(ref_equity_win, TRADING_DAYS_PER_YEAR) if ref_equity_win is not None else {}
        print(f"  기준(일봉, 해당 구간) Sharpe={ref_perf_win.get('sharpe')}, CAGR={ref_perf_win.get('cagr')}, MDD={ref_perf_win.get('max_drawdown')}")

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
            rows.append({"ratio": r, "cycle_len": cyc_len, "days_equiv": days_equiv, **perf})
            print(f"  ratio={r:<5} cycle_len={cyc_len:<5}(~{days_equiv:6.1f}일) "
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
        md_sections.append(f"- 기준(일봉·무스케일·동일구간) Sharpe: **{ref_sharpe:.3f}**" if ref_sharpe is not None else "- 기준 Sharpe: 계산불가")
        md_sections.append("")
        md_sections.append("| ratio | cycle_len(봉) | ≈거래일 | Sharpe | CAGR | MDD |")
        md_sections.append("| ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in rows:
            sh = row.get("sharpe")
            cg = row.get("cagr")
            md_val = row.get("max_drawdown")
            sh_str = "-" if sh is None else f"{sh:.3f}" + (f" ({sh/ref_sharpe*100:.0f}%)" if ref_sharpe else "")
            cg_str = "-" if cg is None else f"{cg*100:.2f}%"
            mdd_str = "-" if md_val is None else f"{md_val*100:.2f}%"
            md_sections.append(
                f"| {row['ratio']} | {row['cycle_len']} | {row['days_equiv']:.1f} | {sh_str} | {cg_str} | {mdd_str} |"
            )
        md_sections.append("")

    outdir = ROOT / "reports" / "profile_sizing"
    out_json = outdir / "final0706_period_scale_cross_tf.json"
    out_json.write_text(json.dumps(all_results, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    lines = [
        "# final0706 US 슬리브 — TF(1m~4h) × 기간요소 축소비율 크로스",
        "",
        f"- 유니버스: {', '.join(universe)} + market_filter/safe_sleeve용 {', '.join(EXTRA_SYMBOLS)}",
        "- 스케일 대상: base_cycle.length, profile.rolling_window, market_filter.ma_len, safe_sleeve.ma_len,"
        " regime_cap.recovery_stage_bars — 전부 같은 ratio를 곱함. rebalance_freq(monthly)·hold_buffer는 유지.",
        "- 기준 Sharpe = 그 TF가 덮는 달력 구간만 잘라낸 일봉·무스케일 성과(레짐 대조군).",
        "- Sharpe/CAGR은 각 TF의 실제 연간 거래봉수로 직접 계산(engine.py 기본 _ann_factor 미사용).",
        "",
    ] + md_sections + ["## 파일", "", f"- JSON: `{out_json.relative_to(ROOT)}`"]
    out_md = outdir / "final0706_period_scale_cross_tf.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nMD: {out_md}")


if __name__ == "__main__":
    main()
