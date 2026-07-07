"""final0706 vs SPY/QQQ/QLD buy&hold 비교.

final0706(원/달러 통화틸트 + 신용스프레드 게이트를 포함한 최종 확정형, 원화 기준)을
실제로 매수 가능한 단일종목 buy&hold 3종과 비교한다:

    - SPY : 시총가중 미국 시장
    - QQQ : 나스닥100
    - QLD : 나스닥100 2배 레버리지

공식 파이프라인(``Yoon1kHandler.build_artifacts``)이 만드는 equity를 그대로 쓰므로
통화틸트·신용게이트 로직을 재구현하지 않는다(공통 인프라 무수정 원칙). 벤치마크는
원화환산(원화 투자자 기준, final0706의 base_currency와 동일 조건)과 참고용 USD
그대로 두 버전을 함께 낸다.

Usage:
    PYTHONPATH=src:scripts python3 scripts/profile_sizing/final0706_benchmark_compare.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from run_kalman_pipeline import load_yfinance  # noqa: E402
from profile_sizing.engine import performance  # noqa: E402
from profile_sizing.yoon2a_bootstrap import _stats, block_indices, pct  # noqa: E402
from trading_lab.strategies.yoon1k import Yoon1kHandler  # noqa: E402

BENCH_SYMBOLS = ["SPY", "QQQ", "QLD"]


def _clean_metric(p: dict[str, Any]) -> dict[str, float | None]:
    keys = ["cagr", "max_drawdown", "sharpe", "total_return", "volatility"]
    return {k: (None if p.get(k) is None else float(p[k])) for k in keys}


def _fmt_pct(v: float | None) -> str:
    return "-" if v is None else f"{v * 100:.2f}%"


def _fmt_num(v: float | None) -> str:
    return "-" if v is None else f"{v:.3f}"


def perf_nav(nav: pd.Series, window: pd.DatetimeIndex | None = None) -> dict[str, Any]:
    s = nav.reindex(window).ffill().dropna() if window is not None else nav.dropna()
    if len(s) < 60:
        return {}
    eq = s / s.iloc[0]
    ret = eq.pct_change().fillna(0.0)
    return performance(eq, ret, "1d")


def paired_bootstrap(
    nav_a: pd.Series, nav_b: pd.Series, *, label_a: str, label_b: str,
    iters: int = 8000, block: int = 20, seed: int = 7,
) -> dict[str, Any]:
    idx = nav_a.dropna().index.intersection(nav_b.dropna().index).sort_values()
    a = nav_a.reindex(idx).ffill()
    b = nav_b.reindex(idx).ffill()
    ra = a.pct_change().fillna(0.0).to_numpy()
    rb = b.pct_change().fillna(0.0).to_numpy()
    rng = np.random.default_rng(seed)
    bidx = block_indices(len(idx), block, iters, rng)
    sa, ca, ma = _stats(ra[bidx])
    sb, cb, mb = _stats(rb[bidx])
    return {
        "label_a": label_a, "label_b": label_b,
        "period": f"{idx[0].date()}~{idx[-1].date()}", "bars": len(idx),
        "prob": {
            "sharpe_a_gt_b": float((sa > sb).mean()),
            "cagr_a_gt_b": float((ca > cb).mean()),
            "mdd_a_shallower": float((ma > mb).mean()),
        },
        "ci95": {
            "excess_sharpe": pct(sa - sb),
            "excess_cagr": pct(ca - cb),
            "excess_mdd": pct(ma - mb),
        },
    }


def main() -> None:
    config = json.loads((ROOT / "configs" / "strategies" / "final0706.json").read_text("utf-8"))
    handler = Yoon1kHandler()
    raw = handler.load_data("PORTFOLIO", config, synthetic=False)
    art = handler.build_artifacts(raw, config, symbol="PORTFOLIO", phase="all", bars_per_year=252)
    strat_nav = art.equity  # KRW 기준, final0706 최종 equity (통화틸트+신용게이트 반영)
    idx = strat_nav.index

    fx = handler._fx(raw, str(config.get("fx_symbol", "KRW=X")))
    fx = pd.Series(fx).reindex(idx).ffill() if fx is not None else None

    bench_krw: dict[str, pd.Series] = {}
    bench_usd: dict[str, pd.Series] = {}
    print("벤치마크 종목 로드 중...", flush=True)
    for sym in BENCH_SYMBOLS:
        px = load_yfinance(sym, "1d", "max")["close"].reindex(idx).ffill()
        usd_nav = (px / px.dropna().iloc[0]).rename(sym)
        bench_usd[sym] = usd_nav
        if fx is not None:
            krw_nav = usd_nav * (fx / fx.dropna().iloc[0])
            bench_krw[sym] = krw_nav.rename(sym)

    train_frac = float(config.get("train_frac", 0.6))
    validation_frac = float(config.get("validation_frac", 0.2))
    n = len(idx)
    t_end = int(n * train_frac)
    v_end = int(n * (train_frac + validation_frac))
    phases = {
        "all": idx,
        "test": idx[v_end:],
        "recent_3y": idx[idx >= (idx[-1] - pd.Timedelta(days=365 * 3))],
    }

    rows = []
    for phase, window in phases.items():
        strat_p = _clean_metric(perf_nav(strat_nav, window))
        rows.append({"phase": phase, "candidate": "final0706", **strat_p})
        for sym in BENCH_SYMBOLS:
            krw_p = _clean_metric(perf_nav(bench_krw[sym], window)) if fx is not None else {}
            usd_p = _clean_metric(perf_nav(bench_usd[sym], window))
            rows.append({"phase": phase, "candidate": f"{sym}_KRW_bnh", **krw_p})
            rows.append({"phase": phase, "candidate": f"{sym}_USD_bnh", **usd_p})

    boot_results = {}
    for sym in BENCH_SYMBOLS:
        if fx is not None:
            boot_results[f"final0706_vs_{sym}_KRW"] = paired_bootstrap(
                strat_nav, bench_krw[sym], label_a="final0706", label_b=f"{sym}_KRW_bnh")

    result = {
        "period": f"{idx[0].date()}~{idx[-1].date()}",
        "bars": n,
        "phases": {k: f"{v[0].date()}~{v[-1].date()}" if len(v) else "-" for k, v in phases.items()},
        "rows": rows,
        "bootstrap": boot_results,
    }
    out_json = ROOT / "reports" / "profile_sizing" / "final0706_benchmark_compare.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    lines = [
        "# final0706 vs SPY/QQQ/QLD buy&hold 비교",
        "",
        f"- Period: `{result['period']}` ({n} bars, 원/달러 KRW 기준)",
        "- final0706 equity는 공식 파이프라인(`Yoon1kHandler.build_artifacts`)이 만든 값을 그대로 사용(재구현 없음).",
        "- `_KRW_bnh` = 원화환산 buy&hold(원화 투자자 기준, final0706과 동일 통화 조건), `_USD_bnh` = 참고용 달러 그대로.",
        "",
    ]
    for phase in phases:
        lines.append(f"## {phase} ({result['phases'][phase]})")
        lines.append("")
        lines.append("| candidate | CAGR | MDD | Sharpe | Total Return |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for r in rows:
            if r["phase"] != phase:
                continue
            lines.append(
                f"| {r['candidate']} | {_fmt_pct(r.get('cagr'))} | {_fmt_pct(r.get('max_drawdown'))} | "
                f"{_fmt_num(r.get('sharpe'))} | {_fmt_pct(r.get('total_return'))} |"
            )
        lines.append("")

    lines.append("## 페어드 부트스트랩 (final0706 vs 원화환산 buy&hold, 8000 iters, block=20)")
    lines.append("")
    lines.append("| vs | P(Sharpe 개선) | P(CAGR 개선) | P(MDD 더 얕음) |")
    lines.append("| --- | ---: | ---: | ---: |")
    for key, b in boot_results.items():
        lines.append(
            f"| {key} | {b['prob']['sharpe_a_gt_b']*100:.1f}% | "
            f"{b['prob']['cagr_a_gt_b']*100:.1f}% | {b['prob']['mdd_a_shallower']*100:.1f}% |"
        )

    lines.extend(["", "## 파일", "", f"- JSON: `{out_json.relative_to(ROOT)}`"])

    out_md = ROOT / "reports" / "profile_sizing" / "final0706_benchmark_compare.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"JSON: {out_json}")
    print(f"MD:   {out_md}")


if __name__ == "__main__":
    main()
