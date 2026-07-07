"""final0706 + QLD 오버레이 비중 스윕 — MDD 상한 -30%에서 CAGR이 얼마나 개선되는지 실측.

방법: 기존 yoon1k_levmix 패턴(core 60% + QLD 40%, 이미 registry 등록)과 동일하게
`Yoon1kLeverageMixHandler`를 재사용해 core_strategy=final0706 + QLD(2x Nasdaq100)
오버레이를 월간 리밸런싱으로 블렌드한다(공통 인프라 재구현 없음). QLD 비중을
0%~100%까지 스윕하며 MDD가 -30%를 넘어서는 지점을 찾는다.

config JSON 파일을 새로 만들지 않고 딕셔너리를 직접 핸들러에 넘겨 스윕한다
(final0706.json 자체는 동결 유지, 실험 산출물만 reports에 남긴다).

Usage:
    PYTHONPATH=src:scripts python3 scripts/profile_sizing/final0706_qld_mix_sweep.py
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

from profile_sizing.yoon2a_bootstrap import _stats, block_indices, pct  # noqa: E402
from trading_lab.strategies.yoon1k_leverage_mix import Yoon1kLeverageMixHandler  # noqa: E402

QLD_WEIGHTS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
MDD_CAP = -0.30


def _fmt_pct(v: float | None) -> str:
    return "-" if v is None else f"{v * 100:.2f}%"


def _fmt_num(v: float | None) -> str:
    return "-" if v is None else f"{v:.3f}"


def make_config(qld_weight: float) -> dict[str, Any]:
    return {
        "version": "final0706_qld_sweep",
        "interval": "1d",
        "period": "max",
        "base_currency": "KRW",
        "fx_symbol": "KRW=X",
        "core_strategy": "final0706",
        "rebalance_freq": "monthly",
        "benchmark_label": "50/50 SPY원화+KODEX200",
        "train_frac": 0.6,
        "validation_frac": 0.2,
        "synthetic_symbols": 6,
        "synthetic_bars": 2000,
        "seed": 7,
        "leverage_mix": {
            "core_weight": round(1.0 - qld_weight, 6),
            "rebalance_freq": "monthly",
            "overlays": [
                {"symbol": "QLD", "label": "QLD 2x Nasdaq100", "currency": "USD", "weight": qld_weight}
            ],
        },
    }


def main() -> None:
    handler = Yoon1kLeverageMixHandler()
    rows = []
    equities: dict[float, pd.Series] = {}
    # load_data는 종목 구성(심볼 목록)에만 의존하고 비중에는 의존하지 않는다 —
    # QLD_WEIGHTS 전부 동일한 심볼 집합이므로 raw 데이터는 한 번만 로드해서 재사용.
    print("데이터 로드 중 (1회)...", flush=True)
    raw = handler.load_data("PORTFOLIO", make_config(0.5), synthetic=False)
    print("스윕 실행 중 (final0706 core weight 0~100% QLD)...", flush=True)
    for w in QLD_WEIGHTS:
        cfg = make_config(w)
        art = handler.build_artifacts(raw, cfg, symbol="PORTFOLIO", phase="all", bars_per_year=252)
        equities[w] = art.equity
        m = art.metrics
        rows.append({
            "qld_weight": w,
            "cagr": m.get("cagr"),
            "max_drawdown": m.get("max_drawdown"),
            "sharpe": m.get("sharpe"),
            "total_return": m.get("total_return"),
        })
        print(f"  QLD {w*100:.0f}%: CAGR {_fmt_pct(m.get('cagr'))} MDD {_fmt_pct(m.get('max_drawdown'))} "
              f"Sharpe {_fmt_num(m.get('sharpe'))}", flush=True)

    df = pd.DataFrame(rows)

    # MDD 상한(-30%) 이내에서 CAGR이 최대인 지점(그리드) + 선형보간으로 정확한 교차점 추정
    within = df[df["max_drawdown"] >= MDD_CAP]
    best_grid = within.loc[within["cagr"].idxmax()] if not within.empty else None

    exceed = df[df["max_drawdown"] < MDD_CAP]
    interp = None
    if not within.empty and not exceed.empty:
        # 그리드에서 인접한 마지막 안전점과 첫 초과점 사이를 선형보간
        w_ok = within["qld_weight"].max()
        w_bad = exceed[exceed["qld_weight"] > w_ok]["qld_weight"].min()
        row_ok = df[df["qld_weight"] == w_ok].iloc[0]
        row_bad = df[df["qld_weight"] == w_bad].iloc[0]
        t = (MDD_CAP - row_ok["max_drawdown"]) / (row_bad["max_drawdown"] - row_ok["max_drawdown"])
        w_star = row_ok["qld_weight"] + t * (row_bad["qld_weight"] - row_ok["qld_weight"])
        cagr_star = row_ok["cagr"] + t * (row_bad["cagr"] - row_ok["cagr"])
        sharpe_star = row_ok["sharpe"] + t * (row_bad["sharpe"] - row_ok["sharpe"])
        interp = {"qld_weight": float(w_star), "cagr": float(cagr_star),
                  "max_drawdown": MDD_CAP, "sharpe": float(sharpe_star)}
        # 보간 지점 실측 재실행(선형보간은 근사이므로 실제 백테스트로 확인)
        cfg_star = make_config(float(w_star))
        art_star = handler.build_artifacts(raw, cfg_star, symbol="PORTFOLIO", phase="all", bars_per_year=252)
        interp["measured"] = {
            "cagr": art_star.metrics.get("cagr"),
            "max_drawdown": art_star.metrics.get("max_drawdown"),
            "sharpe": art_star.metrics.get("sharpe"),
        }

    # 부트스트랩: 보간 지점 vs SPY/QQQ/QLD 원화환산 buy&hold, final0706 원본
    from run_kalman_pipeline import load_yfinance  # noqa: E402

    def paired_bootstrap(nav_a: pd.Series, nav_b: pd.Series, label_a: str, label_b: str) -> dict[str, Any]:
        idx = nav_a.dropna().index.intersection(nav_b.dropna().index).sort_values()
        a = nav_a.reindex(idx).ffill()
        b = nav_b.reindex(idx).ffill()
        ra = a.pct_change().fillna(0.0).to_numpy()
        rb = b.pct_change().fillna(0.0).to_numpy()
        rng = np.random.default_rng(11)
        bidx = block_indices(len(idx), 20, 8000, rng)
        sa, ca, ma = _stats(ra[bidx])
        sb, cb, mb = _stats(rb[bidx])
        return {
            "label_a": label_a, "label_b": label_b,
            "prob": {
                "sharpe_a_gt_b": float((sa > sb).mean()),
                "cagr_a_gt_b": float((ca > cb).mean()),
                "mdd_a_shallower": float((ma > mb).mean()),
            },
        }

    boot = {}
    if interp is not None:
        cfg_star = make_config(float(interp["qld_weight"]))
        art_star = handler.build_artifacts(raw, cfg_star, symbol="PORTFOLIO", phase="all", bars_per_year=252)
        star_nav = art_star.equity
        base_nav = equities[0.0]  # final0706 원본(QLD 0%)
        idx = star_nav.index
        fx = handler._fx(raw, "KRW=X")
        fx = pd.Series(fx).reindex(idx).ffill() if fx is not None else None
        for sym in ["SPY", "QQQ", "QLD"]:
            px = load_yfinance(sym, "1d", "max")["close"].reindex(idx).ffill()
            usd_nav = px / px.dropna().iloc[0]
            krw_nav = usd_nav * (fx / fx.dropna().iloc[0]) if fx is not None else usd_nav
            boot[f"final0706_qld_mix_vs_{sym}_KRW"] = paired_bootstrap(
                star_nav, krw_nav, "final0706_qld_mix", f"{sym}_KRW_bnh")
        boot["final0706_qld_mix_vs_final0706"] = paired_bootstrap(
            star_nav, base_nav, "final0706_qld_mix", "final0706")

    result = {
        "grid": rows,
        "mdd_cap": MDD_CAP,
        "best_within_cap_grid": None if best_grid is None else best_grid.to_dict(),
        "interpolated_at_cap": interp,
        "bootstrap": boot,
    }
    out_json = ROOT / "reports" / "profile_sizing" / "final0706_qld_mix_sweep.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    lines = [
        "# final0706 + QLD 오버레이 비중 스윕 (MDD 상한 -30%)",
        "",
        "- 방법: `Yoon1kLeverageMixHandler`(core+overlay 블렌드, 공식 인프라 재사용)로 "
        "core_strategy=final0706 + QLD(2x Nasdaq100) 오버레이를 월간 리밸런싱 블렌드.",
        f"- MDD 상한: `{MDD_CAP*100:.0f}%`",
        "",
        "## 그리드",
        "",
        "| QLD 비중 | CAGR | MDD | Sharpe |",
        "| --- | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r['qld_weight']*100:.0f}% | {_fmt_pct(r['cagr'])} | {_fmt_pct(r['max_drawdown'])} | "
            f"{_fmt_num(r['sharpe'])} |"
        )
    lines.append("")
    if interp is not None:
        lines.extend([
            "## MDD -30% 교차점 (선형보간 → 실측 재검증)",
            "",
            f"- 보간 QLD 비중: `{interp['qld_weight']*100:.1f}%`",
            f"- 보간 CAGR: `{_fmt_pct(interp['cagr'])}` (MDD 정의상 `-30.00%`)",
            f"- 실측 재검증: CAGR `{_fmt_pct(interp['measured']['cagr'])}`, "
            f"MDD `{_fmt_pct(interp['measured']['max_drawdown'])}`, "
            f"Sharpe `{_fmt_num(interp['measured']['sharpe'])}`",
            "",
        ])
    if boot:
        lines.extend([
            "## 페어드 부트스트랩 (8000 iters, block=20)",
            "",
            "| vs | P(Sharpe 개선) | P(CAGR 개선) | P(MDD 더 얕음) |",
            "| --- | ---: | ---: | ---: |",
        ])
        for key, b in boot.items():
            lines.append(
                f"| {key} | {b['prob']['sharpe_a_gt_b']*100:.1f}% | "
                f"{b['prob']['cagr_a_gt_b']*100:.1f}% | {b['prob']['mdd_a_shallower']*100:.1f}% |"
            )
    lines.extend(["", "## 파일", "", f"- JSON: `{out_json.relative_to(ROOT)}`"])

    out_md = ROOT / "reports" / "profile_sizing" / "final0706_qld_mix_sweep.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nJSON: {out_json}")
    print(f"MD:   {out_md}")


if __name__ == "__main__":
    main()
