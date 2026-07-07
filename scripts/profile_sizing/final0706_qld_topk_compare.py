"""final0706: QLD를 US 슬리브 top_k 유니버스에 추가하는 구성 vs 기존 두 버전.

기존 `final0706_qld`(등록 config)는 QLD를 최상위 core/overlay 블렌드(고정
core_weight=0.635/0.365, 월간 리밸런싱)로 얹은 것이었다. 이번 요청은 그게
아니라 "QLD를 US 슬리브(yoon1j_lp)의 top_k=3 유니버스 후보 중 하나로 넣어서
다른 섹터 ETF와 프로파일 점수(컨트래리언 저가권 스코어)로 경쟁시키는" 구성이다.
universe 리스트에 QLD 한 종목만 추가하고 top_k/hold_buffer/그 외 전부 동일하게
유지 — 공식 ProfilePortfolioHandler 선택 로직을 재구현 없이 그대로 씀.

비교 대상 3종(모두 원화 기준, 공식 핸들러로 계산):
    1) final0706            — 원본(US 유니버스에 QLD 없음)
    2) final0706_qld_topk   — US 유니버스에 QLD 추가(top_k=3 경쟁), KR 슬리브·통화틸트·
                               신용게이트는 원본과 동일
    3) final0706_qld        — 기존 등록 파생(core 63.5%+QLD 36.5% 오버레이 블렌드)

Usage:
    PYTHONPATH=src:scripts python3 scripts/profile_sizing/final0706_qld_topk_compare.py
"""
from __future__ import annotations

import copy
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

from profile_sizing.engine import performance  # noqa: E402
from profile_sizing.yoon2a_bootstrap import _stats, block_indices  # noqa: E402
from trading_lab.strategies.yoon1k import Yoon1kHandler  # noqa: E402
from trading_lab.strategies.yoon1k_leverage_mix import Yoon1kLeverageMixHandler  # noqa: E402

CONFIGS_DIR = ROOT / "configs" / "strategies"


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


def paired_bootstrap(nav_a: pd.Series, nav_b: pd.Series, *, iters: int = 8000,
                      block: int = 20, seed: int = 7) -> dict[str, Any]:
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
        "sharpe_a_gt_b": float((sa > sb).mean()),
        "cagr_a_gt_b": float((ca > cb).mean()),
        "mdd_a_shallower": float((ma > mb).mean()),
    }


def build_qld_topk_config(final0706_cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(final0706_cfg)
    for sub in cfg["sub_strategies"]:
        if sub["currency"] == "USD":
            universe = list(sub["overrides"]["universe"])
            if "QLD" not in universe:
                universe.append("QLD")
            sub["overrides"]["universe"] = universe
    return cfg


def main() -> None:
    final0706_cfg = json.loads((CONFIGS_DIR / "final0706.json").read_text("utf-8"))
    final0706_qld_cfg = json.loads((CONFIGS_DIR / "final0706_qld.json").read_text("utf-8"))
    qld_topk_cfg = build_qld_topk_config(final0706_cfg)

    k_handler = Yoon1kHandler()
    lev_handler = Yoon1kLeverageMixHandler()

    print("final0706 로드 중...", flush=True)
    raw_orig = k_handler.load_data("PORTFOLIO", final0706_cfg, synthetic=False)
    art_orig = k_handler.build_artifacts(raw_orig, final0706_cfg, symbol="PORTFOLIO", phase="all", bars_per_year=252)
    nav_orig = art_orig.equity

    print("final0706_qld_topk(US 유니버스에 QLD 추가) 로드 중...", flush=True)
    raw_topk = k_handler.load_data("PORTFOLIO", qld_topk_cfg, synthetic=False)
    art_topk = k_handler.build_artifacts(raw_topk, qld_topk_cfg, symbol="PORTFOLIO", phase="all", bars_per_year=252)
    nav_topk = art_topk.equity

    print("final0706_qld(기존 오버레이 블렌드) 로드 중...", flush=True)
    raw_qld = lev_handler.load_data("PORTFOLIO", final0706_qld_cfg, synthetic=False)
    art_qld = lev_handler.build_artifacts(raw_qld, final0706_qld_cfg, symbol="PORTFOLIO", phase="all", bars_per_year=252)
    nav_qld = art_qld.equity

    # US 슬리브에서 QLD가 실제로 몇 번 top_k에 편입됐는지(진입 거래 수)로 확인.
    us_trades = art_topk.trades
    qld_entries = int((us_trades["symbol"] == "yoon1j_lp:QLD").sum()) if not us_trades.empty else 0
    us_sleeve_entries = int(us_trades["symbol"].str.startswith("yoon1j_lp:").sum()) if not us_trades.empty else 0

    navs = {
        "final0706 (원본)": nav_orig,
        "final0706_qld_topk (US유니버스+QLD, top_k=3 경쟁)": nav_topk,
        "final0706_qld (core63.5%+QLD36.5% 오버레이)": nav_qld,
    }

    idx = nav_orig.index
    n = len(idx)
    train_frac = float(final0706_cfg.get("train_frac", 0.6))
    validation_frac = float(final0706_cfg.get("validation_frac", 0.2))
    v_end = int(n * (train_frac + validation_frac))
    phases = {
        "all": idx,
        "test": idx[v_end:],
        "recent_3y": idx[idx >= (idx[-1] - pd.Timedelta(days=365 * 3))],
    }

    rows = []
    for phase, window in phases.items():
        for label, nav in navs.items():
            m = _clean_metric(perf_nav(nav, window))
            rows.append({"phase": phase, "candidate": label, **m})

    boot = {
        "qld_topk_vs_원본": paired_bootstrap(nav_topk, nav_orig),
        "qld_overlay_vs_원본": paired_bootstrap(nav_qld, nav_orig),
        "qld_topk_vs_qld_overlay": paired_bootstrap(nav_topk, nav_qld),
    }

    result = {
        "period": f"{idx[0].date()}~{idx[-1].date()}",
        "bars": n,
        "rows": rows,
        "bootstrap": boot,
    }
    out_json = ROOT / "reports" / "profile_sizing" / "final0706_qld_topk_compare.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    lines = [
        "# final0706: QLD를 US top_k 유니버스에 추가 vs 기존 오버레이 블렌드",
        "",
        f"- Period: `{result['period']}` ({n} bars, 원화 기준)",
        "- `final0706_qld_topk` = US 슬리브(yoon1j_lp) universe에 QLD 한 종목만 추가, top_k=3/hold_buffer=2 등 나머지는 원본과 동일"
        " — 공식 ProfilePortfolioHandler 선택 로직(프로파일 컨트래리언 점수) 그대로, KR 슬리브·통화틸트·신용게이트도 원본과 동일.",
        "- `final0706_qld` = 기존 등록 파생(core 63.5%+QLD 36.5% 고정 오버레이, 월간 리밸런싱) 참고용.",
        f"- US 슬리브 전체 진입 거래 {us_sleeve_entries}건 중 QLD 진입 {qld_entries}건"
        f"({(qld_entries / us_sleeve_entries * 100) if us_sleeve_entries else 0:.1f}%).",
        "",
    ]
    for phase in phases:
        lines.append(f"## {phase}")
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

    lines.append("## 페어드 부트스트랩 (8000 iters, block=20)")
    lines.append("")
    lines.append("| vs | P(Sharpe 개선) | P(CAGR 개선) | P(MDD 더 얕음) |")
    lines.append("| --- | ---: | ---: | ---: |")
    for key, b in boot.items():
        lines.append(f"| {key} | {b['sharpe_a_gt_b']*100:.1f}% | {b['cagr_a_gt_b']*100:.1f}% | {b['mdd_a_shallower']*100:.1f}% |")

    lines.extend(["", "## 파일", "", f"- JSON: `{out_json.relative_to(ROOT)}`"])

    out_md = ROOT / "reports" / "profile_sizing" / "final0706_qld_topk_compare.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"JSON: {out_json}")
    print(f"MD:   {out_md}")


if __name__ == "__main__":
    main()
