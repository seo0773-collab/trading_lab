"""final0706_qld: QLD 오버레이가 core 위험게이트를 따라가도록 한 변경의 실측 검증.

배경: final0706_qld는 core(final0706) 63.5% + QLD 36.5%를 매월 고정비중으로
리밸런싱하는 구조였다. core는 자체 신용스프레드 게이트(HYG/IEF)가 발동하면
노출을 0.5배로 줄이고 IEF로 대피하는데, QLD 오버레이는 이 게이트와 무관하게
항상 36.5% 고정이었다 — 위기 국면에서 코어는 방어에 들어가도 레버리지 슬리브는
그대로 노출된 채 남는 구조적 공백이 있었다(사용자 지적으로 발견).

변경: `Yoon1kLeverageMixHandler`에 overlay별 `gate_follow` 옵션을 추가해,
core의 `target_w___CASH__` 신호(게이트 발동 시 1-exposure)를 그대로 재사용,
QLD 비중도 같은 비율로 줄이고 줄인 만큼을 core와 동일한 안전자산(IEF)으로
돌리도록 확장했다(신규 게이트 로직 재구현 없음 — core가 계산한 신호 재사용).

이 스크립트는 변경 전(gate_follow 없음, 기존 고정비중)과 변경 후(gate_follow=true)
를 동일 데이터로 비교 실측하고, 페어드 부트스트랩으로 상대 비교를 검증한다.

Usage:
    PYTHONPATH=src:scripts python3 scripts/profile_sizing/final0706_qld_gate_follow_validate.py
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

from profile_sizing.yoon2a_bootstrap import _stats, block_indices  # noqa: E402
from trading_lab.strategies.yoon1k_leverage_mix import Yoon1kLeverageMixHandler  # noqa: E402


def make_config(*, gate_follow: bool) -> dict[str, Any]:
    overlay: dict[str, Any] = {
        "symbol": "QLD", "label": "QLD 2x Nasdaq100", "currency": "USD", "weight": 0.365,
    }
    if gate_follow:
        overlay["gate_follow"] = True
    return {
        "version": "final0706_qld_gate_follow_check",
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
            "core_weight": 0.635,
            "rebalance_freq": "monthly",
            "overlays": [overlay],
        },
    }


def _fmt_pct(v: float | None) -> str:
    return "-" if v is None else f"{v * 100:.2f}%"


def _fmt_num(v: float | None) -> str:
    return "-" if v is None else f"{v:.3f}"


def paired_bootstrap(nav_a: pd.Series, nav_b: pd.Series) -> dict[str, float]:
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
        "sharpe_gate_gt_fixed": float((sa > sb).mean()),
        "cagr_gate_gt_fixed": float((ca > cb).mean()),
        "mdd_gate_shallower": float((ma > mb).mean()),
    }


def main() -> None:
    handler = Yoon1kLeverageMixHandler()
    print("데이터 로드 중...", flush=True)
    raw = handler.load_data("PORTFOLIO", make_config(gate_follow=True), synthetic=False)

    art_fixed = handler.build_artifacts(
        raw, make_config(gate_follow=False), symbol="PORTFOLIO", phase="all", bars_per_year=252)
    art_gate = handler.build_artifacts(
        raw, make_config(gate_follow=True), symbol="PORTFOLIO", phase="all", bars_per_year=252)

    m_fixed, m_gate = art_fixed.metrics, art_gate.metrics
    print("\n[변경 전: QLD 고정 36.5%]")
    print(f"  CAGR {_fmt_pct(m_fixed.get('cagr'))} MDD {_fmt_pct(m_fixed.get('max_drawdown'))} "
          f"Sharpe {_fmt_num(m_fixed.get('sharpe'))}")
    print("[변경 후: QLD gate_follow=true]")
    print(f"  CAGR {_fmt_pct(m_gate.get('cagr'))} MDD {_fmt_pct(m_gate.get('max_drawdown'))} "
          f"Sharpe {_fmt_num(m_gate.get('sharpe'))}")

    boot = paired_bootstrap(art_gate.equity, art_fixed.equity)
    print("\n[페어드 부트스트랩 (8000 iters, block=20), gate_follow vs 고정비중]")
    print(f"  P(Sharpe 개선) {boot['sharpe_gate_gt_fixed']*100:.1f}%  "
          f"P(CAGR 개선) {boot['cagr_gate_gt_fixed']*100:.1f}%  "
          f"P(MDD 더 얕음) {boot['mdd_gate_shallower']*100:.1f}%")

    result = {
        "fixed": {"cagr": m_fixed.get("cagr"), "max_drawdown": m_fixed.get("max_drawdown"),
                  "sharpe": m_fixed.get("sharpe"), "total_return": m_fixed.get("total_return")},
        "gate_follow": {"cagr": m_gate.get("cagr"), "max_drawdown": m_gate.get("max_drawdown"),
                        "sharpe": m_gate.get("sharpe"), "total_return": m_gate.get("total_return")},
        "bootstrap_gate_vs_fixed": boot,
    }
    out_json = ROOT / "reports" / "profile_sizing" / "final0706_qld_gate_follow_validate.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"\nJSON: {out_json}")


if __name__ == "__main__":
    main()
