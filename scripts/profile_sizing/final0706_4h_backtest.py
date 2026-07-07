"""final0706_4h(US+KR 2조, 통화틸트+신용게이트 포함) 실데이터 백테스트.

configs/strategies/final0706_4h.json을 공식 파이프라인(Yoon1kHandler)으로 그대로
돌려 CAGR/MDD/Sharpe를 산출한다(재구현 없음). Sharpe/CAGR은 4h의 실제 연간
거래봉수(6.5h/day 기준)로 직접 계산 — engine.py의 `_ann_factor`는 1d/1h/1wk/1mo만
인식해 4h를 252로 폴백하므로 신뢰하지 않는다.

Usage:
    PYTHONPATH=src:scripts python3 scripts/profile_sizing/final0706_4h_backtest.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from trading_lab.strategies.yoon1k import Yoon1kHandler  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TRADING_MIN_PER_DAY = 390.0
TRADING_DAYS_PER_YEAR = 252.0
BAR_MINUTES = 240
ANN = (TRADING_MIN_PER_DAY / BAR_MINUTES) * TRADING_DAYS_PER_YEAR


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


def main() -> None:
    cfg = json.loads((ROOT / "configs" / "strategies" / "final0706_4h.json").read_text("utf-8"))
    handler = Yoon1kHandler()

    print("final0706_4h 데이터 로드 중 (US+KR+FX+신용게이트 심볼)...", flush=True)
    raw = handler.load_data("PORTFOLIO", cfg, synthetic=False)
    print(f"로드 완료: {raw.shape}", flush=True)

    art = handler.build_artifacts(raw, cfg, symbol="PORTFOLIO", phase="all", bars_per_year=252)
    equity = art.equity
    print(f"equity span: {equity.index.min()} ~ {equity.index.max()}, bars={len(equity)}")

    perf = perf_from_equity(equity, ANN)
    print(f"\n=== final0706_4h 성과 (전체 구간, 4h 연환산) ===")
    print(f"Sharpe={perf['sharpe']:.3f}  CAGR={perf['cagr']*100:.2f}%  MDD={perf['max_drawdown']*100:.2f}%  "
          f"TotalReturn={perf['total_return']*100:.2f}%")

    print("\nmetrics (핸들러 내장, interval 폴백 252 기준 — 참고용):")
    print(art.metrics)

    result = {"span": [str(equity.index.min()), str(equity.index.max())], "bars": len(equity),
             "perf_4h_ann": perf, "handler_metrics": {k: (None if v is None else float(v) if isinstance(v, (int, float)) else v)
                                                       for k, v in art.metrics.items()}}
    out_json = ROOT / "reports" / "profile_sizing" / "final0706_4h_backtest.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    lines = [
        "# final0706_4h 실데이터 백테스트",
        "",
        f"- 구간: `{equity.index.min()}~{equity.index.max()}` ({len(equity)}봉, 4h)",
        f"- Sharpe={perf['sharpe']:.3f}, CAGR={perf['cagr']*100:.2f}%, MDD={perf['max_drawdown']*100:.2f}%, "
        f"TotalReturn={perf['total_return']*100:.2f}%",
        "- Sharpe/CAGR은 4h 실제 연간 거래봉수로 직접 계산(engine.py 기본 ann factor 미사용).",
        "",
        "## 파일",
        "",
        f"- JSON: `{out_json.relative_to(ROOT)}`",
    ]
    out_md = ROOT / "reports" / "profile_sizing" / "final0706_4h_backtest.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nMD: {out_md}")


if __name__ == "__main__":
    main()
