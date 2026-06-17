#!/usr/bin/env python
"""yoon1c(종목별 섹터 레짐 필터) vs yoon1/yoon1b 비교.

가설: SPY 단일 시장필터는 전 종목을 같은 신호로 방어시켜, 예컨대 헬스케어가 멀쩡한데도
테크發 SPY 약세로 깎인다. 종목이 '자기 섹터 지수'(반도체 SOXX·헬스 XLV·에너지 XLE 등)
추세로 방어하면 더 정밀해져 불필요한 현금화를 줄이고(수익↑) 진짜 약한 섹터만 빠진다(방어 유지)?

핸들러 경로로 그대로 평가(섹터 데이터 로딩·무누수 포함). 주 벤치마크는 SPY. phase=all과
holdout(test) 양쪽 보고. 점수 산출은 동일하므로 차이는 순전히 레짐 필터에서 온다.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/profile_sizing/sector_regime_compare.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from trading_lab.strategies import get_handler  # noqa: E402

VARIANTS = ["yoon1", "yoon1b", "yoon1c"]


def _f(v, pct=False) -> str:
    if v is None:
        return "—"
    return f"{v*100:.1f}%" if pct else f"{v:.3f}"


def evaluate(strategy_id: str) -> list[dict]:
    handler = get_handler(strategy_id)
    cfg = json.loads(
        (ROOT / "configs" / "strategies" / f"{strategy_id}.json").read_text())
    raw = handler.load_data("PORTFOLIO", cfg, synthetic=False)
    rows = []
    for phase in ("all", "test"):
        art = handler.build_artifacts(raw, cfg, symbol="PORTFOLIO",
                                      phase=phase, bars_per_year=252)
        m = art.metrics
        rows.append({
            "variant": strategy_id, "phase": phase,
            "exposure": art.metadata.get("avg_exposure"),
            "cagr": m["cagr"], "mdd": m["max_drawdown"], "sharpe": m["sharpe"],
            "spy_cagr": m.get("buy_hold_cagr"), "spy_mdd": m.get("buy_hold_max_drawdown"),
            "spy_sharpe": m.get("buy_hold_sharpe"),
            "sharpe_vs_spy": (m["sharpe"] - m["buy_hold_sharpe"])
            if (m["sharpe"] is not None and m.get("buy_hold_sharpe") is not None) else None,
        })
    return rows


def main() -> int:
    rows = []
    for v in VARIANTS:
        print(f"평가: {v} ...", flush=True)
        rows += evaluate(v)
        last = rows[-1]
        print(f"  test: 노출 {_f(last['exposure'], True)} CAGR {_f(last['cagr'], True)} "
              f"MDD {_f(last['mdd'], True)} Sharpe {_f(last['sharpe'])}", flush=True)

    md = _markdown(rows)
    outdir = ROOT / "reports" / "profile_sizing"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "sector_regime_compare.md").write_text(md, encoding="utf-8")
    (outdir / "sector_regime_compare.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8")
    print("\n" + md)
    print(f"리포트: {outdir / 'sector_regime_compare.md'}")
    return 0


def _markdown(rows) -> str:
    def get(v, ph):
        for r in rows:
            if r["variant"] == v and r["phase"] == ph:
                return r
        return None
    lines = [
        "# yoon1c(섹터 레짐 필터) vs yoon1/yoon1b",
        "",
        "yoon1=SPY 단일 시장필터, yoon1b=+게인 1.25, yoon1c=게인 1.25 + **하이브리드**"
        "(SPY 시장필터 ∧ 종목별 섹터필터 동시; 둘 다 약세면 0.5×0.5=0.25 이중 축소). "
        "공통: top_k 20·monthly·floor 1.0. 주 벤치=SPY.",
        "",
        "| 전략 | 구간 | 평균 노출 | CAGR | MDD | Sharpe | SPY CAGR | SPY MDD | SPY Sharpe | 전략−SPY Sharpe |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r['variant']} | {r['phase']} | {_f(r['exposure'], True)} | "
            f"{_f(r['cagr'], True)} | {_f(r['mdd'], True)} | {_f(r['sharpe'])} | "
            f"{_f(r['spy_cagr'], True)} | {_f(r['spy_mdd'], True)} | "
            f"{_f(r['spy_sharpe'])} | **{_f(r['sharpe_vs_spy'])}** |")
    # holdout 요약 + 자동 판정(yoon1c가 yoon1b 대비 test Sharpe·CAGR 개선?)
    b, c = get("yoon1b", "test"), get("yoon1c", "test")
    lines += ["", "## holdout(test) 판정"]
    if b and c:
        d_sharpe = (c["sharpe"] - b["sharpe"]) if (c["sharpe"] and b["sharpe"]) else None
        d_cagr = (c["cagr"] - b["cagr"]) if (c["cagr"] and b["cagr"]) else None
        better = (d_sharpe is not None and d_sharpe > 1e-4)
        lines += [
            f"- yoon1b(SPY필터): CAGR {_f(b['cagr'], True)} · MDD {_f(b['mdd'], True)} · "
            f"Sharpe {_f(b['sharpe'])} · 노출 {_f(b['exposure'], True)}",
            f"- yoon1c(섹터필터): CAGR {_f(c['cagr'], True)} · MDD {_f(c['mdd'], True)} · "
            f"Sharpe {_f(c['sharpe'])} · 노출 {_f(c['exposure'], True)}",
            f"- 차이(yoon1c−yoon1b): Sharpe {_f(d_sharpe)} · CAGR {_f(d_cagr, True)}",
            "",
            ("→ **섹터 레짐 필터 우위**: 위험조정 개선." if better
             else "→ **유의미한 우위 없음**: 섹터 필터가 SPY 단일 대비 위험조정을 못 높임."),
        ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
