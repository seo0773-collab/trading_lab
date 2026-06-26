"""yoon1b 유니버스 재설계 A안 — 메가캡 30종 → 섹터 ETF.

가설: 분산·저가권 매수·방어 엔진은 메가캡(=SPY가 이미 담음)보다 **서로 덜 상관된
자산군**(섹터 ETF, +채권/금)에서 진짜 위험조정 엣지를 낼 수 있다. 섹터 ETF는
상장폐지가 거의 없어 생존편향도 작다. 엔진/핸들러 변경 없이 config의 universe만 교체.
top_k를 줄이면 강한(점수상 저가권) 섹터로 로테이션. 주 벤치마크=SPY. test=holdout.

실행:
    PYTHONPATH=src .venv/bin/python scripts/yoon1b_sector_universe_compare.py
"""
from __future__ import annotations

import copy
import json

from trading_lab.paths import ROOT
from trading_lab.strategies import get_handler

# 표준 11 SPDR 섹터(XLRE 2015~·XLC 2018~는 상장 후부터 자동 편입).
SECTORS = ["XLK", "XLF", "XLV", "XLE", "XLI", "XLP", "XLY", "XLU",
           "XLB", "XLRE", "XLC"]
# 비상관 분산 추가(채권·금).
DIVERSIFIED = SECTORS + ["TLT", "GLD"]


def _build(handler, raw, cfg, ph):
    a = handler.build_artifacts(raw, cfg, symbol="PORTFOLIO", phase=ph,
                                bars_per_year=252)
    m = a.metrics
    return dict(
        exp=a.metadata.get("avg_exposure"), nsym=a.metadata.get("n_symbols"),
        cagr=m.get("cagr"), sharpe=m.get("sharpe"), mdd=m.get("max_drawdown"),
        vol=m.get("volatility"),
        spy=(m.get("buy_hold_cagr"), m.get("buy_hold_sharpe"),
             m.get("buy_hold_max_drawdown"), m.get("benchmark_kind")),
        ew=(m.get("ew_index_cagr"), m.get("ew_index_sharpe"),
            m.get("ew_index_max_drawdown")),
    )


def run() -> str:
    handler = get_handler("yoon1b")
    base_cfg = json.loads(
        (ROOT / "configs" / "strategies" / "yoon1b.json").read_text("utf-8")
    )
    lines = ["# yoon1b 유니버스 재설계 A안 — 섹터 ETF\n"]
    lines.append(
        "베이스 엔진=yoon1b(monthly·gain1.25·SPY200MA필터·SMA추세). universe만 교체. "
        "top_k=전체면 사이징만, 작으면 저가권 섹터 로테이션. 주 벤치마크=SPY, "
        "EW=유니버스 등가중. **test=holdout(OOS).**\n"
    )

    runs = [
        ("SECTORS(11)", SECTORS, [11, 6, 4]),
        ("SECTORS+TLT+GLD(13)", DIVERSIFIED, [13, 6]),
    ]
    for uni_name, universe, topks in runs:
        cfg_load = copy.deepcopy(base_cfg)
        cfg_load["universe"] = universe
        print(f"[load] {uni_name} 로딩...", flush=True)
        raw = handler.load_data("PORTFOLIO", cfg_load, synthetic=False)
        lines.append(f"\n## 유니버스 = {uni_name}\n")
        for ph in ("all", "test"):
            lines.append(f"\n### phase={ph}\n")
            lines.append("| variant | nsym | exp | CAGR | vol | Sharpe | MDD |")
            lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
            spy = ew = None
            for tk in topks:
                cfg = copy.deepcopy(cfg_load)
                cfg["top_k"] = tk
                r = _build(handler, raw, cfg, ph)
                spy, ew = r["spy"], r["ew"]
                lbl = f"top_k={tk}" + (" (all)" if tk >= len(universe) else " 로테이션")
                lines.append(
                    f"| {lbl} | {r['nsym']} | {_p(r['exp'])} | {_p(r['cagr'])} | "
                    f"{_p(r['vol'])} | {_f(r['sharpe'])} | {_p(r['mdd'])} |"
                )
            lines.append(
                f"| **SPY (B&H)** | - | 100% | {_p(spy[0])} | - | {_f(spy[1])} | "
                f"{_p(spy[2])} |"
            )
            lines.append(
                f"| EW {uni_name} | - | 100% | {_p(ew[0])} | - | {_f(ew[1])} | "
                f"{_p(ew[2])} |"
            )
    return "\n".join(lines) + "\n"


def _p(v):
    return "-" if v is None else f"{v*100:+.1f}%"


def _f(v):
    return "-" if v is None else f"{v:.3f}"


def main() -> None:
    report = run()
    print(report)
    out = ROOT / "reports" / "profile_sizing" / "sector_universe_compare.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
