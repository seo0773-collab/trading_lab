"""yoon1j_lp/yoon1j_kr top_k 스윕 (실제 등록 override 기준, 프로덕션 파이프라인 그대로).

yoon1j_decorrelate_validate.py 검증 과정에서 잡은 두 버그(시프트 정렬 오류,
원본 base config를 그대로 읽어 override 전 top_k/유니버스를 쓰던 문제)를 반영해,
이번엔 top_k 자체를 스윕한다. `Yoon1kHandler._sub_config`로 실제
`yoon2_final_fx_credit_gate.json`의 override(hold_buffer=2, LP는 VGT 유니버스)를
그대로 적용하고 top_k만 바꿔서 `ProfilePortfolioHandler.build_artifacts`(프로덕션
코드 그대로)로 각 슬리브의 CAGR/MDD/Sharpe를 비교한다.

Usage:
    PYTHONPATH=src:scripts python3 scripts/profile_sizing/yoon1j_topk_sweep.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from trading_lab.strategies.profile_portfolio import ProfilePortfolioHandler  # noqa: E402
from trading_lab.strategies.yoon1k import Yoon1kHandler  # noqa: E402

TOPK_GRID = [1, 2, 3, 4, 5, 6, 8]


def _fmt_pct(v: float | None) -> str:
    return "-" if v is None else f"{v * 100:.2f}%"


def _fmt_num(v: float | None) -> str:
    return "-" if v is None else f"{v:.3f}"


def sweep_universe(sid: str, base_overrides: dict[str, Any]) -> list[dict[str, Any]]:
    handler = ProfilePortfolioHandler()
    universe_size = len(
        base_overrides.get("universe")
        or json.loads((ROOT / "configs" / "strategies" / f"{sid}.json").read_text("utf-8"))["universe"]
    )
    rows = []
    # raw(가격 패널)는 top_k와 무관하므로 universe가 같으면 한 번만 로드해 재사용한다.
    cached_raw = None
    for top_k in TOPK_GRID:
        if top_k > universe_size:
            continue
        overrides = dict(base_overrides)
        overrides["top_k"] = top_k
        config = Yoon1kHandler._sub_config(sid, overrides)
        if cached_raw is None:
            cached_raw = handler.load_data("PORTFOLIO", config, synthetic=False)
        art = handler.build_artifacts(cached_raw, config, symbol="PORTFOLIO", phase="all", bars_per_year=252)
        m = art.metrics
        rows.append({
            "top_k": top_k,
            "cagr": m["cagr"], "max_drawdown": m["max_drawdown"], "sharpe": m["sharpe"],
            "trades": m["trades"], "hit_rate": m["hit_rate"],
        })
        print(
            f"  {sid} top_k={top_k}: CAGR {m['cagr']*100:.2f}% MDD {m['max_drawdown']*100:.2f}% "
            f"Sharpe {m['sharpe']:.3f} (trades={m['trades']})"
        )
    return rows


def main() -> None:
    composite = json.loads((ROOT / "configs" / "strategies" / "yoon2_final_fx_credit_gate.json").read_text("utf-8"))
    overrides_by_id = {s["config"]: (s.get("overrides") or {}) for s in composite["sub_strategies"]}

    results = {}
    for sid in ["yoon1j_lp", "yoon1j_kr"]:
        print(f"=== {sid} top_k 스윕 (hold_buffer/universe는 등록값 고정) ===")
        results[sid] = sweep_universe(sid, overrides_by_id[sid])

    out_json = ROOT / "reports" / "profile_sizing" / "yoon1j_topk_sweep.json"
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    lines = [
        "# yoon1j_lp/yoon1j_kr top_k 스윕 (실제 등록 override 기준)",
        "",
        "- hold_buffer=2, 유니버스는 `yoon2_final_fx_credit_gate.json`의 override 그대로 고정, top_k만 스윕",
        "- 파이프라인은 `ProfilePortfolioHandler.build_artifacts` 그대로(연구용 재구현 아님)",
        "",
    ]
    for sid, rows in results.items():
        best_sharpe = max(rows, key=lambda r: r["sharpe"] if r["sharpe"] is not None else -999)
        lines.extend([
            f"## {sid}",
            "",
            "| top_k | CAGR | MDD | Sharpe | trades | hit_rate |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for r in rows:
            marker = " **←현재 등록값**" if r["top_k"] == 3 else (" **←Sharpe 최고**" if r is best_sharpe else "")
            lines.append(
                f"| {r['top_k']}{marker} | {_fmt_pct(r['cagr'])} | {_fmt_pct(r['max_drawdown'])} | "
                f"{_fmt_num(r['sharpe'])} | {r['trades']} | {_fmt_pct(r['hit_rate'])} |"
            )
        lines.append("")

    out_md = ROOT / "reports" / "profile_sizing" / "yoon1j_topk_sweep.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"JSON: {out_json}")
    print(f"MD:   {out_md}")


if __name__ == "__main__":
    main()
