#!/usr/bin/env python
"""yoon1d(RSI 필터) 멀티포트폴리오 테스트 리포트.

yoon1d = yoon1b + SPY 일봉 RSI(14)의 MA(14)가 50 미만이면 전체 목표 노출을
축소하는 포트폴리오 레벨 레짐 필터다. 같은 손픽 30종목 유니버스에서 yoon1b,
yoon1c(섹터 필터), yoon1d를 비교한다.

주의: 이 리포트는 현재 yoon1 30종목 유니버스 기준이다. PIT/S&P500 시점구성 검증이
아니므로 수익 기대치는 생존편향에 낙관적일 수 있다.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/profile_sizing/yoon1d_rsi_filter_report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from trading_lab.strategies import get_handler  # noqa: E402

VARIANTS = ("yoon1b", "yoon1c", "yoon1d")


def _f(v, pct: bool = False) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    return f"{v*100:.1f}%" if pct else f"{v:.3f}"


def _config(strategy_id: str) -> dict:
    return json.loads(
        (ROOT / "configs" / "strategies" / f"{strategy_id}.json").read_text(
            encoding="utf-8"
        )
    )


def evaluate(strategy_id: str, raw) -> list[dict]:
    handler = get_handler(strategy_id)
    cfg = _config(strategy_id)
    rows = []
    for phase in ("all", "validation", "test"):
        art = handler.build_artifacts(
            raw,
            cfg,
            symbol="PORTFOLIO",
            phase=phase,
            bars_per_year=252,
        )
        m = art.metrics
        fc = art.forecast
        rsi_scale = fc["rsi_filter"] if "rsi_filter" in fc else None
        rsi_active = (
            float((rsi_scale < 1.0).mean()) if rsi_scale is not None else 0.0
        )
        rsi_avg_scale = float(rsi_scale.mean()) if rsi_scale is not None else 1.0
        rows.append({
            "variant": strategy_id,
            "phase": phase,
            "start": str(fc.index[0].date()) if len(fc) else None,
            "end": str(fc.index[-1].date()) if len(fc) else None,
            "bars": int(len(fc)),
            "avg_exposure": art.metadata.get("avg_exposure"),
            "avg_holdings": art.metadata.get("avg_holdings"),
            "trades": m.get("trades"),
            "cagr": m.get("cagr"),
            "mdd": m.get("max_drawdown"),
            "sharpe": m.get("sharpe"),
            "bench_kind": m.get("benchmark_kind"),
            "bench_cagr": m.get("buy_hold_cagr"),
            "bench_mdd": m.get("buy_hold_max_drawdown"),
            "bench_sharpe": m.get("buy_hold_sharpe"),
            "sharpe_vs_bench": (
                m.get("sharpe") - m.get("buy_hold_sharpe")
                if m.get("sharpe") is not None
                and m.get("buy_hold_sharpe") is not None
                else None
            ),
            "rsi_active_ratio": rsi_active,
            "rsi_avg_scale": rsi_avg_scale,
        })
    return rows


def main() -> int:
    print("데이터 로드: yoon1 30종목 멀티포트폴리오 ...", flush=True)
    base_handler = get_handler("yoon1d")
    base_cfg = _config("yoon1d")
    raw = base_handler.load_data("PORTFOLIO", base_cfg, synthetic=False)

    rows = []
    for strategy_id in VARIANTS:
        print(f"평가: {strategy_id} ...", flush=True)
        rows.extend(evaluate(strategy_id, raw))
        test = next(
            r for r in rows if r["variant"] == strategy_id and r["phase"] == "test"
        )
        print(
            f"  test: CAGR {_f(test['cagr'], True)} / "
            f"MDD {_f(test['mdd'], True)} / Sharpe {_f(test['sharpe'])} / "
            f"노출 {_f(test['avg_exposure'], True)}",
            flush=True,
        )

    outdir = ROOT / "reports" / "profile_sizing"
    outdir.mkdir(parents=True, exist_ok=True)
    md = _markdown(rows)
    (outdir / "yoon1d_rsi_filter_report.md").write_text(md, encoding="utf-8")
    (outdir / "yoon1d_rsi_filter_report.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    print("\n" + md)
    print(f"리포트: {outdir / 'yoon1d_rsi_filter_report.md'}")
    return 0


def _get(rows, variant, phase):
    for row in rows:
        if row["variant"] == variant and row["phase"] == phase:
            return row
    return None


def _table(rows, phase: str) -> list[str]:
    lines = [
        f"## {phase} 결과",
        "",
        "| 전략 | 구간 | 평균 노출 | CAGR | MDD | Sharpe | SPY CAGR | SPY MDD | SPY Sharpe | 전략-SPY Sharpe | RSI 축소일 | RSI 평균배율 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        if r["phase"] != phase:
            continue
        label = {
            "yoon1b": "yoon1b 기준",
            "yoon1c": "yoon1c 섹터",
            "yoon1d": "yoon1d RSI",
        }.get(r["variant"], r["variant"])
        lines.append(
            f"| {label} | {r['start']}~{r['end']} | "
            f"{_f(r['avg_exposure'], True)} | {_f(r['cagr'], True)} | "
            f"{_f(r['mdd'], True)} | {_f(r['sharpe'])} | "
            f"{_f(r['bench_cagr'], True)} | {_f(r['bench_mdd'], True)} | "
            f"{_f(r['bench_sharpe'])} | **{_f(r['sharpe_vs_bench'])}** | "
            f"{_f(r['rsi_active_ratio'], True)} | {_f(r['rsi_avg_scale'], True)} |"
        )
    return lines


def _markdown(rows) -> str:
    b = _get(rows, "yoon1b", "test")
    c = _get(rows, "yoon1c", "test")
    d = _get(rows, "yoon1d", "test")
    d_sharpe = d["sharpe"] - b["sharpe"] if b and d else None
    d_cagr = d["cagr"] - b["cagr"] if b and d else None
    d_mdd = d["mdd"] - b["mdd"] if b and d else None
    lines = [
        "# yoon1d RSI 필터 멀티포트폴리오 테스트",
        "",
        "대상: yoon1 30종목 멀티포트폴리오. yoon1d는 yoon1b에 SPY 일봉 RSI(14)의 "
        "SMA(14)가 50 미만일 때 전체 목표 노출을 0.5배로 줄이는 필터를 더한 변형이다. "
        "신호는 전봉 기준이며 warmup 구간은 정상으로 처리한다.",
        "",
        "주의: 이 결과는 손픽 30종목 기준이다. 최신 PIT 검증 결론상 수익 우위는 생존편향에 "
        "낙관적일 수 있으므로, 최종 포지셔닝은 여전히 방어형 배분 관점으로 봐야 한다.",
        "또한 RSI 기준 심볼은 SPY라서 SPY 데이터가 없는 초기 구간은 warmup/정상 상태로 "
        "처리된다. 따라서 최종 판단은 SPY 데이터가 온전히 있는 holdout(test)을 우선한다.",
        "",
        *_table(rows, "test"),
        "",
        "## holdout(test) 판정",
    ]
    if b and d:
        lines.extend([
            f"- yoon1d - yoon1b: CAGR {_f(d_cagr, True)}, MDD {_f(d_mdd, True)}, "
            f"Sharpe {_f(d_sharpe)}.",
            f"- yoon1d RSI 축소 상태: test 일수의 {_f(d['rsi_active_ratio'], True)}, "
            f"평균 RSI 배율 {_f(d['rsi_avg_scale'], True)}, 평균 주식 노출 "
            f"{_f(b['avg_exposure'], True)} -> {_f(d['avg_exposure'], True)}.",
        ])
        if d["sharpe"] > b["sharpe"] and d["mdd"] >= b["mdd"]:
            verdict = "RSI 필터가 yoon1b 대비 위험조정과 낙폭을 동시에 개선했다."
        elif d["mdd"] > b["mdd"] and d["cagr"] < b["cagr"]:
            verdict = "RSI 필터는 낙폭을 줄이는 대신 수익을 포기하는 성격이다."
        elif d["sharpe"] <= b["sharpe"]:
            verdict = "RSI 필터가 holdout에서 yoon1b 대비 위험조정을 높이지 못했다."
        else:
            verdict = "RSI 필터 효과는 혼재되어 추가 스윕이 필요하다."
        lines.append(f"- 결론: {verdict}")
    if c and d:
        lines.append(
            f"- 참고로 yoon1c 대비 yoon1d test Sharpe 차이는 "
            f"{_f(d['sharpe'] - c['sharpe'])}, CAGR 차이는 "
            f"{_f(d['cagr'] - c['cagr'], True)}."
        )

    lines.extend([
        "",
        *_table(rows, "validation"),
        "",
        *_table(rows, "all"),
        "",
        "*이 파일은 `scripts/profile_sizing/yoon1d_rsi_filter_report.py`로 생성됨.*",
    ])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
