"""final0706 비중 편차(weight dispersion) 스터디.

top-K 내부 배분을 score 비례로 하는 기본(score 스킴)은 top-K로 이미 선별된
종목들의 점수가 regime_cap 등으로 서로 비슷하게 압축돼 있어 사실상 거의 균등
배분처럼 보인다(대시보드 관찰). portfolio.py에 opt-in ``weight_power`` 파라미터를
추가해 score**weight_power 로 재정규화하도록 했다(기본 1.0=완전 불변, 다른
전략 영향 없음) — power>1이면 상위 점수 종목에 더 쏠리고(편차 확대), <1이면
더 균등해진다.

본 스크립트는 (1) 실제 top-3 배분의 편차(HHI·최대비중)가 weight_power에 따라
얼마나 벌어지는지 실측하고 (2) final0706 전체 파이프라인(공식 Yoon1kHandler,
재구현 없음)으로 weight_power=2.0/4.0 적용 시 CAGR/MDD/Sharpe가 원본(1.0) 대비
어떻게 바뀌는지 페어드 부트스트랩으로 비교한다. final0706.json 자체는 동결
스냅샷이라 수정하지 않고, sub_strategies overrides에 weight_power만 즉석
주입한 변형 config로 실험한다.

Usage:
    PYTHONPATH=src:scripts python3 scripts/profile_sizing/final0706_weight_dispersion_study.py
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
from trading_lab.strategies.profile_portfolio import ProfilePortfolioHandler  # noqa: E402
from trading_lab.strategies.yoon1k import Yoon1kHandler  # noqa: E402

CONFIGS_DIR = ROOT / "configs" / "strategies"
POWERS = [1.0, 2.0, 4.0]


def _fmt_pct(v: float | None) -> str:
    return "-" if v is None else f"{v * 100:.2f}%"


def _fmt_num(v: float | None) -> str:
    return "-" if v is None else f"{v:.3f}"


def _clean_metric(p: dict[str, Any]) -> dict[str, float | None]:
    keys = ["cagr", "max_drawdown", "sharpe", "total_return", "volatility"]
    return {k: (None if p.get(k) is None else float(p[k])) for k in keys}


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
    }


def build_variant(final0706_cfg: dict[str, Any], power: float) -> dict[str, Any]:
    cfg = copy.deepcopy(final0706_cfg)
    for sub in cfg["sub_strategies"]:
        sub.setdefault("overrides", {})
        sub["overrides"]["weight_power"] = power
    return cfg


def dispersion_stats(sub_id: str, overrides: dict[str, Any], power: float) -> dict[str, float]:
    """단일 슬리브(US yoon1j_lp 또는 KR yoon1j_kr)만 돌려 top-K 실배분 편차를 잰다."""
    handler = ProfilePortfolioHandler()
    cfg = json.loads((CONFIGS_DIR / f"{sub_id}.json").read_text("utf-8"))
    cfg.update(overrides)
    cfg["weight_power"] = power
    universe = list(cfg.get("universe") or [])
    raw = handler.load_data("PORTFOLIO", cfg, synthetic=False)
    art = handler.build_artifacts(raw, cfg, symbol="PORTFOLIO", phase="all", bars_per_year=252)
    wave = art.extras["portfolio_wave"]
    cols = [c for c in universe if c in wave.columns]
    w = wave[cols].to_numpy(dtype=float)
    exposure = w.sum(axis=1)
    mask = exposure > 1e-6
    w, exposure = w[mask], exposure[mask]
    n_active = (w > 1e-9).sum(axis=1)
    wn = np.divide(w, exposure[:, None], out=np.zeros_like(w), where=exposure[:, None] > 0)
    hhi = (wn ** 2).sum(axis=1)
    hhi_equal = 1.0 / np.maximum(n_active, 1)
    top_share = wn.max(axis=1)
    return {
        "mean_hhi": float(np.nanmean(hhi)),
        "mean_hhi_if_equal": float(np.nanmean(hhi_equal)),
        "mean_top_holding_share": float(np.nanmean(top_share)),
        "mean_active_n": float(np.nanmean(n_active)),
        "days": int(mask.sum()),
    }


def main() -> None:
    final0706_cfg = json.loads((CONFIGS_DIR / "final0706.json").read_text("utf-8"))
    k_handler = Yoon1kHandler()

    print("실배분 편차(HHI) 실측 중 (US/KR 슬리브 단독) ...", flush=True)
    disp_rows = []
    us_sub = next(s for s in final0706_cfg["sub_strategies"] if s["currency"] == "USD")
    kr_sub = next(s for s in final0706_cfg["sub_strategies"] if s["currency"] == "KRW")
    for power in POWERS:
        us_stats = dispersion_stats(us_sub["config"], us_sub["overrides"], power)
        kr_stats = dispersion_stats(kr_sub["config"], kr_sub["overrides"], power)
        disp_rows.append({"sleeve": f"US({us_sub['config']})", "weight_power": power, **us_stats})
        disp_rows.append({"sleeve": f"KR({kr_sub['config']})", "weight_power": power, **kr_stats})
        print(f"  power={power}: US mean_hhi={us_stats['mean_hhi']:.3f} "
              f"top_share={us_stats['mean_top_holding_share']:.3f} | "
              f"KR mean_hhi={kr_stats['mean_hhi']:.3f} "
              f"top_share={kr_stats['mean_top_holding_share']:.3f}", flush=True)

    print("전체 파이프라인(Yoon1kHandler) 성과 비교 중 ...", flush=True)
    navs: dict[str, pd.Series] = {}
    for power in POWERS:
        label = f"weight_power={power}" + ("(원본)" if power == 1.0 else "")
        variant_cfg = build_variant(final0706_cfg, power)
        raw = k_handler.load_data("PORTFOLIO", variant_cfg, synthetic=False)
        art = k_handler.build_artifacts(
            raw, variant_cfg, symbol="PORTFOLIO", phase="all", bars_per_year=252)
        navs[label] = art.equity
        print(f"  {label} 완료", flush=True)

    idx = navs["weight_power=1.0(원본)"].index
    train_frac = float(final0706_cfg.get("train_frac", 0.6))
    validation_frac = float(final0706_cfg.get("validation_frac", 0.2))
    n = len(idx)
    v_end = int(n * (train_frac + validation_frac))
    phases = {
        "all": idx,
        "test": idx[v_end:],
        "recent_3y": idx[idx >= (idx[-1] - pd.Timedelta(days=365 * 3))],
    }

    perf_rows = []
    for phase, window in phases.items():
        for label, nav in navs.items():
            m = _clean_metric(perf_nav(nav, window))
            perf_rows.append({"phase": phase, "candidate": label, **m})

    base_label = "weight_power=1.0(원본)"
    boot = {
        f"power{p}_vs_원본": paired_bootstrap(
            navs[f"weight_power={p}"], navs[base_label],
            label_a=f"power{p}", label_b="원본")
        for p in POWERS if p != 1.0
    }

    result = {
        "period": f"{idx[0].date()}~{idx[-1].date()}",
        "bars": n,
        "phases": {k: f"{v[0].date()}~{v[-1].date()}" if len(v) else "-" for k, v in phases.items()},
        "dispersion": disp_rows,
        "performance": perf_rows,
        "bootstrap": boot,
    }
    out_json = ROOT / "reports" / "profile_sizing" / "final0706_weight_dispersion_study.json"
    out_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    lines = [
        "# final0706 비중 편차(weight_power) 스터디",
        "",
        f"- Period: `{result['period']}` ({n} bars, 원화(KRW) 기준)",
        "- 공식 핸들러(`Yoon1kHandler`/`ProfilePortfolioHandler`)로 계산(재구현 없음). "
        "`final0706.json`은 동결 스냅샷이라 수정하지 않고 sub_strategies overrides에 "
        "`weight_power`만 즉석 주입한 변형으로 실험.",
        "- `weight_power`: score^p 재정규화 지수(portfolio.py `_target_weights`). "
        "1.0=원본(불변), >1=상위 점수 종목에 더 쏠림(편차 확대).",
        "",
        "## 1. 실배분 편차 실측 (top-3 내부, 슬리브 단독)",
        "",
        "| sleeve | weight_power | mean HHI | mean HHI(균등가정) | mean 최대종목비중 | mean 보유종목수 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in disp_rows:
        lines.append(
            f"| {r['sleeve']} | {r['weight_power']} | {r['mean_hhi']:.3f} | "
            f"{r['mean_hhi_if_equal']:.3f} | {_fmt_pct(r['mean_top_holding_share'])} | "
            f"{r['mean_active_n']:.2f} |"
        )
    lines += [
        "",
        "(HHI = sum(w_i/exposure)^2, top-3 완전균등이면 1/3≈0.333. 값이 클수록 편중.)",
        "",
        "## 2. 전체 파이프라인 성과 비교",
        "",
    ]
    for phase in phases:
        lines.append(f"### {phase} ({result['phases'][phase]})")
        lines.append("")
        lines.append("| candidate | CAGR | MDD | Sharpe | Total Return |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for r in perf_rows:
            if r["phase"] != phase:
                continue
            lines.append(
                f"| {r['candidate']} | {_fmt_pct(r.get('cagr'))} | {_fmt_pct(r.get('max_drawdown'))} | "
                f"{_fmt_num(r.get('sharpe'))} | {_fmt_pct(r.get('total_return'))} |"
            )
        lines.append("")

    lines.append("## 3. 페어드 블록 부트스트랩 (vs 원본, iters=8000, block=20)")
    lines.append("")
    lines.append("| 비교 | 기간 | P(Sharpe 우위) | P(CAGR 우위) | P(MDD 더 얕음) |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for label, b in boot.items():
        p = b["prob"]
        lines.append(
            f"| {label} | {b['period']} | {p['sharpe_a_gt_b']*100:.1f}% | "
            f"{p['cagr_a_gt_b']*100:.1f}% | {p['mdd_a_shallower']*100:.1f}% |"
        )

    out_md = ROOT / "reports" / "profile_sizing" / "final0706_weight_dispersion_study.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"저장: {out_json}")
    print(f"저장: {out_md}")


if __name__ == "__main__":
    main()
