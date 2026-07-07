"""top3 선택에 상관계수 페널티를 적용했을 때 실제 성과가 개선되는지 검증.

yoon1j_topk_correlation_audit.py에서 확인된 사실: yoon1j_lp(미국 섹터 ETF
유니버스)는 점수 랭킹만으로 top3를 고르면 무작위 3종목 조합보다도 평균 쌍별
상관이 더 높다(선택 0.610 vs 무작위 0.582, 0.7 초과 달이 35.4%). yoon1j_kr은
반대로 무작위보다 낮아(0.399 vs 0.434) 문제가 없다.

이 스크립트는 `_target_weights`(scripts/profile_sizing/portfolio.py)를 건드리지
않고, 그 앞단에서 점수 패널에 상관계수 페널티를 적용한 "보정 점수"를 만들어
동일한 `simulate_portfolio` 파이프라인(ProfilePortfolioHandler.build_artifacts와
동일한 인자)에 흘려보내 실제 CAGR/MDD/Sharpe가 개선되는지 확인한다.

보정 규칙(무누수, day t의 리밸런스는 t-1까지의 수익률만 사용):
    pool = score>0 후보 중 상위 (top_k+extra) 점수
    avg_corr(s) = pool 내 다른 종목들과의 평균 상관(과거 corr_lookback일)
    adjusted_score(s) = score(s) * (1 - clip(corr_penalty * avg_corr(s), 0, 0.9))

Usage:
    PYTHONPATH=src:scripts python3 scripts/profile_sizing/yoon1j_decorrelate_validate.py
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

from profile_sizing.config import config_from_dict  # noqa: E402
from profile_sizing.engine import performance  # noqa: E402
from profile_sizing.portfolio import compute_universe, rebalance_dates, simulate_portfolio  # noqa: E402
from trading_lab.strategies.profile_portfolio import ProfilePortfolioHandler  # noqa: E402

CORR_LOOKBACK = 126
EXTRA_CANDIDATES = 3  # top_k(3) + 3 = 6명 풀에서 페널티 적용
PENALTIES = [0.0, 0.3, 0.5, 0.7, 1.0]


def _clean_metric(p: dict[str, Any]) -> dict[str, float | None]:
    keys = ["cagr", "max_drawdown", "sharpe", "total_return", "volatility"]
    return {k: (None if p.get(k) is None else float(p[k])) for k in keys}


def _fmt_pct(v: float | None) -> str:
    return "-" if v is None else f"{v * 100:.2f}%"


def _fmt_num(v: float | None) -> str:
    return "-" if v is None else f"{v:.3f}"


def decorrelate_scores(
    scores: pd.DataFrame, prices: pd.DataFrame, rebal: set,
    *, corr_lookback: int, extra_candidates: int, corr_penalty: float, top_k: int,
) -> pd.DataFrame:
    """simulate_portfolio는 내부에서 `sig = scores.shift(1)`을 쓴다. 즉 리밸런스일
    t에 실제로 읽히는 값은 인덱스상 t 바로 전날(`t_prev`)의 점수다. 그래서 보정도
    `t`가 아니라 `t_prev`에 써야 결과에 반영된다.

    주의: 단순히 점수에 상관계수 페널티를 곱하면 `_target_weights`의
    `exposure = mean(strict_topk 점수)` 계산까지 같이 오염돼(페널티로 점수 자체가
    낮아지므로) "덜 상관된 종목을 고른 효과"가 아니라 "그냥 전체 노출을 줄인
    효과"와 뒤섞인다(hold_buffer가 피하는 바로 그 함정). 그래서 여기서는
    ① 그리디 선택(최고점수부터, 이후 정규화점수-페널티*평균상관 최대인 후보 추가)으로
    top_k "멤버십"만 바꾸고, ② 선택된 종목의 점수는 원본 그대로 유지하고 나머지
    풀 멤버만 0으로 눌러 배제한다. 이러면 exposure·종목간 배분 모두 기존
    `_target_weights` 로직 그대로, 멤버십만 상관계수 인지형으로 바뀐다.
    """
    if corr_penalty <= 0.0:
        return scores
    returns = prices.pct_change()
    adj = scores.copy()
    pool_size = top_k + extra_candidates
    idx = scores.index
    for t in idx:
        if t not in rebal:
            continue
        pos = idx.get_loc(t)
        if pos == 0:
            continue
        t_prev = idx[pos - 1]
        row = scores.loc[t_prev]
        pool = row[row > 0].nlargest(pool_size)
        if len(pool) <= top_k:
            continue  # 선택 여지가 없으면 원본 유지
        window = returns.loc[:t_prev].tail(corr_lookback)
        sub = window[pool.index].dropna(how="any")
        if len(sub) < corr_lookback // 2:
            continue
        corr = sub.corr()
        normalized = pool / pool.max()
        selected = [pool.index[0]]
        remaining = list(pool.index[1:])
        while len(selected) < top_k and remaining:
            best_sym, best_obj = None, -np.inf
            for s in remaining:
                avg_c = float(corr.loc[s, selected].mean())
                obj = float(normalized[s]) - corr_penalty * avg_c
                if obj > best_obj:
                    best_obj, best_sym = obj, s
            selected.append(best_sym)
            remaining.remove(best_sym)
        for s in pool.index:
            if s not in selected:
                adj.loc[t_prev, s] = 0.0
    return adj


def run_sim(config: dict[str, Any], cfg, panels, scores: pd.DataFrame, prices: pd.DataFrame) -> pd.Series:
    handler = ProfilePortfolioHandler()
    mf = config.get("market_filter") or {}
    mk = handler._market_close(config, cfg, panels)
    ss = config.get("safe_sleeve") or {}
    safe_close = None
    if ss.get("enabled"):
        sc = {}
        for s in ss.get("symbols", []):
            c = handler._load_close(s, cfg, panels)
            if c is not None:
                sc[s] = c
        safe_close = sc or None
    sim = simulate_portfolio(
        scores, prices, cfg, top_k=int(config.get("top_k", 10)),
        rebal_freq=str(config.get("rebalance_freq", "monthly")),
        market_close=mk,
        market_ma_len=int(mf.get("ma_len", 200)),
        market_off_scale=float(mf.get("off_scale", 0.5)),
        exposure_gain=float(config.get("exposure_gain", 1.0)),
        hold_buffer=int(config.get("hold_buffer", 0)),
        safe_close=safe_close,
        safe_ratio=float(ss.get("ratio", 0.0)),
        safe_max=float(ss.get("max", 0.0)),
        safe_ma_len=int(ss.get("ma_len", 100)),
    )
    nav = sim["nav"]
    return (nav / nav.iloc[0]).rename("equity")


def perf_nav(nav: pd.Series) -> dict[str, Any]:
    eq = nav / nav.iloc[0]
    ret = eq.pct_change().fillna(0.0)
    return performance(eq, ret, "1d")


def audit_corr(scores: pd.DataFrame, prices: pd.DataFrame, rebal: set, top_k: int) -> tuple[float, float]:
    import itertools
    returns = prices.pct_change()
    sig = scores.shift(1).fillna(0.0)
    held: set[str] = set()
    vals = []
    for t in scores.index:
        if t not in rebal:
            continue
        row = sig.loc[t]
        avail = row[row > 0]
        if avail.empty:
            continue
        strict = avail.nlargest(min(top_k, len(avail)))
        syms = list(strict.index)
        if len(syms) < 2:
            continue
        window = returns.loc[:t].tail(CORR_LOOKBACK)
        sub = window[syms].dropna(how="any")
        if len(sub) < CORR_LOOKBACK // 2:
            continue
        corr = sub.corr()
        pairs = list(itertools.combinations(syms, 2))
        pv = [corr.loc[a, b] for a, b in pairs if pd.notna(corr.loc[a, b])]
        if pv:
            vals.append(float(np.mean(pv)))
    return (float(np.mean(vals)) if vals else float("nan")), len(vals)


def main() -> None:
    from trading_lab.strategies.yoon1k import Yoon1kHandler

    composite = json.loads((ROOT / "configs" / "strategies" / "yoon2_final_fx_credit_gate.json").read_text("utf-8"))
    overrides_by_id = {s["config"]: s.get("overrides") for s in composite["sub_strategies"]}

    results = {}
    for sid in ["yoon1j_lp", "yoon1j_kr"]:
        print(f"=== {sid} (실제 등록 override 적용: {overrides_by_id[sid]}) ===")
        config = Yoon1kHandler._sub_config(sid, overrides_by_id[sid])
        cfg = config_from_dict(config)
        handler = ProfilePortfolioHandler()
        raw = handler.load_data("PORTFOLIO", config, synthetic=False)
        panels = handler._from_wide(raw)
        scores, prices = compute_universe(panels, cfg)
        rebal = rebalance_dates(scores.index, str(config.get("rebalance_freq", "monthly")))
        top_k = int(config.get("top_k", 10))

        rows = []
        for penalty in PENALTIES:
            adj_scores = decorrelate_scores(
                scores, prices, rebal, corr_lookback=CORR_LOOKBACK,
                extra_candidates=EXTRA_CANDIDATES, corr_penalty=penalty, top_k=top_k,
            )
            avg_corr, n = audit_corr(adj_scores, prices, rebal, top_k)
            nav = run_sim(config, cfg, panels, adj_scores, prices)
            m = _clean_metric(perf_nav(nav))
            rows.append({"corr_penalty": penalty, "avg_selected_corr": avg_corr, "n": n, **m})
            print(
                f"  penalty={penalty}: avg_corr={avg_corr:.3f} CAGR={m['cagr']*100:.2f}% "
                f"MDD={m['max_drawdown']*100:.2f}% Sharpe={m['sharpe']:.3f}"
            )
        results[sid] = rows

    out_json = ROOT / "reports" / "profile_sizing" / "yoon1j_decorrelate_validate.json"
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    lines = ["# yoon1j_lp/yoon1j_kr 상관계수 페널티 선택 검증", "", f"- extra_candidates={EXTRA_CANDIDATES}, corr_lookback={CORR_LOOKBACK}", ""]
    for sid, rows in results.items():
        lines.extend([
            f"## {sid}",
            "",
            "| corr_penalty | 선택 평균상관 | CAGR | MDD | Sharpe |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ])
        for r in rows:
            lines.append(
                f"| {r['corr_penalty']} | {r['avg_selected_corr']:.3f} | {_fmt_pct(r['cagr'])} | "
                f"{_fmt_pct(r['max_drawdown'])} | {_fmt_num(r['sharpe'])} |"
            )
        lines.append("")

    out_md = ROOT / "reports" / "profile_sizing" / "yoon1j_decorrelate_validate.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"JSON: {out_json}")
    print(f"MD:   {out_md}")


if __name__ == "__main__":
    main()
