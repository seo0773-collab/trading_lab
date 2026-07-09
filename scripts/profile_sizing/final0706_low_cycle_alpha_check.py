"""저가권(cumulative_percentile 낮음) 매수가 실제로 순방향 수익률을 예측하는가.

final0706 US 유니버스(11개 섹터 ETF)의 buckets 사이징이 전제하는 컨트래리언
가설 — "cycle_multiple percentile이 낮을수록(자기 SMA200 대비 저가권) 순방향
수익률이 높다" — 을 포트폴리오 성과(집중·리밸런스·regime_cap이 다 섞인 결과)가
아니라 **원신호 자체**로 직접 검증한다. yoon_profile_study.py와 같은 방법론
(NW t-검정 + Bonferroni/BH-FDR, cycle_state.newey_west_mean_test 재사용)을 쓴다.

방법: 각 ETF의 run_pipeline 출력에서 cumulative_percentile(0~1)을 프로덕션과
동일한 buckets 경계([0,20,40,60,80,100))로 5분위화하고, 각 분위·각 horizon
(21/63/126/252일)의 순방향수익률을 NW t-검정한다. 컨트래리언 가설이 맞다면
저분위(0~20%)일수록 평균 순방향수익률이 높고 고분위(80~100%)일수록 낮아야
한다(단조감소).

Usage:
    PYTHONPATH=src:scripts python3 scripts/profile_sizing/final0706_low_cycle_alpha_check.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from profile_sizing.config import config_from_dict  # noqa: E402
from profile_sizing.cycle_state import forward_returns, newey_west_mean_test  # noqa: E402
from profile_sizing.run import run_pipeline  # noqa: E402
from run_kalman_pipeline import load_yfinance  # noqa: E402

CONFIGS_DIR = ROOT / "configs" / "strategies"
UNIVERSE = ["VGT", "XLF", "IYH", "FENY", "FIDU", "FSTA", "IYC", "XLU", "XLB", "FREL", "FCOM"]
HORIZONS = (21, 63, 126, 252)
BUCKET_EDGES = [0, 20, 40, 60, 80, 100]
BUCKET_LABELS = ["0-20%(최저가권)", "20-40%", "40-60%", "60-80%", "80-100%(최고가권)"]
START = "2007-01-29"


def _apply_multiple_testing(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    table["sig_bonferroni"] = False
    table["sig_fdr"] = False
    elig = table["n"] >= 30
    m = int(elig.sum())
    if m == 0:
        return table
    table.loc[elig, "sig_bonferroni"] = table.loc[elig, "p_value"] < 0.05 / m
    p = table.loc[elig, "p_value"].dropna().sort_values()
    thresh = 0.0
    for rank, (idx, pv) in enumerate(p.items(), start=1):
        if pv <= 0.05 * rank / m:
            thresh = pv
    table.loc[elig, "sig_fdr"] = table.loc[elig, "p_value"] <= thresh
    return table


def main() -> None:
    final0706_cfg = json.loads((CONFIGS_DIR / "final0706.json").read_text("utf-8"))
    us_overrides = next(
        s for s in final0706_cfg["sub_strategies"] if s["currency"] == "USD"
    )["overrides"]
    cfg_dict = json.loads((CONFIGS_DIR / "yoon1j_lp.json").read_text("utf-8"))
    cfg_dict.update(us_overrides)
    cfg = config_from_dict(cfg_dict)

    print("데이터 로딩 및 파이프라인 계산 중...", flush=True)
    per_symbol = {}
    for sym in UNIVERSE:
        raw = load_yfinance(sym, cfg.interval, cfg.period)
        out = run_pipeline(raw, cfg)
        fc = out["forecast"]
        idx = fc.index
        pct = pd.Series(fc["cumulative_percentile"].to_numpy(), index=idx) * 100.0
        close = pd.Series(fc["close"].to_numpy(), index=idx)
        fwd = forward_returns(close, HORIZONS)
        df = fwd.copy()
        df["pct"] = pct
        df["bucket"] = pd.cut(df["pct"], BUCKET_EDGES, labels=range(5), include_lowest=True)
        df = df[df.index >= pd.Timestamp(START)]
        per_symbol[sym] = df
        print(f"  {sym} 완료 ({len(df)}봉)", flush=True)

    # (A) 분위별 NW t-검정 — 심볼별
    rows = []
    for sym, df in per_symbol.items():
        for h in HORIZONS:
            fwd_col = f"fwd_{h}"
            for b in range(5):
                vals = df.loc[df["bucket"] == b, fwd_col].to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                n = int(vals.size)
                mean, se, t = newey_west_mean_test(vals, lag=h)
                rows.append({
                    "symbol": sym, "horizon": h, "bucket": b,
                    "bucket_label": BUCKET_LABELS[b], "n": n,
                    "mean_fwd_return": mean, "nw_se": se, "t_stat": t,
                    "p_value": float(2 * (1 - norm.cdf(abs(t)))) if np.isfinite(t) else np.nan,
                })
    cell_table = _apply_multiple_testing(pd.DataFrame(rows))

    # (B) 풀링(전종목 스택) — 분위별 요약 + 단조성 확인
    pooled_rows = []
    for h in HORIZONS:
        fwd_col = f"fwd_{h}"
        pooled_vals_by_bucket = []
        for b in range(5):
            vals = []
            for sym, df in per_symbol.items():
                v = df.loc[df["bucket"] == b, fwd_col].to_numpy(dtype=float)
                vals.append(v[np.isfinite(v)])
            vals = np.concatenate(vals) if vals else np.array([])
            mean, se, t = newey_west_mean_test(vals, lag=h)
            pooled_rows.append({
                "horizon": h, "bucket": b, "bucket_label": BUCKET_LABELS[b],
                "n": int(vals.size), "mean_fwd_return": mean, "t_stat": t,
            })
            pooled_vals_by_bucket.append(mean)
        monotone_decreasing = all(
            pooled_vals_by_bucket[i] >= pooled_vals_by_bucket[i + 1] - 1e-9
            for i in range(4)
        )
        for r in pooled_rows[-5:]:
            r["monotone_decreasing_all_buckets"] = monotone_decreasing
    pooled_table = pd.DataFrame(pooled_rows)

    # (C) IC(스피어만 순위상관: percentile vs 순방향수익률) — 심볼별 + 풀링
    ic_rows = []
    for h in HORIZONS:
        fwd_col = f"fwd_{h}"
        for sym, df in per_symbol.items():
            sub = df[["pct", fwd_col]].dropna()
            if len(sub) < 30:
                continue
            ic, p = spearmanr(sub["pct"], sub[fwd_col])
            ic_rows.append({"symbol": sym, "horizon": h, "n": len(sub), "ic": float(ic), "p_value": float(p)})
        pooled = pd.concat(
            [per_symbol[sym][["pct", fwd_col]].dropna() for sym in UNIVERSE], ignore_index=True)
        ic, p = spearmanr(pooled["pct"], pooled[fwd_col])
        ic_rows.append({"symbol": "POOLED(전종목)", "horizon": h, "n": len(pooled), "ic": float(ic), "p_value": float(p)})
    ic_table = pd.DataFrame(ic_rows)

    n_sig_fdr = int(cell_table["sig_fdr"].sum())
    n_cells = int(len(cell_table))
    print(f"\nFDR 유의 셀: {n_sig_fdr}/{n_cells}")
    print("\n풀링 분위별 평균 순방향수익률 (전종목):")
    print(pooled_table.pivot(index="bucket_label", columns="horizon", values="mean_fwd_return")
          .reindex(BUCKET_LABELS).round(4).to_string())
    print("\n풀링 IC (percentile vs fwd_return, 낮을수록 컨트래리언 가설 지지):")
    print(ic_table[ic_table["symbol"] == "POOLED(전종목)"].round(4).to_string(index=False))

    out = {
        "cell_table": cell_table.assign(
            bucket=lambda d: d["bucket"].astype(int)).to_dict(orient="records"),
        "pooled_table": pooled_table.to_dict(orient="records"),
        "ic_table": ic_table.to_dict(orient="records"),
        "n_sig_fdr": n_sig_fdr, "n_cells": n_cells,
    }
    out_json = ROOT / "reports" / "profile_sizing" / "final0706_low_cycle_alpha_check.json"
    out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    # 마크다운 리포트
    lines = [
        "# final0706 저가권(cumulative_percentile) 컨트래리언 가설 직접검증",
        "",
        f"- Universe: {', '.join(UNIVERSE)} (final0706 US 슬리브 그대로)",
        f"- 기간: {START}~ (봉 인덱스는 종목별 상장일부터 시작하므로 실제 표본 시작은 종목마다 다름)",
        f"- 방법: cycle_state.newey_west_mean_test 재사용, buckets=[0,20,40,60,80,100] "
        "(프로덕션 buckets와 동일 경계), horizon=21/63/126/252일. FDR α=0.05, "
        f"Bonferroni/BH-FDR 동시 표기. 총 {n_cells}개 셀(11종목×4horizon×5분위) 중 "
        f"**FDR 유의 {n_sig_fdr}개**.",
        "",
        "## 1. 풀링(전종목 스택) 분위별 평균 순방향수익률",
        "",
        "컨트래리언 가설이 맞다면 0-20%(최저가권) 행이 가장 크고 80-100% 행이 가장 "
        "작아야 한다(단조감소).",
        "",
    ]
    piv = pooled_table.pivot(index="bucket_label", columns="horizon", values="mean_fwd_return").reindex(BUCKET_LABELS)
    lines.append("| 분위 | fwd21 | fwd63 | fwd126 | fwd252 |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for label in BUCKET_LABELS:
        row = piv.loc[label]
        lines.append(f"| {label} | " + " | ".join(f"{row[h]*100:.2f}%" for h in HORIZONS) + " |")
    for h in HORIZONS:
        mono = pooled_table[pooled_table["horizon"] == h]["monotone_decreasing_all_buckets"].iloc[0]
        lines.append("")
        lines.append(f"- fwd{h}: 단조감소(가설과 일치) = **{mono}**")

    lines += [
        "",
        "## 2. 풀링 IC (Spearman, percentile vs 순방향수익률)",
        "",
        "IC가 뚜렷한 음수면 컨트래리언 가설 지지(저percentile→고수익). 0 근방이면 "
        "무관, 양수면 오히려 모멘텀(고percentile→고수익) 우세.",
        "",
        "| horizon | n | IC | p-value |",
        "| --- | ---: | ---: | ---: |",
    ]
    for _, r in ic_table[ic_table["symbol"] == "POOLED(전종목)"].iterrows():
        lines.append(f"| fwd{int(r['horizon'])} | {int(r['n'])} | {r['ic']:.4f} | {r['p_value']:.4f} |")

    lines += [
        "",
        "## 3. 종목별 IC (fwd63 기준)",
        "",
        "| symbol | n | IC | p-value |",
        "| --- | ---: | ---: | ---: |",
    ]
    sub = ic_table[(ic_table["horizon"] == 63) & (ic_table["symbol"] != "POOLED(전종목)")]
    for _, r in sub.sort_values("ic").iterrows():
        lines.append(f"| {r['symbol']} | {int(r['n'])} | {r['ic']:.4f} | {r['p_value']:.4f} |")

    lines += [
        "",
        "## 4. FDR 유의 셀 상세",
        "",
        "| symbol | horizon | bucket | n | mean_fwd | t-stat | FDR유의 |",
        "| --- | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    sig = cell_table[cell_table["sig_fdr"]].sort_values(["horizon", "symbol", "bucket"])
    if sig.empty:
        lines.append("| (없음) | | | | | | |")
    else:
        for _, r in sig.iterrows():
            lines.append(
                f"| {r['symbol']} | {int(r['horizon'])} | {r['bucket_label']} | {int(r['n'])} | "
                f"{r['mean_fwd_return']*100:.2f}% | {r['t_stat']:.2f} | O |"
            )

    out_md = ROOT / "reports" / "profile_sizing" / "final0706_low_cycle_alpha_check.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n저장: {out_json}")
    print(f"저장: {out_md}")


if __name__ == "__main__":
    main()
