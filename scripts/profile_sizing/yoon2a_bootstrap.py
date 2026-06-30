"""yoon2a 엣지 부트스트랩: Sharpe·MDD·초과수익 신뢰구간.

전진 모의투자(월간이라 표본 ~12/년)로는 엣지의 통계적 유의성을 얻기 어렵다.
대신 이미 존재하는 과거 일별 수익 시리즈를 **페어드 무빙블록 부트스트랩**으로
리샘플해(자기상관 보존), 다음 질문에 통계로 직접 답한다.

  1) yoon2a Sharpe가 0보다 큰가? (엣지 실재)
  2) yoon2a Sharpe가 50/50 벤치를 이기는가? (위험조정 우위) — 페어드 초과
  3) yoon2a MDD가 벤치보다 얕은가? (낙폭 방어) — 페어드 비교

소스: yoon2 코어(Yoon1kHandler) + SSO(26%, USD→KRW) 월간 리밸런스 결합
      = configs/strategies/yoon2a.json 의 leverage_mix와 동일. 벤치 = 핸들러 50/50.
      (yoon2a_sso_reopt.py 와 동일한 결합 경로를 재사용)

사용:  PYTHONPATH=src python scripts/profile_sizing/yoon2a_bootstrap.py
       옵션: --iters 5000 --block 20 --seed 7
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from profile_sizing.engine import performance  # noqa: E402
from profile_sizing.portfolio import rebalance_dates  # noqa: E402
from run_kalman_pipeline import load_yfinance  # noqa: E402
from trading_lab.strategies.yoon1k import Yoon1kHandler  # noqa: E402

SSO_WEIGHT = 0.26  # yoon2a.json leverage_mix 의 SSO 비중


def to_base(nav: pd.Series, fx: pd.Series) -> pd.Series:
    f = fx.reindex(nav.index).ffill()
    return nav * (f / f.dropna().iloc[0])


def combine2(core: pd.Series, sso: pd.Series, w: float, freq: str = "monthly") -> pd.Series:
    """코어/SSO 월간 리밸런스 결합 NAV (yoon2a_sso_reopt.combine2 와 동일)."""
    idx = core.index
    reb = rebalance_dates(idx, freq)
    rc = core.pct_change().fillna(0.0).to_numpy()
    rs = sso.pct_change().fillna(0.0).to_numpy()
    vc, vs = 1.0 - w, w
    out = np.empty(len(idx))
    for i, t in enumerate(idx):
        vc *= 1.0 + rc[i]
        vs *= 1.0 + rs[i]
        out[i] = vc + vs
        if t in reb:
            tot = vc + vs
            vc, vs = tot * (1.0 - w), tot * w
    return pd.Series(out, index=idx)


def build_series() -> tuple[pd.Series, pd.Series, pd.DatetimeIndex]:
    """yoon2a NAV, 50/50 벤치 NAV (공통 인덱스, 1.0 정규화)."""
    cfg = json.loads((ROOT / "configs" / "strategies" / "yoon2.json").read_text("utf-8"))
    h = Yoon1kHandler()
    raw = h.load_data("SYNTH", cfg, synthetic=False)
    art = h.build_artifacts(raw, cfg, symbol="SYNTH", phase="all", bars_per_year=252)
    core_full, bench_full = art.equity, art.benchmark

    sso_close = load_yfinance("SSO", "1d", "max")["close"].dropna()
    fx = load_yfinance("KRW=X", "1d", "max")["close"]

    idx = core_full.index.intersection(sso_close.index)
    idx = idx.intersection(bench_full.dropna().index).sort_values()
    core = core_full.reindex(idx).ffill()
    core = core / core.iloc[0]
    sso = to_base((sso_close.reindex(idx).ffill()).pipe(lambda s: s / s.iloc[0]), fx)
    bench = bench_full.reindex(idx).ffill()
    bench = bench / bench.iloc[0]

    nav = combine2(core, sso, SSO_WEIGHT)
    return nav, bench, idx


def _stats(returns: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """행별 (Sharpe, CAGR, MDD). returns shape = (B, n) 일별 수익."""
    n = returns.shape[1]
    mean = returns.mean(axis=1)
    std = returns.std(axis=1, ddof=1)
    sharpe = np.where(std > 0, mean / std * np.sqrt(252.0), 0.0)
    eq = np.cumprod(1.0 + returns, axis=1)
    cagr = eq[:, -1] ** (252.0 / n) - 1.0
    run_max = np.maximum.accumulate(eq, axis=1)
    mdd = (eq / run_max - 1.0).min(axis=1)
    return sharpe, cagr, mdd


def block_indices(n: int, block: int, iters: int, rng: np.random.Generator) -> np.ndarray:
    """페어드 무빙블록 인덱스 행렬 (iters, n). 두 시리즈에 같은 인덱스 적용 → 결합관계 보존."""
    num_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(iters, num_blocks))
    offsets = np.arange(block)
    idx = (starts[:, :, None] + offsets[None, None, :]).reshape(iters, num_blocks * block)
    return idx[:, :n]


def pct(arr: np.ndarray) -> tuple[float, float, float]:
    return tuple(np.percentile(arr, [2.5, 50, 97.5]))  # type: ignore[return-value]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--block", type=int, default=20, help="블록 길이(거래일). ~1개월 자기상관 보존")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    nav, bench, idx = build_series()
    ra = nav.pct_change().fillna(0.0).to_numpy()
    rb = bench.pct_change().fillna(0.0).to_numpy()
    n = len(ra)

    p_a = performance(nav / nav.iloc[0], nav.pct_change().fillna(0.0), "1d")
    p_b = performance(bench / bench.iloc[0], bench.pct_change().fillna(0.0), "1d")

    print(f"기간 {idx[0].date()}~{idx[-1].date()} · {n}봉 · "
          f"부트스트랩 {args.iters}회 · 블록 {args.block}일\n")
    print("=== 점 추정 (전체 구간) ===")
    print(f"  yoon2a     CAGR {p_a['cagr']*100:6.2f}%  MDD {p_a['max_drawdown']*100:7.2f}%  "
          f"Sharpe {p_a['sharpe']:.3f}")
    print(f"  50/50 벤치 CAGR {p_b['cagr']*100:6.2f}%  MDD {p_b['max_drawdown']*100:7.2f}%  "
          f"Sharpe {p_b['sharpe']:.3f}\n")

    rng = np.random.default_rng(args.seed)
    bidx = block_indices(n, args.block, args.iters, rng)
    ra_bs, rb_bs = ra[bidx], rb[bidx]
    sa, ca, ma = _stats(ra_bs)
    sb, cb, mb = _stats(rb_bs)

    d_sharpe = sa - sb          # 초과 Sharpe (페어드)
    d_cagr = ca - cb            # 초과 CAGR
    d_mdd = ma - mb             # MDD 차 (>0 = yoon2a 가 더 얕음=방어 우위)

    def line(name: str, arr: np.ndarray, scale: float = 1.0, unit: str = "") -> None:
        lo, md, hi = pct(arr * scale)
        print(f"  {name:24s} [{lo:8.3f}, {hi:8.3f}]{unit}  중앙 {md:8.3f}{unit}")

    print("=== 95% 신뢰구간 (부트스트랩) ===")
    line("yoon2a Sharpe", sa)
    line("yoon2a CAGR", ca, 100, "%")
    line("yoon2a MDD", ma, 100, "%")
    print("  --- 페어드(벤치 대비) ---")
    line("초과 Sharpe (a−bench)", d_sharpe)
    line("초과 CAGR (a−bench)", d_cagr, 100, "%")
    line("MDD차 (a−bench, +=방어우위)", d_mdd, 100, "%")

    print("\n=== 가설 확률 (부트스트랩 표본 비율) ===")
    print(f"  P(yoon2a Sharpe > 0)            = {(sa > 0).mean()*100:5.1f}%")
    print(f"  P(yoon2a Sharpe > 벤치)         = {(d_sharpe > 0).mean()*100:5.1f}%")
    print(f"  P(yoon2a MDD 벤치보다 얕음)     = {(d_mdd > 0).mean()*100:5.1f}%")
    print(f"  P(초과CAGR > 0)                 = {(d_cagr > 0).mean()*100:5.1f}%")

    out = ROOT / "reports" / "profile_sizing" / "yoon2a_bootstrap.json"
    out.write_text(json.dumps({
        "period": f"{idx[0].date()}~{idx[-1].date()}", "bars": n,
        "iters": args.iters, "block": args.block, "seed": args.seed,
        "point": {"yoon2a": p_a, "bench": p_b},
        "ci95": {
            "yoon2a_sharpe": pct(sa), "yoon2a_cagr": pct(ca), "yoon2a_mdd": pct(ma),
            "excess_sharpe": pct(d_sharpe), "excess_cagr": pct(d_cagr), "mdd_diff": pct(d_mdd),
        },
        "prob": {
            "sharpe_pos": float((sa > 0).mean()),
            "sharpe_beats_bench": float((d_sharpe > 0).mean()),
            "mdd_shallower": float((d_mdd > 0).mean()),
            "excess_cagr_pos": float((d_cagr > 0).mean()),
        },
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nJSON: {out}")


if __name__ == "__main__":
    main()
