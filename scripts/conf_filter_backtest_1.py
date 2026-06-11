
#!/usr/bin/env python3
"""
conf_filter_backtest.py
=======================
확신도 필터 전략 백테스트: Kalman p_up vs OU-괴리 베이스라인 정면 비교.

핵심 질문: "Kalman의 확신 순위가 OU의 |m-μ| 괴리 순위보다 거래 선별에 더 좋은가?"
이게 Yes여야 Kalman을 유지할 이유가 있다.

설계
----
- 신호원 (둘 다 동일한 매매 메커니즘으로 비교):
    KALMAN : 방향 = sign(p_up - 0.5),       확신 = |p_up - 0.5|
    OU     : 방향 = sign(m_slow - mult_close), 확신 = |mult_close - m_slow|
- 임계값: 확신도의 rolling 분위수 (walk-forward, 과거만 사용)
           -> σ 보정 오차에 둔감, 절대 p_up 해석 불필요
- 중복 신호: 포지션 보유 중 신규 진입 금지 (flat일 때만 진입)
- 청산: horizon봉 고정 보유 + 반대 신호 시 조기 청산 (옵션)
- 비용: 진입/청산 각 fee_bps (수수료+슬리피지 합산)

사용 예 (trading_lab 환경)
--------------------------
    cd ~/trading_lab/scripts
    ../.venv/bin/python ../strategies/conf_filter_backtest.py \\
        --forecast ../reports/BTCUSD_V1_forecast.csv --horizon 24 --fee-bps 10

    # 주식(정규장 1h봉)이면 Sharpe 연환산 보정:
    ../.venv/bin/python ../strategies/conf_filter_backtest.py \\
        --forecast ../reports/AAPL_forecast.csv --bars-per-year 1638
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# 주의: 이 파일은 어떤 프로젝트 모듈도 import하지 않는다.
# indicators/와 scripts/에 동명의 flat_chart.py / kalman.py가 공존하므로
# sys.path 조작을 하지 않는다 (레거시 모듈 오인 import 방지).
ROOT = Path(__file__).resolve().parents[1]
BARS_PER_YEAR = 24 * 365


# ----------------------------------------------------------------------
@dataclass
class BTConfig:
    horizon: int = 24
    fee_bps: float = 10.0          # 한 방향당 (수수료+슬리피지). 왕복 = 2배
    conf_quantile: float = 0.85    # 확신도 rolling 분위수 임계값
    quantile_window: int = 2000    # rolling 분위수 윈도우 (봉)
    long_only: bool = False
    exit_on_opposite: bool = True  # 보유 중 반대 신호 시 조기 청산
    ident_frac: float = 0.4        # 이 비율 이전 구간은 거래 제외 (파라미터 식별 구간)
    edge_mult: float = 0.0         # 기대수익 >= edge_mult × 왕복비용 일 때만 진입 (0=off)


@dataclass
class BTResult:
    name: str
    n_trades: int
    hit_rate: float
    avg_net_bps: float
    total_return: float
    sharpe: float
    max_dd: float
    exposure: float
    equity: pd.Series
    long_n: int = 0
    long_hit: float = np.nan
    short_n: int = 0
    short_hit: float = np.nan


# ----------------------------------------------------------------------
def rolling_conf_threshold(conf: pd.Series, q: float, window: int) -> pd.Series:
    """과거 window봉의 확신도 분위수 (현재 봉 미포함 -> shift(1))."""
    return conf.rolling(window, min_periods=window // 2).quantile(q).shift(1)


def run_backtest(
    df: pd.DataFrame, direction: pd.Series, conf: pd.Series,
    cfg: BTConfig, name: str, expected_edge: pd.Series | None = None,
) -> BTResult:
    """flat일 때만 진입, horizon 고정 보유(+조기 청산 옵션)의 이벤트 루프 백테스트."""
    close = df["close"].to_numpy()
    n = len(df)
    thr = rolling_conf_threshold(conf, cfg.conf_quantile, cfg.quantile_window).to_numpy()
    d = direction.to_numpy()
    c = conf.to_numpy()
    ee = expected_edge.to_numpy() if expected_edge is not None else None

    start = max(int(n * cfg.ident_frac), cfg.quantile_window // 2)
    fee = cfg.fee_bps / 1e4

    pos = 0          # -1 / 0 / +1
    entry_px = 0.0
    entry_t = -1
    bar_ret = np.zeros(n)      # 봉별 전략 수익률 (포지션 보유분)
    trades: list[float] = []   # 거래별 순수익률
    trade_dirs: list[int] = []
    trade_raw: list[float] = []

    for t in range(start, n - 1):
        signal = 0
        if np.isfinite(thr[t]) and np.isfinite(c[t]) and c[t] >= thr[t] and d[t] != 0:
            signal = int(d[t])
            if cfg.long_only and signal < 0:
                signal = 0
            # 비용 인지형 필터: 기대수익이 왕복비용의 edge_mult배 미만이면 미진입
            if signal != 0 and cfg.edge_mult > 0 and ee is not None:
                if not (np.isfinite(ee[t]) and ee[t] >= cfg.edge_mult * 2 * fee):
                    signal = 0

        if pos == 0:
            if signal != 0:
                pos, entry_px, entry_t = signal, close[t], t
                bar_ret[t] -= fee  # 진입 비용
        else:
            # 봉별 수익 귀속
            bar_ret[t] += pos * (close[t] / close[t - 1] - 1.0)
            held = t - entry_t
            opposite = cfg.exit_on_opposite and signal != 0 and signal != pos
            if held >= cfg.horizon or opposite or t == n - 2:
                raw = pos * (close[t] / entry_px - 1.0)
                net = raw - 2 * fee
                trades.append(net)
                trade_dirs.append(pos)
                trade_raw.append(raw)
                bar_ret[t] -= fee  # 청산 비용
                pos = 0
                # 조기 청산 직후 반대 방향 즉시 진입
                if opposite:
                    pos, entry_px, entry_t = signal, close[t], t
                    bar_ret[t] -= fee

    equity = pd.Series(np.cumprod(1 + bar_ret), index=df.index)
    active = bar_ret != 0
    n_tr = len(trades)
    tr = np.array(trades) if n_tr else np.array([0.0])
    raw_arr = np.array(trade_raw) if n_tr else np.array([0.0])

    dirs = np.array(trade_dirs) if n_tr else np.array([0])
    lmask, smask = dirs > 0, dirs < 0
    long_n, short_n = int(lmask.sum()), int(smask.sum())
    long_hit = float((raw_arr[lmask] > 0).mean()) if long_n else np.nan
    short_hit = float((raw_arr[smask] > 0).mean()) if short_n else np.nan

    sd = bar_ret[start:].std()
    sharpe = float(bar_ret[start:].mean() / sd * np.sqrt(BARS_PER_YEAR)) if sd > 0 else 0.0
    dd = float((equity / equity.cummax() - 1.0).min())

    return BTResult(
        name=name,
        n_trades=n_tr,
        hit_rate=float((raw_arr > 0).mean()) if n_tr else np.nan,
        avg_net_bps=float(tr.mean() * 1e4),
        total_return=float(equity.iloc[-1] - 1.0),
        sharpe=sharpe,
        max_dd=dd,
        exposure=float(active[start:].mean()),
        equity=equity,
        long_n=long_n, long_hit=long_hit,
        short_n=short_n, short_hit=short_hit,
    )


# ----------------------------------------------------------------------
def build_signals(df: pd.DataFrame, horizon: int):
    pup = df[f"pup_{horizon}"]
    kal_dir = np.sign(pup - 0.5)
    kal_conf = (pup - 0.5).abs()
    # 기대수익(%): 배수 공간 예측 이동폭. cycle 변화는 k<=72에서 2차항이라 무시.
    kal_edge = (df[f"mhat_{horizon}"] / df["mult_close"] - 1.0).abs()

    ou_dir = np.sign(df["m_slow"] - df["mult_close"])
    ou_conf = (df["mult_close"] - df["m_slow"]).abs()
    return (kal_dir, kal_conf, kal_edge), (ou_dir, ou_conf)


def fmt_row(r: BTResult) -> str:
    return (f"{r.name:<22} 거래수 {r.n_trades:>5}  적중 {r.hit_rate:>6.1%}  "
            f"평균순익 {r.avg_net_bps:>+7.1f}bp  누적 {r.total_return:>+8.1%}  "
            f"Sharpe {r.sharpe:>5.2f}  MDD {r.max_dd:>7.1%}  노출 {r.exposure:>5.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forecast", required=True, help="reports/{name}_forecast.csv 경로")
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--fee-bps", type=float, default=10.0)
    ap.add_argument("--long-only", action="store_true")
    ap.add_argument("--no-early-exit", action="store_true")
    ap.add_argument("--bars-per-year", type=int, default=24 * 365,
                    help="Sharpe 연환산용. 1h봉: 코인 8760(기본), 미국주식 ~1638")
    args = ap.parse_args()
    global BARS_PER_YEAR
    BARS_PER_YEAR = args.bars_per_year

    df = pd.read_csv(args.forecast, index_col=0, parse_dates=True)
    need = ["close", "mult_close", "m_slow", f"pup_{args.horizon}"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f"forecast CSV에 누락: {missing}")
    df = df.dropna(subset=need)

    name = Path(args.forecast).stem.replace("_forecast", "")
    (kal_dir, kal_conf, kal_edge), (ou_dir, ou_conf) = build_signals(df, args.horizon)

    print(f"=== {name}  horizon={args.horizon}  fee={args.fee_bps}bp/side  "
          f"long_only={args.long_only} ===\n")

    # --- 1) 핵심 비교: 동일 분위수 임계값에서 KALMAN vs OU ---
    print("[1] Kalman 확신 vs OU 괴리 (동일 메커니즘 정면 비교)")
    results = {}
    for q in (0.70, 0.80, 0.85, 0.90, 0.95):
        cfg = BTConfig(horizon=args.horizon, fee_bps=args.fee_bps,
                       conf_quantile=q, long_only=args.long_only,
                       exit_on_opposite=not args.no_early_exit)
        rk = run_backtest(df, kal_dir, kal_conf, cfg, f"KALMAN q={q:.2f}")
        ro = run_backtest(df, ou_dir, ou_conf, cfg, f"OU     q={q:.2f}")
        results[q] = (rk, ro)
        print("  " + fmt_row(rk))
        print(f"{'':24}└ 롱 {rk.long_n}건 적중 {rk.long_hit:.1%} / 숏 {rk.short_n}건 적중 {rk.short_hit:.1%}"
              if rk.long_n and rk.short_n else "")
        print("  " + fmt_row(ro))
        print()

    # --- 2) 판정 ---
    print("[2] 판정 (Sharpe 기준, 거래수 30 미만 구간 제외)")
    kal_wins = ou_wins = 0
    for q, (rk, ro) in results.items():
        if rk.n_trades < 30 or ro.n_trades < 30:
            continue
        if rk.sharpe > ro.sharpe:
            kal_wins += 1
        elif ro.sharpe > rk.sharpe:
            ou_wins += 1
    verdict = ("Kalman 확신도가 OU 괴리보다 거래 선별에 유효" if kal_wins > ou_wins
               else "OU 괴리만으로 충분 — Kalman 추가 가치 미확인" if ou_wins > kal_wins
               else "무승부 — 표본/자산 추가 필요")
    print(f"  Kalman 우위 {kal_wins} : OU 우위 {ou_wins}  ->  {verdict}\n")

    # --- 3) 비용 민감도 (q=0.85 고정) ---
    print("[3] 비용 민감도 (KALMAN, q=0.85)")
    for fee in (0.0, 5.0, 10.0, 20.0):
        cfg = BTConfig(horizon=args.horizon, fee_bps=fee, conf_quantile=0.85,
                       long_only=args.long_only,
                       exit_on_opposite=not args.no_early_exit)
        r = run_backtest(df, kal_dir, kal_conf, cfg, f"fee={fee:>4.0f}bp")
        print("  " + fmt_row(r))

    # --- 4) 비용 인지형 기대수익 필터 (KALMAN, q=0.85) ---
    print("\n[4] 기대수익 필터: 진입조건 += 기대수익 >= λ×왕복비용 (KALMAN, q=0.85)")
    rt_cost = 2 * args.fee_bps
    eq = kal_edge.dropna() * 1e4
    print(f"    기대수익 분포(bp): 25% {eq.quantile(.25):.0f} / 중앙 {eq.quantile(.5):.0f} / "
          f"75% {eq.quantile(.75):.0f} / 95% {eq.quantile(.95):.0f}   왕복비용 {rt_cost:.0f}bp")
    best_edge = None
    for em in (0.0, 1.0, 2.0, 3.0, 5.0, 8.0):
        cfg = BTConfig(horizon=args.horizon, fee_bps=args.fee_bps, conf_quantile=0.85,
                       long_only=args.long_only,
                       exit_on_opposite=not args.no_early_exit, edge_mult=em)
        r = run_backtest(df, kal_dir, kal_conf, cfg, f"λ={em:>3.1f}", kal_edge)
        print("  " + fmt_row(r))
        if r.n_trades >= 20 and (best_edge is None or r.sharpe > best_edge.sharpe):
            best_edge = r

    # --- 5) equity plot ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(13, 6))
    q_plot = 0.85
    rk, ro = results[q_plot]
    ax.plot(rk.equity.index, rk.equity, label=f"KALMAN conf q={q_plot}", lw=1.2)
    ax.plot(ro.equity.index, ro.equity, label=f"OU distance q={q_plot}", lw=1.2)
    if best_edge is not None:
        ax.plot(best_edge.equity.index, best_edge.equity,
                label=f"KALMAN +edge filter ({best_edge.name.strip()})", lw=1.4, color="tab:green")
    bh = df["close"] / df["close"].iloc[0]
    ax.plot(df.index, bh, label="buy & hold", lw=0.8, color="gray", alpha=0.7)
    ax.set_title(f"{name} — confidence-filtered strategy equity (h={args.horizon}, "
                 f"fee={args.fee_bps}bp/side)")
    ax.legend(); ax.grid(alpha=0.3)
    out_png = ROOT / "reports" / f"{name}_strategy_h{args.horizon}.png"
    fig.tight_layout(); fig.savefig(out_png, dpi=110); plt.close(fig)
    print(f"\nequity plot -> {out_png}")


if __name__ == "__main__":
    main()
