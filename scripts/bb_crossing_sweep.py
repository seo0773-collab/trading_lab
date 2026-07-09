"""표준 볼린저밴드(bb.txt, TradingView 기본 BB 인디케이터) 평균회귀 크로싱 전략
— 횡보성이 강한 종목을 선별한 뒤 length × stdev(mult) × offset를 스윕해 최적 파라미터를 찾는다.

전략 규칙 (평균회귀, long only):
    - basis = SMA(close, length), dev = mult * stdev_population(close, length)
      upper = basis + dev, lower = basis - dev  (bb.txt와 동일, maType=SMA 고정)
    - offset: Pine의 plot(..., offset=N)은 "계산은 현재 봉까지의 데이터로 하되
      화면에는 N봉 미래에 그린다" = 밴드를 N봉만큼 지연시켜 사용하는 것과 동치.
      offset >= 0만 인과적(causal)이다 — offset < 0은 아직 오지 않은 미래의 밴드값을
      과거 봉에 대응시키는 것이라 실거래로 옮길 수 없다(look-ahead). 그래서 이 스윕은
      offset >= 0만 다룬다.
    - 진입: 전봉 close < 전봉 lower(지연) 이고 당봉 close >= 당봉 lower(지연)
      (하단밴드 이탈 후 재상향 크로스)
    - 청산: 전봉 close < 전봉 basis(지연) 이고 당봉 close >= 당봉 basis(지연)
      (중심선 상향 크로스) — 미청산 시 구간 마지막 봉 종가로 강제청산.

횡보성 선별: ADX(14) 평균이 낮은(추세가 약한) 종목을 유동성/가격 필터와 함께 선정.

Usage:
    .venv/bin/python3 scripts/bb_crossing_sweep.py
출력:
    reports/bb_crossing_universe.csv   — 선정 종목 + 랭킹 지표
    reports/bb_crossing_sweep.json     — 그리드 전체 결과
    reports/bb_crossing_sweep.md       — 요약 리포트
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "profile_sizing"))
from metrics import equity_metrics  # noqa: E402 (scripts/profile_sizing 평행 임포트)
from config import KIS_US_FEE_BPS  # noqa: E402

DATA_DIR = ROOT / "var" / "market_data" / "1d"
OUT_DIR = ROOT / "reports"

SLIPPAGE_BPS = 5.0
ROUNDTRIP_FEE = 2 * (KIS_US_FEE_BPS + SLIPPAGE_BPS) / 1e4  # ~50.2bp 왕복

# --- 스크리닝 파라미터 ---
ADX_WINDOW = 14
SCREEN_LOOKBACK_BARS = 504          # 최근 ~2년, 횡보성 랭킹 기준
MIN_PRICE = 5.0                      # 저가주 노이즈 배제
MIN_DOLLAR_VOL = 5_000_000.0         # 하루 평균 거래대금(최근 구간) 최소치
UNIVERSE_SIZE = 30

# --- 백테스트 윈도우 ---
BACKTEST_BARS = 1260                 # 최근 ~5년
TRAIN_FRAC = 0.7
MAX_GAP_TRADING_DAYS = 10            # 이보다 큰 캘린더갭이 있으면 데이터 품질 의심 -> 제외

# --- 스윕 그리드 (bb.txt의 4개 입력 중 TF=1d 고정, length/stdev/offset 스윕) ---
LENGTHS = (10, 14, 20, 30, 40, 50)
MULTS = (1.5, 2.0, 2.5, 3.0)
OFFSETS = (0, 1, 2, 3, 5)            # 인과적(>=0)만. offset<0은 미래참조라 제외.

WARMUP_BARS = max(LENGTHS) + max(OFFSETS) + 5


def load_ohlcv(path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_parquet(path)
    except Exception:
        return None
    df = df.rename(columns=str.lower)
    if not {"open", "high", "low", "close", "volume"}.issubset(df.columns):
        return None
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def compute_adx(df: pd.DataFrame, window: int = ADX_WINDOW) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(
        alpha=1 / window, adjust=False, min_periods=window).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(
        alpha=1 / window, adjust=False, min_periods=window).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def max_calendar_gap_days(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return 0.0
    deltas = index.to_series().diff().dt.days.dropna()
    return float(deltas.max()) if len(deltas) else 0.0


def screen_universe() -> pd.DataFrame:
    rows = []
    for path in sorted(DATA_DIR.glob("*.parquet")):
        symbol = path.stem
        if "." in symbol:
            # KR(.KS)/JP(.T)/지수(.NYB) 등 — KIS_US_FEE_BPS 비용가정과 맞지 않아 제외.
            continue
        df = load_ohlcv(path)
        if df is None or len(df) < BACKTEST_BARS + WARMUP_BARS:
            continue
        window = df.tail(SCREEN_LOOKBACK_BARS)
        if max_calendar_gap_days(window.index) > MAX_GAP_TRADING_DAYS * 7:  # 주말/공휴일 여유
            continue
        avg_price = float(window["close"].mean())
        avg_dollar_vol = float((window["close"] * window["volume"]).mean())
        if avg_price < MIN_PRICE or avg_dollar_vol < MIN_DOLLAR_VOL:
            continue
        adx = compute_adx(df.tail(SCREEN_LOOKBACK_BARS + ADX_WINDOW * 3))
        mean_adx = float(adx.tail(SCREEN_LOOKBACK_BARS).mean())
        if not np.isfinite(mean_adx):
            continue
        rows.append({
            "symbol": symbol, "mean_adx": mean_adx, "avg_price": avg_price,
            "avg_dollar_vol": avg_dollar_vol, "bars_available": len(df),
        })
    universe = pd.DataFrame(rows).sort_values("mean_adx").reset_index(drop=True)
    return universe.head(UNIVERSE_SIZE)


def bollinger_bands(close: pd.Series, length: int, mult: float, offset: int):
    basis = close.rolling(length).mean()
    dev = mult * close.rolling(length).std(ddof=0)
    upper = basis + dev
    lower = basis - dev
    if offset:
        basis, upper, lower = basis.shift(offset), upper.shift(offset), lower.shift(offset)
    return basis, upper, lower


def simulate(close: pd.Series, basis: pd.Series, lower: pd.Series,
             trade_from: int) -> tuple[pd.DataFrame, pd.Series]:
    """진입: 전봉<lower & 당봉>=lower. 청산: 전봉<basis & 당봉>=basis. trade_from 이전 봉은
    지표 워밍업 구간이라 신호만 계산하고 거래는 발생시키지 않는다."""
    c = close.to_numpy(float)
    b = basis.to_numpy(float)
    lo = lower.to_numpy(float)
    n = len(c)
    entry_cross = np.zeros(n, dtype=bool)
    exit_cross = np.zeros(n, dtype=bool)
    valid = np.isfinite(c) & np.isfinite(b) & np.isfinite(lo)
    prev_valid = np.roll(valid, 1)
    prev_valid[0] = False
    entry_cross[1:] = valid[1:] & prev_valid[1:] & (c[:-1] < lo[:-1]) & (c[1:] >= lo[1:])
    exit_cross[1:] = valid[1:] & prev_valid[1:] & (c[:-1] < b[:-1]) & (c[1:] >= b[1:])

    bar_ret = np.zeros(n)
    trades: list[dict] = []
    pos = 0
    entry_i = -1
    entry_price = np.nan
    for i in range(trade_from, n):
        if pos == 0:
            if entry_cross[i]:
                pos, entry_i, entry_price = 1, i, c[i]
        else:
            if exit_cross[i]:
                net = (c[i] / entry_price - 1.0) - ROUNDTRIP_FEE
                trades.append({
                    "entry_time": close.index[entry_i], "entry_price": entry_price,
                    "exit_time": close.index[i], "exit_price": c[i],
                    "net_return": net, "exit_reason": "basis_cross",
                    "holding_bars": i - entry_i,
                })
                bar_ret[i] += net
                pos = 0
    if pos != 0:
        i = n - 1
        net = (c[i] / entry_price - 1.0) - ROUNDTRIP_FEE
        trades.append({
            "entry_time": close.index[entry_i], "entry_price": entry_price,
            "exit_time": close.index[i], "exit_price": c[i],
            "net_return": net, "exit_reason": "end_of_data",
            "holding_bars": i - entry_i,
        })
        bar_ret[i] += net

    returns = pd.Series(bar_ret, index=close.index)
    equity = (1.0 + returns).cumprod()
    return pd.DataFrame(trades), equity


def slice_windows(close: pd.Series) -> tuple[pd.Series, int, int]:
    """워밍업 포함 전체 슬라이스, train 시작 인덱스, holdout 시작 인덱스를 반환."""
    full = close.tail(BACKTEST_BARS + WARMUP_BARS)
    train_start = WARMUP_BARS
    n_trade = len(full) - WARMUP_BARS
    holdout_start = train_start + int(n_trade * TRAIN_FRAC)
    return full, train_start, holdout_start


def perf_slice(equity: pd.Series, start: int, end: int) -> dict:
    seg = equity.iloc[start:end]
    if len(seg) < 10:
        return {"sharpe": None, "cagr": None, "max_drawdown": None,
                "total_return": None, "n": len(seg)}
    return equity_metrics(seg, ann=252.0)


def run_combo(universe: pd.DataFrame, length: int, mult: float, offset: int) -> dict:
    train_rows, holdout_rows = [], []
    for symbol in universe["symbol"]:
        df = load_ohlcv(DATA_DIR / f"{symbol}.parquet")
        close = df["close"]
        full, train_start, holdout_start = slice_windows(close)
        basis, _, lower = bollinger_bands(full, length, mult, offset)
        trades, equity = simulate(full, basis, lower, trade_from=train_start)
        if trades.empty:
            continue
        tr_perf = perf_slice(equity, train_start, holdout_start)
        ho_perf = perf_slice(equity, holdout_start, len(equity))
        n_train_trades = int(((trades["exit_time"] >= full.index[train_start]) &
                               (trades["exit_time"] < full.index[holdout_start])).sum())
        n_holdout_trades = int((trades["exit_time"] >= full.index[holdout_start]).sum())
        if tr_perf["sharpe"] is not None and n_train_trades >= 1:
            train_rows.append({"symbol": symbol, "trades": n_train_trades, **tr_perf})
        if ho_perf["sharpe"] is not None and n_holdout_trades >= 1:
            holdout_rows.append({"symbol": symbol, "trades": n_holdout_trades, **ho_perf})

    tr_df = pd.DataFrame(train_rows)
    ho_df = pd.DataFrame(holdout_rows)
    return {
        "length": length, "mult": mult, "offset": offset,
        "train_n_symbols": len(tr_df),
        "train_median_sharpe": float(tr_df["sharpe"].median()) if len(tr_df) else None,
        "train_median_trades": float(tr_df["trades"].median()) if len(tr_df) else None,
        "train_median_total_return": float(tr_df["total_return"].median()) if len(tr_df) else None,
        "holdout_n_symbols": len(ho_df),
        "holdout_median_sharpe": float(ho_df["sharpe"].median()) if len(ho_df) else None,
        "holdout_median_trades": float(ho_df["trades"].median()) if len(ho_df) else None,
        "holdout_median_total_return": float(ho_df["total_return"].median()) if len(ho_df) else None,
    }


def buy_hold_baseline(universe: pd.DataFrame) -> dict:
    tr_rows, ho_rows = [], []
    for symbol in universe["symbol"]:
        df = load_ohlcv(DATA_DIR / f"{symbol}.parquet")
        close = df["close"]
        full, train_start, holdout_start = slice_windows(close)
        equity = full / full.iloc[train_start]
        tr_rows.append(perf_slice(equity, train_start, holdout_start))
        ho_rows.append(perf_slice(equity, holdout_start, len(equity)))
    tr_df = pd.DataFrame(tr_rows)
    ho_df = pd.DataFrame(ho_rows)
    return {
        "train_median_sharpe": float(tr_df["sharpe"].median()),
        "train_median_total_return": float(tr_df["total_return"].median()),
        "holdout_median_sharpe": float(ho_df["sharpe"].median()),
        "holdout_median_total_return": float(ho_df["total_return"].median()),
    }


def main() -> None:
    print("1) 횡보성 종목 스크리닝 중 (ADX 낮은 순, 유동성 필터)...", flush=True)
    universe = screen_universe()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    universe.to_csv(OUT_DIR / "bb_crossing_universe.csv", index=False)
    print(f"   선정 {len(universe)}종목: {', '.join(universe['symbol'].tolist())}")

    print("2) buy&hold 베이스라인 계산...", flush=True)
    bh = buy_hold_baseline(universe)
    print(f"   B&H train Sharpe(median)={bh['train_median_sharpe']:.3f} "
          f"holdout Sharpe(median)={bh['holdout_median_sharpe']:.3f}")

    print(f"3) 그리드 스윕: length{LENGTHS} x mult{MULTS} x offset{OFFSETS} "
          f"= {len(LENGTHS)*len(MULTS)*len(OFFSETS)}개 조합 x {len(universe)}종목", flush=True)
    rows = []
    total = len(LENGTHS) * len(MULTS) * len(OFFSETS)
    done = 0
    for length in LENGTHS:
        for mult in MULTS:
            for offset in OFFSETS:
                res = run_combo(universe, length, mult, offset)
                rows.append(res)
                done += 1
                if done % 20 == 0 or done == total:
                    print(f"   {done}/{total} 완료", flush=True)

    grid = pd.DataFrame(rows)
    valid = grid[grid["train_median_sharpe"].notna() & (grid["train_median_trades"] >= 5)]
    ranked = valid.sort_values("train_median_sharpe", ascending=False)
    top5 = ranked.head(5)

    import json
    out_json = OUT_DIR / "bb_crossing_sweep.json"
    out_json.write_text(json.dumps({
        "roundtrip_fee_bps": ROUNDTRIP_FEE * 1e4,
        "universe": universe.to_dict(orient="records"),
        "buy_hold_baseline": bh,
        "grid": grid.to_dict(orient="records"),
        "top5_by_train_sharpe": top5.to_dict(orient="records"),
    }, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    lines = [
        "# 볼린저밴드 평균회귀 크로싱 — 횡보종목 파라미터 스윕",
        "",
        f"- 유니버스: ADX({ADX_WINDOW}) 최근 {SCREEN_LOOKBACK_BARS}봉 평균이 낮은 "
        f"{len(universe)}종목 (유동성/가격 필터 통과)",
        f"- 백테스트: 최근 ~{BACKTEST_BARS}봉, train {int(TRAIN_FRAC*100)}% / holdout "
        f"{int((1-TRAIN_FRAC)*100)}% 시계열 분할, 왕복비용 {ROUNDTRIP_FEE*1e4:.1f}bp "
        "(KIS US 편도수수료+슬리피지 가정)",
        f"- offset은 인과적(>=0)만 스윕({OFFSETS}). offset<0은 미래 밴드값을 과거봉에 "
        "대응시키는 것이라 실거래 불가(look-ahead)하여 제외.",
        f"- B&H 베이스라인: train Sharpe(median)={bh['train_median_sharpe']:.3f}, "
        f"holdout Sharpe(median)={bh['holdout_median_sharpe']:.3f}",
        "",
        "## Train 기준 상위 5개 조합 (median trades>=5)",
        "",
        "| length | mult | offset | train Sharpe | train trades | holdout Sharpe | holdout trades |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, r in top5.iterrows():
        hs = f"{r['holdout_median_sharpe']:.3f}" if r["holdout_median_sharpe"] is not None else "-"
        ht = f"{r['holdout_median_trades']:.0f}" if r["holdout_median_trades"] is not None else "-"
        lines.append(
            f"| {int(r['length'])} | {r['mult']} | {int(r['offset'])} | "
            f"{r['train_median_sharpe']:.3f} | {r['train_median_trades']:.0f} | {hs} | {ht} |"
        )
    lines.append("")

    if len(top5):
        best = top5.iloc[0]
        lines += [
            "## 최적 조합 (train Sharpe 기준)",
            "",
            f"- length={int(best['length'])}, mult={best['mult']}, offset={int(best['offset'])}",
            f"- train median Sharpe={best['train_median_sharpe']:.3f} "
            f"(B&H {bh['train_median_sharpe']:.3f} 대비 "
            f"{'우위' if best['train_median_sharpe'] > bh['train_median_sharpe'] else '열위'})",
        ]
        if best["holdout_median_sharpe"] is not None:
            lines.append(
                f"- holdout median Sharpe={best['holdout_median_sharpe']:.3f} "
                f"(B&H {bh['holdout_median_sharpe']:.3f} 대비 "
                f"{'우위' if best['holdout_median_sharpe'] > bh['holdout_median_sharpe'] else '열위'})"
            )
        lines.append("")

    lines += [
        "## 주의",
        "",
        "- 스크리닝 lookback(최근 2년)이 holdout 구간과 일부 겹친다 — 완전한 "
        "out-of-sample 종목선정은 아니다(현재 횡보 중인 종목을 고른 뒤 과거로 돌아가 검증하는 구조).",
        "- 진입/청산 모두 당봉 종가 체결로 단순화(익일 시가 체결 지연 미반영).",
        "- maType은 bb.txt 기본값 SMA로 고정(스윕 대상 아님).",
        "",
        "## 파일",
        "",
        f"- 유니버스: `{(OUT_DIR / 'bb_crossing_universe.csv').relative_to(ROOT)}`",
        f"- 전체 그리드: `{out_json.relative_to(ROOT)}`",
    ]
    out_md = OUT_DIR / "bb_crossing_sweep.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nMD: {out_md}")
    if len(top5):
        best = top5.iloc[0]
        print(f"\n=== 최적: length={int(best['length'])} mult={best['mult']} "
              f"offset={int(best['offset'])} train Sharpe={best['train_median_sharpe']:.3f} ===")


if __name__ == "__main__":
    main()
