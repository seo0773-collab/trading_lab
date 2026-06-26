"""Volume Profile core — 2D heatmap + 1D rolling levels (POC/VAH/VAL).

heatmap1 전략과 독립 CLI가 공유하는 *순수 함수* 모음.
- ``build_heatmap``: 가격×시간 격자 누적(2D) → PNG/연구용.
- ``rolling_profile_levels``: 바별 롤링 프로파일에서 POC/VAH/VAL 1D 시계열
  (lookahead 없음: 각 t는 t 이하 데이터만 사용) → 신호 입력 + forecast 오버레이.

데이터 소스 무관: 정규화된 OHLCV(open/high/low/close/volume, DatetimeIndex)만 받는다.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# --------------------------------------------------------------------- 격자
def _price_edges(pmin: float, pmax: float, bins: int, scale: str) -> np.ndarray:
    if pmax <= pmin:
        pmax = pmin + 1e-9
    if scale == "log" and pmin > 0:
        return np.geomspace(pmin, pmax, bins + 1)
    return np.linspace(pmin, pmax, bins + 1)


def window_histogram(
    low: np.ndarray,
    high: np.ndarray,
    volume: np.ndarray,
    bins: int,
    *,
    scale: str = "linear",
) -> tuple[np.ndarray, np.ndarray]:
    """[min low, max high]를 bins개 가격 bin으로 나눠 volume 균등분배 누적.

    각 바의 volume을 [low, high]가 걸친 bin들에 겹침 비율로 분배한다.
    반환: (hist[bins], edges[bins+1]).
    """
    pmin = float(np.min(low))
    pmax = float(np.max(high))
    edges = _price_edges(pmin, pmax, bins, scale)
    lo_clip = np.maximum(edges[:-1][None, :], low[:, None])
    hi_clip = np.minimum(edges[1:][None, :], high[:, None])
    overlap = np.clip(hi_clip - lo_clip, 0.0, None)
    span = high - low
    zero = span <= 0
    span_safe = np.where(zero, 1.0, span)[:, None]
    hist = (volume[:, None] * (overlap / span_safe)).sum(axis=0)
    if zero.any():  # high==low 바: 전체 volume을 해당 가격 bin에 투하
        idx = np.clip(np.searchsorted(edges, high[zero], side="right") - 1, 0, bins - 1)
        np.add.at(hist, idx, volume[zero])
    return hist, edges


def value_area(
    hist: np.ndarray, edges: np.ndarray, va_pct: float
) -> tuple[float, float, float]:
    """프로파일에서 POC와 Value Area(VAH/VAL)를 산출.

    POC = 최대 volume bin 중심. VA = POC에서 좌우로 더 큰 이웃을 흡수하며
    누적 volume이 va_pct에 도달할 때까지 확장한 가격대.
    """
    centers = 0.5 * (edges[:-1] + edges[1:])
    total = float(hist.sum())
    if total <= 0.0:
        mid = len(hist) // 2
        return float(centers[mid]), float(edges[mid + 1]), float(edges[mid])
    poc_idx = int(np.argmax(hist))
    lo_idx = hi_idx = poc_idx
    cum = float(hist[poc_idx])
    target = va_pct * total
    n = len(hist)
    while cum < target and (lo_idx > 0 or hi_idx < n - 1):
        below = hist[lo_idx - 1] if lo_idx > 0 else -np.inf
        above = hist[hi_idx + 1] if hi_idx < n - 1 else -np.inf
        if above >= below:
            hi_idx += 1
            cum += float(hist[hi_idx])
        else:
            lo_idx -= 1
            cum += float(hist[lo_idx])
    return float(centers[poc_idx]), float(edges[hi_idx + 1]), float(edges[lo_idx])


# --------------------------------------------------------- 1D 롤링 레벨 (신호용)
def rolling_profile_levels(
    df: pd.DataFrame,
    *,
    lookback: int = 120,
    bins: int = 60,
    va_pct: float = 0.70,
    cumulative: bool = False,
    scale: str = "linear",
) -> pd.DataFrame:
    """바별 롤링/누적 프로파일 → poc/vah/val 시계열 (lookahead 없음).

    cumulative=True면 0..t 확장 윈도우, False면 직전 lookback개 바를 본다.
    warmup(lookback) 이전 바는 NaN.
    """
    low = df["low"].to_numpy(float)
    high = df["high"].to_numpy(float)
    vol = df["volume"].to_numpy(float)
    n = len(df)
    poc = np.full(n, np.nan)
    vah = np.full(n, np.nan)
    val = np.full(n, np.nan)
    warmup = max(2, lookback)
    for t in range(warmup - 1, n):
        start = 0 if cumulative else t - lookback + 1
        sl = slice(max(0, start), t + 1)
        hist, edges = window_histogram(
            low[sl], high[sl], vol[sl], bins, scale=scale
        )
        poc[t], vah[t], val[t] = value_area(hist, edges, va_pct)
    return pd.DataFrame({"poc": poc, "vah": vah, "val": val}, index=df.index)


# ------------------------------------------- 고볼륨 노드(HVN) 지지/저항 (heatmap2)
def profile_nodes(
    hist: np.ndarray,
    edges: np.ndarray,
    *,
    top_n: int = 4,
    min_strength: float = 0.3,
    min_gap_bins: int = 3,
) -> list[tuple[float, float]]:
    """프로파일에서 '색 짙은 구간' = 고볼륨 노드(로컬 피크)를 추출.

    히트맵에서 밝게 보이는 가격대 = 거래가 집중된 합의가격 = 지지/저항 후보.
    조건: (1) 이웃보다 큰 로컬 최대, (2) 최대 bin의 ``min_strength`` 배 이상,
    (3) 서로 ``min_gap_bins`` 이상 떨어진 것만(가까운 봉우리 병합). 강도 상위
    ``top_n``개를 가격 오름차순으로 반환. 반환: [(center_price, strength), ...].
    """
    centers = 0.5 * (edges[:-1] + edges[1:])
    n = len(hist)
    hmax = float(hist.max()) if n else 0.0
    if hmax <= 0.0:
        return []
    thr = min_strength * hmax
    peaks: list[tuple[int, float]] = []
    for i in range(n):
        left = hist[i - 1] if i > 0 else -np.inf
        right = hist[i + 1] if i < n - 1 else -np.inf
        if hist[i] >= left and hist[i] >= right and hist[i] >= thr:
            peaks.append((i, float(hist[i])))
    peaks.sort(key=lambda x: -x[1])  # 강도 내림차순
    chosen: list[tuple[int, float]] = []
    for idx, strength in peaks:
        if all(abs(idx - j) >= min_gap_bins for j, _ in chosen):
            chosen.append((idx, strength))
        if len(chosen) >= top_n:
            break
    chosen.sort(key=lambda x: x[0])  # 가격(인덱스) 오름차순
    return [(float(centers[idx]), strength) for idx, strength in chosen]


def rolling_sr_levels(
    df: pd.DataFrame,
    *,
    lookback: int = 120,
    bins: int = 80,
    scale: str = "log",
    top_n: int = 4,
    min_strength: float = 0.3,
    min_gap_bins: int = 3,
    va_pct: float = 0.70,
    cumulative: bool = False,
    axis: str = "absolute",
) -> pd.DataFrame:
    """바별 롤링 프로파일 HVN에서 현재가 기준 인접 지지/저항 시계열 (lookahead 없음).

    각 t에서 직전 ``lookback``개 바(또는 cumulative=True면 0..t) 프로파일의 HVN 중,
    그 시점 종가(t 이하 데이터) 아래 가장 가까운 노드 = **지지**, 위 가장 가까운
    노드 = **저항**. heatmap1 시뮬엔진 재사용을 위해 컬럼명을 매핑한다:
    ``val=지지``, ``vah=저항``, ``poc``=프로파일 최빈가. warmup 이전·해당 노드
    없음은 NaN.

    ``axis='relative'``면 각 t의 인과 가격범위([cummin, cummax] 또는 윈도우 min/max)
    내 **상대위치(0~1)** 공간에서 프로파일·HVN을 만든 뒤 절대가격으로 역환산한다.
    cumulative=True와 함께 쓰면 전체기간 매물대를 해상도 손실 없이(과거 좁은 구간도
    풀 분해능) 반영한다.
    """
    if str(axis) == "relative":
        return _rolling_sr_relative(
            df, lookback=lookback, bins=bins, scale=scale, top_n=top_n,
            min_strength=min_strength, min_gap_bins=min_gap_bins,
            va_pct=va_pct, cumulative=cumulative,
        )
    low = df["low"].to_numpy(float)
    high = df["high"].to_numpy(float)
    vol = df["volume"].to_numpy(float)
    close = df["close"].to_numpy(float)
    n = len(df)
    poc = np.full(n, np.nan)
    support = np.full(n, np.nan)
    resistance = np.full(n, np.nan)
    warmup = max(2, lookback)
    for t in range(warmup - 1, n):
        start = 0 if cumulative else t - lookback + 1
        sl = slice(max(0, start), t + 1)
        hist, edges = window_histogram(low[sl], high[sl], vol[sl], bins, scale=scale)
        if hist.sum() <= 0.0:
            continue
        poc[t] = value_area(hist, edges, va_pct)[0]
        nodes = profile_nodes(
            hist, edges, top_n=top_n, min_strength=min_strength,
            min_gap_bins=min_gap_bins,
        )
        ref = close[t]
        below = [c for c, _ in nodes if c < ref]
        above = [c for c, _ in nodes if c > ref]
        if below:
            support[t] = max(below)
        if above:
            resistance[t] = min(above)
    return pd.DataFrame(
        {"poc": poc, "vah": resistance, "val": support}, index=df.index
    )


def _rel_scalar(price: float, lo: float, hi: float, scale: str) -> float:
    return float(_rel_positions(np.array([price], dtype=float), lo, hi, scale)[0])


def _inv_rel(r: float, lo: float, hi: float, scale: str) -> float:
    """상대위치 r(0~1) → 절대가격 역환산 (_rel_positions의 역함수)."""
    if scale == "log" and lo > 0 and hi > 0:
        return float(np.exp(np.log(lo) + r * (np.log(hi) - np.log(lo))))
    return float(lo + r * (hi - lo))


def _relative_histogram(
    low_w: np.ndarray, high_w: np.ndarray, vol_w: np.ndarray,
    lo_t: float, hi_t: float, bins: int, scale: str,
) -> tuple[np.ndarray, np.ndarray]:
    """윈도우 가격을 [lo_t, hi_t] 내 상대위치(0~1)로 매핑해 volume 누적."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    L = _rel_positions(low_w, lo_t, hi_t, scale)
    Hh = _rel_positions(high_w, lo_t, hi_t, scale)
    lo_clip = np.maximum(edges[:-1][None, :], L[:, None])
    hi_clip = np.minimum(edges[1:][None, :], Hh[:, None])
    overlap = np.clip(hi_clip - lo_clip, 0.0, None)
    span = Hh - L
    zero = span <= 0
    span_safe = np.where(zero, 1.0, span)[:, None]
    hist = (vol_w[:, None] * (overlap / span_safe)).sum(axis=0)
    if zero.any():
        idx = np.clip((Hh[zero] * bins).astype(int), 0, bins - 1)
        np.add.at(hist, idx, vol_w[zero])
    return hist, edges


def _rolling_sr_relative(
    df: pd.DataFrame,
    *,
    lookback: int,
    bins: int,
    scale: str,
    top_n: int,
    min_strength: float,
    min_gap_bins: int,
    va_pct: float,
    cumulative: bool,
) -> pd.DataFrame:
    """상대위치(0~1) 공간 HVN 지지/저항 → 절대가격 역환산 (rolling_sr_levels axis=relative)."""
    low = df["low"].to_numpy(float)
    high = df["high"].to_numpy(float)
    vol = df["volume"].to_numpy(float)
    close = df["close"].to_numpy(float)
    n = len(df)
    poc = np.full(n, np.nan)
    support = np.full(n, np.nan)
    resistance = np.full(n, np.nan)
    cummin = np.minimum.accumulate(low) if n else low
    cummax = np.maximum.accumulate(high) if n else high
    warmup = 2 if cumulative else max(2, lookback)
    for t in range(warmup - 1, n):
        if cumulative:
            lo_t, hi_t, sl = float(cummin[t]), float(cummax[t]), slice(0, t + 1)
        else:
            sl = slice(max(0, t - lookback + 1), t + 1)
            lo_t, hi_t = float(low[sl].min()), float(high[sl].max())
        if hi_t <= lo_t:
            continue
        hist, edges = _relative_histogram(
            low[sl], high[sl], vol[sl], lo_t, hi_t, bins, scale
        )
        if hist.sum() <= 0.0:
            continue
        poc_r = value_area(hist, edges, va_pct)[0]
        nodes = profile_nodes(
            hist, edges, top_n=top_n, min_strength=min_strength,
            min_gap_bins=min_gap_bins,
        )
        ref_r = _rel_scalar(close[t], lo_t, hi_t, scale)
        below = [c for c, _ in nodes if c < ref_r]
        above = [c for c, _ in nodes if c > ref_r]
        poc[t] = _inv_rel(poc_r, lo_t, hi_t, scale)
        if below:
            support[t] = _inv_rel(max(below), lo_t, hi_t, scale)
        if above:
            resistance[t] = _inv_rel(min(above), lo_t, hi_t, scale)
    return pd.DataFrame(
        {"poc": poc, "vah": resistance, "val": support}, index=df.index
    )


# --------------------------------------------- 시점별 상대위치 히트맵 (인과, 연구용)
def _rel_positions(
    price: np.ndarray, lo: float, hi: float, scale: str
) -> np.ndarray:
    """가격을 [lo, hi] 범위 내 상대위치(0..1)로. log면 로그가격 공간에서 매핑."""
    if scale == "log" and lo > 0 and hi > 0:
        return (np.log(price) - np.log(lo)) / (np.log(hi) - np.log(lo))
    return (price - lo) / (hi - lo)


def build_relative_heatmap(
    df: pd.DataFrame,
    *,
    rows: int = 300,
    max_cols: int | None = None,
    cumulative: bool = True,
    lookback: int = 120,
    scale: str = "linear",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """세로축을 '각 시점까지의 인과 가격범위 내 상대위치(0..1)'로 둔 히트맵.

    열 t는 **0..t 데이터만** 사용한다(미래 범위가 과거 열의 격자를 늘리지 않음).
    덕분에 가격대가 좁았던 과거 구간도 세로 전체 해상도를 써서 디테일이 보존된다.
    트레이드오프: y축은 절대가격이 아니라 '그 시점 범위 내 상대위치'다
    (수평선=동일가격 의미가 사라짐 — 시점마다 0.5가 다른 절대가격).

    ``cumulative=True``면 열 t = 0..t 누적 프로파일, False면 직전 ``lookback`` 윈도우.
    반환: (H[rows, k], rel_edges[rows+1] (0..1), time_axis[k]).
    """
    low = df["low"].to_numpy(float)
    high = df["high"].to_numpy(float)
    vol = df["volume"].to_numpy(float)
    n = len(df)
    edges = np.linspace(0.0, 1.0, rows + 1)
    if n == 0:
        return np.zeros((rows, 0)), edges, np.asarray(df.index)
    cummin = np.minimum.accumulate(low)
    cummax = np.maximum.accumulate(high)
    if max_cols and n > max_cols:
        cols_idx = np.unique(np.linspace(0, n - 1, max_cols).astype(int))
    else:
        cols_idx = np.arange(n)
    H = np.zeros((rows, len(cols_idx)), dtype=np.float64)
    for k, t in enumerate(cols_idx):
        if cumulative:
            lo_t, hi_t, sl = float(cummin[t]), float(cummax[t]), slice(0, t + 1)
        else:
            sl = slice(max(0, t - lookback + 1), t + 1)
            lo_t, hi_t = float(low[sl].min()), float(high[sl].max())
        if hi_t <= lo_t:  # 단일가격(스팬 0) — 표현 불가, 빈 열
            continue
        L = _rel_positions(low[sl], lo_t, hi_t, scale)
        Hh = _rel_positions(high[sl], lo_t, hi_t, scale)
        v = vol[sl]
        lo_clip = np.maximum(edges[:-1][None, :], L[:, None])
        hi_clip = np.minimum(edges[1:][None, :], Hh[:, None])
        overlap = np.clip(hi_clip - lo_clip, 0.0, None)
        span = Hh - L
        zero = span <= 0
        span_safe = np.where(zero, 1.0, span)[:, None]
        col = (v[:, None] * (overlap / span_safe)).sum(axis=0)
        if zero.any():  # high==low 바: 전체 volume을 상대위치 bin에 투하
            idx = np.clip((Hh[zero] * rows).astype(int), 0, rows - 1)
            np.add.at(col, idx, v[zero])
        H[:, k] = col
    return H, edges, np.asarray(df.index)[cols_idx]


# ------------------------------------------------------------- 2D 히트맵 (연구용)
def build_heatmap(
    df: pd.DataFrame,
    *,
    rows: int = 300,
    cumulative: bool = False,
    scale: str = "linear",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """가격×시간 격자 누적. 반환: (H[rows, cols], price_edges[rows+1], time_axis)."""
    low = df["low"].to_numpy(float)
    high = df["high"].to_numpy(float)
    vol = df["volume"].to_numpy(float)
    cols = len(df)
    edges = _price_edges(float(low.min()), float(high.max()), rows, scale)
    H = np.zeros((rows, cols), dtype=np.float64)
    lo_clip = np.maximum(edges[:-1][:, None], low[None, :])
    hi_clip = np.minimum(edges[1:][:, None], high[None, :])
    overlap = np.clip(hi_clip - lo_clip, 0.0, None)  # rows x cols
    span = high - low
    span_safe = np.where(span <= 0, 1.0, span)[None, :]
    H = vol[None, :] * (overlap / span_safe)
    zero = span <= 0
    if zero.any():
        idx = np.clip(np.searchsorted(edges, high[zero], side="right") - 1, 0, rows - 1)
        cols_zero = np.nonzero(zero)[0]
        for r, c, v in zip(idx, cols_zero, vol[zero]):
            H[r, c] += v
    if cumulative:
        H = np.cumsum(H, axis=1)
    return H, edges, np.asarray(df.index)


def render(
    H: np.ndarray,
    price_edges: np.ndarray,
    time_axis: np.ndarray,
    *,
    gamma: float = 0.5,
    percentile_clip: float = 99.0,
    cmap: str = "viridis",
    gaussian_sigma: float = 0.0,
    title: str | None = None,
) -> Any:
    """히트맵 렌더 → matplotlib Figure (분위수 클립 + gamma + 선택 가우시안)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = H.astype(float)
    if gaussian_sigma > 0.0:
        from scipy.ndimage import gaussian_filter

        grid = gaussian_filter(grid, sigma=gaussian_sigma)
    clip = np.percentile(grid, percentile_clip) if grid.max() > 0 else 1.0
    norm = np.clip(grid / clip, 0.0, 1.0) ** gamma
    fig, ax = plt.subplots(figsize=(12, 6))
    mesh = ax.imshow(
        norm, aspect="auto", origin="lower", cmap=cmap,
        extent=[0, grid.shape[1], price_edges[0], price_edges[-1]],
    )
    fig.colorbar(mesh, ax=ax, label="volume (norm)")
    ax.set_xlabel("bar index")
    ax.set_ylabel("price")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------------- CLI
def _load_cli(args: Any) -> pd.DataFrame:
    if args.parquet:
        df = pd.read_parquet(args.parquet)
        df.columns = [str(c).lower() for c in df.columns]
        return df
    import sys

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
    if args.asset_class == "crypto":
        import ccxt  # noqa: F401

        from fetch_ohlcv import fetch_ohlcv

        ex = getattr(__import__("ccxt"), args.exchange)({"enableRateLimit": True})
        ex.load_markets()
        since = int(pd.Timestamp(args.since, tz="UTC").timestamp() * 1000)
        return fetch_ohlcv(ex, args.symbol, args.interval, since)
    from trading_lab.market_data import load_cumulative_yfinance

    return load_cumulative_yfinance(args.symbol, args.interval, args.period)


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Volume Profile heatmap renderer")
    p.add_argument("--parquet", help="직접 로드할 OHLCV parquet 경로")
    p.add_argument("--symbol", default="BTC/USDT")
    p.add_argument("--asset-class", default="equity", choices=["equity", "crypto"])
    p.add_argument("--exchange", default="binance")
    p.add_argument("--interval", default="1d")
    p.add_argument("--period", default="max")
    p.add_argument("--since", default="2021-01-01")
    p.add_argument("--rows", type=int, default=300)
    p.add_argument("--last", type=int, default=0, help="마지막 N개 바만(0=전체)")
    p.add_argument("--cumulative", action="store_true")
    p.add_argument("--scale", default="linear", choices=["linear", "log"])
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--percentile-clip", type=float, default=99.0)
    p.add_argument("--cmap", default="viridis")
    p.add_argument("--gaussian-sigma", type=float, default=1.0)
    p.add_argument("--out", default="var/heatmap.png")
    args = p.parse_args(argv)

    df = _load_cli(args).dropna(subset=["high", "low", "close", "volume"]).sort_index()
    if args.last > 0:
        df = df.iloc[-args.last :]
    H, edges, taxis = build_heatmap(
        df, rows=args.rows, cumulative=args.cumulative, scale=args.scale
    )
    title = f"{args.symbol} {args.interval} VP heatmap ({len(df)} bars" + (
        ", cumulative)" if args.cumulative else ")"
    )
    fig = render(
        H, edges, taxis, gamma=args.gamma, percentile_clip=args.percentile_clip,
        cmap=args.cmap, gaussian_sigma=args.gaussian_sigma, title=title,
    )
    out = __import__("pathlib").Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"saved {out}  H={H.shape}  price[{edges[0]:.2f},{edges[-1]:.2f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
