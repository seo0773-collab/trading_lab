"""칼만 히스토그램 누적프로파일 → 모멘텀 게이트 (yoon3).

macd_raw.txt(Pine v6)의 **칼만 히스토그램**(kalHist = kalMacd − kalSignal)을 계산하고,
그 값의 **누적프로파일(cumulative profile)** 내 백분위(0~1)를 봉마다 산출한다. 백분위를
[g_min, 1.0] 게이트로 매핑해 yoon1b의 종목 점수에 곱한다(블렌드: 저가권 × 모멘텀).

설계 의도:
- **무누수**: 각 봉의 백분위는 시작~현재 봉까지의 값만 누적해 산출한다(과거·현재).
  엔진(`simulate_portfolio`)에서 점수가 한 번 더 shift(1)되므로 체결은 전봉 신호 기준.
- **자기적응**: 히스토그램을 rolling std로 정규화한 z를 고정 범위([-z_clip, z_clip])에
  비닝하므로 종목·국면별 스케일 차이가 사라진다 → 원시 칼만 추세신호 이식이 `floor`
  포화로 무효였던 것과 달리, 곱셈 게이트는 포화 없이 작동한다.
- **블렌드 방향(기본 momentum)**: 백분위가 높을수록(모멘텀이 자기 분포 상위) 게이트가
  열린다 → "싸면서(저가권 점수↑) 모멘텀이 자기 분포를 타고 올라온" 종목에 비중을 싣고,
  여전히 붕괴 중(백분위 하위)인 종목은 억제해 회복 진입 타이밍/노출 공백을 보강한다.

`gate_cfg.enabled`가 거짓이면 호출되지 않으며, 게이트가 적용돼도 warmup 구간은 1.0
(불변)으로 떨어진다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from indicators.kalman import kalman_1d


def _kal_hist(
    close: pd.Series, *, fast: int, slow: int, signal: int,
    q: float, r: float, base: str,
) -> pd.Series:
    """macd_raw.txt의 칼만 히스토그램(kalHist)을 재현한다.

    kalman_base="MACD Line": kalMacd = Kalman(EMA_fast − EMA_slow).
    kalman_base="Fast/Slow EMA": kalMacd = Kalman(EMA_fast) − Kalman(EMA_slow).
    공통: kalSignal = Kalman(EMA(kalMacd, signal)), kalHist = kalMacd − kalSignal.
    """
    c = close.astype(float)
    fast_ema = c.ewm(span=fast, adjust=False, min_periods=fast).mean()
    slow_ema = c.ewm(span=slow, adjust=False, min_periods=slow).mean()
    if str(base) == "Fast/Slow EMA":
        kal_macd = kalman_1d(fast_ema, q, r) - kalman_1d(slow_ema, q, r)
    else:
        kal_macd = kalman_1d(fast_ema - slow_ema, q, r)
    kal_sig_base = kal_macd.ewm(span=signal, adjust=False, min_periods=signal).mean()
    kal_sig = kalman_1d(kal_sig_base, q, r)
    return (kal_macd - kal_sig).rename("kal_hist")


def _cumulative_percentile(
    vals: np.ndarray, bin_count: int, rolling_window: int, z_clip: float,
) -> np.ndarray:
    """정규화 히스토그램 값의 누적프로파일 내 하위 백분위(0~1).

    봉마다 값을 [-z_clip, z_clip] 범위의 bin에 one-hot 누적 → bars 방향 cumsum으로
    누적프로파일을 만든 뒤, 현재 값이 그 분포에서 차지하는 하위 누적비율을 구한다.
    rolling_window>0이면 최근 N봉만 유지(window 밖 제거), 0이면 시작부터 누적.
    profile.py의 `_percentile_at`과 동일한 (below + 자기bin 절반)/total 방식.
    """
    n = len(vals)
    edges = np.linspace(-z_clip, z_clip, bin_count + 1)
    valid = np.isfinite(vals)
    idx = np.clip(np.searchsorted(edges, vals, side="right") - 1, 0, bin_count - 1)

    contrib = np.zeros((n, bin_count), dtype=float)
    rows = np.arange(n)
    contrib[rows[valid], idx[valid]] = 1.0

    cum = np.cumsum(contrib, axis=0)
    if rolling_window and 0 < rolling_window < n:
        prof = cum.copy()
        prof[rolling_window:] = cum[rolling_window:] - cum[:-rolling_window]
    else:
        prof = cum

    total = prof.sum(axis=1)
    cumdist = np.cumsum(prof, axis=1)
    below = np.where(idx > 0, cumdist[rows, np.maximum(idx - 1, 0)], 0.0)
    in_bin = prof[rows, idx]
    pct = np.divide(
        below + 0.5 * in_bin, total,
        out=np.full(n, np.nan), where=total > 0,
    )
    pct[~valid] = np.nan
    return np.clip(pct, 0.0, 1.0)


def momentum_gate(close: pd.Series, gate_cfg: dict) -> pd.Series:
    """종목 종가 → [g_min, 1.0] 모멘텀 게이트 시리즈(close.index 정렬).

    warmup/결측 구간은 1.0(불변)으로 둔다. direction="contrarian"이면 백분위를 뒤집어
    낮은 모멘텀(투매)에 게이트를 연다(실험용, 기본은 momentum).
    """
    g_min = float(gate_cfg.get("g_min", 0.5))
    fast = int(gate_cfg.get("fast_len", 12))
    slow = int(gate_cfg.get("slow_len", 26))
    signal = int(gate_cfg.get("signal_len", 9))
    q = float(gate_cfg.get("kalman_q", 0.01))
    r = float(gate_cfg.get("kalman_r", 0.10))
    base = str(gate_cfg.get("kalman_base", "MACD Line"))
    norm_window = int(gate_cfg.get("norm_window", 252))
    bin_count = int(gate_cfg.get("bin_count", 120))
    rolling_window = int(gate_cfg.get("rolling_window", 0))  # 0 = 누적(expanding)
    z_clip = float(gate_cfg.get("z_clip", 4.0))
    direction = str(gate_cfg.get("direction", "momentum"))

    hist = _kal_hist(
        close, fast=fast, slow=slow, signal=signal, q=q, r=r, base=base,
    )
    sd = hist.rolling(norm_window, min_periods=max(5, norm_window // 4)).std()
    z = (hist / sd.replace(0.0, np.nan)).clip(-z_clip, z_clip)
    pct = _cumulative_percentile(
        z.to_numpy(dtype=float), bin_count, rolling_window, z_clip,
    )
    pct_s = pd.Series(pct, index=close.index)
    if direction == "contrarian":
        pct_s = 1.0 - pct_s
    gate = g_min + (1.0 - g_min) * pct_s
    return gate.fillna(1.0).rename("mom_gate")
