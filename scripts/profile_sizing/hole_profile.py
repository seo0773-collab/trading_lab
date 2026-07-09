"""hole_profile: 계층적 지그재그(L1) 진행-스윙 프로파일 돌파/지속 모멘텀 (단일종목 롱숏).

hole_profile.txt(Pine "Multiscale Market Structure")의 익스트림 프로파일 구조를
hole_profile_features.compute_hole_profile_state로 이식하고, "hole"(분포 희소
구간) 진입을 돌파/지속 신호로 해석한다: 선택 레벨(L1)에서 마지막 확정 피벗
이후 진행 중인 스윙의 |%변동|이 그 레벨의 과거 완결 스윙 분포에서 상위
percentile(entry_percentile, expanding·무누수)에 도달하면 그 방향으로 진입한다.

hole_profile_study.py(2026-07-08) 근거: BTC 일봉 2017-08~2026-07 기준 L1 레벨·
threshold=0.70 게이트가 h=5/10/20 전부 Newey-West t=+2.9~+3.5(FDR 유의, 3/36)로
유일하게 살아남았다(L0 무의미·L2 표본부족/불안정). train/val/test 3분할 전부
edge 방향은 양(+)으로 일관되나 val/test는 표본 축소로 t<1.65(방향만 확정,
개별 구간 유의성은 미확정) — live_eligible=False, 페이퍼 전 보류.

청산: (1) ATR 배수 손절, (2) 구조적 확정 청산 — 진입 근거였던 피벗이 다음
확정 피벗으로 갱신되면(그 레벨에서 스윙이 공식적으로 마무리됐다는 뜻) 다음
봉 시가 청산, (3) percentile 소진(exit_percentile 미만으로 후퇴), (4) 시간청산.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .hole_profile_features import compute_hole_profile_state


@dataclass(frozen=True)
class HoleProfileConfig:
    atr_len: int = 48
    atr_mult: float = 1.0
    level: str = "L1"
    min_profile_samples: int = 30
    profile_bins: int = 40
    entry_percentile: float = 0.70
    exit_percentile: float = 0.30
    min_hold_bars: int = 1
    max_hold_bars: int = 60
    stop_atr_mult: float = 3.0
    take_profit_atr_mult: float = 0.0   # 0 = 비활성(구조/percentile 청산에만 의존)
    contrarian: bool = False   # True면 hole 진입 시 역추세(fade) — 진행 방향 반대로 진입
    long_only: bool = False
    fee_bps_per_side: float = 5.0
    slippage_bps: float = 5.0


def hole_profile_config(raw: dict[str, Any]) -> HoleProfileConfig:
    base = HoleProfileConfig()
    st = raw.get("structure") or {}
    sig = raw.get("signal") or {}
    costs = raw.get("costs") or {}
    return HoleProfileConfig(
        atr_len=int(st.get("atr_len", base.atr_len)),
        atr_mult=float(st.get("atr_mult", base.atr_mult)),
        level=str(st.get("level", base.level)),
        min_profile_samples=int(st.get("min_profile_samples", base.min_profile_samples)),
        profile_bins=int(st.get("profile_bins", base.profile_bins)),
        entry_percentile=float(sig.get("entry_percentile", base.entry_percentile)),
        exit_percentile=float(sig.get("exit_percentile", base.exit_percentile)),
        min_hold_bars=int(sig.get("min_hold_bars", base.min_hold_bars)),
        max_hold_bars=int(sig.get("max_hold_bars", base.max_hold_bars)),
        stop_atr_mult=float(sig.get("stop_atr_mult", base.stop_atr_mult)),
        take_profit_atr_mult=float(
            sig.get("take_profit_atr_mult", base.take_profit_atr_mult)
        ),
        contrarian=bool(sig.get("contrarian", base.contrarian)),
        long_only=bool(sig.get("long_only", base.long_only)),
        fee_bps_per_side=float(costs.get("fee_bps_per_side", base.fee_bps_per_side)),
        slippage_bps=float(costs.get("slippage_bps", base.slippage_bps)),
    )


def _entry_reason(
    d: int, level: str, rank: float, duration: int, pct_move: float, contrarian: bool,
) -> str:
    side = "롱" if d == 1 else "숏"
    mode = "역추세(fade)" if contrarian else "돌파/지속"
    return (
        f"{side} · {mode} · {level} 스윙 백분위 {rank:.2f} · "
        f"{duration}봉/{pct_move:+.1f}% 진행"
    )


def simulate_hole_profile(
    df: pd.DataFrame, state: pd.DataFrame, cfg: HoleProfileConfig
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    op = df["open"].to_numpy(dtype=float)
    hi = df["high"].to_numpy(dtype=float)
    lo = df["low"].to_numpy(dtype=float)
    cl = df["close"].to_numpy(dtype=float)
    atr = state["atr"].to_numpy(dtype=float)
    rank = state["profile_rank"].to_numpy(dtype=float)
    direction = state["direction"].to_numpy(dtype=int)
    pivot_idx = state["pivot_idx"].to_numpy(dtype=int)
    duration = state["duration_bars"].to_numpy(dtype=int)
    pct_move = state["pct_move"].to_numpy(dtype=float)
    index = df.index
    n = len(df)

    allow_short = not cfg.long_only
    cost_frac = (cfg.fee_bps_per_side + cfg.slippage_bps) / 1e4

    valid_rank = np.isfinite(rank)
    warmup = int(np.argmax(valid_rank)) if valid_rank.any() else n

    position = np.zeros(n)
    cost_hits = np.zeros(n)
    trades: list[dict[str, Any]] = []

    i = warmup
    while i < n - 1:
        d = 0
        if valid_rank[i] and direction[i] != 0 and rank[i] >= cfg.entry_percentile:
            d = -int(direction[i]) if cfg.contrarian else int(direction[i])
        if d == -1 and not allow_short:
            d = 0
        if d == 0:
            i += 1
            continue

        entry_j = i + 1
        entry_price = op[entry_j]
        entry_pivot_idx = pivot_idx[i]
        entry_atr = atr[i] if np.isfinite(atr[i]) and atr[i] > 0 else entry_price * 0.02
        stop_price = entry_price - d * cfg.stop_atr_mult * entry_atr
        take_profit_price = (
            entry_price + d * cfg.take_profit_atr_mult * entry_atr
            if cfg.take_profit_atr_mult > 0 else np.nan
        )
        reason_str = _entry_reason(
            d, cfg.level, float(rank[i]), int(duration[i]), float(pct_move[i]),
            cfg.contrarian,
        )

        term_j = term_price = None
        reason = ""
        j = entry_j
        while j < n:
            held = j - entry_j
            if (d == 1 and lo[j] <= stop_price) or (d == -1 and hi[j] >= stop_price):
                term_j, term_price, reason = j, stop_price, "stop_atr"
                break
            if cfg.take_profit_atr_mult > 0 and (
                (d == 1 and hi[j] >= take_profit_price)
                or (d == -1 and lo[j] <= take_profit_price)
            ):
                term_j, term_price, reason = j, take_profit_price, "take_profit_atr"
                break
            if held >= cfg.min_hold_bars:
                if pivot_idx[j] != entry_pivot_idx:
                    nxt = j + 1
                    if nxt < n:
                        term_j, term_price, reason = nxt, op[nxt], "structure_confirmed"
                    else:
                        term_j, term_price, reason = j, cl[j], "end_of_data"
                    break
                if valid_rank[j] and rank[j] < cfg.exit_percentile:
                    nxt = j + 1
                    if nxt < n:
                        term_j, term_price, reason = nxt, op[nxt], "percentile_fade"
                    else:
                        term_j, term_price, reason = j, cl[j], "end_of_data"
                    break
                if cfg.max_hold_bars > 0 and held >= cfg.max_hold_bars:
                    nxt = j + 1
                    if nxt < n:
                        term_j, term_price = nxt, op[nxt]
                    else:
                        term_j, term_price = j, cl[j]
                    reason = "horizon"
                    break
            j += 1
        if term_j is None:
            term_j, term_price, reason = n - 1, cl[n - 1], "end_of_data"

        gross = d * (term_price / entry_price - 1.0)
        trades.append({
            "direction": d,
            "entry_time": index[entry_j],
            "entry_price": entry_price,
            "exit_time": index[term_j],
            "exit_price": term_price,
            "stop_loss_price": stop_price,
            "take_profit_price": take_profit_price,
            "net_return": gross - 2.0 * cost_frac,
            "exit_reason": reason,
            "entry_reason": reason_str,
        })
        position[entry_j: term_j + 1] = d
        cost_hits[entry_j] += cost_frac
        cost_hits[term_j] += cost_frac
        i = term_j + 1

    ret = np.zeros(n)
    ret[1:] = cl[1:] / cl[:-1] - 1.0
    bar_return = position * ret - cost_hits

    trades_df = pd.DataFrame(trades, columns=[
        "direction", "entry_time", "entry_price", "exit_time", "exit_price",
        "stop_loss_price", "take_profit_price", "net_return", "exit_reason",
        "entry_reason",
    ])
    return (
        trades_df,
        pd.Series(position, index=index, name="position"),
        pd.Series(bar_return, index=index, name="bar_return"),
    )


def _histogram_frame(
    values: list[float], *, bins: int, lo: float, hi: float, label_fmt: Any,
) -> pd.DataFrame:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    edges = np.linspace(lo, hi, bins + 1)
    counts = (
        np.zeros(bins, dtype=int) if arr.size == 0
        else np.histogram(arr, bins=edges)[0]
    )
    centers = (edges[:-1] + edges[1:]) / 2.0
    labels = [label_fmt(edges[i], edges[i + 1]) for i in range(bins)]
    return pd.DataFrame({
        "bin_label": labels, "bin_center": centers, "count": counts.astype(int),
    })


def duration_and_pct_move_profiles(
    level_samples: dict[str, list[float]], *, bins: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pine 원본의 "Time Bin"/"Price % Bin" 프로파일 테이블을 그대로 재현.

    duration_profile: 완결 스윙 지속시간(봉) 히스토그램(0~최대, 비대칭).
    pct_move_profile: 완결 스윙 가격변동%(부호 보존) 히스토그램(-최대|~+최대|, 대칭).
    """
    durations = level_samples["durations"]
    moves = level_samples["signed_moves"]
    dur_hi = max(durations) if durations else 1.0
    dur_hi = dur_hi if dur_hi > 0 else 1.0
    duration_profile = _histogram_frame(
        durations, bins=bins, lo=0.0, hi=dur_hi,
        label_fmt=lambda a, b: f"{int(round(a))}~{int(round(b))}봉",
    )
    abs_max = max((abs(m) for m in moves), default=1.0)
    abs_max = abs_max if abs_max > 0 else 1.0
    pct_move_profile = _histogram_frame(
        moves, bins=bins, lo=-abs_max, hi=abs_max,
        label_fmt=lambda a, b: f"{a:+.1f}~{b:+.1f}%",
    )
    return duration_profile, pct_move_profile


def run_hole_profile_pipeline(raw: pd.DataFrame, raw_cfg: dict[str, Any]) -> dict:
    cfg = hole_profile_config(raw_cfg)
    df = raw.dropna(subset=["open", "high", "low", "close"]).sort_index()
    state, samples = compute_hole_profile_state(
        df,
        atr_len=cfg.atr_len,
        atr_mult=cfg.atr_mult,
        level=cfg.level,
        min_profile_samples=cfg.min_profile_samples,
        include_samples=True,
    )
    trades, position, bar_return = simulate_hole_profile(df, state, cfg)

    forecast = pd.DataFrame(index=df.index)
    for col in ("open", "high", "low", "close", "volume"):
        if col in df:
            forecast[col] = np.asarray(df[col], dtype=float)
    forecast = forecast.join(state)
    forecast["position"] = position

    equity = (1.0 + bar_return).cumprod().rename("equity")
    bnh = (df["close"] / df["close"].iloc[0]).rename("buy_hold")

    duration_profile, pct_move_profile = duration_and_pct_move_profiles(
        samples[cfg.level], bins=cfg.profile_bins,
    )

    return {
        "forecast": forecast,
        "trades": trades,
        "position": position,
        "bar_return": bar_return,
        "equity": equity,
        "buy_hold": bnh,
        "duration_profile": duration_profile,
        "pct_move_profile": pct_move_profile,
    }
