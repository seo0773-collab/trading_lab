"""Signal generation for entry variants A (P4) and B (P5) (plan 1, 8)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtest import stop_take_profit
from .config import (
    CostConfig, ExitConfig, SignalConfig, SimilarityEvConfig, StatsConfig,
)
from .events import Event
from .extreme_transition import _geometry, lookup_transition
from .extremes import Extreme
from .stats import expected_values, lookup_stats


@dataclass
class Signal:
    signal_idx: int  # decision bar; entry happens at the next bar open
    entry_variant: str
    direction: str  # predicted_direction
    event: Event
    signal: str = "hold"  # "long" | "short" | "hold"
    filter_reason: str = ""
    confidence: float = float("nan")
    raw_expected_value: float = float("nan")
    pressure_adjusted_expected_value: float = float("nan")
    entry_price_candidate: float = float("nan")
    stop_price_candidate: float = float("nan")
    take_profit_candidate: float = float("nan")
    stats_bucket: str = ""
    continuation_score: float = float("nan")  # P(next-extreme continuation)
    transition_bucket: str = ""
    similarity_expected_return: float = float("nan")
    similarity_ev_lower_bound: float = float("nan")
    similarity_effective_n: float = float("nan")
    similarity_confidence: float = float("nan")
    similarity_fallback: str = ""


def p5_confirmation_index(
    event: Event, plus_ext: list[Extreme], minus_ext: list[Extreme]
) -> int | None:
    """plan 5A.3 / 8 Variant B: the directional W line must confirm a valid P5.

    For a "strong" setup both DI lines must confirm (the directional W line as
    a higher-low / lower-high, and the opposite M line in mirror). For a "weak"
    setup only the directional W line is checked, since the opposite line is
    not a confirming pattern. Returns the (later) confirmation bar, or None
    when the P5 condition is unmet or not yet confirmed.
    """
    pj = event.plus_j + 1
    mj = event.minus_j + 1
    p_next = plus_ext[pj] if pj < len(plus_ext) else None
    m_next = minus_ext[mj] if mj < len(minus_ext) else None
    plus_p3 = event.plus_p[2].value
    minus_p3 = event.minus_p[2].value
    weak = event.tier == "weak"
    if event.direction == "long":
        # directional line +DI(W): next extreme is a higher low (P5 > P3).
        plus_ok = (
            p_next is not None and p_next.kind == "L" and p_next.value > plus_p3
        )
        if weak:
            return p_next.confirmation_idx if plus_ok else None
        # strong also needs -DI(M): next extreme is a lower high (P5 < P3).
        minus_ok = (
            m_next is not None and m_next.kind == "H" and m_next.value < minus_p3
        )
        if plus_ok and minus_ok:
            return max(p_next.confirmation_idx, m_next.confirmation_idx)
        return None
    # short: directional line -DI(W): next extreme is a higher low (P5 > P3).
    minus_ok = (
        m_next is not None and m_next.kind == "L" and m_next.value > minus_p3
    )
    if weak:
        return m_next.confirmation_idx if minus_ok else None
    # strong also needs +DI(M): next extreme is a lower high (P5 < P3).
    plus_ok = (
        p_next is not None and p_next.kind == "H" and p_next.value < plus_p3
    )
    if plus_ok and minus_ok:
        return max(p_next.confirmation_idx, m_next.confirmation_idx)
    return None


def generate_signals(
    df: pd.DataFrame,
    atr_series: pd.Series,
    events: list[Event],
    plus_ext: list[Extreme],
    minus_ext: list[Extreme],
    variant: str,
    sig_cfg: SignalConfig,
    exit_cfg: ExitConfig,
    stats_cfg: StatsConfig,
    stats: dict,
    costs: CostConfig,
    transition_stats: dict | None = None,
    similarity_expectations: dict[tuple[str, int], dict] | None = None,
    similarity_ev_cfg: SimilarityEvConfig | None = None,
) -> list[Signal]:
    n = len(df)
    close = df["close"]
    ma = (
        close.rolling(sig_cfg.ma_filter_len).mean()
        if sig_cfg.ma_filter_len > 0
        else None
    )
    out: list[Signal] = []
    for ev in events:
        if variant == "p5":
            sidx = p5_confirmation_index(ev, plus_ext, minus_ext)
            if sidx is None:
                continue
        else:
            sidx = ev.event_idx
        if sidx >= n - 1:  # no next bar to enter on
            continue

        s = Signal(
            signal_idx=sidx,
            entry_variant=variant,
            direction=ev.direction,
            event=ev,
            confidence=ev.pressure_score,
        )
        directional_line = "plus" if ev.direction == "long" else "minus"
        directional_p4 = ev.plus_p[-1] if ev.direction == "long" else ev.minus_p[-1]
        similarity = (
            (similarity_expectations or {}).get(
                (directional_line, directional_p4.confirmation_idx)
            )
        )
        if similarity is not None:
            s.similarity_expected_return = float(
                similarity["expected_net_return"]
            )
            s.similarity_ev_lower_bound = float(
                similarity["ev_lower_bound"]
            )
            s.similarity_effective_n = float(similarity["effective_n"])
            s.similarity_confidence = float(similarity["confidence"])
            s.similarity_fallback = str(similarity["model_fallback"])
        # plan 6/7: continuation probability for the directional (W) line.
        if transition_stats is not None and sig_cfg.use_transition:
            window = ev.plus_p if ev.direction == "long" else ev.minus_p
            _, feats = _geometry(window)
            tr = lookup_transition(transition_stats, "W", feats, ev.setup_shape)
            if tr is not None:
                s.continuation_score = tr.get("p_continuation", float("nan"))
                s.transition_bucket = tr.get("bucket", "")

        st = lookup_stats(stats, ev.direction, ev.pressure_aligned, stats_cfg)
        if st is not None:
            raw, adjusted = expected_values(
                st, ev.pressure_rr_factor, costs, s.continuation_score
            )
            s.raw_expected_value = raw
            s.pressure_adjusted_expected_value = adjusted
            s.stats_bucket = st["bucket"]

        entry_candidate = float(close.iloc[sidx])
        stop_c, tp_c = stop_take_profit(
            ev.direction, entry_candidate, sidx, df, atr_series, exit_cfg,
            ev.pressure_rr_factor, s.continuation_score,
        )
        s.entry_price_candidate = entry_candidate
        s.stop_price_candidate = stop_c
        s.take_profit_candidate = tp_c if tp_c is not None else float("nan")

        atr_pct = float(atr_series.iloc[sidx]) / entry_candidate if entry_candidate else float("nan")
        reason = ""
        if not np.isfinite(stop_c):
            reason = "invalid_stop"
        elif ev.pressure_score < sig_cfg.pressure_score_min:
            reason = "pressure_score"
        elif ma is not None and not (
            (ev.direction == "long" and entry_candidate > float(ma.iloc[sidx]))
            or (ev.direction == "short" and entry_candidate < float(ma.iloc[sidx]))
        ):
            reason = "ma_filter"
        elif not (sig_cfg.atr_pct_min <= atr_pct <= sig_cfg.atr_pct_max):
            reason = "volatility"
        elif (
            similarity_ev_cfg is not None
            and similarity_ev_cfg.enabled
            and (
                similarity is None
                or s.similarity_effective_n < similarity_ev_cfg.min_neighbors
                or s.similarity_ev_lower_bound
                <= similarity_ev_cfg.entry_margin
            )
        ):
            reason = "similarity_expected_value"
        elif (
            not (similarity_ev_cfg and similarity_ev_cfg.enabled)
            and sig_cfg.require_positive_ev
            and not (
            s.pressure_adjusted_expected_value > 0
            )
        ):
            reason = "expected_value"

        if reason:
            s.signal = "hold"
            s.filter_reason = reason
        else:
            s.signal = ev.direction
        out.append(s)

    out.sort(key=lambda s: s.signal_idx)
    return out
