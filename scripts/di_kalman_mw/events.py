"""M/W pattern classification and setup event generation (plan 1, 5, 5A, 6)."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby

import numpy as np

from .config import PatternConfig
from .extremes import Extreme

EPS = 1e-9


def classify_pattern(window: tuple[Extreme, ...], strict: bool) -> str | None:
    """plan 5A.2: P1..P4 ordering decides W / M / None."""
    kinds = tuple(e.kind for e in window)
    v = [e.value for e in window]
    if kinds == ("L", "H", "L", "H") and v[2] > v[0] and (not strict or v[3] > v[1]):
        return "W"
    if kinds == ("H", "L", "H", "L") and v[2] < v[0] and (not strict or v[3] < v[1]):
        return "M"
    return None


def mw_shape(
    window: tuple[Extreme, ...], parallel_band: float
) -> tuple[str, float]:
    """Right/left channel-width ratio -> diverging / parallel / converging.

    The shape label is *independent* of ``classify_pattern``'s gate (which
    already fixes the P3 vs P1 direction): it only measures whether the
    high-low channel widens, holds, or narrows toward the right (P4) side.
    Assumes a valid W (L,H,L,H) or M (H,L,H,L) window. Returns the label and
    the raw right/left ratio (NaN when the left width is degenerate).
    """
    v = [e.value for e in window]
    if window[0].kind == "L":  # W: L H L H, widths are H - L
        left = v[1] - v[0]
        right = v[3] - v[2]
    else:                       # M: H L H L, widths are H - L
        left = v[0] - v[1]
        right = v[2] - v[3]
    if left <= EPS:
        return "parallel", float("nan")
    ratio = right / left
    if ratio >= 1.0 + parallel_band:
        return "diverging", ratio
    if ratio <= 1.0 - parallel_band:
        return "converging", ratio
    return "parallel", ratio


@dataclass(frozen=True)
class Event:
    event_idx: int  # confirmation bar where the combined setup became true
    direction: str  # "long" | "short"
    plus_j: int  # list index of +DI P4 within the plus extremes list
    minus_j: int
    plus_p: tuple[Extreme, ...]  # P1..P4
    minus_p: tuple[Extreme, ...]
    plus_pattern: str  # "W" | "M" | "" (empty when the line is unconfirmed)
    minus_pattern: str
    plus_extreme_mean_4: float
    minus_extreme_mean_4: float
    di_pressure_spread: float
    long_pressure_score: float
    short_pressure_score: float
    long_rr_factor: float
    short_rr_factor: float
    pressure_score: float
    pressure_rr_factor: float
    pressure_aligned: bool
    # plan 17 shape: diverging / parallel / converging per line and for the
    # directional (W-forming) line that drives the setup.
    plus_shape: str = ""
    minus_shape: str = ""
    plus_width_ratio: float = float("nan")
    minus_width_ratio: float = float("nan")
    setup_shape: str = ""
    setup_width_ratio: float = float("nan")
    # plan 6/8: "strong" = both lines confirmed (W&M); "weak" = only the
    # directional line confirmed, opposite line supplies pressure only.
    tier: str = "strong"


def _pressure_block(
    plus_p: tuple[Extreme, ...], minus_p: tuple[Extreme, ...], direction: str
) -> dict:
    """plan 6: pressure features from the latest 4 extremes of each DI line."""
    pm = float(np.mean([e.value for e in plus_p]))
    mm = float(np.mean([e.value for e in minus_p]))
    denom = max(pm + mm, EPS)
    long_ps = pm / denom
    short_ps = mm / denom
    long_rr = float(np.clip(pm / max(mm, EPS), 0.50, 2.00))
    short_rr = float(np.clip(mm / max(pm, EPS), 0.50, 2.00))
    if direction == "long":
        score, prf = long_ps, long_rr
    else:
        score, prf = short_ps, short_rr
    return {
        "plus_extreme_mean_4": pm,
        "minus_extreme_mean_4": mm,
        "di_pressure_spread": (pm - mm) / denom,
        "long_pressure_score": long_ps,
        "short_pressure_score": short_ps,
        "long_rr_factor": long_rr,
        "short_rr_factor": short_rr,
        "pressure_score": score,
        "pressure_rr_factor": prf,
        "pressure_aligned": score > 0.5,
    }


def build_events(
    plus_ext: list[Extreme],
    minus_ext: list[Extreme],
    cfg: PatternConfig,
) -> list[Event]:
    """Replay extreme confirmations chronologically and emit setup events.

    A pattern instance stays active until the next extreme of the same DI
    line is confirmed (the 4-extreme window shifts). A new event is emitted
    each time a new (plus pattern, minus pattern) pair forms a setup.

    Two tiers are produced (plan 6/8):
      - "strong": both lines confirm the opposing W/M pair.
      - "weak":   only the directional (W-forming) line confirms; the
        opposite line is not yet a confirming pattern but pressure on the
        directional side is >= weak_pressure_min. Enabled by
        cfg.allow_weak_setup.
    Both tiers act only on confirmed extremes (confirmation_idx), so no
    future information leaks into the decision bar (plan 16).
    """
    exts = {"plus": plus_ext, "minus": minus_ext}
    timeline = [
        (e.confirmation_idx, side, j)
        for side in ("plus", "minus")
        for j, e in enumerate(exts[side])
    ]
    timeline.sort()

    current: dict[str, tuple[int, str | None]] = {
        "plus": (-1, None),
        "minus": (-1, None),
    }
    emitted: set[tuple[int, int, str]] = set()
    events: list[Event] = []

    for conf_idx, group in groupby(timeline, key=lambda t: t[0]):
        for _, side, j in group:
            if j >= 3:
                window = tuple(exts[side][j - 3:j + 1])
                current[side] = (j, classify_pattern(window, cfg.strict))
            else:
                current[side] = (j, None)

        pj, pp = current["plus"]
        mj, mp = current["minus"]
        # Both lines need 4 confirmed extremes for the pressure window.
        if pj < 3 or mj < 3:
            continue
        plus_p = tuple(plus_ext[pj - 3:pj + 1])
        minus_p = tuple(minus_ext[mj - 3:mj + 1])
        pm = float(np.mean([e.value for e in plus_p]))
        mm = float(np.mean([e.value for e in minus_p]))
        denom = max(pm + mm, EPS)
        long_ps = pm / denom
        short_ps = mm / denom

        direction: str | None = None
        tier = "strong"
        if pp == "W" and mp == "M":
            direction = "long"
        elif pp == "M" and mp == "W":
            direction = "short"
        elif cfg.allow_weak_setup:
            # weak: directional line is W, opposite line is not yet its M.
            if pp == "W" and mp != "M" and long_ps >= cfg.weak_pressure_min:
                direction, tier = "long", "weak"
            elif mp == "W" and pp != "M" and short_ps >= cfg.weak_pressure_min:
                direction, tier = "short", "weak"
        if direction is None:
            continue
        key = (pj, mj, direction)
        if key in emitted:
            continue
        emitted.add(key)

        plus_shape, plus_wr = (
            mw_shape(plus_p, cfg.parallel_band)
            if pp in ("W", "M") else ("", float("nan"))
        )
        minus_shape, minus_wr = (
            mw_shape(minus_p, cfg.parallel_band)
            if mp in ("W", "M") else ("", float("nan"))
        )
        if direction == "long":
            setup_shape, setup_wr = plus_shape, plus_wr
        else:
            setup_shape, setup_wr = minus_shape, minus_wr

        events.append(
            Event(
                event_idx=conf_idx,
                direction=direction,
                plus_j=pj,
                minus_j=mj,
                plus_p=plus_p,
                minus_p=minus_p,
                plus_pattern=pp or "",
                minus_pattern=mp or "",
                plus_shape=plus_shape,
                minus_shape=minus_shape,
                plus_width_ratio=plus_wr,
                minus_width_ratio=minus_wr,
                setup_shape=setup_shape,
                setup_width_ratio=setup_wr,
                tier=tier,
                **_pressure_block(plus_p, minus_p, direction),
            )
        )
    return events
