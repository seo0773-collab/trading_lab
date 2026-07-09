"""hole_profile 피처: ATR 방향전환 기반 계층적 지그재그(L0→L1→L2) + 익스트림 프로파일.

hole_profile.txt(Pine v6 "Multiscale Market Structure")의 핵심 로직을 그대로 이식한다:

1. ATR 트레일링 방향전환(directional change)으로 원시 피벗열 L0을 만든다
   (``atrLen`` 봉 SMA(TR) × ``atrMult``를 되돌림 폭으로 사용, upMove/downMove
   상태기계 — Pine 코드 라인 344~398과 동일).
2. "피벗의 피벗" 재귀 압축(``f_next_candidate``, 라인 118~239)으로 L0에서 L1을,
   L1에서 L2를 도출한다. 한 번의 신규 L0 이벤트가 L1에 0~2개, 각 L1 포인트가
   다시 L2에 0~2개를 캐스케이드로 밀어 넣을 수 있다(라인 403~471과 동일 순서).
3. 각 레벨에서 완결된 스윙(연속 익스트림 두 점)마다 지속시간(봉수)·가격변동%을
   프로파일 표본으로 기록한다(``f_record_profile``, 라인 88~116).

전략에 필요한 추가 확장(Pine에는 없음): 매 봉마다 "선택 레벨의 마지막 확정
피벗 → 현재가"까지 진행 중인 스윙의 크기(``pct_move``)가, 그 레벨에서 지금까지
완결된 스윙들의 절대 %변동 분포 중 어디에 위치하는지(``profile_rank``, expanding
백분위·미래참조 없음)를 계산한다. 이게 "hole"(분포 희소구간) 진입 여부를
판정하는 신호축이다.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

_LEVEL_INDEX = {"L0": 0, "L1": 1, "L2": 2}


def f_cmp(x: float, y: float, ext_type: int) -> bool:
    """Pine f_cmp: extType==1이면 x>y(고점 비교), -1이면 x<y(저점 비교)."""
    return x > y if ext_type == 1 else x < y


def f_next_candidate(
    src: list[tuple[int, int, float]],
    dst: list[tuple[int, int, float]],
    ext_type: int,
) -> list[tuple[int, int, float]]:
    """Pine f_next_candidate 이식. src/dst 원소=(type, idx, price).

    src에 방금 type=ext_type인 포인트가 추가된 직후 호출된다. dst에 새로
    추가해야 할 포인트 0~2개를 순서대로 반환한다(2개면 [upgrade, candidate]).
    """
    ext_pos = len(src) - 1
    if ext_pos < 4:
        return []

    prev_pos = ext_pos - 2
    new_price = src[ext_pos][2]
    prev_price = src[prev_pos][2]

    ok = f_cmp(prev_price, new_price, ext_type)

    has_prev_next = len(dst) > 0
    prev_next_type = prev_next_idx = prev_next_price = None
    if has_prev_next:
        prev_next_type, prev_next_idx, prev_next_price = dst[-1]
        if prev_next_type != ext_type and not f_cmp(prev_price, prev_next_price, ext_type):
            ok = False

    adj_prev_pos = prev_pos
    prior_pos = ext_pos - 4
    while prior_pos >= 0 and ok:
        prior_price = src[prior_pos][2]
        prior_idx = src[prior_pos][1]
        if f_cmp(prior_price, prev_price, ext_type):
            ok = False
        elif has_prev_next and prior_idx <= prev_next_idx:
            break
        elif prior_price == prev_price:
            adj_prev_pos = prior_pos
        elif f_cmp(prior_price, prev_price, -ext_type):
            break
        prior_pos -= 2

    if not ok:
        return []

    cand_type = ext_type
    cand_idx = src[adj_prev_pos][1]
    cand_price = src[adj_prev_pos][2]

    need_upgrade = has_prev_next and prev_next_type == ext_type
    if not need_upgrade:
        return [(cand_type, cand_idx, cand_price)]

    found_upgrade = False
    up_pos = -1
    up_price = 0.0
    j = ext_pos - 1
    while j >= 0:
        j_idx = src[j][1]
        j_price = src[j][2]
        if j_idx >= cand_idx:
            j -= 2
            continue
        if j_idx <= prev_next_idx:
            break
        if not found_upgrade or not f_cmp(j_price, up_price, ext_type):
            found_upgrade = True
            up_pos = j
            up_price = j_price
        j -= 2

    if not found_upgrade:
        return []
    return [(-ext_type, src[up_pos][1], src[up_pos][2]), (cand_type, cand_idx, cand_price)]


@dataclass
class _LevelState:
    points: list[tuple[int, int, float]] = field(default_factory=list)
    profile_durations: list[float] = field(default_factory=list)
    profile_abs_moves: list[float] = field(default_factory=list)
    profile_signed_moves: list[float] = field(default_factory=list)
    sorted_abs_moves: list[float] = field(default_factory=list)


def _record_profile(level: _LevelState) -> None:
    pts = level.points
    if len(pts) < 2:
        return
    prev_type, prev_idx, prev_price = pts[-2]
    curr_type, curr_idx, curr_price = pts[-1]
    if curr_idx > prev_idx and prev_price != 0:
        dur = float(curr_idx - prev_idx)
        pct_move = (curr_price / prev_price - 1.0) * 100.0
        level.profile_durations.append(dur)
        level.profile_abs_moves.append(abs(pct_move))
        level.profile_signed_moves.append(pct_move)
        bisect.insort(level.sorted_abs_moves, abs(pct_move))


def _cascade(levels: list[_LevelState], level_idx: int, ext_type: int) -> None:
    if level_idx + 1 >= len(levels):
        return
    candidates = f_next_candidate(
        levels[level_idx].points, levels[level_idx + 1].points, ext_type
    )
    for cand in candidates:
        levels[level_idx + 1].points.append(cand)
        _record_profile(levels[level_idx + 1])
        _cascade(levels, level_idx + 1, cand[0])


def compute_hole_profile_state(
    df: pd.DataFrame,
    *,
    atr_len: int = 48,
    atr_mult: float = 1.0,
    level: str = "L1",
    min_profile_samples: int = 30,
    include_samples: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, dict[str, list[float]]]]:
    """L0/L1/L2 계층 지그재그 + 선택 레벨의 진행 중 스윙 백분위를 봉 단위로 산출.

    반환 컬럼: atr, l0_count/l1_count/l2_count(누적 확정 피벗 수 — 진단용),
    pivot_idx/pivot_price/pivot_type(선택 레벨의 마지막 확정 피벗),
    duration_bars(피벗 이후 경과 봉수), pct_move(피벗 대비 현재 종가 변동%,
    부호=진행 방향), direction(±1/0), profile_n(선택 레벨의 완결 스윙 표본수),
    profile_rank(진행 중 |pct_move|의 expanding 백분위 0~1, 표본 부족/미확정시 NaN).

    ``include_samples=True``면 (state_df, samples) 튜플을 반환한다. ``samples``는
    레벨별 완결 스윙 표본 ``{"L0": {"durations": [...], "signed_moves": [...]}, ...}``
    — Pine 원본의 "Time Bin"/"Price % Bin" 프로파일 테이블을 그대로 재현하는 데 쓴다.
    """
    level_idx = _LEVEL_INDEX[level]
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    n = len(df)

    prev_close = np.empty(n, dtype=float)
    prev_close[0] = close[0]
    prev_close[1:] = close[:-1]
    tr = np.maximum(
        high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close))
    )
    atr = pd.Series(tr).rolling(atr_len, min_periods=atr_len).mean().to_numpy() * atr_mult

    levels = [_LevelState(), _LevelState(), _LevelState()]

    up_move = True
    pend_max = np.nan
    pend_min = np.nan
    pend_max_i = 0
    pend_min_i = 0

    l0_count = np.zeros(n, dtype=int)
    l1_count = np.zeros(n, dtype=int)
    l2_count = np.zeros(n, dtype=int)
    pivot_idx = np.full(n, -1, dtype=int)
    pivot_price = np.full(n, np.nan)
    pivot_type = np.zeros(n, dtype=int)
    duration_bars = np.full(n, -1, dtype=int)
    pct_move = np.full(n, np.nan)
    direction = np.zeros(n, dtype=int)
    profile_n = np.zeros(n, dtype=int)
    profile_rank = np.full(n, np.nan)

    chosen = levels[level_idx]

    for i in range(n):
        if not np.isnan(atr[i]):
            if np.isnan(pend_max):
                pend_max = high[i]
                pend_min = low[i]
                pend_max_i = i
                pend_min_i = i

            new_base = False
            base_type = 0
            base_idx = 0
            base_price = 0.0

            if up_move:
                if high[i] > pend_max:
                    pend_max = high[i]
                    pend_max_i = i
                elif low[i] < pend_max - atr[i]:
                    new_base = True
                    base_type, base_idx, base_price = 1, pend_max_i, pend_max
                    up_move = False
                    pend_min, pend_min_i = low[i], i
            else:
                if low[i] < pend_min:
                    pend_min = low[i]
                    pend_min_i = i
                elif high[i] > pend_min + atr[i]:
                    new_base = True
                    base_type, base_idx, base_price = -1, pend_min_i, pend_min
                    up_move = True
                    pend_max, pend_max_i = high[i], i

            if new_base:
                levels[0].points.append((base_type, base_idx, base_price))
                _record_profile(levels[0])
                _cascade(levels, 0, base_type)

        l0_count[i] = len(levels[0].points)
        l1_count[i] = len(levels[1].points)
        l2_count[i] = len(levels[2].points)

        if chosen.points:
            p_type, p_idx, p_price = chosen.points[-1]
            pivot_idx[i] = p_idx
            pivot_price[i] = p_price
            pivot_type[i] = p_type
            dur = i - p_idx
            duration_bars[i] = dur
            n_samples = len(chosen.sorted_abs_moves)
            profile_n[i] = n_samples
            if dur > 0 and p_price != 0:
                move = (close[i] / p_price - 1.0) * 100.0
                pct_move[i] = move
                direction[i] = 1 if move > 0 else (-1 if move < 0 else 0)
                if n_samples >= min_profile_samples:
                    sorted_moves = chosen.sorted_abs_moves
                    lo = bisect.bisect_left(sorted_moves, abs(move))
                    hi = bisect.bisect_right(sorted_moves, abs(move))
                    profile_rank[i] = (lo + hi) / 2.0 / n_samples

    out = pd.DataFrame(index=df.index)
    out["atr"] = atr
    out["l0_count"] = l0_count
    out["l1_count"] = l1_count
    out["l2_count"] = l2_count
    out["pivot_idx"] = pivot_idx
    out["pivot_price"] = pivot_price
    out["pivot_type"] = pivot_type
    out["duration_bars"] = duration_bars
    out["pct_move"] = pct_move
    out["direction"] = direction
    out["profile_n"] = profile_n
    out["profile_rank"] = profile_rank
    if not include_samples:
        return out
    samples = {
        name: {
            "durations": list(st.profile_durations),
            "signed_moves": list(st.profile_signed_moves),
        }
        for name, st in zip(("L0", "L1", "L2"), levels)
    }
    return out, samples
