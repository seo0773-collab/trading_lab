"""4단계 국면 판정 (profile_plan.txt §6).

NORMAL / CAUTION / DEFENSE / RECOVERY 를 상태기계로 산출한다. 각 봉은 과거·현재
값만 본다(무누수). RECOVERY는 DEFENSE 이후의 전이 상태이므로 직전 상태를 기억해야
하며, 그래서 순차 루프로 구현한다(일봉 수천 개라 비용 무시 가능).

판정 신호:
- below_base : cm_close < 1.0 (가격이 base_cycle 아래)
- dist_down  : rolling_mid_50 < cumulative_mid_50 (최근 분포 중심이 장기 아래로 이동)
- recovered  : cm_close > rolling_mid_50 (최근 분포 중심 회복)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NORMAL, CAUTION, DEFENSE, RECOVERY = "NORMAL", "CAUTION", "DEFENSE", "RECOVERY"
REGIME_CODE = {NORMAL: 0, CAUTION: 1, RECOVERY: 2, DEFENSE: 3}


def classify(cycle: pd.DataFrame, profile: pd.DataFrame) -> pd.DataFrame:
    cm_close = np.asarray(cycle["cm_close"], dtype=float)
    rmid = np.asarray(profile["rolling_mid_50"], dtype=float)
    cmid = np.asarray(profile["cumulative_mid_50"], dtype=float)

    n = len(cm_close)
    regimes = np.empty(n, dtype=object)
    recovery_bars = np.zeros(n, dtype=int)

    state = NORMAL
    rec_count = 0
    for i in range(n):
        c, rm, cm = cm_close[i], rmid[i], cmid[i]
        if not (np.isfinite(c) and np.isfinite(rm) and np.isfinite(cm)):
            # warmup/무효 구간: 중립으로 두고 비중 0이 되게 한다.
            regimes[i] = NORMAL
            rec_count = 0
            continue
        below_base = c < 1.0
        dist_down = rm < cm
        recovered = c > rm

        if state == DEFENSE:
            if recovered and c > 1.0 and not dist_down:
                state = NORMAL
            elif recovered:
                state, rec_count = RECOVERY, 0
        elif state == RECOVERY:
            if dist_down and below_base:
                state = DEFENSE
            elif c > 1.0 and not dist_down:
                state = NORMAL
            else:
                rec_count += 1
        else:  # NORMAL / CAUTION
            if dist_down and below_base:
                state = DEFENSE
            elif below_base:
                state = CAUTION
            else:
                state = NORMAL

        regimes[i] = state
        recovery_bars[i] = rec_count if state == RECOVERY else 0

    out = pd.DataFrame(index=cycle.index)
    out["regime"] = regimes
    out["regime_code"] = [REGIME_CODE[r] for r in regimes]
    out["recovery_bars"] = recovery_bars
    return out
