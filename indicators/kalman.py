import numpy as np
import pandas as pd


def _validate_noise(q: float, r: float) -> None:
    if q < 0:
        raise ValueError("q must be greater than or equal to zero")
    if r <= 0:
        raise ValueError("r must be greater than zero")


def kalman_1d(src: pd.Series, q: float = 0.01, r: float = 1.0) -> pd.Series:
    """1D random-walk Kalman filter for slowly moving noisy series."""
    values = src.astype(float).to_numpy()
    _validate_noise(q, r)
    out = np.full_like(values, np.nan, dtype=float)
    est = np.nan
    p = 1.0

    for i, x in enumerate(values):
        if np.isnan(x):
            out[i] = est
            continue
        if np.isnan(est):
            est = x
        else:
            pp = p + q
            denom = pp + r
            k = pp / denom if denom > 0 else 0.0
            est = est + k * (x - est)
            p = (1.0 - k) * pp
        out[i] = est

    return pd.Series(out, index=src.index, name=f"{src.name}_kalman1d")


def len_to_q(length: float) -> float:
    """Convert SMA/EMA length feeling into a q/r-like Kalman process noise value."""
    if length <= 0:
        raise ValueError("length must be greater than zero")
    alpha = 2.0 / (float(length) + 1.0)
    return (alpha * alpha) / max(1.0 - alpha, 1e-12)


def kalman_cv(src: pd.Series, q: float | None = None, r: float = 1.0, equiv_len: float = 200.0) -> pd.Series:
    """2-state constant-velocity Kalman filter: level + slope."""
    if q is None:
        q = len_to_q(equiv_len)
    _validate_noise(q, r)

    values = src.astype(float).to_numpy()
    out = np.full_like(values, np.nan, dtype=float)

    lvl = np.nan
    vel = 0.0
    p00, p01, p10, p11 = 1.0, 0.0, 0.0, 1.0

    for i, x in enumerate(values):
        if np.isnan(x):
            out[i] = lvl
            continue

        if np.isnan(lvl):
            lvl = x
            vel = 0.0
        else:
            pred_lvl = lvl + vel
            pred_vel = vel

            pp00 = p00 + p01 + p10 + p11 + q
            pp01 = p01 + p11
            pp10 = p10 + p11
            pp11 = p11 + q

            s = pp00 + r
            k0 = pp00 / s if s > 0 else 0.0
            k1 = pp10 / s if s > 0 else 0.0
            innov = x - pred_lvl

            lvl = pred_lvl + k0 * innov
            vel = pred_vel + k1 * innov

            p00 = (1.0 - k0) * pp00
            p01 = (1.0 - k0) * pp01
            p10 = pp10 - k1 * pp00
            p11 = pp11 - k1 * pp01

        out[i] = lvl

    return pd.Series(out, index=src.index, name=f"{src.name}_kalman_cv")
