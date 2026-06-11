"""Weighted Gaussian approximation for a profile histogram."""

import numpy as np
import pandas as pd


def fit_gaussian_profile(profile: pd.DataFrame) -> dict[str, float]:
    x = profile["mult"].to_numpy(dtype=float)
    weights = profile["value"].fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
    total = float(weights.sum())
    if total <= 0:
        return {"mu": float("nan"), "sigma": float("nan"), "total": 0.0}

    mu = float(np.sum(x * weights) / total)
    variance = float(np.sum(weights * (x - mu) ** 2) / total)
    return {"mu": mu, "sigma": max(float(np.sqrt(variance)), 1e-9), "total": total}


def add_gaussian_expectation(
    profile: pd.DataFrame, fit: dict[str, float] | None = None
) -> pd.DataFrame:
    fit = fit or fit_gaussian_profile(profile)
    out = profile.copy()
    mu, sigma, total = fit["mu"], fit["sigma"], fit["total"]

    if not np.isfinite(mu) or not np.isfinite(sigma) or total <= 0:
        out["expected"] = 0.0
    else:
        x = out["mult"].to_numpy(dtype=float)
        density = np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        density_total = density.sum()
        out["expected"] = density * total / density_total if density_total > 0 else 0.0

    out["gap"] = out["value"] - out["expected"]
    out["deficit"] = (-out["gap"]).clip(lower=0.0)
    return out


gaussian_fit = fit_gaussian_profile
