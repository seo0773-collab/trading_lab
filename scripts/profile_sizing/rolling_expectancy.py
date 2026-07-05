"""Rolling smooth profile location features and expectancy sizing.

This module turns the causal rolling multiple-profile histogram into a smooth
probability distribution P_t(m), maps the current multiple candle onto that
distribution, then estimates bucketed forward-return expectancy using only
labels that would have been known at each decision bar. Smoothing uses a local
kernel only; it does not fit a symmetric Gaussian distribution, so skewed and
multi-modal profile shapes are preserved.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ProfileSizingConfig
from .engine import (
    buy_hold_equity, equity_from_returns, lot_trades, performance,
    portfolio_returns,
)
from .indicators import compute_cycle
from .profile import contribution_matrix, rolling_profile_matrix


def _gaussian_kernel(sigma: float) -> np.ndarray:
    sigma = float(sigma)
    if sigma <= 0.0:
        return np.array([1.0])
    radius = max(1, int(np.ceil(4.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


def smooth_profile_prob(profile: np.ndarray, sigma_bins: float) -> np.ndarray:
    """Smooth histograms and normalize rows, without imposing symmetry."""
    if profile.ndim != 2:
        raise ValueError("profile must be a 2D bars x bins matrix")
    kernel = _gaussian_kernel(sigma_bins)
    smooth = np.zeros_like(profile, dtype=float)
    for i, row in enumerate(np.asarray(profile, dtype=float)):
        if np.nansum(row) <= 0.0:
            continue
        smooth[i] = np.convolve(np.nan_to_num(row, nan=0.0), kernel, mode="same")
    smooth = np.maximum(smooth, 0.0)
    total = smooth.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        prob = smooth / total[:, None]
    prob[~np.isfinite(prob)] = 0.0
    return prob


def _bin_index(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    bins = len(edges) - 1
    idx = np.searchsorted(edges, values, side="right") - 1
    out = ~np.isfinite(values) | (values < edges[0]) | (values > edges[-1])
    idx = np.clip(idx, 0, bins - 1)
    idx[out] = -1
    return idx


def _cdf_at(prob: np.ndarray, edges: np.ndarray, values: np.ndarray) -> np.ndarray:
    bars, bins = prob.shape
    idx = _bin_index(np.asarray(values, dtype=float), edges)
    cum = np.cumsum(prob, axis=1)
    out = np.full(bars, np.nan)
    rows = np.arange(bars)
    valid = idx >= 0
    if not np.any(valid):
        return out
    below = np.where(idx > 0, cum[rows, np.maximum(idx - 1, 0)], 0.0)
    width = np.maximum(edges[idx + 1] - edges[idx], 1e-12)
    frac = np.clip((values - edges[idx]) / width, 0.0, 1.0)
    val = below + prob[rows, idx] * frac
    row_has_mass = prob.sum(axis=1) > 0.0
    valid &= row_has_mass
    out[valid] = np.clip(val[valid], 0.0, 1.0)
    return out


def _density_at(prob: np.ndarray, edges: np.ndarray, values: np.ndarray) -> np.ndarray:
    idx = _bin_index(np.asarray(values, dtype=float), edges)
    rows = np.arange(prob.shape[0])
    out = np.full(prob.shape[0], np.nan)
    valid = idx >= 0
    out[valid] = prob[rows[valid], idx[valid]]
    return out


def _quantile(prob: np.ndarray, edges: np.ndarray, q: float) -> np.ndarray:
    bars, bins = prob.shape
    cum = np.cumsum(prob, axis=1)
    out = np.full(bars, np.nan)
    for i in range(bars):
        if prob[i].sum() <= 0.0:
            continue
        j = int(np.searchsorted(cum[i], q, side="left"))
        j = min(max(j, 0), bins - 1)
        prev = cum[i, j - 1] if j > 0 else 0.0
        mass = max(prob[i, j], 1e-12)
        frac = np.clip((q - prev) / mass, 0.0, 1.0)
        out[i] = edges[j] + frac * (edges[j + 1] - edges[j])
    return out


def _js_distance(prob: np.ndarray) -> np.ndarray:
    out = np.full(prob.shape[0], np.nan)
    eps = 1e-12
    for i in range(1, prob.shape[0]):
        p = prob[i - 1]
        q = prob[i]
        if p.sum() <= 0.0 or q.sum() <= 0.0:
            continue
        m = 0.5 * (p + q)
        kl_pm = np.sum(np.where(p > 0.0, p * np.log((p + eps) / (m + eps)), 0.0))
        kl_qm = np.sum(np.where(q > 0.0, q * np.log((q + eps) / (m + eps)), 0.0))
        out[i] = float(np.sqrt(max(0.0, 0.5 * (kl_pm + kl_qm))))
    return out


def _profile_turnover(contrib: np.ndarray, roll: np.ndarray, window: int) -> np.ndarray:
    n = len(contrib)
    out = np.full(n, np.nan)
    for i in range(n):
        outgoing = contrib[i - window] if i >= window else 0.0
        delta = contrib[i] - outgoing
        total = roll[i].sum()
        if total > 0.0:
            out[i] = np.abs(delta).sum() / total
    return out


def compute_rolling_profile_features(
    cycle: pd.DataFrame, raw: pd.DataFrame, cfg: ProfileSizingConfig
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return candle-location features and the smoothed rolling probability rows."""
    contrib, edges = contribution_matrix(cycle, raw, cfg.profile)
    roll = rolling_profile_matrix(contrib, cfg.profile.rolling_window)
    prob = smooth_profile_prob(roll, cfg.smoothing.sigma_bins)

    cm_open = np.asarray(cycle["cm_open"], dtype=float)
    cm_high = np.asarray(cycle["cm_high"], dtype=float)
    cm_low = np.asarray(cycle["cm_low"], dtype=float)
    cm_close = np.asarray(cycle["cm_close"], dtype=float)

    out = pd.DataFrame(index=cycle.index)
    out["open_percentile"] = _cdf_at(prob, edges, cm_open)
    out["high_percentile"] = _cdf_at(prob, edges, cm_high)
    out["low_percentile"] = _cdf_at(prob, edges, cm_low)
    out["close_percentile"] = _cdf_at(prob, edges, cm_close)
    out["rolling_percentile"] = out["close_percentile"]
    out["profile_score"] = 2.0 * (0.5 - out["close_percentile"])
    out["candle_mid_percentile"] = (
        out["high_percentile"] + out["low_percentile"]
    ) / 2.0
    out["body_mid_percentile"] = (
        out["open_percentile"] + out["close_percentile"]
    ) / 2.0
    out["range_percentile_width"] = (
        out["high_percentile"] - out["low_percentile"]
    ).clip(lower=0.0)
    denom = np.maximum(cm_high - cm_low, 1e-12)
    close_in_candle = (cm_close - cm_low) / denom
    out["close_in_candle_percentile"] = np.clip(close_in_candle, 0.0, 1.0)
    out["density_at_price"] = _density_at(prob, edges, cm_close)

    centers = (edges[:-1] + edges[1:]) / 2.0
    total = prob.sum(axis=1)
    poc_idx = prob.argmax(axis=1)
    poc = np.where(total > 0.0, centers[poc_idx], np.nan)
    out["poc_mult"] = poc
    out["poc_distance"] = cm_close - poc
    pv = cfg.profile.percentile_value / 100.0
    out["lower_band"] = _quantile(prob, edges, pv)
    out["upper_band"] = _quantile(prob, edges, 1.0 - pv)
    out["profile_width"] = out["upper_band"] - out["lower_band"]

    mid = cfg.profile.bin_count // 2
    lower_mass = prob[:, :mid].sum(axis=1)
    upper_mass = prob[:, mid:].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        skew = upper_mass / lower_mass
    out["profile_skew"] = np.where(np.isfinite(skew), skew, np.nan)
    out["profile_turnover"] = _profile_turnover(
        contrib, roll, cfg.profile.rolling_window
    )
    out["profile_jsd"] = _js_distance(prob)
    return out, prob


def forward_path_stats(
    close: pd.Series, horizon: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return forward return, MFE, and MAE arrays for a fixed horizon."""
    px = np.asarray(close, dtype=float)
    n = len(px)
    fwd = np.full(n, np.nan)
    mfe = np.full(n, np.nan)
    mae = np.full(n, np.nan)
    horizon = max(1, int(horizon))
    for i in range(0, n - horizon):
        path = px[i + 1:i + horizon + 1] / px[i] - 1.0
        fwd[i] = px[i + horizon] / px[i] - 1.0
        mfe[i] = np.nanmax(path)
        mae[i] = np.nanmin(path)
    return fwd, mfe, mae


def _fit_bucket_stats(
    score: np.ndarray,
    fwd: np.ndarray,
    mfe: np.ndarray,
    mae: np.ndarray,
    bins: np.ndarray,
    cost: float,
) -> pd.DataFrame:
    rows = []
    bucket = np.digitize(score, bins, right=False) - 1
    bucket = np.clip(bucket, 0, len(bins) - 2)
    for b in range(len(bins) - 1):
        mask = (bucket == b) & np.isfinite(score) & np.isfinite(fwd)
        vals = fwd[mask] - cost
        wins = vals[vals > 0.0]
        losses = vals[vals <= 0.0]
        n = int(vals.size)
        win_rate = float(wins.size / n) if n else np.nan
        avg_win = float(wins.mean()) if wins.size else 0.0
        avg_loss = float(losses.mean()) if losses.size else 0.0
        expectancy = (
            win_rate * avg_win - (1.0 - win_rate) * abs(avg_loss)
            if n else np.nan
        )
        rows.append({
            "bucket": b,
            "score_lo": float(bins[b]),
            "score_hi": float(bins[b + 1]),
            "n": n,
            "mean_future_return": float(vals.mean()) if n else np.nan,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "expectancy": expectancy,
            "expectancy_bps": expectancy * 10_000.0 if n else np.nan,
            "mfe": float(np.nanmean(mfe[mask])) if n else np.nan,
            "mae": float(np.nanmean(mae[mask])) if n else np.nan,
        })
    return pd.DataFrame(rows)


def walk_forward_expectancy(
    features: pd.DataFrame, close: pd.Series, cfg: ProfileSizingConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate score-bucket expectancy with causal walk-forward refits."""
    ex = cfg.expectancy
    horizon = int(ex.primary_horizon)
    score = features["profile_score"].to_numpy(dtype=float)
    fwd, mfe, mae = forward_path_stats(close, horizon)
    bins = np.asarray(ex.score_bins, dtype=float)
    cost = float(ex.cost_bps_roundtrip) / 10_000.0
    n = len(features)

    out = pd.DataFrame(index=features.index)
    out["expectancy_bps"] = np.nan
    out["expectancy_samples"] = 0
    out["expectancy_bucket"] = np.nan

    table_rows = []
    current_table: pd.DataFrame | None = None
    refit_every = max(1, int(ex.refit_every))
    for t in range(n):
        train_end = t - horizon
        train_start = max(0, train_end - int(ex.train_window))
        if train_end <= train_start:
            continue
        if current_table is None or t % refit_every == 0:
            current_table = _fit_bucket_stats(
                score[train_start:train_end],
                fwd[train_start:train_end],
                mfe[train_start:train_end],
                mae[train_start:train_end],
                bins,
                cost,
            )
            snap = current_table.copy()
            snap.insert(0, "bar", t)
            table_rows.append(snap)
        if not np.isfinite(score[t]):
            continue
        b = int(np.digitize([score[t]], bins, right=False)[0] - 1)
        b = int(np.clip(b, 0, len(bins) - 2))
        row = current_table[current_table["bucket"] == b]
        if row.empty:
            continue
        samples = int(row["n"].iloc[0])
        out.iat[t, out.columns.get_loc("expectancy_bucket")] = b
        out.iat[t, out.columns.get_loc("expectancy_samples")] = samples
        if samples >= int(ex.min_samples):
            out.iat[t, out.columns.get_loc("expectancy_bps")] = float(
                row["expectancy_bps"].iloc[0]
            )

    tables = pd.concat(table_rows, ignore_index=True) if table_rows else pd.DataFrame()
    return out, tables


def expectancy_weights(
    expectancy_bps: pd.Series, cfg: ProfileSizingConfig
) -> pd.DataFrame:
    """Map causal expectancy estimates to gradually rebalanced long-only weights."""
    sz = cfg.expectancy_sizing
    exp = np.asarray(expectancy_bps, dtype=float)
    base = np.full(len(exp), float(sz.neutral_weight))
    valid = np.isfinite(exp)
    pos = valid & (exp > float(sz.min_positive_expectancy_bps))
    if float(sz.expectancy_scale_bps) > 0.0:
        base[pos] = exp[pos] / float(sz.expectancy_scale_bps)
    if sz.long_only:
        base = np.maximum(base, 0.0)
    base = np.clip(base, 0.0, float(sz.max_weight))

    actual = np.zeros(len(base), dtype=float)
    prev = 0.0
    for i, target in enumerate(base):
        diff = target - prev
        if abs(diff) < float(sz.rebalance_threshold):
            new = prev
        else:
            step = float(np.clip(
                diff,
                -float(sz.max_trade_weight_per_bar),
                float(sz.max_trade_weight_per_bar),
            ))
            new = prev + step
        actual[i] = float(np.clip(new, 0.0, float(sz.max_weight)))
        prev = actual[i]

    out = pd.DataFrame(index=expectancy_bps.index)
    out["target_weight"] = base
    out["actual_weight"] = actual
    return out


def run_rolling_expectancy_pipeline(
    raw: pd.DataFrame, cfg: ProfileSizingConfig
) -> dict:
    """End-to-end research pipeline for rolling-profile expectancy sizing."""
    cycle = compute_cycle(raw, cfg.base_cycle)
    features, smooth_prob = compute_rolling_profile_features(cycle, raw, cfg)
    close = pd.Series(np.asarray(raw["close"], dtype=float), index=raw.index)
    exp, exp_tables = walk_forward_expectancy(features, close, cfg)
    weights = expectancy_weights(exp["expectancy_bps"], cfg)

    port_ret = portfolio_returns(close, weights["actual_weight"], cfg)
    equity = equity_from_returns(port_ret)
    bnh = buy_hold_equity(close)
    regime = pd.Series(["EXPECTANCY"] * len(raw), index=raw.index)
    trades = lot_trades(close, weights["actual_weight"], regime, cfg)

    forecast = cycle.join(features).join(exp).join(weights)
    if "volume" in raw:
        forecast["volume"] = np.asarray(raw["volume"], dtype=float)
    return {
        "forecast": forecast,
        "smooth_profile_prob": smooth_prob,
        "expectancy_tables": exp_tables,
        "port_ret": port_ret,
        "equity": equity,
        "buy_hold": bnh,
        "trades": trades,
        "metrics": performance(equity, port_ret, cfg.interval),
    }
