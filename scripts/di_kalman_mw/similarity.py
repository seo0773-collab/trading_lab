"""Train-only weighted-neighbor predictor for P1..P4 -> P5 outcomes."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .pattern_dataset import NUMERIC_FEATURE_COLUMNS


@dataclass(frozen=True)
class SimilarityConfig:
    neighbors: int = 50
    temperature: float = 1.0
    min_neighbors: int = 10


def _weighted_quantile(
    values: np.ndarray, weights: np.ndarray, quantiles: tuple[float, ...]
) -> list[float]:
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    if cumulative[-1] <= 0:
        return [float("nan")] * len(quantiles)
    cumulative /= cumulative[-1]
    return [
        float(np.interp(q, cumulative, values)) for q in quantiles
    ]


class PatternSimilarityModel:
    """Robust-scaled kernel k-NN with explicit sample-confidence outputs."""

    def __init__(self, config: SimilarityConfig | None = None):
        self.config = config or SimilarityConfig()
        self._train: pd.DataFrame | None = None
        self._center: pd.Series | None = None
        self._scale: pd.Series | None = None

    def fit(self, train: pd.DataFrame) -> "PatternSimilarityModel":
        required = list(NUMERIC_FEATURE_COLUMNS) + [
            "line", "pattern", "p5_dv_norm", "continuation",
        ]
        usable = train.dropna(subset=required).copy()
        if usable.empty:
            raise ValueError("no complete pattern rows available for fitting")
        features = usable.loc[:, NUMERIC_FEATURE_COLUMNS].astype(float)
        center = features.median()
        q75 = features.quantile(0.75)
        q25 = features.quantile(0.25)
        scale = (q75 - q25).where((q75 - q25) > 1e-9, 1.0)
        self._train = usable.reset_index(drop=True)
        self._center = center
        self._scale = scale
        return self

    def _candidate_rows(self, row: pd.Series) -> tuple[pd.DataFrame, str]:
        if self._train is None:
            raise RuntimeError("model is not fitted")
        def without_self(frame: pd.DataFrame) -> pd.DataFrame:
            if "instance_id" not in frame or "instance_id" not in row:
                return frame
            return frame[
                frame["instance_id"] != int(row["instance_id"])
            ]

        exact = without_self(self._train[
            (self._train["line"] == row["line"])
            & (self._train["pattern"] == row["pattern"])
        ])
        if len(exact) >= self.config.min_neighbors:
            return exact, "line_pattern"
        pattern = without_self(
            self._train[self._train["pattern"] == row["pattern"]]
        )
        if len(pattern) >= self.config.min_neighbors:
            return pattern, "pattern"
        return without_self(self._train), "global"

    def neighbor_sample(
        self, row: pd.Series
    ) -> tuple[pd.DataFrame, np.ndarray, dict]:
        if self._center is None or self._scale is None:
            raise RuntimeError("model is not fitted")
        values = row.loc[list(NUMERIC_FEATURE_COLUMNS)].astype(float)
        if not np.isfinite(values.to_numpy()).all():
            raise ValueError("prediction row contains non-finite features")
        candidates, fallback = self._candidate_rows(row)
        if candidates.empty:
            raise ValueError("no neighbors remain after self-exclusion")
        matrix = candidates.loc[:, NUMERIC_FEATURE_COLUMNS].astype(float)
        z_train = (matrix - self._center) / self._scale
        z_query = (values - self._center) / self._scale
        distances = np.sqrt(
            np.mean((z_train.to_numpy() - z_query.to_numpy()) ** 2, axis=1)
        )
        count = min(self.config.neighbors, len(candidates))
        selected = np.argsort(distances)[:count]
        d = distances[selected]
        target = candidates.iloc[selected]
        temperature = max(self.config.temperature, 1e-9)
        weights = np.exp(-(d ** 2) / temperature)
        if weights.sum() <= 0:
            weights = np.ones_like(d)
        weights /= weights.sum()
        effective_n = float(1.0 / np.square(weights).sum())
        nearest = float(d[0])
        sample_factor = min(
            1.0, effective_n / max(self.config.min_neighbors, 1)
        )
        metadata = {
            "effective_n": effective_n,
            "nearest_distance": nearest,
            "confidence": float(sample_factor * np.exp(-nearest)),
            "neighbors_used": int(count),
            "model_fallback": fallback,
        }
        return target, weights, metadata

    def predict_one(self, row: pd.Series) -> dict:
        target, weights, metadata = self.neighbor_sample(row)
        p5 = target["p5_dv_norm"].to_numpy(dtype=float)
        quantiles = _weighted_quantile(
            p5, weights, (0.10, 0.25, 0.50, 0.75, 0.90)
        )
        continuation = target["continuation"].astype(float).to_numpy()
        return {
            "q10": quantiles[0],
            "q25": quantiles[1],
            "prediction_median": quantiles[2],
            "q75": quantiles[3],
            "q90": quantiles[4],
            "p_continuation": float(np.dot(weights, continuation)),
            **metadata,
        }


def evaluate_similarity(
    model: PatternSimilarityModel, evaluation: pd.DataFrame
) -> dict:
    errors: list[float] = []
    global_errors: list[float] = []
    brier: list[float] = []
    covered = 0
    predictions = []
    train_global = float(model._train["p5_dv_norm"].median())
    for _, row in evaluation.iterrows():
        try:
            prediction = model.predict_one(row)
        except ValueError:
            continue
        actual = float(row["p5_dv_norm"])
        errors.append(abs(actual - prediction["prediction_median"]))
        global_errors.append(abs(actual - train_global))
        covered += int(prediction["q10"] <= actual <= prediction["q90"])
        brier.append(
            (
                prediction["p_continuation"]
                - float(bool(row["continuation"]))
            ) ** 2
        )
        predictions.append(prediction)
    n = len(errors)
    mae = float(np.mean(errors)) if errors else float("nan")
    global_mae = (
        float(np.mean(global_errors)) if global_errors else float("nan")
    )
    return {
        "n_evaluated": n,
        "similarity_mae": mae,
        "global_mae": global_mae,
        "mae_reduction_pct": (
            float((global_mae - mae) / global_mae * 100.0)
            if n and global_mae > 0 else float("nan")
        ),
        "coverage_10_90": float(covered / n) if n else float("nan"),
        "continuation_brier": (
            float(np.mean(brier)) if brier else float("nan")
        ),
        "mean_effective_n": (
            float(np.mean([p["effective_n"] for p in predictions]))
            if predictions else float("nan")
        ),
        "mean_confidence": (
            float(np.mean([p["confidence"] for p in predictions]))
            if predictions else float("nan")
        ),
    }
