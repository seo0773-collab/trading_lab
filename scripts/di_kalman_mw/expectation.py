"""Similarity-weighted price expectation for P1..P4 entry candidates."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CostConfig, SimilarityEvConfig
from .similarity import (
    PatternSimilarityModel,
    SimilarityConfig,
    _weighted_quantile,
)

PRICE_TARGET = "directional_return_to_p5"


class PriceExpectationModel:
    """Estimate net entry value from realized paths of similar P1..P4 cases."""

    def __init__(self, config: SimilarityEvConfig | None = None):
        self.config = config or SimilarityEvConfig()
        self._model = PatternSimilarityModel(
            SimilarityConfig(
                neighbors=self.config.neighbors,
                temperature=self.config.temperature,
                min_neighbors=self.config.min_neighbors,
            )
        )

    def fit(self, train: pd.DataFrame) -> "PriceExpectationModel":
        usable = train.dropna(subset=[PRICE_TARGET]).copy()
        self._model.fit(usable)
        return self

    def predict_one(
        self, row: pd.Series, costs: CostConfig
    ) -> dict:
        neighbors, weights, metadata = self._model.neighbor_sample(row)
        gross = neighbors[PRICE_TARGET].to_numpy(dtype=float)
        net = gross - costs.round_trip_cost
        q10, q25, median, q75, q90 = _weighted_quantile(
            net, weights, (0.10, 0.25, 0.50, 0.75, 0.90)
        )
        mean = float(np.dot(weights, net))
        win_probability = float(np.dot(weights, net > 0))
        lower = q25 if self.config.lower_quantile == 0.25 else q10
        return {
            "expected_net_return": mean,
            "ev_lower_bound": lower,
            "net_q10": q10,
            "net_q25": q25,
            "net_median": median,
            "net_q75": q75,
            "net_q90": q90,
            "win_probability": win_probability,
            "entry_eligible": bool(
                lower > self.config.entry_margin
                and metadata["effective_n"] >= self.config.min_neighbors
            ),
            **metadata,
        }


def evaluate_price_expectation(
    model: PriceExpectationModel,
    evaluation: pd.DataFrame,
    costs: CostConfig,
) -> dict:
    predictions: list[dict] = []
    realized: list[float] = []
    for _, row in evaluation.dropna(subset=[PRICE_TARGET]).iterrows():
        try:
            prediction = model.predict_one(row, costs)
        except ValueError:
            continue
        predictions.append(prediction)
        realized.append(float(row[PRICE_TARGET]) - costs.round_trip_cost)
    if not predictions:
        return {"n_evaluated": 0}
    predicted = np.array(
        [p["expected_net_return"] for p in predictions], dtype=float
    )
    actual = np.array(realized, dtype=float)
    eligible = np.array(
        [p["entry_eligible"] for p in predictions], dtype=bool
    )
    return {
        "n_evaluated": len(predictions),
        "mae": float(np.mean(np.abs(predicted - actual))),
        "mean_predicted_ev": float(predicted.mean()),
        "mean_realized_net_return": float(actual.mean()),
        "eligible_count": int(eligible.sum()),
        "eligible_realized_net_return": (
            float(actual[eligible].mean()) if eligible.any() else None
        ),
        "eligible_win_rate": (
            float((actual[eligible] > 0).mean()) if eligible.any() else None
        ),
        "mean_effective_n": float(np.mean(
            [p["effective_n"] for p in predictions]
        )),
        "mean_confidence": float(np.mean(
            [p["confidence"] for p in predictions]
        )),
    }


def predict_price_frame(
    model: PriceExpectationModel,
    frame: pd.DataFrame,
    costs: CostConfig,
) -> pd.DataFrame:
    rows: list[dict] = []
    for _, row in frame.iterrows():
        try:
            prediction = model.predict_one(row, costs)
        except ValueError:
            continue
        rows.append({
            "instance_id": int(row["instance_id"]),
            "decision_time": row["decision_time"],
            "decision_idx": int(row["decision_idx"]),
            "decision_split": row["decision_split"],
            "line": row["line"],
            "pattern": row["pattern"],
            "predicted_direction": row["predicted_direction"],
            **prediction,
        })
    return pd.DataFrame(rows)


def expectation_lookup(frame: pd.DataFrame) -> dict[tuple[str, int], dict]:
    if frame.empty:
        return {}
    return {
        (str(row.line), int(row.decision_idx)): row._asdict()
        for row in frame.itertuples(index=False)
    }
