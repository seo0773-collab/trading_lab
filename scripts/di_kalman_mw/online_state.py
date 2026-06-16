"""Online P5-candidate path snapshots and hold/exit/reverse decisions."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import CostConfig, OnlineConfig
from .pattern_dataset import NUMERIC_FEATURE_COLUMNS
from .similarity import _weighted_quantile

ONLINE_FEATURE_COLUMNS = (
    *NUMERIC_FEATURE_COLUMNS,
    "elapsed_bars",
    "candidate_dv_norm",
    "candidate_slope_norm",
)


def build_online_snapshots(
    df: pd.DataFrame,
    plus_kalman: pd.Series,
    minus_kalman: pd.Series,
    pattern_frame: pd.DataFrame,
    labels: np.ndarray,
) -> pd.DataFrame:
    """Create causal P4->current snapshots with remaining return outcomes."""
    rows: list[dict] = []
    for _, base in pattern_frame.iterrows():
        if not bool(base["has_p5"]):
            continue
        start = int(base["decision_idx"])
        end = int(base["outcome_idx"])
        if (
            start < 0 or end <= start or end >= len(df)
            or labels[start] != labels[end]
        ):
            continue
        series = plus_kalman if base["line"] == "plus" else minus_kalman
        scale = max(float(base["mean_leg"]), 1e-9)
        p4 = float(base["p4_value"])
        direction = 1.0 if base["predicted_direction"] == "long" else -1.0
        outcome_price = float(df["close"].iloc[end])
        previous = p4
        for current in range(start, end):
            entry_idx = current + 1
            if entry_idx >= len(df) or labels[entry_idx] != labels[start]:
                break
            candidate = float(series.iloc[current])
            if not np.isfinite(candidate):
                continue
            entry_price = float(df["open"].iloc[entry_idx])
            remaining = direction * (outcome_price / entry_price - 1.0)
            row = {
                column: base[column] for column in NUMERIC_FEATURE_COLUMNS
            }
            row.update({
                "instance_id": int(base["instance_id"]),
                "timestamp": df.index[current],
                "bar_idx": current,
                "split": str(labels[current]),
                "line": base["line"],
                "pattern": base["pattern"],
                "predicted_direction": base["predicted_direction"],
                "elapsed_bars": float(current - start),
                "candidate_dv_norm": (candidate - p4) / scale,
                "candidate_slope_norm": (candidate - previous) / scale,
                "remaining_directional_return": remaining,
            })
            rows.append(row)
            previous = candidate
    return pd.DataFrame(rows)


@dataclass
class OnlinePrediction:
    expected_net_return: float
    lower_bound: float
    upper_bound: float
    effective_n: float
    confidence: float
    nearest_distance: float


class OnlineStateModel:
    """Weighted neighbors over base pattern and current P5-candidate path."""

    def __init__(self, config: OnlineConfig | None = None):
        self.config = config or OnlineConfig()
        self._train: pd.DataFrame | None = None
        self._center: pd.Series | None = None
        self._scale: pd.Series | None = None
        self._z_train: np.ndarray | None = None
        self._targets: np.ndarray | None = None
        self._groups: dict[tuple[str, str] | tuple[str], np.ndarray] = {}

    def fit(self, train: pd.DataFrame) -> "OnlineStateModel":
        required = list(ONLINE_FEATURE_COLUMNS) + [
            "line", "pattern", "remaining_directional_return",
        ]
        usable = train.dropna(subset=required).copy()
        if usable.empty:
            raise ValueError("no online snapshots available for fitting")
        features = usable.loc[:, ONLINE_FEATURE_COLUMNS].astype(float)
        self._center = features.median()
        iqr = features.quantile(0.75) - features.quantile(0.25)
        self._scale = iqr.where(iqr > 1e-9, 1.0)
        self._train = usable.reset_index(drop=True)
        normalized = (
            self._train.loc[:, ONLINE_FEATURE_COLUMNS].astype(float)
            - self._center
        ) / self._scale
        self._z_train = normalized.to_numpy()
        self._targets = self._train[
            "remaining_directional_return"
        ].to_numpy(dtype=float)
        self._groups = {}
        for key, group in self._train.groupby(["line", "pattern"]):
            self._groups[(str(key[0]), str(key[1]))] = group.index.to_numpy()
        for pattern, group in self._train.groupby("pattern"):
            self._groups[(str(pattern),)] = group.index.to_numpy()
        return self

    def predict_one(
        self, row: pd.Series, costs: CostConfig
    ) -> OnlinePrediction:
        if (
            self._train is None
            or self._center is None
            or self._scale is None
            or self._z_train is None
            or self._targets is None
        ):
            raise RuntimeError("model is not fitted")
        indices = self._groups.get(
            (str(row["line"]), str(row["pattern"])), np.array([], dtype=int)
        )
        if len(indices) < self.config.min_neighbors:
            indices = self._groups.get(
                (str(row["pattern"]),), np.array([], dtype=int)
            )
        if len(indices) < self.config.min_neighbors:
            indices = np.arange(len(self._train))
        query = row.loc[list(ONLINE_FEATURE_COLUMNS)].astype(float)
        if not np.isfinite(query.to_numpy()).all():
            raise ValueError("online query contains non-finite features")
        z_query = ((query - self._center) / self._scale).to_numpy()
        distances = np.sqrt(np.mean(
            (self._z_train[indices] - z_query) ** 2, axis=1
        ))
        count = min(self.config.neighbors, len(indices))
        selected = np.argsort(distances)[:count]
        d = distances[selected]
        weights = np.exp(
            -(d ** 2) / max(self.config.temperature, 1e-9)
        )
        if weights.sum() <= 0:
            weights = np.ones_like(d)
        weights /= weights.sum()
        net = (
            self._targets[indices[selected]]
            - costs.round_trip_cost
        )
        q25, q75 = _weighted_quantile(net, weights, (0.25, 0.75))
        effective_n = float(1.0 / np.square(weights).sum())
        nearest = float(d[0])
        confidence = float(
            min(1.0, effective_n / max(self.config.min_neighbors, 1))
            * np.exp(-nearest)
        )
        return OnlinePrediction(
            expected_net_return=float(np.dot(weights, net)),
            lower_bound=q25,
            upper_bound=q75,
            effective_n=effective_n,
            confidence=confidence,
            nearest_distance=nearest,
        )


def decide_position(
    keep: OnlinePrediction,
    config: OnlineConfig,
    *,
    reverse: OnlinePrediction | None = None,
    switch_cost: float = 0.0,
    adverse_bars: int = 0,
) -> str:
    """Return hold/exit/reverse without executing an order."""
    reliable = (
        keep.effective_n >= config.min_neighbors
        and keep.confidence >= config.min_confidence
    )
    if reverse is not None:
        reverse_reliable = (
            reverse.effective_n >= config.min_neighbors
            and reverse.confidence >= config.min_confidence
        )
        if (
            reverse_reliable
            and reverse.lower_bound
            > keep.upper_bound + switch_cost + config.reversal_margin
            and adverse_bars >= config.confirm_bars
        ):
            return "reverse"
    if (
        (not reliable or keep.lower_bound <= config.exit_threshold)
        and adverse_bars >= config.confirm_bars
    ):
        return "exit"
    return "hold"


def evaluate_online_state(
    model: OnlineStateModel,
    evaluation: pd.DataFrame,
    costs: CostConfig,
) -> tuple[dict, pd.DataFrame]:
    rows: list[dict] = []
    errors: list[float] = []
    decision_counts = {"hold": 0, "exit": 0, "reverse": 0}
    for _, group in evaluation.groupby("instance_id", sort=False):
        adverse_bars = 0
        for _, row in group.sort_values("bar_idx").iterrows():
            try:
                prediction = model.predict_one(row, costs)
            except ValueError:
                continue
            realized = (
                float(row["remaining_directional_return"])
                - costs.round_trip_cost
            )
            adverse_bars = (
                adverse_bars + 1
                if prediction.lower_bound <= model.config.exit_threshold
                else 0
            )
            decision = decide_position(
                prediction,
                model.config,
                adverse_bars=adverse_bars,
            )
            decision_counts[decision] += 1
            errors.append(abs(prediction.expected_net_return - realized))
            rows.append({
                "instance_id": int(row["instance_id"]),
                "timestamp": row["timestamp"],
                "split": row["split"],
                "line": row["line"],
                "pattern": row["pattern"],
                "predicted_direction": row["predicted_direction"],
                "elapsed_bars": row["elapsed_bars"],
                "candidate_dv_norm": row["candidate_dv_norm"],
                "expected_net_return": prediction.expected_net_return,
                "ev_lower_bound": prediction.lower_bound,
                "ev_upper_bound": prediction.upper_bound,
                "effective_n": prediction.effective_n,
                "confidence": prediction.confidence,
                "nearest_distance": prediction.nearest_distance,
                "realized_remaining_net_return": realized,
                "adverse_bars": adverse_bars,
                "decision": decision,
            })
    decisions = pd.DataFrame(rows)
    return {
        "n_evaluated": len(rows),
        "mae": float(np.mean(errors)) if errors else None,
        "mean_confidence": (
            float(decisions["confidence"].mean()) if len(decisions) else None
        ),
        "mean_effective_n": (
            float(decisions["effective_n"].mean())
            if len(decisions) else None
        ),
        "decision_counts": decision_counts,
    }, decisions
