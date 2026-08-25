"""Strict-inductive recovered V18 scoring and boundary-aware routing.

The score construction follows the recovered V18 leaf and hierarchy contract,
but every rank is mapped through outer-training inner-OOF reference values.
Operating points, PRT bounds, and the BAB booster are learned without access to
outer-test labels or outer-test EASI improvement values.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

import atopix_ml.recovered_v18_model as recovered_model
from atopix_ml.documented_proposed import LEAF_NAMES
from atopix_ml.recovered_v18_model import LeafBundle
from atopix_ml.strict_proposed import (
    FEATURE_NAMES,
    assemble_hierarchy,
    reference_percentile,
    regression_stratified_splits,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
ComponentRecords = dict[str, list[dict[str, FloatArray]]]

LEAF_TO_MEMBER: Final = {
    "mt_knn": "multithreshold_three",
    "mt_r3": "multithreshold_five",
    "mt_poly": "polynomial_interaction_robust",
    "mt_poly_v3": "polynomial_interaction_quantile",
    "mt_poly_full": "polynomial_full_robust",
}
PRT_LOW_GRID: Final = (0.35, 0.40, 0.45, 0.50, 0.55)
PRT_HIGH_GRID: Final = (0.55, 0.60, 0.65, 0.70, 0.75)
EASI_BINS: Final = (
    (0.0, 50.0),
    (50.0, 65.0),
    (65.0, 75.0),
    (75.0, 85.0),
    (85.0, np.inf),
)


@dataclass(frozen=True, slots=True)
class PRTSelection:
    low: float
    high: float
    accuracy: float
    bin_accuracy_sd: float
    objective: float


@dataclass(frozen=True, slots=True)
class BABFit:
    model: Pipeline | None
    status: str
    n_midband: int
    n_midband_positive: int


class ConstantProbabilityModel:
    """Two-column probability adapter for a single-class inner-training head."""

    def __init__(self, positive_probability: float) -> None:
        self.positive_probability = positive_probability

    def predict_proba(self, X: FloatArray) -> FloatArray:
        positive = np.full(len(X), self.positive_probability, dtype=np.float64)
        return np.column_stack((1.0 - positive, positive))


def _positive_probability(model: Any, X: FloatArray) -> FloatArray:
    return np.asarray(model.predict_proba(X)[:, 1], dtype=np.float64)


def _fit_calibrated_with_small_class_fallback(
    pipeline: Any,
    X: FloatArray,
    labels: IntArray,
) -> Any:
    if np.unique(labels).size < 2:
        return ConstantProbabilityModel(float(np.mean(labels)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = CalibratedClassifierCV(pipeline, method="isotonic", cv=3)
            model.fit(X, labels)
            return model
        except ValueError:
            pipeline.fit(X, labels)
            return pipeline


def predict_leaf_components(
    bundle: LeafBundle,
    X: FloatArray,
) -> list[dict[str, FloatArray]]:
    """Return the recovered pre-rank components for one leaf."""
    leaf = bundle["leaf"]
    records: list[dict[str, FloatArray]] = []
    for threshold_record in bundle["threshold_models"]:
        models = threshold_record["models"]
        if leaf == "mt_knn":
            records.append(
                {
                    "logistic_mean": np.mean(
                        [_positive_probability(model, X) for model in models["logistic"]],
                        axis=0,
                    ),
                    "huber_mean": np.mean(
                        [_positive_probability(model, X) for model in models["huber"]],
                        axis=0,
                    ),
                }
            )
        elif leaf == "mt_r3":
            ordered_models = [*models["logistic"], *models["huber"]]
            records.append(
                {
                    f"classifier_{index}": _positive_probability(model, X)
                    for index, model in enumerate(ordered_models)
                }
            )
        else:
            records.append(
                {
                    "polynomial_mean": np.mean(
                        [_positive_probability(model, X) for model in models["polynomial"]],
                        axis=0,
                    )
                }
            )
    return records


def fit_component_records(
    X_train: FloatArray,
    improvement_train: FloatArray,
    X_score: FloatArray,
) -> ComponentRecords:
    """Fit all recovered leaves and return their pre-rank predictions."""
    original_fit_calibrated = recovered_model._fit_calibrated
    recovered_model._fit_calibrated = _fit_calibrated_with_small_class_fallback
    try:
        return {
            leaf: predict_leaf_components(
                recovered_model.fit_recovered_leaf(
                    leaf,
                    X_train,
                    improvement_train,
                    feature_names=FEATURE_NAMES,
                ),
                X_score,
            )
            for leaf in LEAF_NAMES
        }
    finally:
        recovered_model._fit_calibrated = original_fit_calibrated


def _allocate_like(records: ComponentRecords, n_rows: int) -> ComponentRecords:
    return {
        leaf: [
            {name: np.full(n_rows, np.nan, dtype=np.float64) for name in threshold}
            for threshold in thresholds
        ]
        for leaf, thresholds in records.items()
    }


def _assign_rows(
    destination: ComponentRecords,
    source: ComponentRecords,
    indices: IntArray,
) -> None:
    if tuple(destination) != tuple(source):
        raise RuntimeError("leaf component structures do not align")
    for leaf in destination:
        if len(destination[leaf]) != len(source[leaf]):
            raise RuntimeError(f"threshold component structures do not align for {leaf}")
        for target_threshold, source_threshold in zip(
            destination[leaf], source[leaf], strict=True
        ):
            if tuple(target_threshold) != tuple(source_threshold):
                raise RuntimeError(f"model component structures do not align for {leaf}")
            for name in target_threshold:
                target_threshold[name][indices] = source_threshold[name]


def _rank_pair(reference: FloatArray, scored: FloatArray) -> tuple[FloatArray, FloatArray]:
    return (
        reference_percentile(reference, reference),
        reference_percentile(reference, scored),
    )


def assemble_inductive_leaf(
    leaf: str,
    reference_thresholds: list[dict[str, FloatArray]],
    scored_thresholds: list[dict[str, FloatArray]],
) -> tuple[FloatArray, FloatArray]:
    """Apply every recovered leaf rank through the training reference."""
    if len(reference_thresholds) != len(scored_thresholds) or not reference_thresholds:
        raise ValueError("reference and scored thresholds must align and be non-empty")

    threshold_reference: list[FloatArray] = []
    threshold_scored: list[FloatArray] = []
    for reference_components, scored_components in zip(
        reference_thresholds, scored_thresholds, strict=True
    ):
        if tuple(reference_components) != tuple(scored_components):
            raise ValueError("reference and scored model components must align")
        if leaf in {"mt_knn", "mt_r3"}:
            ranked_pairs = [
                _rank_pair(reference_components[name], scored_components[name])
                for name in reference_components
            ]
            threshold_reference.append(
                np.mean([pair[0] for pair in ranked_pairs], axis=0)
            )
            threshold_scored.append(np.mean([pair[1] for pair in ranked_pairs], axis=0))
        else:
            if tuple(reference_components) != ("polynomial_mean",):
                raise ValueError(f"unexpected polynomial component structure for {leaf}")
            threshold_reference.append(reference_components["polynomial_mean"])
            threshold_scored.append(scored_components["polynomial_mean"])

    ranked_thresholds = [
        _rank_pair(reference, scored)
        for reference, scored in zip(threshold_reference, threshold_scored, strict=True)
    ]
    return (
        np.mean([pair[0] for pair in ranked_thresholds], axis=0),
        np.mean([pair[1] for pair in ranked_thresholds], axis=0),
    )


def strict_fold_scores(
    X: FloatArray,
    improvement: FloatArray,
    *,
    seed: int,
    fold: int,
) -> tuple[IntArray, IntArray, FloatArray, FloatArray]:
    """Return outer indices, training inner-OOF score, and outer-test score."""
    train, test = regression_stratified_splits(improvement, seed=seed)[fold]
    X_train = X[train]
    improvement_train = improvement[train]
    reference: ComponentRecords | None = None
    visits = np.zeros(len(train), dtype=np.int64)
    for inner_train, inner_validation in regression_stratified_splits(
        improvement_train, seed=seed + 1000
    ):
        predicted = fit_component_records(
            X_train[inner_train],
            improvement_train[inner_train],
            X_train[inner_validation],
        )
        if reference is None:
            reference = _allocate_like(predicted, len(train))
        _assign_rows(reference, predicted, inner_validation)
        visits[inner_validation] += 1
    if reference is None or not np.all(visits == 1):
        raise RuntimeError(f"inner OOF reference is incomplete for seed={seed} fold={fold}")
    for thresholds in reference.values():
        for components in thresholds:
            if any(not np.isfinite(values).all() for values in components.values()):
                raise RuntimeError("inner OOF component predictions are incomplete")

    scored = fit_component_records(X_train, improvement_train, X[test])
    reference_leaves: dict[str, FloatArray] = {}
    scored_leaves: dict[str, FloatArray] = {}
    for leaf in LEAF_NAMES:
        reference_leaf, scored_leaf = assemble_inductive_leaf(
            leaf, reference[leaf], scored[leaf]
        )
        member = LEAF_TO_MEMBER[leaf]
        reference_leaves[member] = reference_leaf
        scored_leaves[member] = scored_leaf
    reference_stages, scored_stages = assemble_hierarchy(
        reference_leaves, scored_leaves
    )
    return train, test, reference_stages["score"], scored_stages["score"]


def _candidate_threshold_rows(
    scores: FloatArray,
    labels: IntArray,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    prevalence = float(np.mean(labels))
    for threshold in np.unique(scores):
        predicted = scores >= threshold
        true_positive = int(np.count_nonzero(predicted & (labels == 1)))
        true_negative = int(np.count_nonzero(~predicted & (labels == 0)))
        false_positive = int(np.count_nonzero(predicted & (labels == 0)))
        false_negative = int(np.count_nonzero(~predicted & (labels == 1)))
        sensitivity = true_positive / (true_positive + false_negative)
        specificity = true_negative / (true_negative + false_positive)
        rows.append(
            {
                "threshold": float(threshold),
                "youden": float(sensitivity + specificity - 1.0),
                "f1": float(f1_score(labels, predicted, zero_division=0)),
                "prevalence_distance": float(abs(np.mean(predicted) - prevalence)),
            }
        )
    return rows


def select_operating_points(
    scores: FloatArray,
    labels: IntArray,
) -> dict[str, dict[str, float | str]]:
    """Select the primary and general sensitivity operating points."""
    score_values = np.asarray(scores, dtype=np.float64)
    label_values = np.asarray(labels, dtype=np.int64)
    if score_values.ndim != 1 or score_values.shape != label_values.shape:
        raise ValueError("scores and labels must be aligned one-dimensional arrays")
    if not np.isfinite(score_values).all() or np.unique(label_values).size != 2:
        raise ValueError("operating-point selection requires finite scores and both classes")
    candidates = _candidate_threshold_rows(score_values, label_values)

    def maximize(metric: str) -> dict[str, float]:
        best = max(row[metric] for row in candidates)
        return max(
            (row for row in candidates if row[metric] == best),
            key=lambda row: row["threshold"],
        )

    def minimize(metric: str) -> dict[str, float]:
        best = min(row[metric] for row in candidates)
        return max(
            (row for row in candidates if row[metric] == best),
            key=lambda row: row["threshold"],
        )

    youden = maximize("youden")
    max_f1 = maximize("f1")
    prevalence_match = minimize("prevalence_distance")
    return {
        "youden": {
            "threshold": youden["threshold"],
            "objective": "youden",
            "objective_value": youden["youden"],
        },
        "fixed_0_5": {
            "threshold": 0.5,
            "objective": "fixed",
            "objective_value": 0.5,
        },
        "prevalence_match": {
            "threshold": prevalence_match["threshold"],
            "objective": "absolute_prevalence_difference",
            "objective_value": prevalence_match["prevalence_distance"],
        },
        "max_f1": {
            "threshold": max_f1["threshold"],
            "objective": "f1",
            "objective_value": max_f1["f1"],
        },
    }


def apply_prt_midpoint(scores: FloatArray, low: float, high: float) -> IntArray:
    """Route outside the ambiguity band and split its midpoint."""
    if high <= low:
        raise ValueError("PRT high bound must exceed the low bound")
    score_values = np.asarray(scores, dtype=np.float64)
    predicted = np.zeros(len(score_values), dtype=np.int64)
    predicted[score_values > high] = 1
    midband = (score_values >= low) & (score_values <= high)
    predicted[midband] = (score_values[midband] >= (low + high) / 2.0).astype(
        np.int64
    )
    return predicted


def _per_bin_accuracy(
    labels: IntArray,
    predicted: IntArray,
    improvement: FloatArray,
) -> list[float]:
    accuracies = []
    for low, high in EASI_BINS:
        mask = (improvement >= low) & (improvement < high)
        if np.any(mask):
            accuracies.append(float(accuracy_score(labels[mask], predicted[mask])))
    return accuracies


def select_prt_band(
    scores: FloatArray,
    labels: IntArray,
    improvement: FloatArray,
    *,
    alpha: float = 0.5,
    low_grid: tuple[float, ...] = PRT_LOW_GRID,
    high_grid: tuple[float, ...] = PRT_HIGH_GRID,
) -> PRTSelection:
    """Choose the first lexicographic band maximizing accuracy - alpha*bin-SD."""
    score_values = np.asarray(scores, dtype=np.float64)
    label_values = np.asarray(labels, dtype=np.int64)
    improvement_values = np.asarray(improvement, dtype=np.float64)
    if not (
        score_values.ndim == label_values.ndim == improvement_values.ndim == 1
        and score_values.shape == label_values.shape == improvement_values.shape
    ):
        raise ValueError("scores, labels, and improvement must be aligned vectors")

    best: PRTSelection | None = None
    for low in low_grid:
        for high in high_grid:
            if high <= low:
                continue
            predicted = apply_prt_midpoint(score_values, low, high)
            accuracy = float(accuracy_score(label_values, predicted))
            bin_accuracies = _per_bin_accuracy(
                label_values, predicted, improvement_values
            )
            bin_sd = float(np.std(bin_accuracies)) if bin_accuracies else 1.0
            selection = PRTSelection(
                low=float(low),
                high=float(high),
                accuracy=accuracy,
                bin_accuracy_sd=bin_sd,
                objective=accuracy - alpha * bin_sd,
            )
            if best is None or selection.objective > best.objective:
                best = selection
    if best is None:
        raise ValueError("PRT grids contain no valid low/high pair")
    return best


def fit_bab(
    X_train: FloatArray,
    labels_train: IntArray,
    inner_scores: FloatArray,
    low: float,
    high: float,
) -> BABFit:
    """Fit the preserved AdaBoost-logistic booster on outer-training midband rows."""
    midband = (inner_scores >= low) & (inner_scores <= high)
    mid_labels = np.asarray(labels_train[midband], dtype=np.int64)
    n_midband = int(np.count_nonzero(midband))
    n_positive = int(np.count_nonzero(mid_labels == 1))
    if n_midband < 10:
        return BABFit(None, "fallback_midband_n_lt_10", n_midband, n_positive)
    if np.unique(mid_labels).size < 2:
        return BABFit(None, "fallback_midband_single_class", n_midband, n_positive)

    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            (
                "classifier",
                AdaBoostClassifier(
                    estimator=LogisticRegression(
                        C=0.5,
                        max_iter=2000,
                        random_state=42,
                    ),
                    n_estimators=10,
                    learning_rate=0.3,
                    algorithm="SAMME",
                    random_state=42,
                ),
            ),
        ]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model.fit(X_train[midband], mid_labels)
        except (TypeError, ValueError) as error:
            return BABFit(
                None,
                f"fallback_fit_error_{type(error).__name__}",
                n_midband,
                n_positive,
            )
    return BABFit(model, "fitted", n_midband, n_positive)


def apply_bab(
    base_decisions: IntArray,
    outer_scores: FloatArray,
    X_test: FloatArray,
    low: float,
    high: float,
    fit: BABFit,
) -> tuple[IntArray, int]:
    """Replace PRT midpoint decisions only within the test ambiguity band."""
    decisions = np.asarray(base_decisions, dtype=np.int64).copy()
    midband = (outer_scores >= low) & (outer_scores <= high)
    if fit.model is not None and np.any(midband):
        decisions[midband] = np.asarray(
            fit.model.predict(X_test[midband]), dtype=np.int64
        )
    return decisions, int(np.count_nonzero(midband))


def classification_metrics(labels: IntArray, predicted: IntArray) -> dict[str, float | int]:
    """Return the binary performance contract used in the report."""
    y = np.asarray(labels, dtype=np.int64)
    decision = np.asarray(predicted, dtype=np.int64)
    true_positive = int(np.count_nonzero((y == 1) & (decision == 1)))
    true_negative = int(np.count_nonzero((y == 0) & (decision == 0)))
    false_positive = int(np.count_nonzero((y == 0) & (decision == 1)))
    false_negative = int(np.count_nonzero((y == 1) & (decision == 0)))
    positive_denominator = true_positive + false_negative
    negative_denominator = true_negative + false_positive
    sensitivity = (
        true_positive / positive_denominator if positive_denominator else float("nan")
    )
    specificity = (
        true_negative / negative_denominator if negative_denominator else float("nan")
    )
    balanced_accuracy = (
        (sensitivity + specificity) / 2.0
        if positive_denominator and negative_denominator
        else float("nan")
    )
    return {
        "n": len(y),
        "n_positive": int(np.count_nonzero(y == 1)),
        "n_negative": int(np.count_nonzero(y == 0)),
        "accuracy": float(accuracy_score(y, decision)),
        "balanced_accuracy": balanced_accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": float(f1_score(y, decision, zero_division=0)),
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }
