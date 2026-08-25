"""Executable reconstruction of the document-supported proposed framework.

The V13-V18 cohort-level rank hierarchy is recovered exactly. The leaf-training
functions implement the most specific surviving document contract, but are a
candidate reconstruction because the five original leaf-writer sources are not
preserved.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    PolynomialFeatures,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
LeafName = Literal["mt_knn", "mt_r3", "mt_poly", "mt_poly_v3", "mt_poly_full"]
Preprocessing = Literal["knn_robust", "knn_quantile", "knn_standard"]
PolynomialVariant = Literal["interaction_robust", "interaction_quantile", "full_robust"]

LR_C_VALUES: Final = (0.3, 1.0, 3.0)
SGD_RANDOM_STATES: Final = (42, 7, 123)
SGD_ALPHA_VALUES: Final = (1e-4, 1e-3)
POLYNOMIAL_C_VALUES: Final = (0.3, 1.0, 3.0)
LEAF_NAMES: Final[tuple[LeafName, ...]] = (
    "mt_knn",
    "mt_r3",
    "mt_poly",
    "mt_poly_v3",
    "mt_poly_full",
)
LEAF_THRESHOLDS: Final = {
    "mt_knn": (65, 75, 85),
    "mt_r3": (55, 65, 75, 85, 95),
    "mt_poly": (65, 75, 85),
    "mt_poly_v3": (65, 75, 85),
    "mt_poly_full": (65, 75, 85),
}


@dataclass(frozen=True, slots=True)
class HistoricalStages:
    """Exactly recovered cohort-level V13-V18 hierarchy."""

    v13: FloatArray
    v14: FloatArray
    v15: FloatArray
    v16: FloatArray
    v17: FloatArray
    v18: FloatArray


def ordinal_rank(values: FloatArray) -> FloatArray:
    """Historical V5 ordinal rank: ``(argsort(argsort(x)) + 1) / (n + 1)``."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("rank input must be a non-empty finite vector")
    return (values.argsort().argsort() + 1).astype(np.float64) / (len(values) + 1)


def rank_average(*vectors: FloatArray) -> FloatArray:
    if not vectors or any(
        np.asarray(vector).shape != np.asarray(vectors[0]).shape for vector in vectors
    ):
        raise ValueError("rank-average inputs must have the same non-empty shape")
    rank_sum = np.zeros(np.asarray(vectors[0]).shape, dtype=np.float64)
    for vector in vectors:
        rank_sum += ordinal_rank(np.asarray(vector, dtype=np.float64))
    return rank_sum / len(vectors)


def assemble_historical_hierarchy(leaves: Mapping[str, FloatArray]) -> HistoricalStages:
    """Apply the preserved transductive V13-V18 rank hierarchy exactly."""
    if set(leaves) != set(LEAF_NAMES):
        raise ValueError(f"historical hierarchy requires exactly these leaves: {LEAF_NAMES}")
    mt_knn, mt_r3, mt_poly, mt_poly_v3, mt_poly_full = (
        np.asarray(leaves[name], dtype=np.float64) for name in LEAF_NAMES
    )
    v13 = rank_average(mt_knn, mt_r3, mt_poly, mt_poly_v3)
    v14 = (
        2 * ordinal_rank(mt_knn)
        + 2 * ordinal_rank(mt_r3)
        + 3 * ordinal_rank(mt_poly)
        + 2 * ordinal_rank(mt_poly_v3)
    ) / 9
    v15 = (v13 + v14) / 2
    v16 = rank_average(mt_knn, mt_r3, mt_poly_v3, mt_poly_full, v15)
    v17 = rank_average(mt_knn, mt_r3, mt_poly_full, v15, v16)
    v18 = rank_average(v16, v17)
    return HistoricalStages(v13=v13, v14=v14, v15=v15, v16=v16, v17=v17, v18=v18)


def _preprocessing(kind: Preprocessing, n_train: int) -> list[tuple[str, object]]:
    imputer = KNNImputer(n_neighbors=7, weights="distance")
    if kind == "knn_robust":
        scaler: object = RobustScaler()
    elif kind == "knn_quantile":
        scaler = QuantileTransformer(
            n_quantiles=min(50, n_train),
            output_distribution="normal",
            random_state=42,
        )
    elif kind == "knn_standard":
        scaler = StandardScaler()
    else:
        raise ValueError(f"unknown preprocessing: {kind}")
    return [("imputer", imputer), ("scaler", scaler)]


def _calibrated_probability(
    pipeline: Pipeline,
    X_train: FloatArray,
    labels: IntArray,
    X_score: FloatArray,
) -> FloatArray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            calibrated = CalibratedClassifierCV(pipeline, method="isotonic", cv=3)
            calibrated.fit(X_train, labels)
            return calibrated.predict_proba(X_score)[:, 1].astype(np.float64)
        except ValueError:
            pipeline.fit(X_train, labels)
            return pipeline.predict_proba(X_score)[:, 1].astype(np.float64)


def _fit_lr_sgd_head(
    X_train: FloatArray,
    labels: IntArray,
    X_score: FloatArray,
    *,
    preprocessing: Preprocessing,
    rank_individual_models: bool,
) -> FloatArray:
    counts = np.bincount(labels, minlength=2)
    if np.min(counts) < 4:
        return np.full(len(X_score), float(np.mean(labels)), dtype=np.float64)

    logistic_predictions = []
    for c_value in LR_C_VALUES:
        pipeline = Pipeline(
            [
                *_preprocessing(preprocessing, len(X_train)),
                (
                    "classifier",
                    LogisticRegression(
                        C=c_value,
                        max_iter=4000,
                        random_state=42,
                    ),
                ),
            ]
        )
        logistic_predictions.append(_calibrated_probability(pipeline, X_train, labels, X_score))

    huber_predictions = []
    huber_preprocessing: Preprocessing = (
        "knn_standard" if preprocessing == "knn_robust" else preprocessing
    )
    for random_state in SGD_RANDOM_STATES:
        for alpha in SGD_ALPHA_VALUES:
            pipeline = Pipeline(
                [
                    *_preprocessing(huber_preprocessing, len(X_train)),
                    (
                        "classifier",
                        SGDClassifier(
                            loss="modified_huber",
                            alpha=alpha,
                            max_iter=2000,
                            tol=1e-4,
                            random_state=random_state,
                        ),
                    ),
                ]
            )
            huber_predictions.append(_calibrated_probability(pipeline, X_train, labels, X_score))

    if rank_individual_models:
        return rank_average(*logistic_predictions, *huber_predictions)
    return rank_average(np.mean(logistic_predictions, axis=0), np.mean(huber_predictions, axis=0))


def _fit_polynomial_head(
    X_train: FloatArray,
    labels: IntArray,
    X_score: FloatArray,
    *,
    variant: PolynomialVariant,
) -> FloatArray:
    counts = np.bincount(labels, minlength=2)
    if np.min(counts) < 4:
        return np.full(len(X_score), float(np.mean(labels)), dtype=np.float64)

    if variant == "interaction_quantile":
        first_scaler: object = QuantileTransformer(
            n_quantiles=min(50, len(X_train)),
            output_distribution="normal",
            random_state=42,
        )
        interaction_only = True
    else:
        first_scaler = RobustScaler()
        interaction_only = variant == "interaction_robust"

    predictions = []
    for c_value in POLYNOMIAL_C_VALUES:
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("first_scaler", first_scaler),
                (
                    "polynomial",
                    PolynomialFeatures(
                        degree=2,
                        interaction_only=interaction_only,
                        include_bias=False,
                    ),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        C=c_value,
                        penalty="l1",
                        solver="liblinear",
                        max_iter=6000,
                        random_state=42,
                    ),
                ),
            ]
        )
        predictions.append(_calibrated_probability(pipeline, X_train, labels, X_score))
    return np.mean(predictions, axis=0)


def fit_documented_leaf(
    leaf: LeafName,
    X_train: FloatArray,
    improvement_train: FloatArray,
    X_score: FloatArray,
) -> FloatArray:
    if leaf not in LEAF_NAMES:
        raise ValueError(f"unknown documented leaf: {leaf}")
    thresholds = LEAF_THRESHOLDS[leaf]
    predictions = []
    for threshold in thresholds:
        labels = (improvement_train >= threshold).astype(np.int64)
        if leaf == "mt_knn":
            prediction = _fit_lr_sgd_head(
                X_train,
                labels,
                X_score,
                preprocessing="knn_robust",
                rank_individual_models=False,
            )
        elif leaf == "mt_r3":
            prediction = _fit_lr_sgd_head(
                X_train,
                labels,
                X_score,
                preprocessing="knn_quantile",
                rank_individual_models=True,
            )
        else:
            variants: dict[LeafName, PolynomialVariant] = {
                "mt_poly": "interaction_robust",
                "mt_poly_v3": "interaction_quantile",
                "mt_poly_full": "full_robust",
            }
            prediction = _fit_polynomial_head(
                X_train,
                labels,
                X_score,
                variant=variants[leaf],
            )
        predictions.append(prediction)
    return rank_average(*predictions)
