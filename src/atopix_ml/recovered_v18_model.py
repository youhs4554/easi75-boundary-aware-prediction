"""Fit and score the selected recovered V18 leaf-model package."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

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

from atopix_ml.documented_proposed import (
    LEAF_THRESHOLDS,
    LR_C_VALUES,
    POLYNOMIAL_C_VALUES,
    SGD_ALPHA_VALUES,
    SGD_RANDOM_STATES,
    HistoricalStages,
    LeafName,
    assemble_historical_hierarchy,
    rank_average,
)

FloatArray = NDArray[np.float64]
LeafBundle = dict[str, Any]


@dataclass(frozen=True, slots=True)
class RecoveredV18BatchPrediction:
    leaves: dict[str, FloatArray]
    stages: HistoricalStages


def _fit_calibrated(pipeline: Pipeline, X: FloatArray, labels: NDArray[np.int64]) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = CalibratedClassifierCV(pipeline, method="isotonic", cv=3)
        model.fit(X, labels)
    return model


def _knn_preprocessing(scaler: object) -> list[tuple[str, object]]:
    return [
        ("imputer", KNNImputer(n_neighbors=7, weights="distance")),
        ("scaler", scaler),
    ]


def _fit_linear_threshold(
    leaf: LeafName,
    X: FloatArray,
    labels: NDArray[np.int64],
) -> dict[str, list[Any]]:
    if leaf == "mt_knn":
        lr_scaler = RobustScaler
        sgd_scaler = StandardScaler
    elif leaf == "mt_r3":
        lr_scaler = lambda: QuantileTransformer(  # noqa: E731
            n_quantiles=min(50, len(X)),
            output_distribution="normal",
            random_state=42,
        )
        sgd_scaler = lr_scaler
    else:
        raise ValueError(f"not a linear recovered leaf: {leaf}")

    logistic_models = []
    for c_value in LR_C_VALUES:
        pipeline = Pipeline(
            [
                *_knn_preprocessing(lr_scaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=c_value,
                        penalty="l2",
                        solver="lbfgs",
                        max_iter=4000,
                        random_state=42,
                    ),
                ),
            ]
        )
        logistic_models.append(_fit_calibrated(pipeline, X, labels))

    huber_models = []
    for random_state in SGD_RANDOM_STATES:
        for alpha in SGD_ALPHA_VALUES:
            pipeline = Pipeline(
                [
                    *_knn_preprocessing(sgd_scaler()),
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
            huber_models.append(_fit_calibrated(pipeline, X, labels))
    return {"logistic": logistic_models, "huber": huber_models}


def _fit_polynomial_threshold(
    leaf: LeafName,
    X: FloatArray,
    labels: NDArray[np.int64],
) -> dict[str, list[Any]]:
    if leaf == "mt_poly_v3":
        scaler_factory = lambda: QuantileTransformer(  # noqa: E731
            n_quantiles=min(50, len(X)),
            output_distribution="normal",
            random_state=42,
        )
    else:
        scaler_factory = RobustScaler
    interaction_only = leaf != "mt_poly_full"

    models = []
    for c_value in POLYNOMIAL_C_VALUES:
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", scaler_factory()),
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
        models.append(_fit_calibrated(pipeline, X, labels))
    return {"polynomial": models}


def fit_recovered_leaf(
    leaf: LeafName,
    X: FloatArray,
    improvement: FloatArray,
    *,
    feature_names: tuple[str, ...],
) -> LeafBundle:
    X = np.asarray(X, dtype=np.float64)
    improvement = np.asarray(improvement, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] != improvement.shape[0]:
        raise ValueError("X and improvement must be aligned two-dimensional training data")
    if X.shape[1] != len(feature_names):
        raise ValueError("feature_names must match the training matrix width")

    threshold_models = []
    for threshold in LEAF_THRESHOLDS[leaf]:
        labels = (improvement >= threshold).astype(np.int64)
        if leaf in {"mt_knn", "mt_r3"}:
            models = _fit_linear_threshold(leaf, X, labels)
        else:
            models = _fit_polynomial_threshold(leaf, X, labels)
        threshold_models.append({"threshold": threshold, "models": models})

    return {
        "schema_version": "1.0",
        "artifact_type": "selected_recovered_v18_leaf_model",
        "leaf": leaf,
        "feature_names": feature_names,
        "n_training_rows": len(X),
        "threshold_models": threshold_models,
        "batch_rank_contract": (
            "Predictions are ordinal-ranked within the supplied batch. "
            "Single-patient scoring is not defined by the historical model."
        ),
    }


def _positive_probability(model: Any, X: FloatArray) -> FloatArray:
    return np.asarray(model.predict_proba(X)[:, 1], dtype=np.float64)


def predict_recovered_leaf(bundle: LeafBundle, X: FloatArray) -> FloatArray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != len(bundle["feature_names"]):
        raise ValueError("X must be two-dimensional and match the stored feature order")
    if len(X) < 2:
        raise ValueError("historical batch-rank scoring requires at least two rows")

    leaf = bundle["leaf"]
    threshold_predictions = []
    for record in bundle["threshold_models"]:
        models = record["models"]
        if leaf == "mt_knn":
            logistic_mean = np.mean(
                [_positive_probability(model, X) for model in models["logistic"]],
                axis=0,
            )
            huber_mean = np.mean(
                [_positive_probability(model, X) for model in models["huber"]],
                axis=0,
            )
            threshold_prediction = rank_average(logistic_mean, huber_mean)
        elif leaf == "mt_r3":
            threshold_prediction = rank_average(
                *[
                    _positive_probability(model, X)
                    for model in [*models["logistic"], *models["huber"]]
                ]
            )
        else:
            threshold_prediction = np.mean(
                [_positive_probability(model, X) for model in models["polynomial"]],
                axis=0,
            )
        threshold_predictions.append(threshold_prediction)
    return rank_average(*threshold_predictions)


def predict_recovered_v18(
    leaf_bundles: dict[str, LeafBundle],
    X: FloatArray,
) -> RecoveredV18BatchPrediction:
    leaves = {leaf: predict_recovered_leaf(bundle, X) for leaf, bundle in leaf_bundles.items()}
    return RecoveredV18BatchPrediction(
        leaves=leaves,
        stages=assemble_historical_hierarchy(leaves),
    )
