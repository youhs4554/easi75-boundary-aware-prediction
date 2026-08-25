from __future__ import annotations

import hashlib
import json
import math
import warnings
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    PolynomialFeatures,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

from atopix_ml.v6_reanalysis.thresholds import select_highest_observed_youden

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
Preprocessing = Literal["knn_robust", "knn_quantile"]
PolynomialVariant = Literal["interaction_robust", "interaction_quantile", "full_robust"]

OUTER_SEEDS: Final = (42, 7, 123, 456, 789, 1, 2, 3, 4, 5, 100, 200, 300, 400, 500)
FEATURE_NAMES: Final = (
    "Baseline Trunk EASI",
    "fe_allergen_panel_mean",
    (
        "음식/식품 관련 알레르겐\n"
        "(Egg White, Milk, Soy bean, Maize, Sesame, Crab, Shrimp, Potato, "
        "Apple, Cacao, Peach, Mackerel)"
    ),
    "Age",
    "Baseline H&N EASI",
    "fe_hn_easi_ratio",
    "Scalp eczema",
    "fe_allergen_panel_count_nonzero",
    "fe_allergy_burden",
    "WBC count",
    "CRP",
    "Albumin",
    "AST",
    "ALT",
    "Creatinine",
    "eGFR",
    "Total IgE",
    "Onset age",
    "Hx of Other allergies",
    "Previous treatment of cyclosporine",
    "fe_prev_treatment_count",
    "Hx of Dyslipidemia",
    "fe_subcomp_max",
)
MEMBER_NAMES: Final = (
    "multithreshold_three",
    "multithreshold_five",
    "polynomial_interaction_robust",
    "polynomial_interaction_quantile",
    "polynomial_full_robust",
)
STAGE_NAMES: Final = (
    "stage_uniform",
    "stage_weighted",
    "stage_mean",
    "stage_a",
    "stage_b",
    "score",
)


@dataclass(frozen=True, slots=True)
class FoldResult:
    outer_seed: int
    outer_fold: int
    train_indices: IntArray
    test_indices: IntArray
    inner_scores: FloatArray
    outer_scores: FloatArray
    outer_decisions: IntArray
    threshold: float
    threshold_youden: float
    threshold_sensitivity: float
    threshold_specificity: float
    threshold_tie_count: int
    member_scores: dict[str, FloatArray]
    stage_scores: dict[str, FloatArray]


def regression_stratified_splits(
    improvement: FloatArray,
    *,
    n_splits: int = 5,
    seed: int,
) -> tuple[tuple[IntArray, IntArray], ...]:
    from sklearn.model_selection import StratifiedKFold

    bins = np.quantile(improvement, np.linspace(0, 1, 11)[1:-1])
    labels = np.digitize(improvement, bins)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return tuple(
        (train.astype(np.int64), test.astype(np.int64))
        for train, test in splitter.split(np.zeros(len(improvement)), labels)
    )


def reference_percentile(reference: FloatArray, values: FloatArray) -> FloatArray:
    """Map scores through a training-reference empirical CDF.

    Every scored value is transformed independently, so adding, removing, or
    reordering other scored patients cannot change its result.
    """

    reference = np.asarray(reference, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if reference.ndim != 1 or values.ndim != 1 or reference.size == 0:
        raise ValueError("reference and values must be one-dimensional; reference cannot be empty")
    if not np.isfinite(reference).all() or not np.isfinite(values).all():
        raise ValueError("reference and values must contain only finite numbers")
    ordered = np.sort(reference)
    left = np.searchsorted(ordered, values, side="left")
    right = np.searchsorted(ordered, values, side="right")
    return (left + right + 1.0) / (2.0 * (len(reference) + 1.0))


def _rank_pair(reference: FloatArray, values: FloatArray) -> tuple[FloatArray, FloatArray]:
    return reference_percentile(reference, reference), reference_percentile(reference, values)


def assemble_hierarchy(
    reference_members: dict[str, FloatArray],
    scored_members: dict[str, FloatArray],
) -> tuple[dict[str, FloatArray], dict[str, FloatArray]]:
    if tuple(reference_members) != MEMBER_NAMES or tuple(scored_members) != MEMBER_NAMES:
        raise ValueError("member order or names do not match the frozen hierarchy contract")
    reference_size = len(next(iter(reference_members.values())))
    scored_size = len(next(iter(scored_members.values())))
    if any(len(values) != reference_size for values in reference_members.values()):
        raise ValueError("reference member vectors must have equal length")
    if any(len(values) != scored_size for values in scored_members.values()):
        raise ValueError("scored member vectors must have equal length")

    ranked_reference: dict[str, FloatArray] = {}
    ranked_scored: dict[str, FloatArray] = {}
    for name in MEMBER_NAMES:
        ranked_reference[name], ranked_scored[name] = _rank_pair(
            reference_members[name],
            scored_members[name],
        )

    first_four = MEMBER_NAMES[:4]
    uniform_reference = np.mean([ranked_reference[name] for name in first_four], axis=0)
    uniform_scored = np.mean([ranked_scored[name] for name in first_four], axis=0)
    weights = np.asarray((2.0, 2.0, 3.0, 2.0))
    weighted_reference = np.average(
        [ranked_reference[name] for name in first_four], axis=0, weights=weights
    )
    weighted_scored = np.average(
        [ranked_scored[name] for name in first_four], axis=0, weights=weights
    )
    mean_reference = (uniform_reference + weighted_reference) / 2.0
    mean_scored = (uniform_scored + weighted_scored) / 2.0

    mean_reference_rank, mean_scored_rank = _rank_pair(mean_reference, mean_scored)
    stage_a_reference = np.mean(
        [
            ranked_reference["multithreshold_three"],
            ranked_reference["multithreshold_five"],
            ranked_reference["polynomial_interaction_quantile"],
            ranked_reference["polynomial_full_robust"],
            mean_reference_rank,
        ],
        axis=0,
    )
    stage_a_scored = np.mean(
        [
            ranked_scored["multithreshold_three"],
            ranked_scored["multithreshold_five"],
            ranked_scored["polynomial_interaction_quantile"],
            ranked_scored["polynomial_full_robust"],
            mean_scored_rank,
        ],
        axis=0,
    )

    stage_a_reference_rank, stage_a_scored_rank = _rank_pair(
        stage_a_reference,
        stage_a_scored,
    )
    stage_b_reference = np.mean(
        [
            ranked_reference["multithreshold_three"],
            ranked_reference["multithreshold_five"],
            ranked_reference["polynomial_full_robust"],
            mean_reference_rank,
            stage_a_reference_rank,
        ],
        axis=0,
    )
    stage_b_scored = np.mean(
        [
            ranked_scored["multithreshold_three"],
            ranked_scored["multithreshold_five"],
            ranked_scored["polynomial_full_robust"],
            mean_scored_rank,
            stage_a_scored_rank,
        ],
        axis=0,
    )
    stage_a_final_reference, stage_a_final_scored = _rank_pair(
        stage_a_reference,
        stage_a_scored,
    )
    stage_b_final_reference, stage_b_final_scored = _rank_pair(
        stage_b_reference,
        stage_b_scored,
    )
    score_reference = (stage_a_final_reference + stage_b_final_reference) / 2.0
    score_scored = (stage_a_final_scored + stage_b_final_scored) / 2.0

    reference_stages = {
        "stage_uniform": uniform_reference,
        "stage_weighted": weighted_reference,
        "stage_mean": mean_reference,
        "stage_a": stage_a_reference,
        "stage_b": stage_b_reference,
        "score": score_reference,
    }
    scored_stages = {
        "stage_uniform": uniform_scored,
        "stage_weighted": weighted_scored,
        "stage_mean": mean_scored,
        "stage_a": stage_a_scored,
        "stage_b": stage_b_scored,
        "score": score_scored,
    }
    return reference_stages, scored_stages


def _preprocessing_steps(kind: Preprocessing, n_train: int) -> list[tuple[str, object]]:
    if kind == "knn_robust":
        return [
            ("imputer", KNNImputer(n_neighbors=7, weights="distance")),
            ("scaler", RobustScaler()),
        ]
    if kind == "knn_quantile":
        return [
            ("imputer", KNNImputer(n_neighbors=7, weights="distance")),
            (
                "scaler",
                QuantileTransformer(
                    n_quantiles=min(50, n_train),
                    output_distribution="normal",
                    random_state=42,
                ),
            ),
        ]
    raise ValueError(f"unknown preprocessing kind: {kind}")


def _calibrated_predictions(
    pipeline: Pipeline,
    X_train: FloatArray,
    labels: IntArray,
    X_score: FloatArray,
) -> FloatArray:
    counts = np.bincount(labels, minlength=2)
    if np.min(counts) < 3:
        return np.full(len(X_score), float(np.mean(labels)), dtype=np.float64)
    calibrated = CalibratedClassifierCV(pipeline, method="isotonic", cv=3, n_jobs=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calibrated.fit(X_train, labels)
        return calibrated.predict_proba(X_score)[:, 1].astype(np.float64)


def _fit_threshold_head(
    X_train: FloatArray,
    labels: IntArray,
    X_score: FloatArray,
    *,
    preprocessing: Preprocessing,
) -> FloatArray:
    predictions: list[FloatArray] = []
    for c_value in (0.3, 1.0, 3.0):
        pipeline = Pipeline(
            [
                *_preprocessing_steps(preprocessing, len(X_train)),
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
        predictions.append(_calibrated_predictions(pipeline, X_train, labels, X_score))
    for random_state in (42, 7, 123):
        for alpha in (1e-4, 1e-3):
            pipeline = Pipeline(
                [
                    *_preprocessing_steps(preprocessing, len(X_train)),
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
            predictions.append(_calibrated_predictions(pipeline, X_train, labels, X_score))
    return np.mean(predictions, axis=0)


def _fit_multithreshold(
    X_train: FloatArray,
    improvement_train: FloatArray,
    X_score: FloatArray,
    *,
    thresholds: tuple[int, ...],
    weights: tuple[float, ...],
    preprocessing: Preprocessing,
) -> FloatArray:
    if len(thresholds) != len(weights) or not math.isclose(sum(weights), 1.0):
        raise ValueError("threshold weights must align and sum to one")
    predictions = [
        _fit_threshold_head(
            X_train,
            (improvement_train >= threshold).astype(np.int64),
            X_score,
            preprocessing=preprocessing,
        )
        for threshold in thresholds
    ]
    return np.average(predictions, axis=0, weights=np.asarray(weights))


def _fit_polynomial(
    X_train: FloatArray,
    labels: IntArray,
    X_score: FloatArray,
    *,
    variant: PolynomialVariant,
) -> FloatArray:
    if variant == "interaction_robust":
        first_scaler: object = RobustScaler()
        interaction_only = True
    elif variant == "interaction_quantile":
        first_scaler = QuantileTransformer(
            n_quantiles=min(50, len(X_train)),
            output_distribution="normal",
            random_state=42,
        )
        interaction_only = True
    elif variant == "full_robust":
        first_scaler = RobustScaler()
        interaction_only = False
    else:
        raise ValueError(f"unknown polynomial variant: {variant}")

    predictions: list[FloatArray] = []
    for c_value in (0.05, 0.1, 0.3, 1.0):
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
                ("second_scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=c_value,
                        penalty="l1",
                        solver="saga",
                        max_iter=6000,
                        random_state=42,
                    ),
                ),
            ]
        )
        predictions.append(_calibrated_predictions(pipeline, X_train, labels, X_score))
    return np.mean(predictions, axis=0)


def fit_members(
    X_train: FloatArray,
    improvement_train: FloatArray,
    X_score: FloatArray,
) -> dict[str, FloatArray]:
    labels = (improvement_train >= 75.0).astype(np.int64)
    return {
        "multithreshold_three": _fit_multithreshold(
            X_train,
            improvement_train,
            X_score,
            thresholds=(65, 75, 85),
            weights=(0.25, 0.60, 0.15),
            preprocessing="knn_robust",
        ),
        "multithreshold_five": _fit_multithreshold(
            X_train,
            improvement_train,
            X_score,
            thresholds=(55, 65, 75, 85, 95),
            weights=(0.10, 0.20, 0.40, 0.20, 0.10),
            preprocessing="knn_quantile",
        ),
        "polynomial_interaction_robust": _fit_polynomial(
            X_train,
            labels,
            X_score,
            variant="interaction_robust",
        ),
        "polynomial_interaction_quantile": _fit_polynomial(
            X_train,
            labels,
            X_score,
            variant="interaction_quantile",
        ),
        "polynomial_full_robust": _fit_polynomial(
            X_train,
            labels,
            X_score,
            variant="full_robust",
        ),
    }


def run_outer_fold(
    X: FloatArray,
    labels: IntArray,
    improvement: FloatArray,
    *,
    outer_seed: int,
    outer_fold: int,
) -> FoldResult:
    if X.shape != (119, 23) or labels.shape != (119,) or improvement.shape != (119,):
        raise ValueError("the strict analysis requires the fixed 119-patient, 23-predictor cohort")
    if not np.array_equal(labels, (improvement >= 75.0).astype(np.int64)):
        raise ValueError("binary labels must equal EASI improvement >= 75")
    outer_splits = regression_stratified_splits(improvement, seed=outer_seed)
    if outer_fold not in range(len(outer_splits)):
        raise ValueError("outer_fold must be between zero and four")
    train_indices, test_indices = outer_splits[outer_fold]
    X_train = X[train_indices]
    improvement_train = improvement[train_indices]
    labels_train = labels[train_indices]

    inner_member_scores = {
        name: np.full(len(train_indices), np.nan, dtype=np.float64) for name in MEMBER_NAMES
    }
    inner_splits = regression_stratified_splits(
        improvement_train,
        seed=outer_seed + 1000,
    )
    validation_visits = np.zeros(len(train_indices), dtype=np.int64)
    for inner_train, inner_validation in inner_splits:
        predictions = fit_members(
            X_train[inner_train],
            improvement_train[inner_train],
            X_train[inner_validation],
        )
        for name in MEMBER_NAMES:
            inner_member_scores[name][inner_validation] = predictions[name]
        validation_visits[inner_validation] += 1
    if not np.all(validation_visits == 1):
        raise RuntimeError("inner validation folds do not partition the outer-training patients")
    if any(not np.isfinite(values).all() for values in inner_member_scores.values()):
        raise RuntimeError("inner member predictions are incomplete or non-finite")

    outer_member_scores = fit_members(X_train, improvement_train, X[test_indices])
    inner_stages, outer_stages = assemble_hierarchy(inner_member_scores, outer_member_scores)
    selection = select_highest_observed_youden(inner_stages["score"], labels_train)
    outer_scores = outer_stages["score"]
    outer_decisions = (outer_scores >= selection.threshold).astype(np.int64)
    return FoldResult(
        outer_seed=outer_seed,
        outer_fold=outer_fold,
        train_indices=train_indices,
        test_indices=test_indices,
        inner_scores=inner_stages["score"],
        outer_scores=outer_scores,
        outer_decisions=outer_decisions,
        threshold=selection.threshold,
        threshold_youden=selection.youden,
        threshold_sensitivity=selection.sensitivity,
        threshold_specificity=selection.specificity,
        threshold_tie_count=selection.tie_count,
        member_scores=outer_member_scores,
        stage_scores=outer_stages,
    )


def fold_result_payload(result: FoldResult, *, config_sha256: str) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "1.0",
        "method_status": "strict_inductive_reimplementation_not_exact_frozen_reproduction",
        "config_sha256": config_sha256,
        "outer_seed": result.outer_seed,
        "outer_fold": result.outer_fold,
        "outer_fold_id": f"seed-{result.outer_seed}-fold-{result.outer_fold}",
        "outer_train_indices": result.train_indices.tolist(),
        "outer_test_indices": result.test_indices.tolist(),
        "inner_oof_scores": result.inner_scores.tolist(),
        "outer_test_scores": result.outer_scores.tolist(),
        "outer_test_decisions": result.outer_decisions.tolist(),
        "threshold": result.threshold,
        "threshold_youden": result.threshold_youden,
        "threshold_sensitivity": result.threshold_sensitivity,
        "threshold_specificity": result.threshold_specificity,
        "threshold_tie_count": result.threshold_tie_count,
        "member_scores": {name: result.member_scores[name].tolist() for name in MEMBER_NAMES},
        "stage_scores": {name: result.stage_scores[name].tolist() for name in STAGE_NAMES},
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["payload_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def validate_fold_payload(payload: dict[str, object], *, config_sha256: str) -> None:
    if payload.get("config_sha256") != config_sha256:
        raise ValueError("fold payload belongs to a different experiment configuration")
    copied = dict(payload)
    expected_hash = copied.pop("payload_sha256", None)
    canonical = json.dumps(copied, sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(canonical).hexdigest() != expected_hash:
        raise ValueError("fold payload checksum mismatch")
    train = set(int(value) for value in copied["outer_train_indices"])  # type: ignore[arg-type]
    test = set(int(value) for value in copied["outer_test_indices"])  # type: ignore[arg-type]
    if train & test or train | test != set(range(119)):
        raise ValueError("outer train/test partition is invalid")
    scores = np.asarray(copied["outer_test_scores"], dtype=np.float64)
    decisions = np.asarray(copied["outer_test_decisions"], dtype=np.int64)
    if len(scores) != len(test) or len(decisions) != len(test):
        raise ValueError("outer prediction length mismatch")
    if not np.isfinite(scores).all() or not np.all(np.isin(decisions, (0, 1))):
        raise ValueError("outer predictions are invalid")


def mean_repeat_auroc(labels: IntArray, score_matrix: FloatArray) -> float:
    return float(np.mean([roc_auc_score(labels, score_matrix[:, column]) for column in range(15)]))
