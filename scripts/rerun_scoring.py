"""Fold-level scoring state for the proposed framework.

``atopix_ml.strict_recovered_router.strict_fold_scores`` refits every leaf each
time it is called, which is what the performance runs want and what the
attribution analysis cannot afford: attribution needs to evaluate the score
function on tens of thousands of synthetic covariate rows per patient.

This module splits that single function into its two parts — the fitted state
of a fold, and the pointwise map from covariates to a score — without changing
either.  The map is exactly the one the performance run uses:

* the leaves are fitted once on the outer-training partition;
* the rank references are the outer-training inner out-of-fold component
  predictions, fixed for the fold;
* every rank is then a per-row lookup against those fixed references, so the
  score of one patient does not depend on which other rows are scored with it.

:func:`fold_state` asserts that property by reproducing the outer-test score of
``strict_fold_scores`` exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

import atopix_ml.recovered_v18_model as recovered_model
from atopix_ml.documented_proposed import LEAF_NAMES
from atopix_ml.strict_proposed import assemble_hierarchy, regression_stratified_splits
from atopix_ml.strict_recovered_router import (
    LEAF_TO_MEMBER,
    _allocate_like,
    _assign_rows,
    _fit_calibrated_with_small_class_fallback,
    assemble_inductive_leaf,
    predict_leaf_components,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(slots=True)
class FoldState:
    seed: int
    fold: int
    train: IntArray
    test: IntArray
    inner_scores: FloatArray
    outer_scores: FloatArray
    reference: dict[str, list[dict[str, FloatArray]]]
    bundles: dict[str, Any]
    feature_names: tuple[str, ...]

    def score(self, X_rows: FloatArray) -> FloatArray:
        """Score arbitrary covariate rows through this fold's fixed map."""
        rows = np.asarray(X_rows, dtype=np.float64)
        if rows.ndim == 1:
            rows = rows.reshape(1, -1)
        if rows.shape[1] != len(self.feature_names):
            raise ValueError("row width does not match the fold's predictor order")
        reference_leaves: dict[str, FloatArray] = {}
        scored_leaves: dict[str, FloatArray] = {}
        for leaf in LEAF_NAMES:
            reference_leaf, scored_leaf = assemble_inductive_leaf(
                leaf,
                self.reference[leaf],
                predict_leaf_components(self.bundles[leaf], rows),
            )
            member = LEAF_TO_MEMBER[leaf]
            reference_leaves[member] = reference_leaf
            scored_leaves[member] = scored_leaf
        _, scored_stages = assemble_hierarchy(reference_leaves, scored_leaves)
        return np.asarray(scored_stages["score"], dtype=np.float64)


def _fit_bundles(
    X_train: FloatArray, improvement_train: FloatArray, feature_names: tuple[str, ...]
) -> dict[str, Any]:
    original = recovered_model._fit_calibrated
    recovered_model._fit_calibrated = _fit_calibrated_with_small_class_fallback
    try:
        return {
            leaf: recovered_model.fit_recovered_leaf(
                leaf, X_train, improvement_train, feature_names=feature_names
            )
            for leaf in LEAF_NAMES
        }
    finally:
        recovered_model._fit_calibrated = original


def fold_state(
    X: FloatArray,
    improvement: FloatArray,
    *,
    seed: int,
    fold: int,
    feature_names: tuple[str, ...],
    split: tuple[IntArray, IntArray] | None = None,
) -> FoldState:
    """Fit one outer fold and return its reusable scoring state.

    ``split`` overrides the outer partition.  The partition is a function of the
    whole observed-improvement vector, so the leakage probe has to hold it fixed
    while corrupting the held-out outcome values; passing it explicitly is how
    that is done without altering the fitting path.
    """
    train, test = (
        split if split is not None else regression_stratified_splits(improvement, seed=seed)[fold]
    )
    X_train = X[train]
    improvement_train = improvement[train]

    reference: dict[str, list[dict[str, FloatArray]]] | None = None
    visits = np.zeros(len(train), dtype=np.int64)
    for inner_train, inner_validation in regression_stratified_splits(
        improvement_train, seed=seed + 1000
    ):
        inner_bundles = _fit_bundles(
            X_train[inner_train], improvement_train[inner_train], feature_names
        )
        predicted = {
            leaf: predict_leaf_components(inner_bundles[leaf], X_train[inner_validation])
            for leaf in LEAF_NAMES
        }
        if reference is None:
            reference = _allocate_like(predicted, len(train))
        _assign_rows(reference, predicted, inner_validation)
        visits[inner_validation] += 1
    if reference is None or not np.all(visits == 1):
        raise RuntimeError(
            f"inner out-of-fold reference is incomplete for seed={seed} fold={fold}"
        )
    for thresholds in reference.values():
        for components in thresholds:
            if any(not np.isfinite(values).all() for values in components.values()):
                raise RuntimeError("inner out-of-fold component predictions are incomplete")

    bundles = _fit_bundles(X_train, improvement_train, feature_names)
    reference_leaves: dict[str, FloatArray] = {}
    for leaf in LEAF_NAMES:
        reference_leaf, _ = assemble_inductive_leaf(
            leaf, reference[leaf], predict_leaf_components(bundles[leaf], X_train[:2])
        )
        reference_leaves[LEAF_TO_MEMBER[leaf]] = reference_leaf
    reference_stages, _ = assemble_hierarchy(reference_leaves, reference_leaves)

    state = FoldState(
        seed=seed,
        fold=fold,
        train=train,
        test=test,
        inner_scores=np.asarray(reference_stages["score"], dtype=np.float64),
        outer_scores=np.zeros(len(test), dtype=np.float64),
        reference=reference,
        bundles=bundles,
        feature_names=feature_names,
    )
    state.outer_scores = state.score(X[test])
    return state


def verify_against_reference_implementation(
    state: FoldState, X: FloatArray, improvement: FloatArray
) -> dict[str, float | bool]:
    """Check the split state reproduces the shared scoring function exactly."""
    from atopix_ml.strict_recovered_router import strict_fold_scores

    train, test, inner, outer = strict_fold_scores(
        X, improvement, seed=state.seed, fold=state.fold
    )
    return {
        "train_indices_match": bool(np.array_equal(train, state.train)),
        "test_indices_match": bool(np.array_equal(test, state.test)),
        "inner_score_exact_match": bool(np.array_equal(inner, state.inner_scores)),
        "outer_score_exact_match": bool(np.array_equal(outer, state.outer_scores)),
        "max_abs_outer_difference": float(np.max(np.abs(outer - state.outer_scores))),
    }
