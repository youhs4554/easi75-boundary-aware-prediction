#!/usr/bin/env python3
"""Re-estimate the pre-specified comparison models under the same protocol.

Every comparator is fitted on the same 119 patients, the same predictor matrix,
the same 15 x 5 outer partitions, and the same inner partitions as the proposed
framework, and each takes its decision threshold from its own outer-training
inner out-of-fold predictions.  Nothing is selected on held-out performance.

Twelve models are estimated.  Publication placement — five in the main
comparison, seven in the supplementary table — is recorded per model but has no
effect on fitting, evaluation, or the multiplicity family.

One comparator, the nested-tuned multi-kernel support vector machine, searches a
fixed 168-cell kernel and hyperparameter grid inside each outer-training
partition.  It is the only comparator that receives an inner tuning loop; that
asymmetry is deliberate and is recorded in its roster entry.

Usage:
    python run_comparators.py --root <repo> --output-dir <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
import platform
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC

from atopix_ml.comparison_models import fit_comparison_model
from atopix_ml.repro_data import array_sha256, load_cohort
from atopix_ml.strict_proposed import OUTER_SEEDS, regression_stratified_splits
from atopix_ml.strict_recovered_router import classification_metrics
from atopix_ml.v6_reanalysis.thresholds import select_highest_observed_youden

NESTED_SVM: Final = "nested_tuned_multikernel_svm"

#: model id -> (printed name, publication placement, notes)
ROSTER: Final = {
    "conventional_logistic_c1": ("Logistic regression", "main", ""),
    "random_forest": ("Random forest", "main", ""),
    "rbf_svm": ("Support vector machine (RBF kernel)", "main", ""),
    "lightgbm": ("LightGBM", "main", ""),
    "xgboost": ("XGBoost", "main", ""),
    "linear_svm": ("Linear SVM", "supplementary", ""),
    NESTED_SVM: (
        "Nested-tuned multi-kernel SVM",
        "supplementary",
        "the only comparator given an inner hyperparameter search (168 fixed cells)",
    ),
    "adaboosted_logistic": ("AdaBoosted logistic regression", "supplementary", ""),
    "extremely_randomized_trees": ("Extremely randomized trees", "supplementary", ""),
    "gaussian_naive_bayes": ("Gaussian naive Bayes", "supplementary", ""),
    "knn": ("k-nearest neighbours", "supplementary", ""),
    "single_multitask_learner": ("Single multi-task learner", "supplementary", ""),
}
MODEL_IDS: Final = (
    "conventional_logistic_c1",
    "rbf_svm",
    "random_forest",
    "lightgbm",
    "xgboost",
    "single_multitask_learner",
)


def canonical_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        + b"\n"
    )


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_bytes(payload))
    temporary.replace(path)


def svm_candidate_configs() -> list[dict[str, Any]]:
    """The fixed 168-cell kernel and hyperparameter grid, in fixed order."""
    configs: list[dict[str, Any]] = []
    for class_weight in (None, "balanced"):
        for c_value in (0.01, 0.1, 1.0, 10.0, 100.0):
            configs.append({"kernel": "linear", "C": c_value, "class_weight": class_weight})
    for class_weight in (None, "balanced"):
        for c_value in (0.1, 1.0, 10.0, 100.0):
            for gamma in ("scale", 0.01, 0.1, 1.0):
                configs.append(
                    {
                        "kernel": "rbf",
                        "C": c_value,
                        "gamma": gamma,
                        "class_weight": class_weight,
                    }
                )
    for class_weight in (None, "balanced"):
        for c_value in (0.1, 1.0, 10.0):
            for gamma in ("scale", 0.01, 0.1):
                for degree in (2, 3):
                    for coef0 in (0.0, 1.0):
                        configs.append(
                            {
                                "kernel": "poly",
                                "C": c_value,
                                "gamma": gamma,
                                "degree": degree,
                                "coef0": coef0,
                                "class_weight": class_weight,
                            }
                        )
    for class_weight in (None, "balanced"):
        for c_value in (0.1, 1.0, 10.0):
            for gamma in ("scale", 0.01, 0.1):
                for coef0 in (-1.0, 0.0, 1.0):
                    configs.append(
                        {
                            "kernel": "sigmoid",
                            "C": c_value,
                            "gamma": gamma,
                            "coef0": coef0,
                            "class_weight": class_weight,
                        }
                    )
    return configs


def _svm_pipeline(config: dict[str, Any], *, random_state: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            (
                "svc",
                SVC(
                    **config,
                    probability=False,
                    random_state=random_state,
                    cache_size=500,
                    max_iter=10_000,
                ),
            ),
        ]
    )


def _calibrated_svm(config: dict[str, Any], *, random_state: int) -> Any:
    return CalibratedClassifierCV(
        _svm_pipeline(config, random_state=random_state), method="sigmoid", cv=3
    )


def _select_svm_config(
    X: np.ndarray, labels: np.ndarray, improvement: np.ndarray, *, seed: int
) -> tuple[dict[str, Any], float]:
    configs = svm_candidate_configs()
    prepared = []
    for train, validation in regression_stratified_splits(improvement, seed=seed):
        imputer = SimpleImputer(strategy="median")
        scaler = RobustScaler()
        X_train = scaler.fit_transform(imputer.fit_transform(X[train]))
        X_validation = scaler.transform(imputer.transform(X[validation]))
        prepared.append((X_train, labels[train], X_validation, labels[validation]))
    best_config, best_score = None, -np.inf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for config in configs:
            aurocs = []
            for X_train, y_train, X_validation, y_validation in prepared:
                model = _svm_pipeline(config, random_state=seed)
                model.fit(X_train, y_train)
                decision = model.named_steps["svc"].decision_function(X_validation)
                aurocs.append(float(roc_auc_score(y_validation, decision)))
            mean_auroc = float(np.mean(aurocs))
            if mean_auroc > best_score:
                best_config, best_score = dict(config), mean_auroc
    if best_config is None:
        raise RuntimeError("the SVM candidate grid produced no fitted configuration")
    return best_config, best_score


def _fit_scores(
    model_id: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    improvement_train: np.ndarray,
    X_score: np.ndarray,
    feature_names: tuple[str, ...],
    *,
    seed: int,
    svm_config: dict[str, Any] | None = None,
) -> np.ndarray:
    if model_id == NESTED_SVM:
        if svm_config is None:
            raise RuntimeError("the nested SVM needs a configuration selected on training data")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = _calibrated_svm(svm_config, random_state=seed)
            model.fit(X_train, y_train)
            return np.asarray(model.predict_proba(X_score)[:, 1], dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = fit_comparison_model(
            model_id,
            X_train,
            y_train,
            improvement_train,
            feature_names=feature_names,
        )
        return np.asarray(fitted.predict_proba(X_score)[:, 1], dtype=np.float64)


def run_fold(
    model_id: str,
    X: np.ndarray,
    labels: np.ndarray,
    improvement: np.ndarray,
    feature_names: tuple[str, ...],
    seed: int,
    fold: int,
) -> dict[str, Any]:
    # Parallelism belongs to the fold pool, not to the estimators inside a fold.
    # Left unpinned, the tree bags request their own threads per member and the
    # host is oversubscribed by more than an order of magnitude.
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = "1"
    started = time.monotonic()
    train, test = regression_stratified_splits(improvement, seed=seed)[fold]
    X_train, y_train = X[train], labels[train]
    improvement_train = improvement[train]
    inner_seed = seed + 1000

    svm_config, svm_selection_score = None, None
    if model_id == NESTED_SVM:
        svm_config, svm_selection_score = _select_svm_config(
            X_train, y_train, improvement_train, seed=inner_seed
        )

    inner_scores = np.full(len(train), np.nan, dtype=np.float64)
    visits = np.zeros(len(train), dtype=np.int64)
    for inner_train, inner_validation in regression_stratified_splits(
        improvement_train, seed=inner_seed
    ):
        inner_scores[inner_validation] = _fit_scores(
            model_id,
            X_train[inner_train],
            y_train[inner_train],
            improvement_train[inner_train],
            X_train[inner_validation],
            feature_names,
            seed=seed,
            svm_config=svm_config,
        )
        visits[inner_validation] += 1
    if not np.all(visits == 1) or not np.isfinite(inner_scores).all():
        raise RuntimeError(f"inner out-of-fold scores are incomplete: {model_id}/{seed}/{fold}")

    outer_scores = _fit_scores(
        model_id,
        X_train,
        y_train,
        improvement_train,
        X[test],
        feature_names,
        seed=seed,
        svm_config=svm_config,
    )
    if outer_scores.shape != (len(test),) or not np.isfinite(outer_scores).all():
        raise RuntimeError(f"held-out scores are invalid: {model_id}/{seed}/{fold}")

    selection = select_highest_observed_youden(inner_scores, y_train)
    decisions = (outer_scores >= selection.threshold).astype(np.int64)
    return {
        "schema_version": "1.0",
        "model_id": model_id,
        "seed": seed,
        "fold": fold,
        "test_indices": test.tolist(),
        "test_index_sha256": array_sha256(test, ">i8"),
        "inner_score_sha256_float64": array_sha256(inner_scores, ">f8"),
        "outer_scores": outer_scores.tolist(),
        "outer_score_sha256_float64": array_sha256(outer_scores, ">f8"),
        "threshold": selection.threshold,
        "inner_youden": selection.youden,
        "inner_sensitivity": selection.sensitivity,
        "inner_specificity": selection.specificity,
        "threshold_tie_count": selection.tie_count,
        "outer_decisions": decisions.tolist(),
        "svm_selected_config": svm_config,
        "svm_selection_mean_inner_auroc": svm_selection_score,
        "wall_seconds": time.monotonic() - started,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--seed-count", type=int, default=15)
    parser.add_argument("--models", nargs="+", default=list(MODEL_IDS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = args.data.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cohort = load_cohort(data_path)
    seeds = tuple(OUTER_SEEDS[: args.seed_count])
    labels, improvement, X = cohort.labels, cohort.improvement, cohort.X
    models = tuple(args.models)
    unknown = [model for model in models if model not in ROSTER]
    if unknown:
        raise RuntimeError(f"unknown comparators requested: {unknown}")

    config = {
        "schema_version": "1.0",
        "estimate": "comparison_models",
        "specification": cohort.specification,
        "count_fill": "zero",
        "protocol": f"{len(seeds)} regression-stratified seeds x 5 outer folds; n=119",
        "outer_seeds": list(seeds),
        "inner_seed_offset": 1000,
        "operating_point": (
            "highest observed threshold maximizing the Youden index on each "
            "outer-training partition's inner out-of-fold predictions"
        ),
        "aggregation": (
            "discrimination from the mean of the 15 held-out scores; classification "
            "from the majority of the 15 held-out decisions"
        ),
        "fairness_clauses": [
            "same patients, same predictor matrix, same outer and inner partitions",
            "one pre-declared configuration per comparator; nothing chosen on held-out data",
            "thresholds selected inside the outer-training partition only",
            "publication placement does not affect fitting, evaluation, or multiplicity",
        ],
        # The run identity covers the whole roster, not the subset a given
        # process happens to fit, so that fitting the models one at a time
        # produces shards belonging to a single run.
        "roster": {
            model_id: {
                "print_name": ROSTER[model_id][0],
                "placement": ROSTER[model_id][1],
                "note": ROSTER[model_id][2],
            }
            for model_id in MODEL_IDS
        },
        "excluded_models": {
            "tabular_foundation_model": "excluded from the comparison family by instruction",
            "mlp": "not part of the pre-specified comparator family",
        },
        "nested_svm_candidate_count": len(svm_candidate_configs()),
        "feature_names": list(cohort.feature_names),
        "feature_matrix_sha256_float64": cohort.feature_sha256,
        "label_sha256_int64": cohort.label_sha256,
        "improvement_sha256_float64": cohort.improvement_sha256,
        "dataset_files": cohort.source_files,
        "boundary_definition": "65 <= observed EASI improvement < 85",
        "boundary_status": (
            "retrospective outcome stratum defined by the observed result; not "
            "identifiable from baseline information"
        ),
    }
    config_hash = hashlib.sha256(canonical_bytes(config)).hexdigest()
    config["config_sha256"] = config_hash
    config_path = output_dir / "config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text())
        if existing.get("config_sha256") != config_hash:
            raise RuntimeError(f"output directory holds a different run: {config_path}")
    atomic_json(config_path, config)

    payloads: dict[tuple[str, int, int], dict[str, Any]] = {}
    pending: list[tuple[str, int, int]] = []
    for model_id in models:
        for seed in seeds:
            for fold in range(5):
                shard = output_dir / "shards" / model_id / f"seed-{seed}-fold-{fold}.json"
                if shard.exists():
                    payload = json.loads(shard.read_text())
                    if payload.get("config_sha256") != config_hash:
                        raise RuntimeError(f"stale shard: {shard}")
                    payloads[(model_id, seed, fold)] = payload
                else:
                    pending.append((model_id, seed, fold))

    started = time.monotonic()
    completed = 0
    # A forked worker inherits the parent's OpenMP runtime state, and several
    # comparators (the boosted-tree and bagged-tree bags) start their own thread
    # pools inside that worker.  That combination deadlocks: workers spin at full
    # CPU and never return a fold.  Spawning gives each worker a clean runtime.
    with ProcessPoolExecutor(
        max_workers=args.jobs, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        futures = {
            executor.submit(
                run_fold, model_id, X, labels, improvement, cohort.feature_names, seed, fold
            ): (model_id, seed, fold)
            for model_id, seed, fold in pending
        }
        for future in as_completed(futures):
            model_id, seed, fold = futures[future]
            payload = future.result()
            payload["config_sha256"] = config_hash
            atomic_json(
                output_dir / "shards" / model_id / f"seed-{seed}-fold-{fold}.json", payload
            )
            payloads[(model_id, seed, fold)] = payload
            completed += 1
            if completed % 25 == 0 or payload["wall_seconds"] > 30:
                print(
                    f"[progress] {completed}/{len(pending)} last={model_id} "
                    f"s={payload['wall_seconds']:.1f}",
                    flush=True,
                )

    n = len(labels)
    boundary = cohort.boundary
    rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    patient_columns: dict[str, np.ndarray] = {
        "patient_index": np.arange(n),
        "y_easi75": labels,
        "easi_improvement_pct": improvement,
        "boundary_65_to_lt85": boundary,
    }
    for model_id in models:
        score_matrix = np.full((n, len(seeds)), np.nan)
        decision_matrix = np.full((n, len(seeds)), -1, dtype=np.int64)
        visits = np.zeros((n, len(seeds)), dtype=np.int64)
        kernels: list[str] = []
        for column, seed in enumerate(seeds):
            for fold in range(5):
                payload = payloads[(model_id, seed, fold)]
                test = np.asarray(payload["test_indices"], dtype=np.int64)
                score_matrix[test, column] = payload["outer_scores"]
                decision_matrix[test, column] = payload["outer_decisions"]
                visits[test, column] += 1
                threshold_rows.append(
                    {
                        "model_id": model_id,
                        "seed": seed,
                        "fold": fold,
                        "threshold": payload["threshold"],
                        "inner_youden": payload["inner_youden"],
                        "selected_kernel": (payload["svm_selected_config"] or {}).get("kernel"),
                    }
                )
                if payload["svm_selected_config"]:
                    kernels.append(payload["svm_selected_config"]["kernel"])
        if not np.all(visits == 1):
            raise RuntimeError(f"incomplete patient coverage for {model_id}")
        patient_score = score_matrix.mean(axis=1)
        vote_count = (decision_matrix == 1).sum(axis=1)
        patient_decision = (vote_count * 2 > len(seeds)).astype(np.int64)
        patient_columns[f"{model_id}_patient_average_score"] = patient_score
        patient_columns[f"{model_id}_vote_count"] = vote_count
        patient_columns[f"{model_id}_decision"] = patient_decision

        overall = classification_metrics(labels, patient_decision)
        boundary_metrics = classification_metrics(labels[boundary], patient_decision[boundary])
        rows.append(
            {
                "model_id": model_id,
                "print_name": ROSTER[model_id][0],
                "placement": ROSTER[model_id][1],
                "decision_mechanism": "fold-local inner out-of-fold Youden threshold",
                "auroc": float(roc_auc_score(labels, patient_score)),
                "auprc": float(average_precision_score(labels, patient_score)),
                "brier": float(brier_score_loss(labels, patient_score)),
                "boundary_auroc": float(
                    roc_auc_score(labels[boundary], patient_score[boundary])
                ),
                **{f"overall_{k}": v for k, v in overall.items()},
                **{f"boundary_{k}": v for k, v in boundary_metrics.items()},
                "selected_kernel_counts": json.dumps(
                    {kernel: kernels.count(kernel) for kernel in sorted(set(kernels))}
                )
                if kernels
                else "",
                "patient_average_score_sha256_float64": array_sha256(patient_score, ">f8"),
            }
        )
        print(f"[model] {model_id} auroc={rows[-1]['auroc']:.6f}", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "model_metrics.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(output_dir / "fold_thresholds.csv", index=False)
    pd.DataFrame(patient_columns).to_parquet(
        output_dir / "patient_scores_and_decisions.parquet", index=False
    )
    atomic_json(
        output_dir / "metrics.json",
        {
            "schema_version": "1.0",
            "config_sha256": config_hash,
            "specification": cohort.specification,
            "models": rows,
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "packages": {
                    name: importlib.metadata.version(name)
                    for name in ("numpy", "pandas", "scikit-learn", "scipy", "lightgbm", "xgboost")
                },
                "created_at_utc": datetime.now(UTC).isoformat(),
                "wall_seconds": time.monotonic() - started,
            },
        },
    )
    print(f"[done] models={len(rows)} seconds={time.monotonic() - started:.1f}", flush=True)


if __name__ == "__main__":
    main()
