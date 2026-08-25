#!/usr/bin/env python3
"""Re-estimate the proposed framework under the established protocol.

Protocol, unchanged from the preserved estimate: 15 outer seeds x 5
regression-stratified outer folds (75 fits), inner-seed offset 1000, operating
points selected inside the outer-training partition only, patient-average score
for discrimination and patient-level majority vote for classification.

Each fold additionally carries a leakage probe.  Every learned quantity of the
fold is recomputed a second time from outcome vectors whose held-out entries
have been replaced with corrupted values, holding the outer partition fixed.
Because the fitting path indexes only the training rows, every hash must be
identical; a difference would mean a held-out outcome reached a fitted quantity.

Usage:
    python run_proposed.py --root <repo> --output-dir <dir> --specification allergens
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import time
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
from rerun_scoring import fold_state, verify_against_reference_implementation
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from atopix_ml.repro_data import array_sha256, load_cohort
from atopix_ml.strict_proposed import OUTER_SEEDS, regression_stratified_splits
from atopix_ml.strict_recovered_router import (
    EASI_BINS,
    PRT_HIGH_GRID,
    PRT_LOW_GRID,
    apply_bab,
    apply_prt_midpoint,
    classification_metrics,
    fit_bab,
    select_operating_points,
    select_prt_band,
)

POLICIES: Final = (
    "youden",
    "fixed_0_5",
    "prevalence_match",
    "max_f1",
    "prt_midpoint",
    "prt_bab",
)
THRESHOLD_POLICIES: Final = ("youden", "fixed_0_5", "prevalence_match", "max_f1")
CALIBRATION_MAPS: Final = ("platt", "isotonic")
BIN_LABELS: Final = ("0_to_lt50", "50_to_lt65", "65_to_lt75", "75_to_lt85", "85_plus")


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


def _fit_calibration_maps(scores: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    return {
        "isotonic": IsotonicRegression(
            y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip"
        ).fit(scores, labels),
        "platt": LogisticRegression(C=1e6, solver="lbfgs", max_iter=10000).fit(
            scores.reshape(-1, 1), labels
        ),
    }


def _apply_calibration_map(name: str, model: Any, scores: np.ndarray) -> np.ndarray:
    if name == "isotonic":
        return np.asarray(model.predict(scores), dtype=np.float64)
    return np.asarray(model.predict_proba(scores.reshape(-1, 1))[:, 1], dtype=np.float64)


def _learned_quantities(
    X: np.ndarray,
    labels: np.ndarray,
    improvement: np.ndarray,
    feature_names: tuple[str, ...],
    seed: int,
    fold: int,
    split: tuple[np.ndarray, np.ndarray],
) -> dict[str, Any]:
    """Every quantity of one fold that is learned from data, plus its outputs."""
    state = fold_state(
        X,
        improvement,
        seed=seed,
        fold=fold,
        feature_names=feature_names,
        split=split,
    )
    train, test = state.train, state.test
    labels_train = labels[train]
    thresholds = select_operating_points(state.inner_scores, labels_train)
    decisions = {
        policy: (state.outer_scores >= float(thresholds[policy]["threshold"])).astype(np.int64)
        for policy in THRESHOLD_POLICIES
    }
    prt = select_prt_band(state.inner_scores, labels_train, improvement[train])
    midpoint = apply_prt_midpoint(state.outer_scores, prt.low, prt.high)
    decisions["prt_midpoint"] = midpoint
    bab_fit = fit_bab(X[train], labels_train, state.inner_scores, prt.low, prt.high)
    decisions["prt_bab"], n_test_midband = apply_bab(
        midpoint, state.outer_scores, X[test], prt.low, prt.high, bab_fit
    )

    maps = _fit_calibration_maps(state.inner_scores, labels_train)
    calibrated = {
        name: _apply_calibration_map(name, model, state.outer_scores)
        for name, model in maps.items()
    }
    inner_calibrated = {
        name: _apply_calibration_map(name, model, state.inner_scores)
        for name, model in maps.items()
    }
    for name, values in calibrated.items():
        ordered = values[np.argsort(state.outer_scores, kind="stable")]
        if not bool(np.all(np.diff(ordered) >= -1e-12)):
            raise RuntimeError(f"calibration map is not non-decreasing: {name}/{seed}/{fold}")

    return {
        "state": state,
        "thresholds": thresholds,
        "prt": prt,
        "n_test_midband": n_test_midband,
        "bab_fit": bab_fit,
        "decisions": decisions,
        "calibrated": calibrated,
        "inner_calibrated": inner_calibrated,
        "fingerprint": {
            "inner_scores": array_sha256(state.inner_scores, ">f8"),
            "outer_scores": array_sha256(state.outer_scores, ">f8"),
            "thresholds": hashlib.sha256(
                canonical_bytes({k: v["threshold"] for k, v in thresholds.items()})
            ).hexdigest(),
            "prt_bounds": hashlib.sha256(
                canonical_bytes([prt.low, prt.high, prt.objective])
            ).hexdigest(),
            "decisions": hashlib.sha256(
                canonical_bytes({k: v.tolist() for k, v in decisions.items()})
            ).hexdigest(),
            "calibrated": hashlib.sha256(
                canonical_bytes({k: v.tolist() for k, v in calibrated.items()})
            ).hexdigest(),
        },
    }


def _corrupt_held_out(
    labels: np.ndarray, improvement: np.ndarray, test: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Replace held-out outcome values with values that cannot be the truth."""
    rng = np.random.default_rng(9_000_000 + seed)
    corrupted_labels = labels.copy()
    corrupted_improvement = improvement.copy()
    corrupted_labels[test] = 1 - labels[test]
    corrupted_improvement[test] = rng.uniform(-500.0, -100.0, size=len(test))
    return corrupted_labels, corrupted_improvement


def run_fold(
    X: np.ndarray,
    labels: np.ndarray,
    improvement: np.ndarray,
    feature_names: tuple[str, ...],
    seed: int,
    fold: int,
    *,
    verify_reference: bool,
) -> dict[str, Any]:
    started = time.monotonic()
    split = regression_stratified_splits(improvement, seed=seed)[fold]
    primary = _learned_quantities(
        X, labels, improvement, feature_names, seed, fold, split
    )
    state = primary["state"]
    train, test = state.train, state.test

    corrupted_labels, corrupted_improvement = _corrupt_held_out(
        labels, improvement, test, seed
    )
    probe = _learned_quantities(
        X, corrupted_labels, corrupted_improvement, feature_names, seed, fold, split
    )
    leakage_probe = {
        key: primary["fingerprint"][key] == probe["fingerprint"][key]
        for key in primary["fingerprint"]
    }
    leakage_probe["all_learned_quantities_independent_of_held_out_outcome"] = all(
        leakage_probe.values()
    )

    reference_parity = (
        verify_against_reference_implementation(state, X, improvement)
        if verify_reference
        else None
    )

    prt = primary["prt"]
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "seed": seed,
        "fold": fold,
        "train_indices": train.tolist(),
        "test_indices": test.tolist(),
        "test_index_sha256": array_sha256(test, ">i8"),
        "inner_score_sha256_float64": array_sha256(state.inner_scores, ">f8"),
        "outer_score_sha256_float64": array_sha256(state.outer_scores, ">f8"),
        "outer_scores": state.outer_scores.tolist(),
        "thresholds": primary["thresholds"],
        "prt": {
            "low": prt.low,
            "high": prt.high,
            "accuracy": prt.accuracy,
            "bin_accuracy_sd": prt.bin_accuracy_sd,
            "objective": prt.objective,
            "alpha": 0.5,
        },
        "band": {
            "low": prt.low,
            "high": prt.high,
            "midpoint": float((prt.low + prt.high) / 2.0),
            "n_test_midband": primary["n_test_midband"],
        },
        "bab": {
            "status": primary["bab_fit"].status,
            "n_train_midband": primary["bab_fit"].n_midband,
            "n_train_midband_positive": primary["bab_fit"].n_midband_positive,
            "n_test_midband": primary["n_test_midband"],
            "n_test_decisions_changed_vs_midpoint": int(
                np.count_nonzero(
                    primary["decisions"]["prt_bab"]
                    != primary["decisions"]["prt_midpoint"]
                )
            ),
        },
        "outer_decisions": {
            policy: values.tolist() for policy, values in primary["decisions"].items()
        },
        "calibrated_outer_probabilities": {
            name: values.tolist() for name, values in primary["calibrated"].items()
        },
        "calibration_training_brier": {
            name: float(brier_score_loss(labels[train], primary["inner_calibrated"][name]))
            for name in CALIBRATION_MAPS
        },
        "fold_score_auroc": float(roc_auc_score(labels[test], state.outer_scores)),
        "leakage_probe": leakage_probe,
        "reference_implementation_parity": reference_parity,
        "wall_seconds": time.monotonic() - started,
    }
    return payload


def bootstrap_auroc(
    labels: np.ndarray,
    scores: np.ndarray,
    reference: np.ndarray | None = None,
    *,
    draws: int = 10_000,
    seed: int = 42,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(draws):
        indices = rng.integers(0, len(labels), len(labels))
        if np.unique(labels[indices]).size != 2:
            continue
        value = roc_auc_score(labels[indices], scores[indices])
        if reference is not None:
            value -= roc_auc_score(labels[indices], reference[indices])
        estimates.append(float(value))
    return tuple(float(v) for v in np.quantile(estimates, [0.025, 0.975]))


def bootstrap_accuracy(
    labels: np.ndarray,
    predicted: np.ndarray,
    reference: np.ndarray | None = None,
    *,
    draws: int = 10_000,
    seed: int = 42,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(draws):
        indices = rng.integers(0, len(labels), len(labels))
        value = float(np.mean(labels[indices] == predicted[indices]))
        if reference is not None:
            value -= float(np.mean(labels[indices] == reference[indices]))
        estimates.append(value)
    return tuple(float(v) for v in np.quantile(estimates, [0.025, 0.975]))


def expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, bins: int = 5
) -> tuple[float, list[dict[str, Any]]]:
    """Equal-count (quantile) bins; the bin definition is reported with the value."""
    order = np.argsort(probabilities, kind="stable")
    groups = np.array_split(order, bins)
    total = len(labels)
    error = 0.0
    rows: list[dict[str, Any]] = []
    for index, group in enumerate(groups):
        if len(group) == 0:
            continue
        mean_predicted = float(probabilities[group].mean())
        observed = float(labels[group].mean())
        error += len(group) / total * abs(observed - mean_predicted)
        rows.append(
            {
                "bin": index,
                "n": int(len(group)),
                "low": float(probabilities[group].min()),
                "high": float(probabilities[group].max()),
                "mean_predicted": mean_predicted,
                "observed_rate": observed,
            }
        )
    return float(error), rows


def calibration_fit(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    """Slope and intercept of ``logit(y) ~ a + b logit(p)``, with 95% intervals."""
    eps = 1e-12
    clipped = np.clip(probabilities, eps, 1.0 - eps)
    logit = np.log(clipped / (1.0 - clipped))
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=10000).fit(
        logit.reshape(-1, 1), labels
    )
    slope = float(model.coef_[0][0])
    intercept = float(model.intercept_[0])
    rng = np.random.default_rng(42)
    slopes, intercepts = [], []
    for _ in range(2000):
        indices = rng.integers(0, len(labels), len(labels))
        if np.unique(labels[indices]).size != 2:
            continue
        try:
            resampled = LogisticRegression(C=1e6, solver="lbfgs", max_iter=10000).fit(
                logit[indices].reshape(-1, 1), labels[indices]
            )
        except ValueError:
            continue
        slopes.append(float(resampled.coef_[0][0]))
        intercepts.append(float(resampled.intercept_[0]))
    return {
        "calibration_slope": slope,
        "calibration_slope_ci_low": float(np.quantile(slopes, 0.025)),
        "calibration_slope_ci_high": float(np.quantile(slopes, 0.975)),
        "calibration_intercept": intercept,
        "calibration_intercept_ci_low": float(np.quantile(intercepts, 0.025)),
        "calibration_intercept_ci_high": float(np.quantile(intercepts, 0.975)),
        "mean_predicted": float(probabilities.mean()),
        "observed_rate": float(labels.mean()),
        "calibration_in_the_large": float(probabilities.mean() - labels.mean()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--leaf-cutoffs",
        default="reported",
        help="which improvement cut-offs each learner is fitted at: 'reported' for the "
        "published set, 'single' for 75%% alone, 'three' for 65/75/85, or an explicit "
        "comma-separated list such as '65,70,75,80,85'. Everything else about the "
        "architecture is unchanged, so a difference between values isolates the cut-offs.",
    )
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--seed-count", type=int, default=15)
    parser.add_argument("--skip-reference-parity", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = args.data.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    # The cut-off table is a module constant, so an arm is set by rebinding it before
    # anything is fitted. The value used is recorded in the run's config.
    from atopix_ml import documented_proposed, recovered_v18_model
    if args.leaf_cutoffs != "reported":
        named = {"single": (75,), "three": (65, 75, 85)}
        replacement = named.get(args.leaf_cutoffs)
        if replacement is None:
            replacement = tuple(int(v) for v in args.leaf_cutoffs.split(","))
            if not replacement or any(not 0 < v < 100 for v in replacement):
                raise SystemExit(f"unusable cut-off list: {args.leaf_cutoffs!r}")
        table = {leaf: replacement for leaf in documented_proposed.LEAF_THRESHOLDS}
        documented_proposed.LEAF_THRESHOLDS = table
        recovered_v18_model.LEAF_THRESHOLDS = table

    cohort = load_cohort(data_path)
    seeds = tuple(OUTER_SEEDS[: args.seed_count])
    labels, improvement, X = cohort.labels, cohort.improvement, cohort.X
    # The legacy specification exists to prove the harness reproduces the
    # preserved estimate; only there is a per-fold parity check meaningful.
    verify_reference = False

    config = {
        "schema_version": "1.0",
        "estimate": "proposed_framework",
        "specification": cohort.specification,
        "count_fill": "zero",
        "leaf_cutoffs": args.leaf_cutoffs,
        "leaf_threshold_table": {
            leaf: list(values)
            for leaf, values in recovered_v18_model.LEAF_THRESHOLDS.items()
        },
        "protocol": f"{len(seeds)} regression-stratified seeds x 5 outer folds; n=119",
        "outer_seeds": list(seeds),
        "inner_seed_offset": 1000,
        "aggregation": (
            "discrimination from the mean of the 15 held-out scores; classification "
            "from the majority of the 15 held-out decisions"
        ),
        "primary_operating_point": "prt_bab",
        "comparator_matched_operating_point": "youden",
        "sensitivity_policies": [p for p in POLICIES if p not in ("prt_bab", "youden")],
        "prt": {
            "low_grid": list(PRT_LOW_GRID),
            "high_grid": list(PRT_HIGH_GRID),
            "objective": "accuracy - 0.5 * SD(five observed-improvement-bin accuracies)",
            "tie_policy": "first lexicographic low/high pair",
            "bounds": "inclusive ambiguity band; the boundary corrector replaces midpoint labels inside the selected band",
        },
        "calibration": {
            "maps": list(CALIBRATION_MAPS),
            "primary_map": None,
            "primary_map_note": (
                "the reported estimate is the raw score; neither map is preferred, and "
                "both are reported symmetrically"
            ),
            "contract": (
                "fitted on outer-training inner out-of-fold scores and outer-training "
                "labels only, then applied unchanged to held-out scores"
            ),
            "ece_bins": "5 equal-count bins",
        },
        "boundary_definition": "65 <= observed EASI improvement < 85",
        "boundary_status": (
            "retrospective outcome stratum defined by the observed result; not "
            "identifiable from baseline information"
        ),
        "leakage_control": (
            "held-out labels and held-out observed improvement never enter score "
            "fitting, operating-point selection, ambiguity-band selection "
            "or calibration-map fitting; each fold carries a probe that "
            "recomputes every learned quantity from corrupted held-out outcomes and "
            "requires identical hashes"
        ),
        "feature_names": list(cohort.feature_names),
        "n_features": len(cohort.feature_names),
        "feature_matrix_sha256_float64": cohort.feature_sha256,
        "label_sha256_int64": cohort.label_sha256,
        "improvement_sha256_float64": cohort.improvement_sha256,
        "dataset_files": cohort.source_files,
        "missing_by_feature": cohort.missing_by_feature,
    }
    config_hash = hashlib.sha256(canonical_bytes(config)).hexdigest()
    config["config_sha256"] = config_hash
    config_path = output_dir / "config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text())
        if existing.get("config_sha256") != config_hash:
            raise RuntimeError(f"output directory holds a different run: {config_path}")
    atomic_json(config_path, config)

    payloads: dict[tuple[int, int], dict[str, Any]] = {}
    pending: list[tuple[int, int]] = []
    for seed in seeds:
        for fold in range(5):
            shard = output_dir / "shards" / f"seed-{seed}-fold-{fold}.json"
            if shard.exists():
                payload = json.loads(shard.read_text())
                if payload.get("config_sha256") != config_hash:
                    raise RuntimeError(f"stale shard: {shard}")
                payloads[(seed, fold)] = payload
            else:
                pending.append((seed, fold))

    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(
                run_fold,
                X,
                labels,
                improvement,
                cohort.feature_names,
                seed,
                fold,
                verify_reference=verify_reference,
            ): (seed, fold)
            for seed, fold in pending
        }
        for future in as_completed(futures):
            seed, fold = futures[future]
            payload = future.result()
            payload["config_sha256"] = config_hash
            atomic_json(output_dir / "shards" / f"seed-{seed}-fold-{fold}.json", payload)
            payloads[(seed, fold)] = payload
            print(
                f"[fold] seed={seed} fold={fold} "
                f"auroc={payload['fold_score_auroc']:.4f} "
                f"prt={payload['prt']['low']:.2f}/{payload['prt']['high']:.2f} "
                f"band={payload['band']['midpoint']:.3f} "
                f"probe={payload['leakage_probe']['all_learned_quantities_independent_of_held_out_outcome']} "
                f"s={payload['wall_seconds']:.1f}",
                flush=True,
            )

    n = len(labels)
    score_matrix = np.full((n, len(seeds)), np.nan)
    calibrated_matrices = {
        name: np.full((n, len(seeds)), np.nan) for name in CALIBRATION_MAPS
    }
    decision_matrices = {policy: np.full((n, len(seeds)), -1, dtype=np.int64) for policy in POLICIES}
    visits = np.zeros((n, len(seeds)), dtype=np.int64)
    fold_rows: list[dict[str, Any]] = []
    operating_point_rows: list[dict[str, Any]] = []
    for column, seed in enumerate(seeds):
        for fold in range(5):
            payload = payloads[(seed, fold)]
            test = np.asarray(payload["test_indices"], dtype=np.int64)
            score_matrix[test, column] = payload["outer_scores"]
            for name in CALIBRATION_MAPS:
                calibrated_matrices[name][test, column] = payload[
                    "calibrated_outer_probabilities"
                ][name]
            for policy in POLICIES:
                decision_matrices[policy][test, column] = payload["outer_decisions"][policy]
            visits[test, column] += 1
            fold_rows.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "n_test": len(test),
                    "fold_score_auroc": payload["fold_score_auroc"],
                    "prt_low": payload["prt"]["low"],
                    "prt_high": payload["prt"]["high"],
                    "prt_objective": payload["prt"]["objective"],
                    **payload["band"],
                    **payload["bab"],
                    "leakage_probe_passed": payload["leakage_probe"][
                        "all_learned_quantities_independent_of_held_out_outcome"
                    ],
                    "wall_seconds": payload["wall_seconds"],
                }
            )
            for policy in THRESHOLD_POLICIES:
                operating_point_rows.append(
                    {"seed": seed, "fold": fold, "policy": policy, **payload["thresholds"][policy]}
                )
    if not np.all(visits == 1) or not np.isfinite(score_matrix).all():
        raise RuntimeError("the held-out ledger is incomplete")

    patient_average = score_matrix.mean(axis=1)
    majority = {
        policy: (matrix.sum(axis=1) >= (len(seeds) // 2 + 1)).astype(np.int64)
        for policy, matrix in decision_matrices.items()
    }
    boundary = cohort.boundary
    overall_auroc = float(roc_auc_score(labels, patient_average))
    ci_low, ci_high = bootstrap_auroc(labels, patient_average)

    policy_results = []
    interval_rows = []
    for policy in POLICIES:
        decision = majority[policy]
        overall = classification_metrics(labels, decision)
        boundary_metrics = classification_metrics(labels[boundary], decision[boundary])
        overall_ci = bootstrap_accuracy(labels, decision)
        boundary_ci = bootstrap_accuracy(labels[boundary], decision[boundary])
        policy_results.append(
            {
                "policy": policy,
                "role": (
                    "primary"
                    if policy == "prt_bab"
                    else "comparator_matched"
                    if policy == "youden"
                    else "sensitivity"
                ),
                "overall_auroc_threshold_invariant": overall_auroc,
                **{f"overall_{k}": v for k, v in overall.items()},
                **{f"boundary_{k}": v for k, v in boundary_metrics.items()},
                "overall_accuracy_ci_low": overall_ci[0],
                "overall_accuracy_ci_high": overall_ci[1],
                "boundary_accuracy_ci_low": boundary_ci[0],
                "boundary_accuracy_ci_high": boundary_ci[1],
            }
        )
        for label, (low, high) in zip(BIN_LABELS, EASI_BINS, strict=True):
            mask = (improvement >= low) & (improvement < high)
            interval_rows.append(
                {
                    "policy": policy,
                    "interval": label,
                    "low_inclusive": low,
                    "high_exclusive": high,
                    **classification_metrics(labels[mask], decision[mask]),
                }
            )

    calibration_summary: dict[str, Any] = {}
    reliability_rows: list[dict[str, Any]] = []
    estimates = {"raw_score": patient_average} | {
        name: calibrated_matrices[name].mean(axis=1) for name in CALIBRATION_MAPS
    }
    for name, values in estimates.items():
        error, rows = expected_calibration_error(labels, values)
        entry = {
            "auroc": float(roc_auc_score(labels, values)),
            "brier": float(brier_score_loss(labels, values)),
            "expected_calibration_error": error,
            **calibration_fit(labels, values),
        }
        if name != "raw_score":
            entry["boundary_brier"] = float(
                brier_score_loss(labels[boundary], values[boundary])
            )
            boundary_error, _ = expected_calibration_error(
                labels[boundary], values[boundary]
            )
            entry["boundary_expected_calibration_error"] = boundary_error
        calibration_summary[name] = entry
        for row in rows:
            reliability_rows.append({"estimate": name, **row})

    probe_summary = {
        "folds": len(payloads),
        "folds_passed": int(sum(row["leakage_probe_passed"] for row in fold_rows)),
        "all_folds_passed": bool(all(row["leakage_probe_passed"] for row in fold_rows)),
    }
    reference_parity = [
        payload["reference_implementation_parity"]
        for payload in payloads.values()
        if payload.get("reference_implementation_parity")
    ]
    metrics = {
        "schema_version": "1.0",
        "estimate": "proposed_framework",
        "specification": cohort.specification,
        "config_sha256": config_hash,
        "n_patients": int(n),
        "n_positive": int(labels.sum()),
        "n_negative": int((labels == 0).sum()),
        "n_seeds": len(seeds),
        "n_outer_folds": len(seeds) * 5,
        "discrimination": {
            "estimator": "mean of the 15 held-out scores",
            "auroc": overall_auroc,
            "auroc_ci_low": ci_low,
            "auroc_ci_high": ci_high,
            "auprc": float(average_precision_score(labels, patient_average)),
            "boundary_auroc": float(
                roc_auc_score(labels[boundary], patient_average[boundary])
            ),
            "mean_repeat_auroc": float(
                np.mean([roc_auc_score(labels, score_matrix[:, i]) for i in range(len(seeds))])
            ),
            "patient_average_score_sha256_float64": array_sha256(patient_average, ">f8"),
        },
        "classification": {
            "estimator": "majority of the 15 held-out decisions",
            "policy_results": policy_results,
        },
        "calibration": calibration_summary,
        "band_summary": {
            "boundary_corrector_fitted": True,
            "operating_point": "training-fitted boundary corrector inside the selected band",
            "midpoint_min": min(row["midpoint"] for row in fold_rows),
            "midpoint_max": max(row["midpoint"] for row in fold_rows),
            "distinct_midpoints": len({row["midpoint"] for row in fold_rows}),
            "total_test_midband_assignments": int(
                sum(row["n_test_midband"] for row in fold_rows)
            ),
            "fitted_folds": int(sum(row["status"] == "fitted" for row in fold_rows)),
            "fallback_folds": int(sum(row["status"] != "fitted" for row in fold_rows)),
            "total_test_decisions_changed_vs_midpoint": int(
                sum(row["n_test_decisions_changed_vs_midpoint"] for row in fold_rows)
            ),
        },
        "leakage_probe": probe_summary,
        # The parity check compares this harness against the preserved implementation
        # and only runs on the legacy specification, which the reported run is not. It is
        # reported with the reason rather than as a check that ran on nothing.
        "reference_implementation_parity": {
            "status": (
                "not applicable: the reference implementation covers the legacy "
                "specification only"
            )
            if not reference_parity
            else "checked",
            "checked_folds": len(reference_parity),
            "all_exact": bool(
                all(row["outer_score_exact_match"] for row in reference_parity)
            )
            if reference_parity
            else None,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "pandas", "scikit-learn", "scipy", "pyarrow")
            },
            "created_at_utc": datetime.now(UTC).isoformat(),
            "wall_seconds": time.monotonic() - started,
        },
    }

    patient_frame = {
        "patient_index": np.arange(n),
        "y_easi75": labels,
        "easi_improvement_pct": improvement,
        "boundary_65_to_lt85": boundary,
        "patient_average_score": patient_average,
    }
    for name in CALIBRATION_MAPS:
        patient_frame[f"{name}_patient_average_probability"] = estimates[name]
    for policy in POLICIES:
        patient_frame[f"{policy}_vote_count"] = decision_matrices[policy].sum(axis=1)
        patient_frame[f"{policy}_decision"] = majority[policy]
    for column, seed in enumerate(seeds):
        patient_frame[f"seed_{seed}_score"] = score_matrix[:, column]

    pd.DataFrame(fold_rows).to_csv(output_dir / "fold_summary.csv", index=False)
    pd.DataFrame(operating_point_rows).to_csv(
        output_dir / "fold_operating_points.csv", index=False
    )
    pd.DataFrame(policy_results).to_csv(output_dir / "policy_metrics.csv", index=False)
    pd.DataFrame(interval_rows).to_csv(output_dir / "interval_metrics.csv", index=False)
    pd.DataFrame(reliability_rows).to_csv(output_dir / "reliability_bins.csv", index=False)
    pd.DataFrame(patient_frame).to_parquet(
        output_dir / "patient_scores_and_decisions.parquet", index=False
    )
    atomic_json(output_dir / "metrics.json", metrics)
    print(
        f"[done] specification={cohort.specification} auroc={overall_auroc:.6f} "
        f"probe_passed={probe_summary['all_folds_passed']} "
        f"seconds={time.monotonic() - started:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
