#!/usr/bin/env python3
"""Mean absolute SHAP attribution for the re-estimated proposed framework.

Scope is deliberately narrow: the mean absolute contribution of each predictor
and the ranking that follows from it.  No stratified split, no per-patient
cases, no waterfall decomposition.

The explainer is applied to the **actual score function**, not to a surrogate
fitted to its outputs.  For a given fold the rank references are fixed to that
fold's outer-training inner out-of-fold component predictions, so the score is a
pointwise map from one patient's covariates to one number, and KernelSHAP can
perturb covariates and read the real score back.

Attribution runs on its own protocol and says so.  It uses the first ``--seeds``
of the frozen seed list in their pre-specified order — not seeds chosen by any
result — because each patient needs thousands of score evaluations.  Every
number this script writes is labelled with that protocol so it can never be
quoted as if it came from the 15-seed performance run.

Stability is measured, not assumed: the seeds are split into two disjoint halves,
attribution is aggregated within each half, and the agreement of the resulting
top-k rankings is reported.

Usage:
    python run_attribution.py --root <repo> --output-dir <dir> --seeds 4 --nsamples 256
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from rerun_scoring import fold_state
from scipy import stats

from atopix_ml.repro_data import array_sha256, load_cohort
from atopix_ml.strict_proposed import OUTER_SEEDS

BACKGROUND_ROWS = 16


def canonical_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        + b"\n"
    )


def _background(X_train: np.ndarray, *, seed: int, fold: int) -> np.ndarray:
    """A label-free deterministic subsample of the outer-training rows.

    Observed missingness is preserved: every member pipeline imputes internally,
    so the background carries the same missing-data pattern the model sees.
    """
    rng = np.random.default_rng(seed * 100 + fold)
    size = min(BACKGROUND_ROWS, len(X_train))
    chosen = rng.choice(len(X_train), size=size, replace=False)
    return X_train[np.sort(chosen)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--nsamples", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--label-map", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    import shap

    args = parse_args()
    data_path = args.data.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cohort = load_cohort(data_path)
    seeds = tuple(OUTER_SEEDS[: args.seeds])
    feature_names = list(cohort.feature_names)
    n_features = len(feature_names)

    started = time.monotonic()
    contributions = np.full((len(cohort.labels), len(seeds), n_features), np.nan)
    additivity_errors: list[float] = []
    fold_rows: list[dict[str, Any]] = []
    for column, seed in enumerate(seeds):
        for fold in range(5):
            fold_started = time.monotonic()
            state = fold_state(
                cohort.X,
                cohort.improvement,
                seed=seed,
                fold=fold,
                feature_names=cohort.feature_names,
            )
            background = _background(cohort.X[state.train], seed=seed, fold=fold)
            # KernelSHAP samples perturbation coalitions from NumPy's legacy global
            # generator. The original analysis omitted this seed, which caused small
            # run-to-run drift in absolute values. Freeze it per fold in the release.
            np.random.seed(seed * 100 + fold)
            explainer = shap.KernelExplainer(state.score, background, link="identity")
            values = explainer.shap_values(
                cohort.X[state.test],
                nsamples=args.nsamples,
                l1_reg=f"num_features({n_features})",
                silent=True,
            )
            values = np.asarray(values, dtype=np.float64)
            if values.shape != (len(state.test), n_features):
                raise RuntimeError(f"unexpected attribution shape: {values.shape}")
            contributions[state.test, column, :] = values
            reconstructed = float(explainer.expected_value) + values.sum(axis=1)
            error = float(np.max(np.abs(reconstructed - state.outer_scores)))
            additivity_errors.append(error)
            fold_rows.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "n_test": len(state.test),
                    "expected_value": float(explainer.expected_value),
                    "max_additivity_error": error,
                    "wall_seconds": time.monotonic() - fold_started,
                }
            )
            print(
                f"[attribution] seed={seed} fold={fold} "
                f"additivity_error={error:.2e} s={time.monotonic() - fold_started:.1f}",
                flush=True,
            )

    if np.isnan(contributions).any():
        raise RuntimeError("the attribution ledger is incomplete")

    per_patient = np.nanmean(contributions, axis=1)
    mean_absolute = np.abs(per_patient).mean(axis=0)
    order = np.argsort(-mean_absolute)

    label_map: dict[str, str] = {}
    if args.label_map and args.label_map.exists():
        frame = pd.read_csv(args.label_map)
        label_map = dict(zip(frame["internal_name"], frame["print_label"], strict=True))

    importance = pd.DataFrame(
        {
            "rank": np.arange(1, n_features + 1),
            "internal_name": [feature_names[i] for i in order],
            "print_label": [
                label_map.get(feature_names[i], feature_names[i]) for i in order
            ],
            "mean_abs_shap": mean_absolute[order],
            "mean_shap": per_patient.mean(axis=0)[order],
            "share_of_total_abs": (mean_absolute / mean_absolute.sum())[order],
        }
    )
    importance.to_csv(output_dir / "feature_importance.csv", index=False)

    # Stability: two disjoint halves of the seed list, aggregated separately.
    half = len(seeds) // 2
    if half < 1:
        raise RuntimeError("stability measurement needs at least two seeds")
    halves = {
        "first_half": list(range(half)),
        "second_half": list(range(half, 2 * half)),
    }
    half_importance = {}
    for name, columns in halves.items():
        subset = np.abs(np.nanmean(contributions[:, columns, :], axis=1)).mean(axis=0)
        half_importance[name] = subset
    first, second = half_importance["first_half"], half_importance["second_half"]
    first_order = list(np.argsort(-first))
    second_order = list(np.argsort(-second))
    stability_rows = []
    for k in (3, 5, 10, n_features):
        overlap = len(set(first_order[:k]) & set(second_order[:k]))
        stability_rows.append(
            {
                "top_k": k,
                "shared_members": overlap,
                "jaccard": overlap / (2 * k - overlap) if k else float("nan"),
                "overlap_fraction": overlap / k,
                # Positions that the SAME predictors occupy in the two rankings.
                "kendall_tau_within_k": float(
                    stats.kendalltau(
                        [first_order.index(i) for i in first_order[:k]],
                        [second_order.index(i) for i in first_order[:k]],
                    ).statistic
                ),
            }
        )
    stability = pd.DataFrame(stability_rows)
    stability.to_csv(output_dir / "rank_stability.csv", index=False)
    # Both correlations compare the two halves' importance vectors predictor by
    # predictor. Correlating the two *orderings* (lists of predictor indices)
    # instead would be meaningless: position i of one list and position i of the
    # other refer to different predictors.
    spearman = float(stats.spearmanr(first, second).statistic)
    full_tau = float(stats.kendalltau(first, second).statistic)
    tau_within_all = float(
        stability.loc[stability["top_k"] == n_features, "kendall_tau_within_k"].iloc[0]
    )
    if not np.isclose(full_tau, tau_within_all, atol=1e-9):
        raise RuntimeError(
            "the full-ranking rank correlation disagrees with the same comparison in the "
            f"stability table: {full_tau} vs {tau_within_all}"
        )
    pd.DataFrame(
        {
            "internal_name": feature_names,
            "first_half_mean_abs_shap": first,
            "second_half_mean_abs_shap": second,
        }
    ).to_csv(output_dir / "half_importance.csv", index=False)
    # Keep the per-seed, per-patient contributions so the stability statistics can
    # be re-derived without refitting.
    np.savez_compressed(
        output_dir / "contributions.npz",
        contributions=contributions,
        seeds=np.asarray(seeds, dtype=np.int64),
        feature_names=np.asarray(feature_names, dtype=object),
    )

    per_seed = {}
    for column, seed in enumerate(seeds):
        per_seed[str(seed)] = np.abs(contributions[:, column, :]).mean(axis=0).tolist()
    pd.DataFrame(
        {"internal_name": feature_names, **{f"seed_{k}_mean_abs_shap": v for k, v in per_seed.items()}}
    ).to_csv(output_dir / "per_seed_importance.csv", index=False)

    pd.DataFrame(
        per_patient, columns=[f"shap_{name}" for name in feature_names]
    ).assign(
        patient_index=np.arange(len(cohort.labels)),
        y_easi75=cohort.labels,
    ).to_parquet(output_dir / "patient_attributions.parquet", index=False)
    pd.DataFrame(fold_rows).to_csv(output_dir / "fold_attribution.csv", index=False)

    config = {
        "schema_version": "1.0",
        "analysis": "mean absolute SHAP attribution for the proposed framework",
        "scope": (
            "mean absolute contribution per predictor and the resulting ranking only; "
            "no stratified split, no per-patient case decomposition"
        ),
        "explainer": "shap.KernelExplainer(link=identity) applied to the exact score function",
        "surrogate_model": "none — the explainer evaluates the score function itself",
        "protocol": (
            f"{len(seeds)} regression-stratified seeds x 5 outer folds; n={len(cohort.labels)}"
        ),
        "protocol_warning": (
            "ATTRIBUTION PROTOCOL ONLY. Every number in this directory comes from a "
            f"{len(seeds)}-seed run and must never be mixed with, or quoted alongside, the "
            "15-seed performance estimates as if they shared a protocol."
        ),
        "seed_selection": (
            f"the first {len(seeds)} of the frozen seed list in its pre-specified order; "
            "not selected by any result"
        ),
        "outer_seeds": list(seeds),
        "inner_seed_offset": 1000,
        "nsamples": args.nsamples,
        "perturbation_seed": "seed * 100 + fold",
        "l1_reg": f"num_features({n_features})",
        "background": (
            f"label-free deterministic subsample of at most {BACKGROUND_ROWS} outer-training "
            "rows per fold, seeded by seed*100+fold; observed missingness preserved because "
            "every member pipeline imputes internally"
        ),
        "leakage_control": (
            "every background row comes from the outer-training partition of the same fold; "
            "held-out labels and held-out observed improvement never enter the explainer"
        ),
        "aggregation": (
            "per-patient contribution averaged over the repeats in which the patient was "
            "held out, then the absolute value averaged over patients"
        ),
        "estimator_separation": (
            "attribution explains the continuous score. It does not explain accuracy, F1, "
            "or any decision-rule quantity, and no contribution is reported as a change in "
            "discrimination."
        ),
        "specification": cohort.specification,
        "feature_names": feature_names,
        "feature_matrix_sha256_float64": cohort.feature_sha256,
        "label_sha256_int64": cohort.label_sha256,
        "stability": {
            "design": "seed list split into two disjoint halves, aggregated separately",
            "halves": {name: [int(seeds[i]) for i in columns] for name, columns in halves.items()},
            "spearman_across_all_predictors": spearman,
            "kendall_tau_full_ranking": full_tau,
            "top_k_overlap": stability_rows,
        },
        "additivity": {
            "max_error_over_folds": float(np.max(additivity_errors)),
            "mean_error_over_folds": float(np.mean(additivity_errors)),
        },
        "mean_abs_shap_sha256_float64": array_sha256(mean_absolute, ">f8"),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "shap": shap.__version__,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "wall_seconds": time.monotonic() - started,
        },
    }
    config["config_sha256"] = hashlib.sha256(canonical_bytes(config)).hexdigest()
    (output_dir / "config.json").write_text(json.dumps(config, indent=1, sort_keys=True) + "\n")
    print(importance.head(12).to_string(index=False))
    print(stability.to_string(index=False))
    print(f"spearman={spearman:.4f} kendall_tau={full_tau:.4f}")


if __name__ == "__main__":
    main()
