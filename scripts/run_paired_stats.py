#!/usr/bin/env python3
"""Paired comparisons between the proposed framework and each comparator.

Discrimination and classification are kept apart throughout, because they are
different estimands computed from different quantities:

* discrimination — the mean of the 15 held-out scores; compared by DeLong's test
  for two correlated ROC curves and by a paired patient bootstrap;
* classification — the majority of the 15 held-out decisions; compared by the
  exact McNemar test.

The multiplicity family is the five comparators the manuscript reports, which
were prespecified as the panel; earlier exploratory models are not part of it and
are not published.  Holm's step-down correction is applied within each of the two
families separately, since they answer different questions with different test
statistics.

Usage:
    python run_paired_stats.py --proposed <dir> --comparators <dir> --output-dir <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 42


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    ranks = np.empty(len(x), dtype=np.float64)
    index = 0
    while index < len(x):
        stop = index
        while stop < len(x) and sorted_x[stop] == sorted_x[index]:
            stop += 1
        ranks[index:stop] = 0.5 * (index + stop - 1) + 1
        index = stop
    output = np.empty(len(x), dtype=np.float64)
    output[order] = ranks
    return output


def delong_test(labels: np.ndarray, first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    """DeLong's test for two AUROCs estimated on the same patients.

    Implements the fast midrank form of DeLong, DeLong and Clarke-Pearson (1988)
    as given by Sun and Xu (2014).
    """
    positive = labels == 1
    negative = ~positive
    m, n = int(positive.sum()), int(negative.sum())
    if m == 0 or n == 0:
        raise ValueError("both outcome classes are required")
    scores = np.vstack([np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)])
    k = scores.shape[0]
    positive_scores = scores[:, positive]
    negative_scores = scores[:, negative]

    tx = np.empty((k, m))
    ty = np.empty((k, n))
    tz = np.empty((k, m + n))
    for r in range(k):
        tx[r] = _midrank(positive_scores[r])
        ty[r] = _midrank(negative_scores[r])
        tz[r] = _midrank(np.concatenate([positive_scores[r], negative_scores[r]]))
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    # np.cov of a (k, n) matrix returns (k, k); reshape defensively so that a
    # degenerate case cannot silently produce a non-scalar variance.
    s01 = np.asarray(np.cov(v01), dtype=np.float64).reshape(k, k)
    s10 = np.asarray(np.cov(v10), dtype=np.float64).reshape(k, k)
    covariance = s01 / m + s10 / n
    contrast = np.array([[1.0, -1.0]])
    variance = float(np.squeeze(contrast @ covariance @ contrast.T))
    difference = float(aucs[0] - aucs[1])
    if variance <= 0:
        z, p_value = 0.0, 1.0
    else:
        z = difference / np.sqrt(variance)
        p_value = float(2.0 * stats.norm.sf(abs(z)))
    return {
        "auroc_first": float(aucs[0]),
        "auroc_second": float(aucs[1]),
        "auroc_difference": difference,
        "standard_error": float(np.sqrt(variance)) if variance > 0 else 0.0,
        "z": float(z),
        "p_value": p_value,
    }


def paired_bootstrap_auroc_difference(
    labels: np.ndarray, first: np.ndarray, second: np.ndarray, *, indices: np.ndarray
) -> dict[str, float]:
    """Outcome-stratified paired bootstrap on shared resample indices."""
    estimates = []
    for draw in indices:
        if np.unique(labels[draw]).size != 2:
            continue
        estimates.append(
            float(roc_auc_score(labels[draw], first[draw]) - roc_auc_score(labels[draw], second[draw]))
        )

    estimates = np.asarray(estimates)
    return {
        "bootstrap_difference_mean": float(estimates.mean()),
        "bootstrap_ci_low": float(np.quantile(estimates, 0.025)),
        "bootstrap_ci_high": float(np.quantile(estimates, 0.975)),
        "bootstrap_draws_used": int(len(estimates)),
    }


def bootstrap_auroc_interval(
    labels: np.ndarray, scores: np.ndarray, *, indices: np.ndarray
) -> tuple[float, float]:
    estimates = [
        float(roc_auc_score(labels[draw], scores[draw]))
        for draw in indices
        if np.unique(labels[draw]).size == 2
    ]
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def exact_mcnemar(
    labels: np.ndarray, first: np.ndarray, second: np.ndarray
) -> dict[str, float | int]:
    first_correct = first == labels
    second_correct = second == labels
    first_only = int(np.count_nonzero(first_correct & ~second_correct))
    second_only = int(np.count_nonzero(~first_correct & second_correct))
    discordant = first_only + second_only
    p_value = (
        float(stats.binomtest(first_only, discordant, 0.5).pvalue) if discordant else 1.0
    )
    return {
        "proposed_only_correct": first_only,
        "comparator_only_correct": second_only,
        "discordant": discordant,
        "exact_p_value": p_value,
    }


def holm(p_values: dict[str, float]) -> dict[str, float]:
    """Holm's step-down correction, monotone-enforced and capped at one."""
    items = sorted(p_values.items(), key=lambda item: item[1])
    total = len(items)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (key, value) in enumerate(items):
        candidate = (total - rank) * value
        running = max(running, candidate)
        adjusted[key] = float(min(1.0, running))
    return adjusted


def classification_interval(
    labels: np.ndarray, decision: np.ndarray, *, indices: np.ndarray
) -> dict[str, tuple[float, float]]:
    """Patient-level bootstrap intervals for the three classification summaries.

    Discrimination carried intervals from the start and classification did not, which
    left one of the two primary estimands reported as bare point estimates resting on
    65 positives. The resample is the same one the discrimination contrasts use, so an
    interval here is on the same footing as the interval beside it.
    """
    sensitivity, specificity, f1 = [], [], []
    for draw in indices:
        y, d = labels[draw], decision[draw]
        tp = int(np.count_nonzero((y == 1) & (d == 1)))
        tn = int(np.count_nonzero((y == 0) & (d == 0)))
        fp = int(np.count_nonzero((y == 0) & (d == 1)))
        fn = int(np.count_nonzero((y == 1) & (d == 0)))
        if not (tp + fn) or not (tn + fp):
            continue
        sensitivity.append(tp / (tp + fn))
        specificity.append(tn / (tn + fp))
        f1.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    out: dict[str, tuple[float, float]] = {}
    for name, sample in (
        ("sensitivity", sensitivity),
        ("specificity", specificity),
        ("f1", f1),
    ):
        values = np.asarray(sample)
        out[name] = (
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        )
    return out


def classification_row(
    labels: np.ndarray, decision: np.ndarray, *, indices: np.ndarray | None = None
) -> dict[str, Any]:
    tp = int(np.count_nonzero((labels == 1) & (decision == 1)))
    tn = int(np.count_nonzero((labels == 0) & (decision == 0)))
    fp = int(np.count_nonzero((labels == 0) & (decision == 1)))
    fn = int(np.count_nonzero((labels == 1) & (decision == 0)))
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    f1 = (
        2 * precision * sensitivity / (precision + sensitivity)
        if (precision + sensitivity) > 0
        else 0.0
    )
    return {
        "n": int(len(labels)),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "accuracy": float((tp + tn) / len(labels)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "ppv": float(precision),
        "npv": float(tn / (tn + fn)) if (tn + fn) else float("nan"),
        "f1": float(f1),
        **(
            {
                f"{name}_ci_{bound}": value
                for name, pair in classification_interval(
                    labels, decision, indices=indices
                ).items()
                for bound, value in zip(("low", "high"), pair, strict=True)
            }
            if indices is not None
            else {}
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposed", type=Path, required=True)
    parser.add_argument("--comparators", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--comparator-set",
        choices=("all", "main"),
        default="all",
        help=(
            "which comparators are evaluated and therefore form the multiplicity family; "
            "'main' restricts both to the panel reported in the main text"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    proposed_path = args.proposed / "patient_scores_and_decisions.parquet"
    comparator_path = args.comparators / "patient_scores_and_decisions.parquet"
    proposed = pd.read_parquet(proposed_path)
    comparators = pd.read_parquet(comparator_path)
    comparator_metrics = pd.read_csv(args.comparators / "model_metrics.csv")
    proposed_metrics = json.loads((args.proposed / "metrics.json").read_text())

    labels = proposed["y_easi75"].to_numpy(dtype=np.int64)
    boundary = proposed["boundary_65_to_lt85"].to_numpy(dtype=bool)
    if not np.array_equal(labels, comparators["y_easi75"].to_numpy(dtype=np.int64)):
        raise RuntimeError("the two ledgers disagree about the outcome vector")

    proposed_score = proposed["patient_average_score"].to_numpy(dtype=np.float64)
    # Two operating points on one score: the framework's own, and the rule every
    # comparator uses. They are separate vectors and must keep separate names.
    proposed_decision = proposed["prt_bab_decision"].to_numpy(dtype=np.int64)
    proposed_youden = proposed["youden_decision"].to_numpy(dtype=np.int64)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    shared_indices = rng.integers(0, len(labels), (BOOTSTRAP_DRAWS, len(labels)))
    boundary_positions = np.flatnonzero(boundary)
    boundary_rng = np.random.default_rng(BOOTSTRAP_SEED)
    boundary_indices = boundary_rng.integers(
        0, len(boundary_positions), (BOOTSTRAP_DRAWS, len(boundary_positions))
    )

    model_ids = [
        column[: -len("_patient_average_score")]
        for column in comparators.columns
        if column.endswith("_patient_average_score")
    ]
    name_of = dict(zip(comparator_metrics["model_id"], comparator_metrics["print_name"], strict=True))
    placement_of = dict(zip(comparator_metrics["model_id"], comparator_metrics["placement"], strict=True))
    if args.comparator_set == "main":
        model_ids = [m for m in model_ids if placement_of.get(m) == "main"]
        if not model_ids:
            raise RuntimeError("no comparator is marked 'main'; nothing to evaluate")

    discrimination_rows: list[dict[str, Any]] = []
    classification_rows: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []

    proposed_ci = bootstrap_auroc_interval(labels, proposed_score, indices=shared_indices)
    discrimination_rows.append(
        {
            "model_id": "proposed_framework",
            "print_name": "Proposed framework",
            "placement": "main",
            "role": "proposed",
            "auroc": float(roc_auc_score(labels, proposed_score)),
            "auroc_ci_low": proposed_ci[0],
            "auroc_ci_high": proposed_ci[1],
            "auroc_difference_vs_proposed": np.nan,
            "difference_ci_low": np.nan,
            "difference_ci_high": np.nan,
            "delong_p_value": np.nan,
            "bootstrap_ci_low": np.nan,
            "bootstrap_ci_high": np.nan,
        }
    )
    classification_rows.append(
        {
            "model_id": "proposed_framework",
            "print_name": "Proposed framework",
            "placement": "main",
            "role": "proposed",
            "decision_mechanism": "training-fold ambiguity band with boundary corrector",
            **classification_row(labels, proposed_decision, indices=shared_indices),
            "mcnemar_exact_p_value": np.nan,
        }
    )
    classification_rows.append(
        {
            "model_id": "proposed_framework_youden",
            "print_name": "Proposed framework (fold-local Youden)",
            "placement": "main",
            "role": "proposed_matched_operating_point",
            "decision_mechanism": "fold-local inner out-of-fold Youden threshold",
            **classification_row(labels, proposed_youden, indices=shared_indices),
            "mcnemar_exact_p_value": np.nan,
        }
    )

    delong_p: dict[str, float] = {}
    mcnemar_p: dict[str, float] = {}
    for model_id in model_ids:
        score = comparators[f"{model_id}_patient_average_score"].to_numpy(dtype=np.float64)
        decision = comparators[f"{model_id}_decision"].to_numpy(dtype=np.int64)
        test = delong_test(labels, proposed_score, score)
        boot = paired_bootstrap_auroc_difference(
            labels, proposed_score, score, indices=shared_indices
        )
        ci = bootstrap_auroc_interval(labels, score, indices=shared_indices)
        delong_p[model_id] = test["p_value"]
        discrimination_rows.append(
            {
                "model_id": model_id,
                "print_name": name_of.get(model_id, model_id),
                "placement": placement_of.get(model_id, ""),
                "role": "comparator",
                "auroc": float(roc_auc_score(labels, score)),
                "auroc_ci_low": ci[0],
                "auroc_ci_high": ci[1],
                "auroc_difference_vs_proposed": test["auroc_difference"],
                "difference_ci_low": boot["bootstrap_ci_low"],
                "difference_ci_high": boot["bootstrap_ci_high"],
                "delong_standard_error": test["standard_error"],
                "delong_z": test["z"],
                "delong_p_value": test["p_value"],
                "bootstrap_ci_low": boot["bootstrap_ci_low"],
                "bootstrap_ci_high": boot["bootstrap_ci_high"],
            }
        )

        mcnemar = exact_mcnemar(labels, proposed_decision, decision)
        mcnemar_p[model_id] = mcnemar["exact_p_value"]
        classification_rows.append(
            {
                "model_id": model_id,
                "print_name": name_of.get(model_id, model_id),
                "placement": placement_of.get(model_id, ""),
                "role": "comparator",
                "decision_mechanism": "fold-local inner out-of-fold Youden threshold",
                **classification_row(labels, decision, indices=shared_indices),
                "mcnemar_exact_p_value": mcnemar["exact_p_value"],
                "mcnemar_proposed_only_correct": mcnemar["proposed_only_correct"],
                "mcnemar_comparator_only_correct": mcnemar["comparator_only_correct"],
                "mcnemar_discordant": mcnemar["discordant"],
            }
        )
        boundary_rows.append(
            {
                "model_id": model_id,
                "print_name": name_of.get(model_id, model_id),
                "placement": placement_of.get(model_id, ""),
                "stratum": "observed EASI improvement 65 to <85",
                "stratum_status": (
                    "retrospective: defined by the observed outcome, not identifiable "
                    "from baseline information"
                ),
                "auroc": float(roc_auc_score(labels[boundary], score[boundary])),
                **classification_row(labels[boundary], decision[boundary], indices=boundary_indices),
            }
        )

    boundary_rows.insert(
        0,
        {
            "model_id": "proposed_framework",
            "print_name": "Proposed framework",
            "placement": "main",
            "stratum": "observed EASI improvement 65 to <85",
            "stratum_status": (
                "retrospective: defined by the observed outcome, not identifiable "
                "from baseline information"
            ),
            "auroc": float(roc_auc_score(labels[boundary], proposed_score[boundary])),
            **classification_row(labels[boundary], proposed_decision[boundary], indices=boundary_indices),
        },
    )

    delong_adjusted = holm(delong_p)
    mcnemar_adjusted = holm(mcnemar_p)
    for row in discrimination_rows:
        row["holm_adjusted_delong_p_value"] = delong_adjusted.get(row["model_id"], np.nan)
    for row in classification_rows:
        row["holm_adjusted_mcnemar_p_value"] = mcnemar_adjusted.get(row["model_id"], np.nan)

    discrimination = pd.DataFrame(discrimination_rows)
    classification = pd.DataFrame(classification_rows)
    boundary_frame = pd.DataFrame(boundary_rows)
    discrimination.to_csv(output_dir / "discrimination_comparison.csv", index=False)
    classification.to_csv(output_dir / "classification_comparison.csv", index=False)
    boundary_frame.to_csv(output_dir / "boundary_stratum_comparison.csv", index=False)

    provenance = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "estimand_separation": {
            "discrimination": "mean of the 15 held-out scores; DeLong plus paired patient bootstrap",
            "classification": "majority of the 15 held-out decisions; exact McNemar",
        },
        "operating_point_asymmetry": (
            "The proposed framework's main classification row uses the midpoint of the "
            "ambiguity band its training partitions selected, which is part of the method "
            "under study. Every comparator instead takes a single cut-point maximizing "
            "the Youden index on "
            "its outer-training inner out-of-fold predictions. The two mechanisms are "
            "therefore not equivalent. A second row applies that same single-cut-point "
            "rule to the proposed score, so a like-for-like classification comparison is "
            "available alongside the method's own decision layer. The discrimination "
            "comparison is threshold-free and is unaffected by this asymmetry."
        ),
        "multiplicity": (
            f"Holm step-down within each family; family = the {len(model_ids)} comparators "
            "prespecified and evaluated under this protocol. A sixth comparator, a "
            "regression on continuous improvement thresholded at 75%, was fitted after "
            "this family was declared and is reported alongside them unadjusted and "
            "outside it; see results/_continuous/"
        ),
        "comparator_set": args.comparator_set,
        "comparators_evaluated": list(model_ids),
        "bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "contract": "shared resample indices across all comparisons",
        },
        "boundary_stratum": {
            "definition": "65 <= observed EASI improvement < 85",
            "status": (
                "retrospective outcome stratum; defined by the observed result and not "
                "identifiable from baseline information"
            ),
            "n": int(boundary.sum()),
            "n_positive": int(labels[boundary].sum()),
        },
        "inputs": {
            str(proposed_path): file_sha256(proposed_path),
            str(comparator_path): file_sha256(comparator_path),
        },
        "proposed_config_sha256": proposed_metrics["config_sha256"],
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=1, sort_keys=True) + "\n"
    )
    print(
        discrimination[
            ["print_name", "auroc", "auroc_difference_vs_proposed", "delong_p_value",
             "holm_adjusted_delong_p_value"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
