#!/usr/bin/env python3
"""Interval estimates for the framework's own component decomposition.

These are not model comparisons and do not belong to the comparator multiplicity
family.  They answer a different question: of the difference between a
conventional single-endpoint model and the score this framework produces, how
much is attributable to each component, and how precisely can this cohort
resolve it.

Stages, each adding one component to the one above it:

1. conventional single-endpoint regularised logistic regression
2. one learner fitted across adjacent improvement cut-offs
3. five learners combined into the response score
4. the score plus an ambiguity band, without the boundary corrector
5. the complete framework

Stages 3 to 5 share one score, so their discrimination is identical by
construction and only the accuracy column separates them.

Intervals are paired patient bootstraps on shared resample indices; they are
descriptive and carry no multiplicity adjustment, because the decomposition is
one argument rather than a family of hypotheses.

Usage:
    python run_ablation_stages.py --results <dir> --output-dir <dir>
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 42


def paired_delta(
    y: np.ndarray, later: np.ndarray, earlier: np.ndarray, indices: np.ndarray
) -> dict[str, float]:
    observed = float(roc_auc_score(y, later) - roc_auc_score(y, earlier))
    draws = []
    for row in indices:
        if len(np.unique(y[row])) < 2:
            continue
        draws.append(roc_auc_score(y[row], later[row]) - roc_auc_score(y[row], earlier[row]))
    values = np.asarray(draws, dtype=float)
    centred = values - values.mean()
    two_sided = float(np.mean(np.abs(centred) >= abs(observed)))
    return {
        "delta_auroc": observed,
        "delta_ci_low": float(np.percentile(values, 2.5)),
        "delta_ci_high": float(np.percentile(values, 97.5)),
        "paired_bootstrap_p": min(1.0, max(two_sided, 1.0 / (len(values) + 1))),
    }


def band_accuracy(y: np.ndarray, decision: np.ndarray, mask: np.ndarray) -> float:
    return float(np.mean(decision[mask] == y[mask]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument(
        "--ablation-members",
        default=None,
        help="where the stage-2 learner is fitted. It is a stage of the framework, "
        "not one of the reported comparators, so it is kept out of the comparator set.",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    results = Path(args.results)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    proposed = pd.read_parquet(results / "proposed" / "patient_scores_and_decisions.parquet")
    comparators = pd.read_parquet(results / "comparators" / "patient_scores_and_decisions.parquet")
    members_dir = Path(args.ablation_members) if args.ablation_members else results / "comparators"
    members = pd.read_parquet(members_dir / "patient_scores_and_decisions.parquet")
    y = proposed["y_easi75"].to_numpy(dtype=int)
    band = proposed["boundary_65_to_lt85"].to_numpy(dtype=bool)

    stage_scores = {
        "stage1_conventional_logistic": comparators[
            "conventional_logistic_c1_patient_average_score"
        ].to_numpy(dtype=float),
        "stage2_neighboring_cutoffs": members[
            "single_multitask_learner_patient_average_score"
        ].to_numpy(dtype=float),
        "stage3_five_learner_score": proposed["patient_average_score"].to_numpy(dtype=float),
    }
    stage_decisions = {
        "stage1_conventional_logistic": comparators[
            "conventional_logistic_c1_decision"
        ].to_numpy(dtype=int),
        "stage2_neighboring_cutoffs": members[
            "single_multitask_learner_decision"
        ].to_numpy(dtype=int),
        "stage3_five_learner_score": proposed["youden_decision"].to_numpy(dtype=int),
        "stage4_band_without_corrector": proposed["prt_midpoint_decision"].to_numpy(dtype=int),
        "stage5_complete_framework": proposed["prt_bab_decision"].to_numpy(dtype=int),
    }

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(y), (BOOTSTRAP_DRAWS, len(y)))

    labels = {
        "stage1_conventional_logistic": "Conventional single-endpoint logistic regression",
        "stage2_neighboring_cutoffs": "One learner fitted across adjacent cut-offs",
        "stage3_five_learner_score": "Five learners combined into the response score",
        "stage4_band_without_corrector": "Ambiguity band, without the boundary corrector",
        "stage5_complete_framework": "Complete framework",
    }

    rows: list[dict[str, Any]] = []
    for key, label in labels.items():
        score_key = key if key in stage_scores else "stage3_five_learner_score"
        score = stage_scores[score_key]
        decision = stage_decisions[key]
        row: dict[str, Any] = {
            "stage": key,
            "label": label,
            "auroc": float(roc_auc_score(y, score)),
            "shares_score_with_stage3": key not in stage_scores,
            "accuracy_overall": float(np.mean(decision == y)),
            "accuracy_near_threshold": band_accuracy(y, decision, band),
        }
        rows.append(row)

    consecutive = [
        ("stage2_neighboring_cutoffs", "stage1_conventional_logistic"),
        ("stage3_five_learner_score", "stage2_neighboring_cutoffs"),
        ("stage3_five_learner_score", "stage1_conventional_logistic"),
    ]
    deltas = []
    for later, earlier in consecutive:
        block = paired_delta(y, stage_scores[later], stage_scores[earlier], indices)
        block.update({"later": later, "earlier": earlier})
        deltas.append(block)

    payload = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "note": (
            "internal component decomposition of the proposed framework; not a comparison "
            "against competing models and not part of the comparator multiplicity family"
        ),
        "stages": rows,
        "discrimination_deltas": deltas,
        "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED},
    }
    (out / "ablation_stages.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    pd.DataFrame(rows).to_csv(out / "ablation_stages.csv", index=False)

    for row in rows:
        print(f"  {row['label']:<52s} AUROC {row['auroc']:.3f}  "
              f"acc {row['accuracy_overall']:.3f}  near-threshold {row['accuracy_near_threshold']:.3f}")
    print()
    for block in deltas:
        print(f"  {block['earlier']} -> {block['later']}: "
              f"{block['delta_auroc']:+.4f} [{block['delta_ci_low']:+.4f}, "
              f"{block['delta_ci_high']:+.4f}] p={block['paired_bootstrap_p']:.3f}")


if __name__ == "__main__":
    main()
