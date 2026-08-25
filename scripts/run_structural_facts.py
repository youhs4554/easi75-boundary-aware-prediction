#!/usr/bin/env python3
"""Structural facts about the model that the reporting did not previously state.

Three groups, all deterministic given the frozen configuration:

``effective_weights``
    What the six-line hierarchy does to the five member ranks.  Three of the
    lines average the same four members and differ only in weight, so the first
    three lines collapse exactly to one weighted average; the later lines apply
    a rank transform to intermediate nodes, so the whole is only approximately
    linear.  Both are reported: the exact collapse of the first three lines, and
    a linearized reading of the whole.

``capacity``
    Predictor count per learner after its feature map, against the size of the
    partition each learner is fitted on, and events per variable.

``sensitisation_definitions``
    The two sensitisation thresholds in play and where each is used, so the
    manuscript can state both instead of one.

Usage:
    python run_structural_facts.py --root <repo> --output-dir <dir>
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

MEMBER_LABELS = (
    "multithreshold_three",
    "multithreshold_five",
    "polynomial_interaction_robust",
    "polynomial_interaction_quantile",
    "polynomial_full_robust",
)
PRINT_LABELS = {
    "multithreshold_three": "Robust raw features",
    "multithreshold_five": "Quantile raw features",
    "polynomial_interaction_robust": "Robust interactions",
    "polynomial_interaction_quantile": "Quantile interactions",
    "polynomial_full_robust": "Full second-order",
}


def effective_weights() -> dict[str, Any]:
    """Collapse the hierarchy onto the member ranks."""
    zero = Fraction(0)
    uniform = np.array([Fraction(1, 4)] * 4 + [zero], dtype=object)
    weighted = np.array(
        [Fraction(2, 9), Fraction(2, 9), Fraction(3, 9), Fraction(2, 9), zero], dtype=object
    )
    mean_node = (uniform + weighted) / 2

    # Later lines rank the intermediate nodes, so this is a linearized reading.
    include_a = np.array([Fraction(1), Fraction(1), zero, Fraction(1), Fraction(1)], dtype=object)
    stage_a = (include_a + mean_node) / 5
    include_b = np.array([Fraction(1), Fraction(1), zero, zero, Fraction(1)], dtype=object)
    stage_b = (include_b + mean_node + stage_a) / 5
    score = (stage_a + stage_b) / 2

    # The same collapse with the highlighted 2:2:3:2 line replaced by a plain mean.
    mean_node_flat = (uniform + uniform) / 2
    stage_a_flat = (include_a + mean_node_flat) / 5
    stage_b_flat = (include_b + mean_node_flat + stage_a_flat) / 5
    score_flat = (stage_a_flat + stage_b_flat) / 2

    return {
        "exact_collapse_of_first_three_lines": {
            "statement": (
                "the first three lines are all linear in the same four member ranks, so their "
                "combination is exactly one weighted average of those four"
            ),
            "weights": {
                PRINT_LABELS[name]: float(value)
                for name, value in zip(MEMBER_LABELS[:4], mean_node[:4], strict=True)
            },
            "uniform_reference": 0.25,
            "max_absolute_departure_from_uniform": float(
                max(abs(value - Fraction(1, 4)) for value in mean_node[:4])
            ),
        },
        "linearized_whole": {
            "statement": (
                "the later lines rank intermediate nodes, so this reading is first order; it "
                "shows which members the structure emphasises, not an exact identity"
            ),
            "weights": {
                PRINT_LABELS[name]: float(value)
                for name, value in zip(MEMBER_LABELS, score, strict=True)
            },
            "uniform_reference": 0.2,
        },
        "effect_of_the_highlighted_weighting": {
            "statement": (
                "replacing the highlighted 2:2:3:2 line with a plain mean and recomputing the "
                "whole collapse moves the final weights by at most this much"
            ),
            "weights_with_the_weighting": {
                PRINT_LABELS[name]: float(value)
                for name, value in zip(MEMBER_LABELS, score, strict=True)
            },
            "weights_without_it": {
                PRINT_LABELS[name]: float(value)
                for name, value in zip(MEMBER_LABELS, score_flat, strict=True)
            },
            "max_absolute_weight_change": float(
                max(abs(a - b) for a, b in zip(score, score_flat, strict=True))
            ),
        },
    }


def capacity(n_patients: int, n_predictors: int, n_events: int) -> dict[str, Any]:
    outer_train = int(round(n_patients * 0.8))
    inner_train = int(round(outer_train * 0.8))
    interactions_only = n_predictors + n_predictors * (n_predictors - 1) // 2
    full_second_order = 2 * n_predictors + n_predictors * (n_predictors - 1) // 2
    return {
        "n_patients": n_patients,
        "n_predictors": n_predictors,
        "n_events": n_events,
        "n_non_events": n_patients - n_events,
        "events_per_variable": n_events / n_predictors,
        "non_events_per_variable": (n_patients - n_events) / n_predictors,
        "outer_training_partition_size": outer_train,
        "inner_training_partition_size": inner_train,
        "per_learner": {
            "Robust raw features": {"n_terms": n_predictors, "map": "identity"},
            "Quantile raw features": {"n_terms": n_predictors, "map": "identity"},
            "Robust interactions": {
                "n_terms": interactions_only,
                "map": "degree-2 interaction expansion",
            },
            "Quantile interactions": {
                "n_terms": interactions_only,
                "map": "degree-2 interaction expansion",
            },
            "Full second-order": {
                "n_terms": full_second_order,
                "map": "full degree-2 expansion",
            },
        },
        "max_terms_to_outer_training_ratio": full_second_order / outer_train,
        "note": (
            "the three expanded learners carry an L1 penalty, which is what makes fitting in "
            "these spaces possible; the counts are the size of the space, not the number of "
            "terms the penalty retains"
        ),
    }


def sensitisation_definitions() -> dict[str, Any]:
    return {
        "descriptive_tables": {
            "threshold": "class 1 or above",
            "used_for": (
                "the baseline characteristic tables, which report the proportion of patients "
                "sensitised to at least one allergen in each panel"
            ),
        },
        "modelled_predictors": {
            "threshold": "class 2 or above",
            "used_for": (
                "the sensitisation predictors entering every model, where a higher threshold "
                "was judged the more clinically meaningful definition of a positive test"
            ),
        },
        "statement_for_the_manuscript": (
            "Two sensitisation thresholds are used and each is stated where it applies: the "
            "descriptive tables count a class of 1 or above as positive, whereas every "
            "modelled sensitisation predictor counts a class of 2 or above."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "effective_weights": effective_weights(),
        "capacity": capacity(n_patients=119, n_predictors=23, n_events=65),
        "sensitisation_definitions": sensitisation_definitions(),
    }
    (out / "structural_facts.json").write_text(json.dumps(payload, indent=2, sort_keys=True))

    ew = payload["effective_weights"]
    print("first three lines collapse to:", ew["exact_collapse_of_first_three_lines"]["weights"])
    print("  max departure from uniform:",
          round(ew["exact_collapse_of_first_three_lines"]["max_absolute_departure_from_uniform"], 4))
    print("linearized whole:", {k: round(v, 4) for k, v in ew["linearized_whole"]["weights"].items()})
    print("  effect of the highlighted weighting:",
          round(ew["effect_of_the_highlighted_weighting"]["max_absolute_weight_change"], 4))
    cap = payload["capacity"]
    print(f"EPV {cap['events_per_variable']:.2f}; outer-train {cap['outer_training_partition_size']}; "
          f"largest space {cap['per_learner']['Full second-order']['n_terms']} terms "
          f"(ratio {cap['max_terms_to_outer_training_ratio']:.1f})")


if __name__ == "__main__":
    main()
