"""Hosmer-Lemeshow calibration of the pooled held-out predictions.

Reads the frozen per-patient held-out predictions written by ``run_proposed.py``
and adds a Hosmer-Lemeshow goodness-of-fit assessment and grouped calibration
data to the calibration summary already stored in ``metrics.json``. No model is
refitted here; the predictions are consumed exactly as produced.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

ESTIMATES = {
    "raw_score": "patient_average_score",
    "platt": "platt_patient_average_probability",
    "isotonic": "isotonic_patient_average_probability",
}
PRINT_NAME = {
    "raw_score": "Response score, as produced",
    "platt": "After logistic recalibration",
    "isotonic": "After isotonic recalibration",
}


def equal_count_groups(probabilities: np.ndarray, groups: int) -> list[np.ndarray]:
    """Split patients into ``groups`` equal-count strata of predicted risk."""
    order = np.argsort(probabilities, kind="stable")
    return [g for g in np.array_split(order, groups) if len(g) > 0]


def hosmer_lemeshow(
    labels: np.ndarray, probabilities: np.ndarray, groups: int = 10
) -> dict[str, Any]:
    """Hosmer-Lemeshow C statistic over equal-count strata of predicted risk."""
    statistic = 0.0
    for index in equal_count_groups(probabilities, groups):
        n_group = len(index)
        observed = float(labels[index].sum())
        expected = float(probabilities[index].sum())
        mean_predicted = expected / n_group
        denominator = expected * (1.0 - mean_predicted)
        if denominator <= 0:
            continue
        statistic += (observed - expected) ** 2 / denominator
    used = len(equal_count_groups(probabilities, groups))
    return {
        "groups": used,
        "hosmer_lemeshow_chi2": float(statistic),
        "degrees_of_freedom": used - 2,
        "p_value": float(chi2.sf(statistic, used - 2)),
        "degrees_of_freedom_unfitted": used,
        "p_value_unfitted": float(chi2.sf(statistic, used)),
    }


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    """Wilson score interval, which stays inside [0, 1] at small group sizes."""
    if total == 0:
        return (float("nan"), float("nan"))
    z = float(norm.ppf(0.975))
    phat = successes / total
    denominator = 1.0 + z**2 / total
    centre = (phat + z**2 / (2 * total)) / denominator
    half = z * np.sqrt(phat * (1 - phat) / total + z**2 / (4 * total**2)) / denominator
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


def group_table(
    labels: np.ndarray, probabilities: np.ndarray, groups: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for position, index in enumerate(equal_count_groups(probabilities, groups)):
        observed = int(labels[index].sum())
        n_group = int(len(index))
        low, high = wilson_interval(observed, n_group)
        rows.append(
            {
                "group": position + 1,
                "n": n_group,
                "predicted_low": float(probabilities[index].min()),
                "predicted_high": float(probabilities[index].max()),
                "mean_predicted": float(probabilities[index].mean()),
                "observed_events": observed,
                "expected_events": float(probabilities[index].sum()),
                "observed_rate": observed / n_group,
                "observed_ci_low": low,
                "observed_ci_high": high,
            }
        )
    return rows


def _ece(labels: np.ndarray, probabilities: np.ndarray, bins: int = 5) -> float:
    """Equal-count bins, the same definition the reported value uses."""
    order = np.argsort(probabilities, kind="stable")
    total = len(labels)
    error = 0.0
    for group in np.array_split(order, bins):
        if len(group) == 0:
            continue
        error += len(group) / total * abs(
            float(labels[group].mean()) - float(probabilities[group].mean())
        )
    return error


def _brier(labels: np.ndarray, probabilities: np.ndarray) -> float:
    return float(np.mean((probabilities - labels) ** 2))


def scalar_intervals(
    labels: np.ndarray, probabilities: np.ndarray, draws: int = 10_000, seed: int = 20260816
) -> dict[str, float]:
    """Patient-level percentile bootstrap for the two scalars that shipped bare.

    Supplementary Table S4 carried intervals on the slope and the intercept and none on
    the Brier score or the calibration error, against the paper's own convention that an
    estimate carries an interval. Same resampling, same draw count.
    """
    rng = np.random.default_rng(seed)
    briers, errors = [], []
    for _ in range(draws):
        take = rng.integers(0, len(labels), len(labels))
        if len(np.unique(labels[take])) < 2:
            continue
        briers.append(_brier(labels[take], probabilities[take]))
        errors.append(_ece(labels[take], probabilities[take]))
    low_b, high_b = np.percentile(briers, [2.5, 97.5])
    low_e, high_e = np.percentile(errors, [2.5, 97.5])
    return {
        "brier_ci_low": float(low_b),
        "brier_ci_high": float(high_b),
        "expected_calibration_error_ci_low": float(low_e),
        "expected_calibration_error_ci_high": float(high_e),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--groups", type=int, default=10)
    args = parser.parse_args()

    results = args.results.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_parquet(results / "proposed" / "patient_scores_and_decisions.parquet")
    metrics = json.loads((results / "proposed" / "metrics.json").read_text())
    labels = predictions["y_easi75"].to_numpy(dtype=int)

    summary_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    for key, column in ESTIMATES.items():
        probabilities = predictions[column].to_numpy(dtype=float)
        stored = metrics["calibration"][key]
        primary = hosmer_lemeshow(labels, probabilities, args.groups)
        sensitivity = hosmer_lemeshow(labels, probabilities, 5)
        summary_rows.append(
            {
                "estimate": key,
                "print_name": PRINT_NAME[key],
                "n": int(len(labels)),
                "brier": stored["brier"],
                "expected_calibration_error": stored["expected_calibration_error"],
                **scalar_intervals(labels, probabilities),
                "calibration_slope": stored["calibration_slope"],
                "calibration_slope_ci_low": stored["calibration_slope_ci_low"],
                "calibration_slope_ci_high": stored["calibration_slope_ci_high"],
                "calibration_intercept": stored["calibration_intercept"],
                "calibration_intercept_ci_low": stored["calibration_intercept_ci_low"],
                "calibration_intercept_ci_high": stored["calibration_intercept_ci_high"],
                "mean_predicted": stored["mean_predicted"],
                "observed_rate": stored["observed_rate"],
                "calibration_in_the_large": stored["calibration_in_the_large"],
                **{f"hl_{k}": v for k, v in primary.items()},
                **{f"hl5_{k}": v for k, v in sensitivity.items()},
            }
        )
        for row in group_table(labels, probabilities, args.groups):
            group_rows.append({"estimate": key, **row})

    summary = pd.DataFrame(summary_rows)
    groups = pd.DataFrame(group_rows)
    summary.to_csv(output_dir / "calibration_summary.csv", index=False)
    groups.to_csv(output_dir / "calibration_groups.csv", index=False)
    (output_dir / "provenance.json").write_text(
        json.dumps(
            {
                "source_predictions": str(
                    results / "proposed" / "patient_scores_and_decisions.parquet"
                ),
                "source_metrics_config_sha256": metrics["config_sha256"],
                "n_patients": int(len(labels)),
                "n_positive": int(labels.sum()),
                "groups_primary": args.groups,
                "groups_sensitivity": 5,
                "grouping": "equal-count strata of predicted risk",
                "statistic": "Hosmer-Lemeshow C",
                "note": (
                    "Brier score, expected calibration error, calibration slope and "
                    "intercept are read unchanged from metrics.json; only the "
                    "Hosmer-Lemeshow statistic and the grouped calibration data are "
                    "computed here."
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
