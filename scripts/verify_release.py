#!/usr/bin/env python3
"""Fail if regenerated outputs drift from the final manuscript contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_MODELS = {
    "conventional_logistic_c1": (0.7626780626780627, 0.6461538461538462, 0.6111111111111112, 0.65625),
    "rbf_svm": (0.6854700854700855, 0.5538461538461539, 0.6851851851851852, 0.6101694915254238),
    "random_forest": (0.6581196581196582, 0.6461538461538462, 0.5925925925925926, 0.6511627906976745),
    "lightgbm": (0.6247863247863249, 0.5538461538461539, 0.5925925925925926, 0.5853658536585366),
    "xgboost": (0.5792022792022793, 0.6615384615384615, 0.46296296296296297, 0.6277372262773723),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(observed: float, expected: float, name: str, tolerance: float = 1e-12) -> None:
    if not np.isclose(observed, expected, rtol=0, atol=tolerance):
        raise RuntimeError(f"{name}: observed {observed!r}, expected {expected!r}")


def compare_numeric_csv(observed_path: Path, reference_path: Path, key: str | None = None) -> None:
    observed = pd.read_csv(observed_path)
    reference = pd.read_csv(reference_path)
    if key:
        observed = observed[observed[key].isin(reference[key])].set_index(key).loc[reference[key]]
        reference = reference.set_index(key)
    if len(observed) != len(reference):
        raise RuntimeError(f"row-count drift: {observed_path}")
    numeric = [column for column in reference.columns if column in observed.columns
               and pd.api.types.is_numeric_dtype(reference[column])
               and pd.api.types.is_numeric_dtype(observed[column])]
    for column in numeric:
        left = observed[column].to_numpy(float)
        right = reference[column].to_numpy(float)
        if not np.allclose(left, right, rtol=0, atol=2e-12, equal_nan=True):
            raise RuntimeError(f"aggregate drift: {observed_path.name}:{column}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    results = args.results.resolve()
    repo = Path(__file__).resolve().parents[1]

    metrics = json.loads((results / "proposed/metrics.json").read_text())
    close(metrics["discrimination"]["auroc"], 0.823931623931624, "proposed AUROC")
    close(metrics["discrimination"]["auroc_ci_low"], 0.7466084100639259, "proposed CI low")
    close(metrics["discrimination"]["auroc_ci_high"], 0.8917478569560338, "proposed CI high")
    primary = next(row for row in metrics["classification"]["policy_results"] if row["policy"] == "prt_bab")
    for key, expected in (("overall_sensitivity", 0.7384615384615385),
                          ("overall_specificity", 0.7777777777777778),
                          ("overall_f1", 0.768)):
        close(primary[key], expected, f"proposed {key}")
    if not metrics["leakage_probe"]["all_folds_passed"]:
        raise RuntimeError("one or more held-out leakage probes failed")

    comparison = pd.read_csv(results / "comparators/model_metrics.csv").set_index("model_id")
    for model_id, expected in EXPECTED_MODELS.items():
        observed = comparison.loc[model_id]
        for column, value in zip(("auroc", "overall_sensitivity", "overall_specificity", "overall_f1"), expected, strict=True):
            close(float(observed[column]), value, f"{model_id} {column}", tolerance=2e-12)

    paired = pd.read_csv(results / "paired_stats/discrimination_comparison.csv").set_index("model_id")
    logistic = paired.loc["conventional_logistic_c1"]
    close(float(logistic["auroc_difference_vs_proposed"]), 0.061253561253561295, "logistic AUROC delta")
    close(float(logistic["delong_p_value"]), 0.012, "logistic DeLong p", tolerance=5e-4)

    reference_results = repo / "artifacts/reference/results"
    compare_numeric_csv(results / "paired_stats/discrimination_comparison.csv",
                        reference_results / "discrimination_comparison.csv", "model_id")
    compare_numeric_csv(results / "paired_stats/classification_comparison.csv",
                        reference_results / "classification_comparison.csv", "model_id")
    compare_numeric_csv(results / "paired_stats/boundary_stratum_comparison.csv",
                        reference_results / "boundary_stratum_comparison.csv", "model_id")
    compare_numeric_csv(results / "calibration/calibration_summary.csv",
                        reference_results / "calibration_summary.csv", "estimate")
    compare_numeric_csv(results / "ablation/ablation_stages.csv",
                        reference_results / "ablation_stages.csv")

    endpoint = pd.read_csv(results / "endpoint/endpoint_sensitivity.csv")
    expected_endpoint = {
        70.0: (25, 90, 0.695, 0.567, 0.690, 0.680),
        72.5: (13, 78, 0.733, 0.615, 0.707, 0.696),
        74.0: (7, 72, 0.783, 0.667, 0.745, 0.727),
        75.0: (0, 65, 0.824, 0.738, 0.778, 0.768),
        76.0: (4, 61, 0.812, 0.754, 0.759, 0.760),
        77.5: (8, 57, 0.797, 0.754, 0.726, 0.735),
        80.0: (17, 48, 0.739, 0.729, 0.648, 0.648),
    }
    for row in endpoint.itertuples():
        expected = expected_endpoint[row.easi_improvement_cutoff_pct]
        observed = (row.labels_changed_vs_75, row.responders, row.auroc,
                    row.sensitivity, row.specificity, row.f1)
        if observed[:2] != expected[:2] or any(round(float(a), 3) != b for a, b in zip(observed[2:], expected[2:], strict=True)):
            raise RuntimeError(f"endpoint-sensitivity drift at {row.easi_improvement_cutoff_pct}%")

    importance = pd.read_csv(results / "attribution/feature_importance.csv")
    expected_top = {
        "Number of positive allergen tests": 0.0990736891339661,
        "Food allergen sensitisation": 0.0871771719251998,
        "Allergen panel, mean": 0.0834276546837722,
        "Atopic comorbidity count": 0.0637518198459972,
        "Total IgE": 0.0631773008638698,
    }
    observed_top = set(importance.head(5)["print_label"])
    if observed_top != set(expected_top):
        raise RuntimeError(f"attribution top-five membership drift: {sorted(observed_top)}")
    for label, value in expected_top.items():
        observed = float(importance.loc[importance["print_label"] == label, "mean_abs_shap"].iloc[0])
        close(observed, value, f"SHAP {label}", tolerance=0.002)

    required_figures = [
        "figure1_framework.jpg", "figure2_outcome_distribution.png",
        "figure3_discrimination.png", "figure4_patient_outcomes.png",
        "figure5_shap.png", "figureS1_calibration.png", "figureS2_validation.png",
    ]
    for name in required_figures:
        if not (results / "figures" / name).is_file():
            raise RuntimeError(f"missing figure: {name}")
    if sha256(results / "figures/figure1_framework.jpg") != "7f9e659d1f633dd0a05ab85e72bbec0537f59874c1b73a7d7c5635a08a36d75b":
        raise RuntimeError("Figure 1 does not match the author-supplied artifact")

    reference = repo / "artifacts/reference"
    for line in (reference / "manifest.sha256").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        if sha256(reference / relative) != expected:
            raise RuntimeError(f"reference artifact checksum drift: {relative}")

    required_tables = {"table1.csv", "table2.csv", "tableS1.csv", "tableS2.csv", "tableS3.csv",
                       "tableS4.csv", "tableS6.csv", "tableS7.csv", "tableS8.csv", "tableS9.csv"}
    observed_tables = {path.name for path in (results / "tables").glob("*.csv")}
    if not required_tables.issubset(observed_tables):
        raise RuntimeError(f"missing tables: {sorted(required_tables - observed_tables)}")

    forbidden_suffixes = {".xlsx", ".xls", ".parquet", ".docx"}
    leaked = [path for path in repo.rglob("*") if path.is_file()
              and ".venv" not in path.parts and "outputs" not in path.parts
              and path.suffix.lower() in forbidden_suffixes]
    if leaked:
        raise RuntimeError(f"restricted or participant-level files are present in the release: {leaked}")

    print("PASS: metrics, attribution, figures, tables, leakage probes, and privacy gates")


if __name__ == "__main__":
    main()
