#!/usr/bin/env python3
"""Build the model-result tables used in the final manuscript."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

DISPLAY_ORDER = [
    "conventional_logistic_c1",
    "rbf_svm",
    "random_forest",
    "lightgbm",
    "xgboost",
]


def ci_text(value: float, low: float, high: float) -> str:
    return f"{value:.3f} ({low:.3f}–{high:.3f})"


def bootstrap_ci(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(42)
    values = []
    for _ in range(10_000):
        index = rng.integers(0, len(y), len(y))
        if np.unique(y[index]).size == 2:
            values.append(roc_auc_score(y[index], score[index]))
    return tuple(float(v) for v in np.quantile(values, [0.025, 0.975]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results, out = args.results.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)

    repo = Path(__file__).resolve().parents[1]
    for name in ("table1.csv", "tableS1.csv", "tableS2.csv"):
        shutil.copyfile(repo / "artifacts/reference/tables" / name, out / name)

    metrics = json.loads((results / "proposed/metrics.json").read_text())
    primary = next(row for row in metrics["classification"]["policy_results"] if row["policy"] == "prt_bab")
    disc = pd.read_csv(results / "paired_stats/discrimination_comparison.csv").set_index("model_id")
    cls = pd.read_csv(results / "paired_stats/classification_comparison.csv").set_index("model_id")
    rows = [{
        "Model": "Our framework",
        "AUROC (95% CI)": ci_text(metrics["discrimination"]["auroc"], metrics["discrimination"]["auroc_ci_low"], metrics["discrimination"]["auroc_ci_high"]),
        "Sensitivity": f"{primary['overall_sensitivity']:.3f}",
        "Specificity": f"{primary['overall_specificity']:.3f}",
        "F1": f"{primary['overall_f1']:.3f}",
    }]
    for model_id in DISPLAY_ORDER:
        d, c = disc.loc[model_id], cls.loc[model_id]
        rows.append({
            "Model": d["print_name"],
            "AUROC (95% CI)": ci_text(d["auroc"], d["auroc_ci_low"], d["auroc_ci_high"]),
            "Sensitivity": f"{c['sensitivity']:.3f}",
            "Specificity": f"{c['specificity']:.3f}",
            "F1": f"{c['f1']:.3f}",
        })
    pd.DataFrame(rows).to_csv(out / "table2.csv", index=False)

    disc_reset = disc.loc[DISPLAY_ORDER].reset_index()
    cls_reset = cls.loc[DISPLAY_ORDER].reset_index()
    s3 = pd.DataFrame({
        "Comparator": disc_reset["print_name"],
        "Delta AUROC (95% CI)": [
            f"{v:+.3f} ({lo:.3f}–{hi:.3f})" for v, lo, hi in zip(
                disc_reset["auroc_difference_vs_proposed"], disc_reset["difference_ci_low"], disc_reset["difference_ci_high"], strict=True)
        ],
        "DeLong p": disc_reset["delong_p_value"],
        "Holm DeLong p": disc_reset["holm_adjusted_delong_p_value"],
        "Framework only / comparator only correct": [
            f"{a} / {b}" for a, b in zip(cls_reset["mcnemar_proposed_only_correct"], cls_reset["mcnemar_comparator_only_correct"], strict=True)
        ],
        "Holm McNemar p": cls_reset["holm_adjusted_mcnemar_p_value"],
    })
    s3.to_csv(out / "tableS3.csv", index=False)

    pd.read_csv(results / "calibration/calibration_summary.csv").to_csv(out / "tableS4.csv", index=False)
    pd.read_csv(repo / "artifacts/reference/results/feature_importance.csv").to_csv(out / "tableS6.csv", index=False)

    proposed = pd.read_parquet(results / "proposed/patient_scores_and_decisions.parquet")
    comparators = pd.read_parquet(results / "comparators/patient_scores_and_decisions.parquet")
    boundary = proposed["boundary_65_to_lt85"].to_numpy(bool)
    y = proposed["y_easi75"].to_numpy(int)[boundary]
    bframe = pd.read_csv(results / "paired_stats/boundary_stratum_comparison.csv").set_index("model_id")
    s7_rows = []
    for model_id, name, score_column in [
        ("proposed_framework", "Our framework", "patient_average_score"),
        *[(mid, disc.loc[mid, "print_name"], f"{mid}_patient_average_score") for mid in DISPLAY_ORDER],
    ]:
        score = (proposed if model_id == "proposed_framework" else comparators)[score_column].to_numpy(float)[boundary]
        area = float(roc_auc_score(y, score))
        low, high = bootstrap_ci(y, score)
        row = bframe.loc[model_id]
        s7_rows.append({"Model": name, "AUROC (95% CI)": ci_text(area, low, high),
                        "Sensitivity": f"{row['sensitivity']:.3f}", "Specificity": f"{row['specificity']:.3f}", "F1": f"{row['f1']:.3f}"})
    pd.DataFrame(s7_rows).to_csv(out / "tableS7.csv", index=False)

    pd.read_csv(results / "ablation/ablation_stages.csv").to_csv(out / "tableS8.csv", index=False)
    pd.read_csv(results / "endpoint/endpoint_sensitivity.csv").to_csv(out / "tableS9.csv", index=False)


if __name__ == "__main__":
    main()
