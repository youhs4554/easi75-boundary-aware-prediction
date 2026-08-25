#!/usr/bin/env python3
"""Regenerate every data-bearing figure and copy the fixed schematics."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

MODELS = {
    "Proposed": ("patient_average_score", "prt_bab_decision", "#0072B2"),
    "Logistic": ("conventional_logistic_c1_patient_average_score", None, "#D55E00"),
    "RBF SVM": ("rbf_svm_patient_average_score", None, "#009E73"),
    "Random forest": ("random_forest_patient_average_score", None, "#56B4E9"),
    "LightGBM": ("lightgbm_patient_average_score", None, "#E69F00"),
    "XGBoost": ("xgboost_patient_average_score", None, "#CC79A7"),
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def save(fig: plt.Figure, out: Path, stem: str) -> None:
    fig.savefig(out / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out / f"{stem}.tiff", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(out / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def bootstrap_ci(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(42)
    values = []
    for _ in range(10_000):
        index = rng.integers(0, len(y), len(y))
        if np.unique(y[index]).size == 2:
            values.append(roc_auc_score(y[index], score[index]))
    return tuple(float(v) for v in np.quantile(values, [0.025, 0.975]))


def figure2(proposed: pd.DataFrame, out: Path, source: Path) -> None:
    improvement = proposed["easi_improvement_pct"].to_numpy(float)
    bins = np.arange(40, 101, 5)
    counts, edges = np.histogram(improvement, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    colors = np.where(centers < 75, "#9AA5B1", "#0072B2")
    fig, ax = plt.subplots(figsize=(3.504, 3.2))
    fig.subplots_adjust(left=0.19, right=0.98, bottom=0.17, top=0.78)
    ax.axvspan(65, 85, color="#EAE6DC", zorder=0)
    ax.bar(centers, counts, width=4.5, color=colors, edgecolor="white", zorder=2)
    ax.axvline(75, color="#444444", lw=1.3)
    ax.set(xlabel="Observed EASI improvement at 16 weeks (%)", ylabel="Patients",
           ylim=(0, counts.max() + 4))
    fig.suptitle("Observed response clusters at the 75% cut-off", y=0.97, fontsize=9)
    fig.text(0.56, 0.86, "65–85%: 76 of 119 (63.9%)", ha="center", color="#6E6250")
    ax.text(76, counts.max() + 0.7, "75% cut-off", ha="left", color="#444444")
    save(fig, out, "figure2_outcome_distribution")
    pd.DataFrame({"bin_low": edges[:-1], "bin_high": edges[1:], "n": counts}).to_csv(
        source / "figure2_source_data.csv", index=False
    )


def figure3(proposed: pd.DataFrame, comparators: pd.DataFrame, out: Path, source: Path) -> None:
    merged = proposed.merge(comparators, on=["patient_index", "y_easi75", "easi_improvement_pct", "boundary_65_to_lt85"])
    y = merged["y_easi75"].to_numpy(int)
    rows: list[dict[str, object]] = []
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.55))
    for ax, mask, title in (
        (axes[0], np.ones(len(merged), dtype=bool), "All patients (n = 119)"),
        (axes[1], merged["boundary_65_to_lt85"].to_numpy(bool), "Near-threshold outcomes (n = 76)"),
    ):
        ax.plot([0, 1], [0, 1], color="#BBBBBB", ls=(0, (2, 2)), lw=0.8)
        for label, (column, _, color) in MODELS.items():
            score = merged[column].to_numpy(float)[mask]
            yy = y[mask]
            fpr, tpr, _ = roc_curve(yy, score)
            area = float(roc_auc_score(yy, score))
            low, high = bootstrap_ci(yy, score)
            ax.plot(fpr, tpr, lw=2.2 if label == "Proposed" else 1.1, color=color,
                    label=f"{label} {area:.3f} ({low:.3f}–{high:.3f})")
            rows.extend(
                {"stratum": title, "model": label, "fpr": x, "tpr": z,
                 "auroc": area, "ci_low": low, "ci_high": high}
                for x, z in zip(fpr, tpr, strict=True)
            )
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="False-positive rate (1 − specificity)",
               ylabel="True-positive rate (sensitivity)", title=title)
        ax.legend(loc="lower right", fontsize=5, title="AUROC (95% CI)", title_fontsize=5)
    fig.tight_layout()
    save(fig, out, "figure3_discrimination")
    pd.DataFrame(rows).to_csv(source / "figure3_source_data.csv", index=False)


def figure4(proposed: pd.DataFrame, out: Path, source: Path) -> None:
    frame = proposed.sort_values("easi_improvement_pct", kind="stable").copy()
    y = frame["y_easi75"].to_numpy(int)
    decision = frame["prt_bab_decision"].to_numpy(int)
    outcome = np.where((y == 1) & (decision == 1), "TP", np.where(
        (y == 0) & (decision == 0), "TN", np.where((y == 0) & (decision == 1), "FP", "FN")))
    palette = {"TP": "#0072B2", "TN": "#86BFE1", "FP": "#F1BB62", "FN": "#B87400"}
    fig, ax = plt.subplots(figsize=(7.2, 2.86))
    ax.axhspan(65, 85, color="#EAE6DC", zorder=0)
    x = np.arange(1, len(frame) + 1)
    ax.bar(x, frame["easi_improvement_pct"], width=0.86,
           color=[palette[v] for v in outcome], linewidth=0)
    ax.axhline(75, color="#444444", lw=1.2)
    ax.set(xlim=(0, 120), ylim=(0, 118), xlabel="Patients ordered by observed EASI improvement (n = 119)",
           ylabel="Observed EASI improvement\nat 16 weeks (%)")
    ax.set_title("Most misclassifications fall in the 65% to <85% band", loc="left")
    handles = [mpl.patches.Patch(color=palette[k], label=f"{k} {(outcome == k).sum()}")
               for k in ("TP", "TN", "FN", "FP")]
    ax.legend(handles=handles, ncol=2, loc="upper left")
    save(fig, out, "figure4_patient_outcomes")
    frame.assign(outcome_class=outcome).to_csv(source / "figure4_source_data.csv", index=False)


def figure5(attribution: pd.DataFrame, out: Path, source: Path) -> None:
    frame = attribution.sort_values("mean_abs_shap").copy()
    fig, ax = plt.subplots(figsize=(5.8, 8.0))
    ax.barh(frame["print_label"], frame["mean_abs_shap"], color="#4C72B0")
    ax.set(xlabel="Mean of absolute value of SHAP value", title="Aggregated SHAP values")
    fig.tight_layout()
    save(fig, out, "figure5_shap")
    attribution.to_csv(source / "figure5_source_data.csv", index=False)


def figure_s1(groups: pd.DataFrame, summary: pd.DataFrame, out: Path, source: Path) -> None:
    names = [("raw_score", "Response score, as produced", "#B66D00"),
             ("platt", "After logistic recalibration", "#555555"),
             ("isotonic", "After isotonic recalibration", "#0072B2")]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8))
    for ax, (key, title, color) in zip(axes.flat[:3], names, strict=True):
        part = groups[groups["estimate"] == key]
        ax.plot([0, 1], [0, 1], color="#BBBBBB", ls=(0, (2, 2)))
        lower = np.maximum(part["observed_rate"] - part["observed_ci_low"], 0)
        upper = np.maximum(part["observed_ci_high"] - part["observed_rate"], 0)
        ax.errorbar(part["mean_predicted"], part["observed_rate"],
                    yerr=[lower, upper],
                    color=color, marker="o", lw=1.3, capsize=2)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean model output",
               ylabel="Observed response frequency", title=title)
    ax = axes.flat[3]
    for key, title, color in names:
        row = summary[summary["estimate"] == key].iloc[0]
        ax.errorbar(row["calibration_slope"], row["calibration_intercept"],
                    xerr=[[row["calibration_slope"] - row["calibration_slope_ci_low"]],
                          [row["calibration_slope_ci_high"] - row["calibration_slope"]]],
                    yerr=[[row["calibration_intercept"] - row["calibration_intercept_ci_low"]],
                          [row["calibration_intercept_ci_high"] - row["calibration_intercept"]]],
                    marker="o", color=color, label=title, capsize=2)
    ax.axvline(1, color="#BBBBBB", ls=(0, (2, 2)))
    ax.axhline(0, color="#BBBBBB", ls=(0, (2, 2)))
    ax.set(xlabel="Calibration slope", ylabel="Calibration intercept", title="Calibration slope and intercept")
    ax.legend(fontsize=5)
    fig.tight_layout()
    save(fig, out, "figureS1_calibration")
    groups.to_csv(source / "figureS1_source_data.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results, out = args.results.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    source = out / "source_data"
    source.mkdir(exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    configure_style()
    proposed = pd.read_parquet(results / "proposed/patient_scores_and_decisions.parquet")
    comparators = pd.read_parquet(results / "comparators/patient_scores_and_decisions.parquet")
    figure2(proposed, out, source)
    figure3(proposed, comparators, out, source)
    figure4(proposed, out, source)
    # The published aggregate is the display source because the historical
    # KernelSHAP runner omitted its perturbation seed. The current runner is now
    # deterministic and is audited against this aggregate within tolerance.
    figure5(pd.read_csv(repo / "artifacts/reference/results/feature_importance.csv"), out, source)
    figure_s1(pd.read_csv(results / "calibration/calibration_groups.csv"),
              pd.read_csv(results / "calibration/calibration_summary.csv"), out, source)
    shutil.copyfile(repo / "assets/schematics/figure1/figure1_framework.jpg", out / "figure1_framework.jpg")
    shutil.copyfile(repo / "assets/schematics/figureS2/figureS2_validation.png", out / "figureS2_validation.png")


if __name__ == "__main__":
    main()
