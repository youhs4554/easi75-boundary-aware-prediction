#!/usr/bin/env python3
"""Analyses added for the revision, computed from the existing run outputs.

Every quantity here is derived from files already written by the estimation
runs.  Nothing is refitted, so nothing here can disagree with the numbers
already in the ledger; the point is to report quantities the earlier reporting
computed but did not surface.

Five groups:

``score_band``
    Performance inside and outside the ambiguity band, using the band bounds
    the folds actually selected.  Unlike the retrospective outcome stratum, this
    stratification is a function of the score alone and is therefore available
    at baseline for a new patient.

``score_deciles``
    Observed response frequency by decile of the score, with Wilson intervals.
    The rule-in / rule-out reading of the model.

``repeat_variability``
    The pooled estimate against the mean of the fifteen per-seed estimates.

``error_location``
    Whether misclassifications fall in the near-threshold band more often than
    the share of the cohort that band holds.

``band``
    The band each training partition selected, the operating point its midpoint
    supplies, and how many held-out predictions fall inside it.

Usage:
    python run_revision_analyses.py --results <dir> --output-dir <dir>
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score, roc_auc_score

from atopix_ml.strict_recovered_router import PRT_LOW_GRID

SEED_ORDER: tuple[int, ...] = (42, 7, 123, 456, 789, 1, 2, 3, 4, 5, 100, 200, 300, 400, 500)
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 42


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return (float("nan"), float("nan"))
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denominator
    half = z * np.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return (float(centre - half), float(centre + half))


def bootstrap_auroc_ci(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(y)
    draws = []
    for _ in range(BOOTSTRAP_DRAWS):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        draws.append(roc_auc_score(y[idx], score[idx]))
    if not draws:
        return (float("nan"), float("nan"))
    return (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))


def classification_block(y: np.ndarray, decision: np.ndarray) -> dict[str, Any]:
    tp = int(np.sum((decision == 1) & (y == 1)))
    tn = int(np.sum((decision == 0) & (y == 0)))
    fp = int(np.sum((decision == 1) & (y == 0)))
    fn = int(np.sum((decision == 0) & (y == 1)))
    n = tp + tn + fp + fn
    accuracy = (tp + tn) / n if n else float("nan")
    low, high = wilson(tp + tn, n)
    return {
        "n": n,
        "n_positive": tp + fn,
        "n_negative": tn + fp,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "accuracy": accuracy,
        "accuracy_ci_low": low,
        "accuracy_ci_high": high,
        "sensitivity": tp / (tp + fn) if (tp + fn) else float("nan"),
        "specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
        "f1": float(f1_score(y, decision, zero_division=0)),
    }


def score_band_analysis(
    patients: pd.DataFrame, band_low: float, band_high: float
) -> dict[str, Any]:
    """Performance inside and outside the ambiguity band on the score scale."""
    score = patients["patient_average_score"].to_numpy(dtype=float)
    y = patients["y_easi75"].to_numpy(dtype=int)
    decision = patients["prt_bab_decision"].to_numpy(dtype=int)
    inside = (score >= band_low) & (score <= band_high)

    blocks: dict[str, Any] = {
        "band_low": band_low,
        "band_high": band_high,
        "definition": (
            "patient-average score within the fold-median ambiguity band; a function of "
            "the score alone and therefore available at baseline for a new patient"
        ),
    }
    for label, mask in (("inside_band", inside), ("outside_band", ~inside)):
        block = classification_block(y[mask], decision[mask])
        block["response_rate"] = float(np.mean(y[mask])) if mask.any() else float("nan")
        if len(np.unique(y[mask])) == 2:
            block["auroc"] = float(roc_auc_score(y[mask], score[mask]))
        else:
            block["auroc"] = float("nan")
        blocks[label] = block

    inside_correct = (decision == y)[inside]
    outside_correct = (decision == y)[~inside]
    table = [
        [int(np.sum(~inside_correct)), int(np.sum(inside_correct))],
        [int(np.sum(~outside_correct)), int(np.sum(outside_correct))],
    ]
    blocks["accuracy_difference_fisher_p"] = float(stats.fisher_exact(table)[1])
    blocks["accuracy_difference"] = float(np.mean(inside_correct) - np.mean(outside_correct))
    return blocks


def score_decile_analysis(patients: pd.DataFrame, n_groups: int = 10) -> dict[str, Any]:
    score = patients["patient_average_score"].to_numpy(dtype=float)
    y = patients["y_easi75"].to_numpy(dtype=int)
    order = np.argsort(score, kind="mergesort")
    groups = np.array_split(order, n_groups)
    rows = []
    for position, members in enumerate(groups, start=1):
        successes = int(np.sum(y[members]))
        total = int(len(members))
        low, high = wilson(successes, total)
        rows.append(
            {
                "group": position,
                "n": total,
                "score_min": float(np.min(score[members])),
                "score_max": float(np.max(score[members])),
                "n_responder": successes,
                "observed_response_rate": successes / total,
                "wilson_ci_low": low,
                "wilson_ci_high": high,
            }
        )
    bottom, top = rows[0], rows[-1]
    top_two_n = rows[-1]["n"] + rows[-2]["n"]
    top_two_responder = rows[-1]["n_responder"] + rows[-2]["n_responder"]
    bottom_two_n = rows[0]["n"] + rows[1]["n"]
    bottom_two_responder = rows[0]["n_responder"] + rows[1]["n_responder"]
    return {
        "n_groups": n_groups,
        "groups": rows,
        "bottom_group_response_rate": bottom["observed_response_rate"],
        "top_group_response_rate": top["observed_response_rate"],
        "bottom_two_groups": {
            "n": bottom_two_n,
            "n_responder": bottom_two_responder,
            "response_rate": bottom_two_responder / bottom_two_n,
            "wilson_ci": wilson(bottom_two_responder, bottom_two_n),
        },
        "top_two_groups": {
            "n": top_two_n,
            "n_responder": top_two_responder,
            "response_rate": top_two_responder / top_two_n,
            "wilson_ci": wilson(top_two_responder, top_two_n),
        },
        "cohort_response_rate": float(np.mean(y)),
    }


def repeat_variability(patients: pd.DataFrame) -> dict[str, Any]:
    y = patients["y_easi75"].to_numpy(dtype=int)
    per_seed = []
    for seed in SEED_ORDER:
        column = f"seed_{seed}_score"
        per_seed.append(
            {
                "seed": seed,
                "auroc": float(roc_auc_score(y, patients[column].to_numpy(dtype=float))),
            }
        )
    values = np.array([row["auroc"] for row in per_seed], dtype=float)
    pooled = float(roc_auc_score(y, patients["patient_average_score"].to_numpy(dtype=float)))
    return {
        "pooled_auroc": pooled,
        "per_seed": per_seed,
        "mean_per_seed_auroc": float(values.mean()),
        "sd_per_seed_auroc": float(values.std(ddof=1)),
        "min_per_seed_auroc": float(values.min()),
        "max_per_seed_auroc": float(values.max()),
        "pooling_gain": pooled - float(values.mean()),
        "note": (
            "the pooled estimate is the area under the curve of the averaged score and "
            "exceeds the average of the fifteen per-seed areas; both are reported so the "
            "gain from averaging is visible rather than folded into one number"
        ),
    }


def error_location(patients: pd.DataFrame) -> dict[str, Any]:
    y = patients["y_easi75"].to_numpy(dtype=int)
    decision = patients["prt_bab_decision"].to_numpy(dtype=int)
    band = patients["boundary_65_to_lt85"].to_numpy(dtype=bool)
    wrong = decision != y
    n_total = len(y)
    n_band = int(band.sum())
    share = n_band / n_total
    errors_in = int(np.sum(wrong & band))
    errors_total = int(wrong.sum())
    binomial = stats.binomtest(errors_in, errors_total, share, alternative="greater")
    table = [
        [errors_in, n_band - errors_in],
        [errors_total - errors_in, (n_total - n_band) - (errors_total - errors_in)],
    ]
    return {
        "n_total": n_total,
        "n_in_band": n_band,
        "band_share_of_cohort": share,
        "n_errors_total": errors_total,
        "n_errors_in_band": errors_in,
        "expected_errors_in_band_if_uniform": errors_total * share,
        "observed_share_of_errors_in_band": errors_in / errors_total,
        "binomial_one_sided_p": float(binomial.pvalue),
        # The question has a direction — are errors enriched inside the band — so the
        # test is one-sided. The two-sided value is carried as well, because the paper
        # states elsewhere that its tests are two-sided and the exception has to be
        # visible rather than implied.
        "binomial_two_sided_p": float(
            stats.binomtest(errors_in, errors_total, share).pvalue
        ),
        "error_rate_in_band": errors_in / n_band,
        "error_rate_outside_band": (errors_total - errors_in) / (n_total - n_band),
        "fisher_p": float(stats.fisher_exact(table)[1]),
        "reading": (
            "the band holds most of the cohort, so most misclassifications fall in it; "
            "the enrichment beyond that share is not separable from chance at this size"
        ),
    }


def band_statistics(shard_dir: Path) -> dict[str, Any]:
    """The band selected within each training partition, and how it was used.

    The band supplies the operating point through its midpoint, and it identifies the
    patients whose score is ambiguous. It no longer routes anyone to a separate
    decision rule, so nothing here counts routed patients.
    """
    lows, highs, midpoints, test_midband, test_n = [], [], [], [], []
    for path in sorted(shard_dir.glob("*.json")):
        shard = json.loads(path.read_text())
        lows.append(shard["prt"]["low"])
        highs.append(shard["prt"]["high"])
        midpoints.append(shard["band"]["midpoint"])
        test_midband.append(shard["band"]["n_test_midband"])
        test_n.append(len(shard["test_indices"]))
    total_slots = int(np.sum(test_n))
    return {
        "n_folds": len(lows),
        "band_low_median": float(np.median(lows)),
        "band_high_median": float(np.median(highs)),
        "band_low_unique": sorted({float(v) for v in lows}),
        "band_high_unique": sorted({float(v) for v in highs}),
        "midpoint_median": float(np.median(midpoints)),
        "midpoint_min": float(np.min(midpoints)),
        "midpoint_max": float(np.max(midpoints)),
        "n_distinct_midpoints": len({float(v) for v in midpoints}),
        # The lower bound is chosen from a grid whose smallest value is 0.35. How often
        # the selection lands on that smallest value says whether the bound was set by
        # the data or by where the grid stops, and the answer belongs in the Methods.
        "n_folds_low_at_grid_floor": int(sum(1 for v in lows if v == min(PRT_LOW_GRID))),
        "low_grid_floor": float(min(PRT_LOW_GRID)),
        "n_folds_selecting_median_pair": int(
            sum(
                1
                for low, high in zip(lows, highs, strict=True)
                if low == float(np.median(lows)) and high == float(np.median(highs))
            )
        ),
        "total_held_out_slots": total_slots,
        "total_inside_band_slots": int(np.sum(test_midband)),
        "inside_band_fraction_of_slots": float(np.sum(test_midband) / total_slots),
        "mean_inside_band_per_fold": float(np.mean(test_midband)),
    }


def net_label_change(patients: pd.DataFrame) -> dict[str, Any]:
    """The reported operating point against the rule every comparator uses.

    The reported rule takes the midpoint of a band selected within each training
    partition; the comparators take the cut-point maximising the Youden index in
    theirs. Both are chosen on training data alone, so this is the like-for-like
    comparison of operating points on one score.
    """
    reported = patients["prt_bab_decision"].to_numpy(dtype=int)
    single = patients["youden_decision"].to_numpy(dtype=int)
    y = patients["y_easi75"].to_numpy(dtype=int)
    differ = reported != single
    return {
        "n_patients_with_different_label": int(differ.sum()),
        "patient_indices": patients.loc[differ, "patient_index"].astype(int).tolist(),
        "n_changed_to_correct": int(np.sum(differ & (reported == y))),
        "n_changed_to_incorrect": int(np.sum(differ & (reported != y))),
        "routing_metrics": classification_block(y, reported),
        "single_cut_point_metrics": classification_block(y, single),
        "note": (
            "the two operating points are selected by different rules on the same "
            "training partitions, and after the patient-level vote across the fifteen "
            "repeats they disagree on this many patients"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    results = Path(args.results)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    patients = pd.read_parquet(results / "proposed" / "patient_scores_and_decisions.parquet")
    band = band_statistics(results / "proposed" / "shards")

    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source": "existing run outputs; nothing refitted",
        "n_patients": int(len(patients)),
        "band": band,
        "net_label_change": net_label_change(patients),
        "score_band": score_band_analysis(
            patients, band["band_low_median"], band["band_high_median"]
        ),
        "score_deciles": score_decile_analysis(patients, n_groups=10),
        "score_quintiles": score_decile_analysis(patients, n_groups=5),
        "repeat_variability": repeat_variability(patients),
        "error_location": error_location(patients),
    }

    (out / "revision_analyses.json").write_text(json.dumps(payload, indent=2, sort_keys=True))

    deciles = pd.DataFrame(payload["score_deciles"]["groups"])
    deciles.to_csv(out / "score_deciles.csv", index=False)
    pd.DataFrame(payload["repeat_variability"]["per_seed"]).to_csv(
        out / "per_seed_auroc.csv", index=False
    )

    print(f"[band] {band['band_low_median']:.2f}-{band['band_high_median']:.2f}; "
          f"operating point {band['midpoint_min']:.3f}-{band['midpoint_max']:.3f} "
          f"over {band['n_distinct_midpoints']} distinct values; "
          f"{band['inside_band_fraction_of_slots']:.1%} of held-out slots inside the band")
    print(f"[net] patients whose final label differs from the single cut-point: "
          f"{payload['net_label_change']['n_patients_with_different_label']}")
    band = payload["score_band"]
    print(f"[band] inside n={band['inside_band']['n']} acc={band['inside_band']['accuracy']:.3f} | "
          f"outside n={band['outside_band']['n']} acc={band['outside_band']['accuracy']:.3f} | "
          f"Fisher p={band['accuracy_difference_fisher_p']:.3f}")
    rv = payload["repeat_variability"]
    print(f"[repeat] pooled {rv['pooled_auroc']:.4f} vs per-seed mean "
          f"{rv['mean_per_seed_auroc']:.4f} +/- {rv['sd_per_seed_auroc']:.4f} "
          f"(gain {rv['pooling_gain']:+.4f})")
    el = payload["error_location"]
    print(f"[errors] {el['n_errors_in_band']}/{el['n_errors_total']} in a band holding "
          f"{el['band_share_of_cohort']:.1%}; binomial p={el['binomial_one_sided_p']:.3f}, "
          f"Fisher p={el['fisher_p']:.3f}")
    sd = payload["score_deciles"]
    print(f"[deciles] bottom two {sd['bottom_two_groups']['n_responder']}/"
          f"{sd['bottom_two_groups']['n']}; top two {sd['top_two_groups']['n_responder']}/"
          f"{sd['top_two_groups']['n']}")


if __name__ == "__main__":
    main()
