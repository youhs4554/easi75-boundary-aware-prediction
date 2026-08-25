#!/usr/bin/env python3
"""Evaluate fixed held-out predictions under alternative EASI cut-offs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

THRESHOLDS = (70.0, 72.5, 74.0, 75.0, 76.0, 77.5, 80.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposed", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_parquet(args.proposed / "patient_scores_and_decisions.parquet").sort_values("patient_index")
    improvement = frame["easi_improvement_pct"].to_numpy()
    score = frame["patient_average_score"].to_numpy()
    decision = frame["prt_bab_decision"].to_numpy()
    recorded = frame["y_easi75"].to_numpy()
    if not ((improvement >= 75.0).astype(int) == recorded).all():
        raise RuntimeError("recorded EASI-75 labels do not equal improvement >= 75")

    rows = []
    for threshold in THRESHOLDS:
        y = (improvement >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, decision, labels=[0, 1]).ravel()
        rows.append(
            {
                "easi_improvement_cutoff_pct": threshold,
                "labels_changed_vs_75": int((y != recorded).sum()),
                "responders": int(y.sum()),
                "auroc": float(roc_auc_score(y, score)),
                "sensitivity": float(tp / (tp + fn)),
                "specificity": float(tn / (tn + fp)),
                "f1": float(f1_score(y, decision)),
                "true_positive": int(tp),
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
            }
        )
    baseline = next(row for row in rows if row["easi_improvement_cutoff_pct"] == 75.0)
    if abs(baseline["auroc"] - 0.823931623931624) > 1e-12:
        raise RuntimeError("the 75% reference result does not reproduce AUROC 0.824")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_dir / "endpoint_sensitivity.csv", index=False)
    (args.output_dir / "endpoint_sensitivity.json").write_text(
        json.dumps({"analysis": "fixed predictions; no model refitting", "rows": rows}, indent=2)
        + "\n"
    )


if __name__ == "__main__":
    main()
