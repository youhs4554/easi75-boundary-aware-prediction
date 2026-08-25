# Reproducibility

## Frozen analysis contract

- Cohort: 119 patients, 65 EASI-75 responders and 54 non-responders.
- Inputs: 23 pre-treatment predictors.
- Validation: 15 fixed seeds × 5 regression-stratified outer folds.
- Inner validation: five folds within each outer-training partition.
- Discrimination: mean of 15 held-out scores for each patient.
- Classification: majority vote of 15 held-out decisions for each patient.
- Framework decision: direct labels outside a fold-selected ambiguity band and
  an AdaBoost-SAMME logistic corrector inside the band.
- Uncertainty: 10,000 patient-level bootstrap resamples, seed 42.
- Multiplicity: Holm correction over the five displayed comparators, separately
  for discrimination and classification.

The loader checks the workbook SHA-256 and the reconstructed feature and label
hashes. Every proposed-model fold also reruns after corrupting its held-out
outcomes; every learned-quantity hash must remain unchanged.

## One command

```bash
uv sync --frozen
uv run python scripts/run_all.py \
  --data data/private/raw_data_v5_260810.xlsx \
  --output outputs/reproduction \
  --jobs 8
```

The full run includes the four-seed KernelSHAP analysis and takes longer than
the performance experiment. For a quick code-path check, add
`--skip-attribution`; this does not constitute full reproduction.

## Individual stages

```bash
uv run python scripts/run_proposed.py --data DATA.xlsx --output-dir outputs/reproduction/proposed --jobs 8
uv run python scripts/run_comparators.py --data DATA.xlsx --output-dir outputs/reproduction/comparators --jobs 8
uv run python scripts/run_paired_stats.py --proposed outputs/reproduction/proposed --comparators outputs/reproduction/comparators --output-dir outputs/reproduction/paired_stats --comparator-set main
uv run python scripts/run_attribution.py --data DATA.xlsx --output-dir outputs/reproduction/attribution --label-map configs/feature_labels.csv
uv run python scripts/make_figures.py --results outputs/reproduction --output outputs/reproduction/figures
uv run python scripts/build_tables.py --results outputs/reproduction --output outputs/reproduction/tables
uv run python scripts/verify_release.py --results outputs/reproduction
```

All commands are resumable: completed fold shards are reused only when their
configuration hash matches the current run.

## Expected headline results

| Model | AUROC | Sensitivity | Specificity | F1 |
|---|---:|---:|---:|---:|
| Proposed framework | 0.824 | 0.738 | 0.778 | 0.768 |
| Logistic regression | 0.763 | 0.646 | 0.611 | 0.656 |
| RBF SVM | 0.685 | 0.554 | 0.685 | 0.610 |
| Random forest | 0.658 | 0.646 | 0.593 | 0.651 |
| LightGBM | 0.625 | 0.554 | 0.593 | 0.585 |
| XGBoost | 0.579 | 0.662 | 0.463 | 0.628 |

Small last-bit differences in tree-ensemble probabilities may occur with
different thread scheduling, but the frozen environment reproduces the printed
metrics and all verification tolerances.
