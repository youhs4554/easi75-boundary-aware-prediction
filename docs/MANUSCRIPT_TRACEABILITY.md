# Manuscript traceability

Audit target before the author-supplied Figure 1 override:

```text
Manuscript_AD_prediction_boundary_aware_clinician_revised_PEER_REVIEWED_RED_GREEN_BOUNDARY_CORRECTOR_NATIVE_EQUATION_YELLOW_FIGURE3B_TABLES7_AUROC_95CI_FINAL_LEGEND_DEFAULT (26.08.24).docx
SHA-256: 305a1706c3527108a088a9916d0643a08e6f0049e986085d8241c871f568bed2
```

The author-supplied Figure 1 dated 2026-08-25 supersedes the image embedded in
that DOCX. Its SHA-256 is
`7f9e659d1f633dd0a05ab85e72bbec0537f59874c1b73a7d7c5635a08a36d75b`.

| Display item | Regenerated from | Public reference |
|---|---|---|
| Figure 1 | Fixed author-supplied schematic | `artifacts/reference/figures/figure1_framework.jpg` |
| Figure 2 | restricted continuous outcomes | `figure2_outcome_distribution.png` |
| Figure 3 | proposed/comparator held-out scores | `figure3_discrimination.png` |
| Figure 4 | restricted outcomes + framework decisions | `figure4_patient_outcomes.png` |
| Figure 5 | four-seed score SHAP analysis | `figure5_shap.png` |
| Supplementary Figure S1 | held-out scores + fold-fitted calibration maps | `figureS1_calibration.png` |
| Supplementary Figure S2 | Fixed validation schematic | `figureS2_validation.png` |
| Table 1 / S1 | protected aggregate clinical tables | `artifacts/reference/tables/` |
| Table 2 | proposed and five comparator metrics | `scripts/build_tables.py` |
| Table S2 | frozen 23-predictor dictionary | `artifacts/reference/tables/tableS2.csv` |
| Table S3 | paired DeLong/bootstrap/McNemar analyses | `scripts/run_paired_stats.py` |
| Table S4 | calibration analysis | `scripts/run_calibration_hosmer.py` |
| Table S6 | score SHAP attribution | `scripts/run_attribution.py` |
| Table S7 | near-threshold outcome stratum | `scripts/build_tables.py` |
| Table S8 | incremental pipeline configurations | `scripts/run_ablation_stages.py` |
| Table S9 | endpoint-definition sensitivity | `scripts/run_endpoint_sensitivity.py` |
| Table S5 / S10 specifications | frozen implementation and configuration records | `docs/MODELS.md`, `configs/` |
