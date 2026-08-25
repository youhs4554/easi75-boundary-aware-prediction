# Release audit

## Reproduction status

- Proposed framework: 75/75 folds rerun; patient-average score and all six
  decision vectors match the preserved final run exactly.
- Leakage probe: 75/75 folds passed after held-out outcomes were corrupted.
- Comparators: all five displayed models and the ablation-only multi-task learner
  rerun; patient decisions match exactly. Random-forest probabilities differ by
  at most `4.45e-16` from parallel floating-point reduction.
- Calibration: all reported estimates reproduce exactly.
- Paired statistics: the reported AUROC differences, DeLong tests, patient
  bootstrap intervals, McNemar discordances, and Holm corrections reproduce.
- Ablation: all five stages reproduce (AUROC 0.763, 0.786, 0.824, 0.824,
  0.824; near-threshold accuracy 0.592, 0.658, 0.711, 0.711, 0.737).
- Endpoint-definition sensitivity: all seven cut-offs reproduce from fixed
  held-out predictions.
- Figures: Figures 2–5 and Supplementary Figure S1 are regenerated in PNG,
  TIFF, PDF, and SVG with source-data CSVs; Figure 1 and Supplementary Figure S2
  are fixed non-data artifacts.
- Privacy: no workbook, Parquet file, DOCX, patient identifier, row-level
  prediction, or row-level attribution is tracked.

## Historical SHAP reproducibility gap

The original KernelSHAP runner did not seed NumPy's perturbation sampler. A full
fresh rerun preserved the same top-five variable set but moved absolute values
by roughly `1e-4` to `1e-3` and swapped the two nearly tied fourth/fifth ranks.
The paper already limits interpretation to top-group membership rather than
adjacent rank order.

The release fixes this by setting a deterministic fold-specific perturbation
seed. The aggregate CSV actually used in the paper is retained as the Figure 5
and Supplementary Table S6 source; fresh SHAP runs are checked for top-five
membership and a 0.002 absolute-value tolerance. Participant-level SHAP arrays
are not public.

## Manuscript-package findings

These findings concern the final DOCX, not the executable model results:

1. The final DOCX lists and cites Supplementary Tables S5 and S10 but contains
   neither table body. Their base-learner and comparator specifications are
   reconstructed in `docs/MODELS.md`, `configs/`, and the implementation.
2. Table 1's IGA percentages use denominators inconsistent with their printed
   counts. From the printed counts and n=65/54, the responder percentages are
   27.7%, 32.3%, and 40.0%, and the non-responder percentages are 18.5%, 27.8%,
   and 53.7%; the final DOCX prints 28.6%, 33.3%, 41.3% and 18.9%, 28.3%, 54.7%.
3. Supplementary Table S1's onset-age categories mix the available-data
   denominator overall (n=109) with full-group denominators in the responder and
   non-responder columns.
4. The author-supplied Figure 1 is preserved verbatim as requested. Its right
   green outside-band box visibly reads `s_i < U*`; the executable and Methods
   correctly implement `s_i > U*` for direct responder assignment.

These items are documented rather than silently changed because the request was
to release the code and final artifacts, not to modify the manuscript or the
author-supplied Figure 1.

## Figure QA

The Python figure source passes the publication preflight with no failures:
editable SVG/PDF text, 600-dpi TIFF export, publication-safe font stack, no
cross-backend rendering, no row sampling, and no unsafe colormap. The single
remaining automated warning is a false positive: NumPy's random generator is
used for the declared 10,000-draw patient bootstrap, not simulated display data.
