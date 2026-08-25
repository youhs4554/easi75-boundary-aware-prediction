# Model specification

## Proposed framework

All five learners use the same 23 predictors. Robust-raw and quantile-raw
learners use L2 logistic models with modified Huber alternatives; the three
expanded learners use L1 logistic models. Each learner fits the 65%, 75%, and
85% improvement tasks, and the quantile-raw learner additionally fits 55% and
95%. Candidate outputs are calibrated, transformed to training-reference
percentile ranks, averaged across cut-offs, and combined through the fixed
hierarchy in `documented_proposed.py`.

The ambiguity-band grids are:

```text
G_L = {0.35, 0.40, 0.45, 0.50, 0.55}
G_U = {0.55, 0.60, 0.65, 0.70, 0.75}
```

The training objective is overall accuracy minus 0.5 times the standard
deviation of accuracy across five observed-improvement bins. Inside the chosen
band, the corrector is AdaBoost-SAMME with ten logistic weak learners,
`C=0.5`, learning rate `0.3`, and random seed `42`.

## Displayed comparators

- regularized logistic regression;
- radial-basis-function support vector machine;
- random forest (three fixed seeds);
- LightGBM (three fixed seeds);
- XGBoost (three fixed seeds).

Every comparator uses the same patients, predictors, outer/inner partitions,
and training-only preprocessing. Its classification threshold maximizes the
Youden index on outer-training inner out-of-fold predictions. Exact settings are
stored in `configs/comparator_specifications.json` and implemented in
`src/atopix_ml/comparison_models.py`.

The single multi-task learner is run only to reconstruct the framework ablation
table; it is not part of the five-comparator multiplicity family.
