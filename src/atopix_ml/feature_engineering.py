"""EDA-driven feature engineering for EASI-75 prediction.

Insights from reports/eda/EDA_REPORT.md:
  - Top univariate AUROC: 음식/식품 알레르겐(0.69), Total IgE (0.63),
    Baseline Erythema/Lower EASI/Scalp eczema (~0.60), Baseline EASI (0.59).
  - Top mutual info: Hx Other allergies, FHx Allergic rhinitis, Dyslipidemia,
    Previous cyclosporine, Trunk EASI.
  - Many baseline EASI components correlated; aggregate composites may help.
  - Allergy/comorbidity burden distributed across many binary features.
  - Lab values (LDH, IgE, eosinophil) high variance, log-transform candidate.

Engineered features:
  1. severity_total = sum(Baseline H&N + Upper + Trunk + Lower EASI)
  2. severity_max = max of the 4 EASI components
  3. erythema_papulation_excoriation_lichenification_sum (severity composite 2)
  4. allergy_burden = Hx_AllergicRhinitis + Hx_Asthma + Hx_OtherAllergies
  5. fhx_burden = FHx_AD + FHx_AllergicRhinitis + FHx_Asthma
  6. comorbidity_burden = Hx_DM + Hx_HTN + Hx_Dyslipidemia + Hx_Severe... + Hx_Auto + Hx_psych
  7. prev_treatment_count = sum of 5 previous treatment binary flags
  8. allergen_panel_total = sum of 7 allergen panel scores (NaN-safe mean);
     allergen_panel_count = number of categories at or above class 2
  9. log_LDH, log_IgE, log_eosinophil (handle skewness)
  10. easi_per_bsa = Baseline EASI / Baseline BSA (clip to avoid div0)
  11. iga_x_easi = Baseline IGA × Baseline EASI (interaction)
  12. age_x_disease_duration = Age × Disease duration
  13. h_n_easi_ratio = Baseline H&N EASI / Baseline EASI
  14. early_onset = (Onset age <= 12) flag
"""
from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

#: MAST class at or above which a category counts as sensitised.
#:
#: The sensitisation-count variable counts allergen categories whose maximum
#: panel intensity reaches this class. Class 2 is the threshold used
#: throughout the analysis; the two panel variables that use the maximum
#: class as a continuous value (``fe_allergen_panel_mean``,
#: ``fe_allergen_panel_max``) are threshold-independent and unaffected by it.
SENSITISATION_CLASS_THRESHOLD: Final[int] = 2

#: Category keywords identifying the seven MAST panel columns.
PANEL_CATEGORY_KEYWORDS: Final[tuple[str, ...]] = (
    '진드기', '실내 환경', '동물털', '곰팡이', '꽃가루', '음식/식품', '벌독',
)

#: Header markers that disqualify a keyword match from being a panel column.
#: ``갯수 (n)`` marks a per-category count of sensitised allergens — an
#: aggregate over the category, not the category's intensity score.
PANEL_EXCLUDED_MARKERS: Final[tuple[str, ...]] = ('갯수', '(n)')

#: Header marker identifying the per-category sensitised-allergen count
#: columns. Present only when the feature matrix was built to carry them.
COUNT_COLUMN_MARKER: Final[str] = '갯수'

#: How the allergen-burden predictor is defined.
#:
#: ``'categories'``
#:     Number of panel categories sensitised, i.e. categories whose maximum
#:     class reaches ``SENSITISATION_CLASS_THRESHOLD``. Range 0-7. Computable
#:     from the maximum-class columns alone.
#: ``'allergens'``
#:     Total number of individual allergens testing positive at or above that
#:     class, summed over the seven categories. Requires the per-category
#:     count columns to be present in the input matrix.
#:
#: Both are emitted when :func:`build_engineered_features` is called with
#: ``allergen_burden='both'``; the default emits the configured one only.
ALLERGEN_BURDEN_DEFAULT: Final[str] = 'allergens'


def build_engineered_features(
    X: pd.DataFrame,
    *,
    allergen_burden: str = ALLERGEN_BURDEN_DEFAULT,
) -> pd.DataFrame:
    """Add engineered features to a copy of X. Original columns preserved.

    Parameters
    ----------
    X
        Baseline feature matrix.
    allergen_burden
        Which allergen-burden predictor to emit under the name
        ``fe_allergen_panel_count_nonzero``:

        * ``'allergens'`` — total positive allergens across the panel.
        * ``'categories'`` — number of sensitised categories.
        * ``'both'`` — emit the category count under the established name and
          the allergen total as ``fe_allergen_positive_total``. Intended for
          sensitivity analysis, not for the primary specification.

        ``'allergens'`` and ``'both'`` require the per-category count columns
        in ``X``; a :class:`ValueError` is raised when they are absent, rather
        than silently returning a differently-defined variable.
    """
    if allergen_burden not in ('allergens', 'categories', 'both'):
        raise ValueError(
            f"allergen_burden must be 'allergens', 'categories' or 'both' — "
            f"got {allergen_burden!r}"
        )
    out = X.copy()
    n = len(out)

    def col(name):
        return X[name] if name in X.columns else pd.Series(np.full(n, np.nan), index=X.index)

    # 1. Severity total / max (across 4 anatomical EASI components)
    easi_components = [
        'Baseline H&N EASI', 'Baseline Upper EASI',
        'Baseline Trunk EASI', 'Baseline Lower EASI',
    ]
    available = [c for c in easi_components if c in X.columns]
    if available:
        comp_df = X[available]
        out['fe_severity_total'] = comp_df.sum(axis=1, skipna=True)
        out['fe_severity_max'] = comp_df.max(axis=1, skipna=True)
        out['fe_severity_mean'] = comp_df.mean(axis=1, skipna=True)
        out['fe_severity_std'] = comp_df.std(axis=1, skipna=True)

    # 2. Severity component sum (E/P/Ex/L)
    sub_components = [
        'Baseline Erythema score', 'Baseline Papulation score',
        'Baseline Excoriation score', 'Baseline Lichenification score',
    ]
    available_sub = [c for c in sub_components if c in X.columns]
    if available_sub:
        out['fe_subcomp_sum'] = X[available_sub].sum(axis=1, skipna=True)
        out['fe_subcomp_max'] = X[available_sub].max(axis=1, skipna=True)

    # 3. Allergy burden (personal Hx)
    allergy_hx = ['Hx of Allergic rhinitis', 'Hx of Asthma', 'Hx of Other allergies']
    avail = [c for c in allergy_hx if c in X.columns]
    if avail:
        out['fe_allergy_burden'] = X[avail].sum(axis=1, skipna=True)

    # 4. Family history burden
    fhx = ['FHx of Atopic dermatitis', 'FHx of Allergic rhinitis', 'FHx of Asthma']
    avail = [c for c in fhx if c in X.columns]
    if avail:
        out['fe_fhx_burden'] = X[avail].sum(axis=1, skipna=True)

    # 5. Comorbidity burden
    comorb = [
        'Hx of DM', 'Hx of HTN', 'Hx of Dyslipidemia',
        'Hx of Severe skin infection', 'Hx of Autoimmune diseases',
        'Hx of psychological disorders',
    ]
    avail = [c for c in comorb if c in X.columns]
    if avail:
        out['fe_comorbidity_burden'] = X[avail].sum(axis=1, skipna=True)

    # 6. Previous treatment count
    prev_tx = [
        'Previous treatment of topical corticosteroid',
        'Previous treatment of total calcineurin inhibitor',
        'Previous treatment of systemic corticosteroid',
        'Previous treatment of cyclosporine',
        'Previous treatment of MTX',
    ]
    avail = [c for c in prev_tx if c in X.columns]
    if avail:
        out['fe_prev_treatment_count'] = X[avail].sum(axis=1, skipna=True)

    # 7. Allergen panel summary (use mean to handle NaN gracefully)
    #
    # The panel columns are the seven per-category MAST maximum-intensity
    # scores (class 0-6). Keyword matching alone is not sufficient: some
    # source revisions also carry per-category *count* columns whose headers
    # share the same category keywords. Those are aggregates, not intensity
    # scores, and must never join the panel — including them would silently
    # redefine every derived variable below. They are excluded explicitly.
    panel_cols = [
        c for c in X.columns
        if any(k in c for k in PANEL_CATEGORY_KEYWORDS)
        and not any(m in c for m in PANEL_EXCLUDED_MARKERS)
    ]
    count_cols = [
        c for c in X.columns
        if isinstance(c, str) and COUNT_COLUMN_MARKER in c
        and any(k in c for k in PANEL_CATEGORY_KEYWORDS)
    ]
    if panel_cols:
        # Threshold-free panel summaries — identical under either definition
        # of the burden predictor below.
        out['fe_allergen_panel_mean'] = X[panel_cols].mean(axis=1, skipna=True)
        out['fe_allergen_panel_max'] = X[panel_cols].max(axis=1, skipna=True)

        # Number of categories sensitised, where sensitisation is a panel
        # score at or above SENSITISATION_CLASS_THRESHOLD.
        n_categories = (X[panel_cols] >= SENSITISATION_CLASS_THRESHOLD).sum(axis=1)

        if allergen_burden == 'categories':
            out['fe_allergen_panel_count_nonzero'] = n_categories
        else:
            if len(count_cols) != 7:
                raise ValueError(
                    f"allergen_burden={allergen_burden!r} needs the seven "
                    f"per-category count columns in X — found {len(count_cols)}. "
                    "Build the feature matrix with the counts retained."
                )
            # Total individual allergens positive at or above the threshold,
            # summed across the seven categories.
            n_allergens = X[count_cols].sum(axis=1, skipna=True)
            # A category with no reading contributes nothing to the sum; keep
            # the all-missing case missing rather than reporting it as zero.
            all_missing = X[count_cols].isna().all(axis=1)
            n_allergens = n_allergens.mask(all_missing)
            if allergen_burden == 'allergens':
                out['fe_allergen_panel_count_nonzero'] = n_allergens
            else:  # 'both'
                out['fe_allergen_panel_count_nonzero'] = n_categories
                out['fe_allergen_positive_total'] = n_allergens

    # 8. Lab log-transforms (heavy tails)
    for src, dst in [('LDH', 'fe_log_LDH'), ('Total IgE', 'fe_log_IgE'),
                     ('Total eosinophil count', 'fe_log_eos')]:
        if src in X.columns:
            v = pd.to_numeric(X[src], errors='coerce')
            out[dst] = np.log1p(v.clip(lower=0))

    # 9. EASI per BSA (intensity adjusted by extent)
    if 'Baseline EASI' in X.columns and 'Baseline BSA' in X.columns:
        bsa = pd.to_numeric(X['Baseline BSA'], errors='coerce').clip(lower=1)
        out['fe_easi_per_bsa'] = pd.to_numeric(X['Baseline EASI'], errors='coerce') / bsa

    # 10. IGA × EASI interaction (clinical impression × measurement)
    if 'Baseline IGA' in X.columns and 'Baseline EASI' in X.columns:
        out['fe_iga_x_easi'] = (
            pd.to_numeric(X['Baseline IGA'], errors='coerce') *
            pd.to_numeric(X['Baseline EASI'], errors='coerce')
        )

    # 11. Age × disease duration (cumulative exposure)
    if 'Age' in X.columns and 'Disease duration' in X.columns:
        out['fe_age_x_duration'] = (
            pd.to_numeric(X['Age'], errors='coerce') *
            pd.to_numeric(X['Disease duration'], errors='coerce')
        )

    # 12. H&N EASI ratio of total (face/neck dominance)
    if 'Baseline H&N EASI' in X.columns and 'Baseline EASI' in X.columns:
        total = pd.to_numeric(X['Baseline EASI'], errors='coerce').clip(lower=0.1)
        out['fe_hn_easi_ratio'] = pd.to_numeric(X['Baseline H&N EASI'], errors='coerce') / total

    # 13. Early onset flag (childhood-onset AD)
    if 'Onset age' in X.columns:
        out['fe_early_onset'] = (pd.to_numeric(X['Onset age'], errors='coerce') <= 12).astype(int)

    # 14. Phenotype one-hot (if Phenotype present and has multiple values)
    if 'Phenotype' in X.columns:
        ph = pd.to_numeric(X['Phenotype'], errors='coerce').fillna(-1).astype(int)
        for ph_val in sorted(ph.unique()):
            if ph_val < 0:
                continue
            out[f'fe_phenotype_{ph_val}'] = (ph == ph_val).astype(int)

    return out


def load_fe_subset(base_subset: str = 'D'):
    """Load preprocessing bundle, optionally select subset, then add engineered features."""
    X = pd.read_parquet('reports/preprocessing_bundle_X.parquet')
    y_bin = pd.read_parquet('reports/preprocessing_bundle_y_easi75.parquet').to_numpy().reshape(-1).astype(int)
    y_reg = pd.read_parquet('reports/preprocessing_bundle_y_improvement_pct.parquet').to_numpy().reshape(-1).astype(float)
    fd = pd.read_csv('reports/feature_dictionary.csv')
    flag = f'subset_{base_subset.upper()}'
    cols = [c for c in X.columns if c in set(fd.loc[fd[flag], 'feature'].tolist())]
    X_sub = X[cols]
    X_eng = build_engineered_features(X_sub)
    return X_eng, y_bin, y_reg


if __name__ == '__main__':
    X_eng, y_bin, y_reg = load_fe_subset('D')
    print(f'Original D: 56 features. Engineered: {X_eng.shape[1]} features')
    eng_cols = [c for c in X_eng.columns if c.startswith('fe_')]
    print(f'Engineered features ({len(eng_cols)}):')
    for c in eng_cols:
        v = X_eng[c]
        print(f'  {c}: NaN={v.isna().sum()} mean={v.mean():.3f} std={v.std():.3f}')
