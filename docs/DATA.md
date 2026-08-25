# Data access and privacy

The frozen workbook contains human participant data and a worksheet with direct
identifiers. It is not committed, archived in GitHub Releases, or covered by the
software license.

## Public repository contents

- analysis and visualization code;
- exact dependency lockfile;
- aggregate manuscript tables and performance summaries;
- publication figures;
- fixed, non-data schematics;
- checksums and a manuscript-to-code traceability map.

## Restricted contents

- source Excel workbook;
- participant identifiers;
- row-level predictors and outcomes;
- patient-level held-out scores, decisions, and SHAP values;
- fitted fold models that could expose participant information.

Authorized investigators can place the frozen workbook in `data/private/` and
run the complete pipeline locally. Generated patient-level files are written
under the gitignored `outputs/` directory.

## Data Availability text requiring author confirmation

> The participant-level data supporting this study are not publicly available
> because they contain sensitive clinical information and direct identifiers in
> the source record. The analysis code, aggregate results, and figure/table
> generation materials are available in the accompanying public repository.
> The institutionally approved route, eligibility criteria, and contact point
> for access to de-identified participant-level data remain to be confirmed.

Before submission, the authors must confirm the data controller, ethics/consent
basis, access-review process, permitted data-use terms, and a durable controlled
access route. “Available upon reasonable request” should not be used without
those details.
