# Data

The study workbook contains human participant data and direct identifiers in a
source worksheet. It is therefore not distributed in this repository.

Authorized users should place the frozen workbook at:

```text
data/private/raw_data_v5_260810.xlsx
```

Expected SHA-256:

```text
720187458e0ad68c8d53514a4eac3df7b9ab770cce802d06b58208cbc676d721
```

The loader reads only `Sheet1`, reconstructs the 23 baseline predictors, and
checks the source, feature, outcome, cohort-size, and class-count contracts
before any model is fitted. See [docs/DATA.md](../docs/DATA.md) for the access
and privacy rationale.
