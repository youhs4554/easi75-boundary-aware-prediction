# Reference artifacts

This directory contains the aggregate outputs and display items used to verify a
fresh restricted-data run without distributing participant-level information.

- `figures/`: final manuscript images; Figure 1 is the author-supplied override.
- `tables/`: CSV extraction of every table body actually present in the final
  DOCX (Table 1, Table 2, and S1–S4/S6–S9).
- `results/`: aggregate metrics and statistical summaries.
- `manifest.sha256`: checksums for every reference file.

The absence of S5 and S10 CSVs reflects their absence from the final DOCX. Their
configurations are preserved in `configs/` and `docs/MODELS.md`.
