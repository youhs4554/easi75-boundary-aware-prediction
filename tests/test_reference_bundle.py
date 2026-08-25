from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_author_supplied_figure1_is_frozen() -> None:
    path = ROOT / "artifacts/reference/figures/figure1_framework.jpg"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "7f9e659d1f633dd0a05ab85e72bbec0537f59874c1b73a7d7c5635a08a36d75b"
    )


def test_reference_tables_are_complete() -> None:
    expected = {"table1", "table2", "tableS1", "tableS2", "tableS3", "tableS4",
                "tableS6", "tableS7", "tableS8", "tableS9"}
    found = {path.stem for path in (ROOT / "artifacts/reference/tables").glob("*.csv")}
    assert found == expected
    for path in (ROOT / "artifacts/reference/tables").glob("*.csv"):
        assert not pd.read_csv(path).empty


def test_reference_manifest() -> None:
    base = ROOT / "artifacts/reference"
    for line in (base / "manifest.sha256").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        assert hashlib.sha256((base / relative).read_bytes()).hexdigest() == expected


def test_no_participant_level_artifacts_are_distributed() -> None:
    forbidden = {".xlsx", ".xls", ".parquet", ".docx"}
    paths = [path for path in ROOT.rglob("*") if path.is_file()
             and ".venv" not in path.parts and "outputs" not in path.parts]
    assert not [path for path in paths if path.suffix.lower() in forbidden]
