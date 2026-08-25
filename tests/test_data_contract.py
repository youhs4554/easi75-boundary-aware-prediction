from __future__ import annotations

import os
from pathlib import Path

import pytest

from atopix_ml.repro_data import FEATURE_SHA256, load_cohort


def test_restricted_data_contract_when_available() -> None:
    value = os.environ.get("EASI75_DATA")
    if not value:
        pytest.skip("set EASI75_DATA to run the restricted-data integration test")
    cohort = load_cohort(Path(value))
    assert cohort.X.shape == (119, 23)
    assert int(cohort.labels.sum()) == 65
    assert cohort.feature_sha256 == FEATURE_SHA256
