from __future__ import annotations

from atopix_ml.strict_proposed import FEATURE_NAMES, OUTER_SEEDS
from atopix_ml.strict_recovered_router import PRT_HIGH_GRID, PRT_LOW_GRID


def test_published_protocol_is_frozen() -> None:
    assert len(FEATURE_NAMES) == 23
    assert tuple(OUTER_SEEDS) == (42, 7, 123, 456, 789, 1, 2, 3, 4, 5, 100, 200, 300, 400, 500)
    assert tuple(PRT_LOW_GRID) == (0.35, 0.40, 0.45, 0.50, 0.55)
    assert tuple(PRT_HIGH_GRID) == (0.55, 0.60, 0.65, 0.70, 0.75)
