"""Load the restricted cohort and reconstruct the published 23 predictors.

The source workbook is intentionally not distributed. Authorized users place it
at ``data/private/raw_data_v5_260810.xlsx`` or pass an explicit path. The loader
rejects any workbook whose content or reconstructed predictor matrix differs
from the analysis frozen for the paper.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from atopix_ml.feature_engineering import build_engineered_features
from atopix_ml.strict_proposed import FEATURE_NAMES

SOURCE_SHA256: Final = "720187458e0ad68c8d53514a4eac3df7b9ab770cce802d06b58208cbc676d721"
FEATURE_SHA256: Final = "a9abbb72a31267c9f8288f55c0e2301a125c10f46001450ac1eb977088b2989a"
LABEL_SHA256: Final = "fd4c79a8ebb2cf3347ea4b3b2d1e619406b483cb9615499db0f893c5793afe5d"
# Direct openpyxl decoding of the frozen workbook. Four values differ from the
# archived Parquet serialization by one floating-point ULP (<8e-15); no patient
# changes stratum, fold, or endpoint because of that representation detail.
IMPROVEMENT_SHA256: Final = "3f32a667da8282052828ab346e5d4f6fe433dae5048cf2eb2188706e1ac97461"

RENAME_MAP: Final = {
    "진드기류\n(D. pteronyssinus, D. farinae, Storage mite, Acarus siro) 중 최대 allergen 강도 (class 0~6)":
        "진드기류\n(D. pteronyssinus, D. farinae, Storage mite, Acarus siro)",
    "실내 환경 알레르겐\n(House dust, Cockroach) 중 최대 allergen 강도 (class 0~6)":
        "실내 환경 알레르겐\n(House dust, Cockroach)",
    "동물털, 상피류\n(Cat, Dog, Horse, Guinea pig, Sheep, Rabbit, Hamster) 중 최대 allergen 강도 (class 0~6)":
        "동물털, 상피류\n(Cat, Dog, Horse, Guinea pig, Sheep, Rabbit, Hamster)",
    "곰팡이류\n(Cladosporium, Aspergillus, Alternaria, Penicillium notatum, Candida) 중 최대 allergen 강도 (class 0~6)":
        "곰팡이류\n(Cladosporium, Aspergillus, Alternaria, Penicillium notatum)",
    "수목, 잡초, 잔디 꽃가루 pollen 그룹 중 최대 allergen 강도 (class 0~6)":
        "수목, 잡초, 잔디 꽃가루 pollen 그룹",
    "음식/식품 관련 알레르겐\n(Egg White, Milk, Soy bean, Maize, Sesame, Crab, Shrimp, Potato, Apple, Cacao, Peach, Mackerel 등) 중 최대 allergen 강도 (class 0~6)":
        "음식/식품 관련 알레르겐\n(Egg White, Milk, Soy bean, Maize, Sesame, Crab, Shrimp, Potato, Apple, Cacao, Peach, Mackerel)",
    "벌독, 기타 특수 알레르겐\n(Honey bee venom, Common wasp venom, Latex) 중 최대 allergen 강도 (class 0~6)":
        "벌독, 기타 특수 알레르겐\n(Honey bee venom, Common wasp venom, Latex)",
    "Cr": "Creatinine",
}


def array_sha256(values: np.ndarray, dtype: str) -> str:
    return hashlib.sha256(np.asarray(values, dtype=dtype).tobytes()).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class Cohort:
    specification: str
    feature_names: tuple[str, ...]
    X: np.ndarray
    labels: np.ndarray
    improvement: np.ndarray
    feature_sha256: str
    label_sha256: str
    improvement_sha256: str
    source_files: dict[str, str]
    missing_by_feature: dict[str, int]

    @property
    def boundary(self) -> np.ndarray:
        return (self.improvement >= 65.0) & (self.improvement < 85.0)


def load_cohort(source: Path, specification: str = "allergens", count_fill: str = "zero") -> Cohort:
    """Return the exact cohort used by the final paper.

    Other historical feature specifications are deliberately excluded from this
    public release. They were exploratory and are not needed to reproduce the
    paper. ``count_fill`` is kept only for the reported missingness convention.
    """
    source = Path(source).resolve()
    if specification != "allergens" or count_fill != "zero":
        raise ValueError("the published run requires specification='allergens' and count_fill='zero'")
    source_hash = file_sha256(source)
    if source_hash != SOURCE_SHA256:
        raise RuntimeError(f"unexpected source workbook SHA-256: {source_hash}")

    raw = pd.read_excel(source, sheet_name="Sheet1")
    raw = raw.rename(columns={c: c.strip() if isinstance(c, str) else c for c in raw.columns})
    missing_headers = [name for name in RENAME_MAP if name not in raw.columns]
    if missing_headers:
        raise RuntimeError(f"source workbook is missing {len(missing_headers)} required headers")
    raw = raw.rename(columns=RENAME_MAP)
    for column in raw.columns:
        converted = pd.to_numeric(raw[column], errors="coerce")
        if converted.notna().any() or raw[column].isna().all():
            raw[column] = converted

    engineered = build_engineered_features(raw, allergen_burden="allergens")
    frame = engineered[list(FEATURE_NAMES)].copy()
    frame["fe_allergen_panel_count_nonzero"] = frame[
        "fe_allergen_panel_count_nonzero"
    ].fillna(0)

    X = frame.to_numpy(dtype=np.float64)
    labels = raw["EAIS-75 achievement"].to_numpy(dtype=np.int64)
    improvement = raw["EASI improvement"].to_numpy(dtype=np.float64)

    feature_hash = array_sha256(X, ">f8")
    label_hash = array_sha256(labels, ">i8")
    improvement_hash = array_sha256(improvement, ">f8")
    observed = (feature_hash, label_hash, improvement_hash)
    expected = (FEATURE_SHA256, LABEL_SHA256, IMPROVEMENT_SHA256)
    if X.shape != (119, 23) or int(labels.sum()) != 65 or observed != expected:
        raise RuntimeError(
            "cohort contract mismatch: "
            f"shape={X.shape}, positives={int(labels.sum())}, hashes={observed}"
        )
    if not np.array_equal(labels, (improvement >= 75.0).astype(np.int64)):
        raise RuntimeError("EASI-75 labels do not match the 75% outcome definition")

    return Cohort(
        specification=specification,
        feature_names=tuple(FEATURE_NAMES),
        X=X,
        labels=labels,
        improvement=improvement,
        feature_sha256=feature_hash,
        label_sha256=label_hash,
        improvement_sha256=improvement_hash,
        source_files={source.name: source_hash},
        missing_by_feature={name: int(frame[name].isna().sum()) for name in frame.columns},
    )
