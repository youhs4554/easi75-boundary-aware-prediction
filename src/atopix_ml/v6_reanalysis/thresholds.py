from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

_FLOAT_TOLERANCE: Final = 0.0


class ThresholdContractError(ValueError):
    code: str
    detail: str

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True, slots=True)
class YoudenCandidate:
    threshold: float
    youden: float
    sensitivity: float
    specificity: float
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int

    def to_json_record(self) -> dict[str, float | int]:
        return {
            "false_negative": self.false_negative,
            "false_positive": self.false_positive,
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
            "threshold": self.threshold,
            "true_negative": self.true_negative,
            "true_positive": self.true_positive,
            "youden": self.youden,
        }


@dataclass(frozen=True, slots=True)
class YoudenSelection:
    threshold: float
    youden: float
    sensitivity: float
    specificity: float
    tie_count: int
    candidate_count: int
    candidate_ledger_sha256: str
    candidates: tuple[YoudenCandidate, ...]


def _parse_inputs(
    scores: NDArray[np.float64], labels: NDArray[np.int64]
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    score_array = np.asarray(scores, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.int64)
    if score_array.ndim != 1 or label_array.ndim != 1:
        raise ThresholdContractError(
            "invalid_dimension", "scores and labels must be one-dimensional"
        )
    if score_array.size != label_array.size or score_array.size == 0:
        raise ThresholdContractError(
            "length_mismatch", "scores and labels must have equal nonzero length"
        )
    if not np.all(np.isfinite(score_array)):
        raise ThresholdContractError("nonfinite_score", "scores must be finite")
    if not np.all(np.isin(label_array, (0, 1))):
        raise ThresholdContractError("nonbinary_label", "labels must contain only zero and one")
    if np.unique(label_array).size != 2:
        raise ThresholdContractError("endpoint_class_missing", "both endpoint classes are required")
    return score_array, label_array


def select_highest_observed_youden(
    scores: NDArray[np.float64], labels: NDArray[np.int64]
) -> YoudenSelection:
    score_array, label_array = _parse_inputs(scores, labels)
    positives = label_array == 1
    negatives = label_array == 0
    positive_count = int(np.count_nonzero(positives))
    negative_count = int(np.count_nonzero(negatives))
    candidates: list[YoudenCandidate] = []
    for threshold in np.unique(score_array):
        decisions = score_array >= threshold
        true_positive = int(np.count_nonzero(decisions & positives))
        true_negative = int(np.count_nonzero(~decisions & negatives))
        false_positive = negative_count - true_negative
        false_negative = positive_count - true_positive
        sensitivity = true_positive / positive_count
        specificity = true_negative / negative_count
        candidates.append(
            YoudenCandidate(
                threshold=float(threshold),
                youden=float(sensitivity + specificity - 1.0),
                sensitivity=float(sensitivity),
                specificity=float(specificity),
                true_positive=true_positive,
                true_negative=true_negative,
                false_positive=false_positive,
                false_negative=false_negative,
            )
        )
    best_youden = max(candidate.youden for candidate in candidates)
    tied = tuple(
        candidate
        for candidate in candidates
        if abs(candidate.youden - best_youden) <= _FLOAT_TOLERANCE
    )
    selected = max(tied, key=lambda candidate: candidate.threshold)
    payload = json.dumps(
        [candidate.to_json_record() for candidate in candidates],
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return YoudenSelection(
        threshold=selected.threshold,
        youden=selected.youden,
        sensitivity=selected.sensitivity,
        specificity=selected.specificity,
        tie_count=len(tied),
        candidate_count=len(candidates),
        candidate_ledger_sha256=hashlib.sha256(payload).hexdigest(),
        candidates=tuple(candidates),
    )
