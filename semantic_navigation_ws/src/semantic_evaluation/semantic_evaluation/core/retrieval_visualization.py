"""Pure helpers used to explain retrieval evidence in the operator GUI."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def object_evidence(
    object_id: str,
    label: str,
    candidate: Mapping[str, Any] | None,
) -> tuple[str, ...]:
    """Return components to which one detected object contributed."""
    if candidate is None:
        return ()
    normalized_id = str(object_id).strip()
    normalized_label = str(label).strip().casefold()
    matched_ids = {
        str(value).strip()
        for value in _sequence(candidate.get("matched_object_ids"))
        if str(value).strip()
    }
    matched_labels = {
        str(value).strip().casefold()
        for value in _sequence(candidate.get("matched_object_labels"))
        if str(value).strip()
    }
    evidence: list[str] = []
    if (
        (normalized_id and normalized_id in matched_ids)
        or (normalized_label and normalized_label in matched_labels)
    ):
        evidence.append("object_match")

    best_crop_id = str(candidate.get("best_crop_object_id", "")).strip()
    best_crop_label = str(
        candidate.get("best_crop_object_label", "")
    ).strip().casefold()
    if (
        (normalized_id and normalized_id == best_crop_id)
        or (normalized_label and normalized_label == best_crop_label)
    ):
        evidence.append("crop_similarity")
    return tuple(evidence)


def _sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value
    return ()
