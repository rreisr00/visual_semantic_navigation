"""Evaluation rows for the inter-room semantic-contamination experiment."""
from __future__ import annotations

from collections.abc import Sequence

from semantic_navigation_core.contamination import contamination_metrics
from semantic_navigation_core.types import SemanticNode


def summarize_observation_contamination(nodes: Sequence[SemanticNode]) -> dict:
    """Return campaign-ready CRDR/COR counts and rates for semantic nodes."""
    metrics = contamination_metrics(
        observation
        for node in nodes
        for observation in node.observations
    )
    return {
        "localized_detections": metrics.localized_detections,
        "cross_room_detections": metrics.cross_room_detections,
        "cross_room_detection_rate": metrics.cross_room_detection_rate,
        "classified_observations": metrics.classified_observations,
        "contaminated_observations": metrics.contaminated_observations,
        "contaminated_observation_rate": metrics.contaminated_observation_rate,
    }
