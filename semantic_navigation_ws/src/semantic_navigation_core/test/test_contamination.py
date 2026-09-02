"""Tests for room assignment and observation-purity metrics."""
import pytest

from semantic_navigation_core.contamination import (
    annotate_observation_rooms,
    classify_purity,
    contamination_metrics,
)
from semantic_navigation_core.rooms import Room
from semantic_navigation_core.types import Observation, ObjectObservation


def test_confidence_weighted_purity_and_rooms():
    rooms = [
        Room("office", 0, 0, 2, 2),
        Room("hall", 2, 0, 4, 2),
    ]
    observation = Observation(
        "view",
        camera_position=(1.0, 1.0, 0.5),
        objects=[
            ObjectObservation("desk", confidence=0.8, map_position=(1, 1, 0)),
            ObjectObservation("chair", confidence=0.2, map_position=(3, 1, 0)),
        ],
    )
    annotate_observation_rooms(observation, rooms)
    assert observation.camera_room == "office"
    assert observation.observation_room == "office"
    assert observation.purity == pytest.approx(0.8)
    assert observation.contamination_class == "clean"
    assert [item.room_id for item in observation.objects] == ["office", "hall"]


def test_thresholds_and_rates():
    assert classify_purity(0.80) == "clean"
    assert classify_purity(0.50) == "mixed"
    assert classify_purity(0.49) == "contaminated"
    observation = Observation(
        "view", camera_room="office", purity=0.4,
        contamination_class="contaminated",
        objects=[ObjectObservation("chair", room_id="hall")],
    )
    metrics = contamination_metrics([observation])
    assert metrics.cross_room_detection_rate == 1.0
    assert metrics.contaminated_observation_rate == 1.0
