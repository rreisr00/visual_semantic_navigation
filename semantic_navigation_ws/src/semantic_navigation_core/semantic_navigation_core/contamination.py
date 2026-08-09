"""Room-aware observation purity and inter-room contamination metrics."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

from semantic_navigation_core.rooms import Room, room_of_point
from semantic_navigation_core.types import Observation

CLEAN_PURITY_THRESHOLD = 0.80
MIXED_PURITY_THRESHOLD = 0.50


def classify_purity(purity: float | None) -> str:
    """Classify purity using the fixed TFM protocol thresholds."""
    if purity is None:
        return "unknown"
    if purity >= CLEAN_PURITY_THRESHOLD:
        return "clean"
    if purity >= MIXED_PURITY_THRESHOLD:
        return "mixed"
    return "contaminated"


@dataclass(frozen=True)
class ObservationRoomEvidence:
    camera_room: str | None
    object_rooms: tuple[str | None, ...]
    observation_room: str | None
    purity: float | None
    contamination_class: str
    transition_zone: bool


def analyze_room_evidence(
    camera_position: tuple[float, float, float] | None,
    object_evidence: Iterable[
        tuple[tuple[float, float, float] | None, float]
    ],
    rooms: Sequence[Room],
) -> ObservationRoomEvidence:
    """Compute a reproducible room assignment from localized detections."""
    camera_room = (
        room_of_point(camera_position[0], camera_position[1], rooms)
        if camera_position is not None else None
    )
    room_evidence: dict[str, float] = defaultdict(float)
    object_rooms: list[str | None] = []
    known_evidence = 0.0
    for position, confidence in object_evidence:
        object_room = (
            room_of_point(position[0], position[1], rooms)
            if position is not None else None
        )
        object_rooms.append(object_room)
        if object_room is not None:
            weight = max(0.0, float(confidence))
            room_evidence[object_room] += weight
            known_evidence += weight
    observation_room = (
        max(room_evidence, key=room_evidence.get) if room_evidence else None
    )
    purity = (
        room_evidence.get(camera_room, 0.0) / known_evidence
        if camera_room is not None and known_evidence > 0.0 else None
    )
    return ObservationRoomEvidence(
        camera_room=camera_room,
        object_rooms=tuple(object_rooms),
        observation_room=observation_room,
        purity=purity,
        contamination_class=classify_purity(purity),
        transition_zone=bool(
            camera_position is not None
            and any(
                room.in_transition_zone(camera_position[0], camera_position[1])
                for room in rooms
            )
        ),
    )


def annotate_observation_rooms(
    observation: Observation,
    rooms: Sequence[Room],
) -> Observation:
    """Annotate camera/object rooms and confidence-weighted visual purity.

    Purity(o, R) is the confidence mass of localized detections belonging to
    the camera room R divided by all localized detection confidence mass.
    Unlocalized detections do not enter the denominator.
    """
    evidence = analyze_room_evidence(
        observation.camera_position,
        [
            (
                obj.map_position or (
                    obj.position_3d if obj.position_3d_frame == "map" else None
                ),
                obj.confidence,
            )
            for obj in observation.objects
        ],
        rooms,
    )
    observation.camera_room = evidence.camera_room
    for obj, room_id in zip(observation.objects, evidence.object_rooms):
        obj.room_id = room_id
    observation.observation_room = evidence.observation_room
    observation.purity = evidence.purity
    observation.contamination_class = evidence.contamination_class
    observation.transition_zone = evidence.transition_zone
    return observation


@dataclass(frozen=True)
class ContaminationMetrics:
    localized_detections: int = 0
    cross_room_detections: int = 0
    classified_observations: int = 0
    contaminated_observations: int = 0

    @property
    def cross_room_detection_rate(self) -> float:
        return (
            self.cross_room_detections / self.localized_detections
            if self.localized_detections else float("nan")
        )

    @property
    def contaminated_observation_rate(self) -> float:
        return (
            self.contaminated_observations / self.classified_observations
            if self.classified_observations else float("nan")
        )


def contamination_metrics(
    observations: Iterable[Observation],
) -> ContaminationMetrics:
    localized = cross_room = classified = contaminated = 0
    for observation in observations:
        if observation.purity is not None:
            classified += 1
            contaminated += observation.contamination_class == "contaminated"
        for obj in observation.objects:
            if obj.room_id is None or observation.camera_room is None:
                continue
            localized += 1
            cross_room += obj.room_id != observation.camera_room
    return ContaminationMetrics(localized, cross_room, classified, contaminated)
