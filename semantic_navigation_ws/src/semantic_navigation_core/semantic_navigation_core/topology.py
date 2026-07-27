"""Pure policies for semantic-node creation and topological connections."""
from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, hypot, sin
from typing import Sequence

from semantic_navigation_core.types import SemanticNode


def wrap_angle(angle: float) -> float:
    return atan2(sin(angle), cos(angle))


@dataclass(frozen=True)
class NodeCreationPolicy:
    minimum_translation_m: float = 1.5
    minimum_rotation_rad: float = 0.7853981633974483
    minimum_time_s: float = 2.0
    duplicate_distance_m: float = 0.5
    maximum_edge_distance_m: float = 4.0

    def should_create(
        self,
        previous_position: tuple[float, float] | None,
        previous_yaw: float | None,
        previous_timestamp: float | None,
        position: tuple[float, float],
        yaw: float,
        timestamp: float,
        manual: bool = False,
    ) -> bool:
        if manual or previous_position is None:
            return True
        translation = hypot(position[0] - previous_position[0], position[1] - previous_position[1])
        rotation = abs(wrap_angle(yaw - (previous_yaw or 0.0)))
        elapsed = timestamp - (previous_timestamp or timestamp)
        return (
            elapsed >= self.minimum_time_s
            and (
                translation >= self.minimum_translation_m
                or rotation >= self.minimum_rotation_rad
            )
        )


def nearest_node(
    nodes: Sequence[SemanticNode],
    position: tuple[float, float],
    scene_id: str,
    maximum_distance_m: float,
) -> tuple[SemanticNode | None, float]:
    """Nearest same-scene node within the inclusive distance threshold."""
    best = None
    best_distance = float("inf")
    for node in nodes:
        if node.scene_id != scene_id:
            continue
        distance = hypot(node.position[0] - position[0], node.position[1] - position[1])
        if distance < best_distance:
            best, best_distance = node, distance
    if best_distance > maximum_distance_m:
        return None, best_distance
    return best, best_distance

