"""Plain data types shared by the semantic navigation core — no ROS imports."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Waypoint:
    """A semantic waypoint decoupled from any ROS message type.

    Attributes:
        node_id: Unique graph node identifier.
        position: (x, y, z) map-frame translation.
        orientation: (x, y, z, w) map-frame quaternion.
        embedding: L2-normalised visual embedding.
        objects: Detected object class names (empty in ``siglip_pure``).
    """

    node_id: str
    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float]
    embedding: np.ndarray
    objects: list[str] = field(default_factory=list)


@dataclass
class RankedWaypoint:
    """A waypoint paired with its retrieval score."""

    waypoint: Waypoint
    score: float
