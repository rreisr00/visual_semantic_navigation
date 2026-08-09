"""Plain data types shared by the semantic navigation core — no ROS imports."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

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


# ── Offline / multi-view extension (additive; Waypoint stays the ROS contract) ─ #


@dataclass
class SpatialRelation:
    """A 2D spatial relation hypothesis between two detected objects.

    These are *visual hypotheses* derived from 2D bounding boxes in a single
    image — not confirmed physical relations. ``subject`` and ``obj`` are
    object class labels (YOLO names).
    """

    subject: str
    predicate: str
    obj: str
    confidence: float = 1.0
    subject_id: str = ""
    object_id: str = ""
    reference_frame: str = "camera_optical_frame"
    source_observation_id: str = ""
    relation_type: str = "visual_2d_hypothesis"
    timestamp: float = 0.0


@dataclass
class ObjectObservation:
    """One detected object inside an observation.

    Attributes:
        label: Detector class name (e.g. ``"cup"``).
        confidence: Detector confidence in [0, 1] (1.0 when unknown, e.g. for
            labels reconstructed from the knowledge graph, which stores none).
        box: Pixel-space (x1, y1, x2, y2) box, or None when unknown.
        embedding: Optional L2-normalised SigLIP embedding of the object crop.
    """

    label: str
    confidence: float = 1.0
    box: tuple[float, float, float, float] | None = None
    embedding: np.ndarray | None = None
    object_id: str = ""
    position_2d: tuple[float, float] | None = None
    position_3d: tuple[float, float, float] | None = None
    position_3d_frame: str = ""
    map_position: tuple[float, float, float] | None = None
    room_id: str | None = None
    observation_ids: list[str] = field(default_factory=list)
    associated_node_ids: list[str] = field(default_factory=list)
    last_seen: float = 0.0


@dataclass
class Observation:
    """A single visual observation (one camera frame) of a semantic node."""

    observation_id: str
    embedding: np.ndarray | None = None
    image_path: str = ""
    objects: list[ObjectObservation] = field(default_factory=list)
    relations: list[SpatialRelation] = field(default_factory=list)
    timestamp: float = 0.0
    camera_frame: str = ""
    camera_position: tuple[float, float, float] | None = None
    camera_orientation: tuple[float, float, float, float] | None = None
    depth_camera_frame: str = ""
    depth_camera_position: tuple[float, float, float] | None = None
    depth_camera_orientation: tuple[float, float, float, float] | None = None
    requested_yaw: float = 0.0
    measured_yaw: float = 0.0
    angular_error: float = 0.0
    image_valid: bool = True
    depth_valid: bool = False
    camera_room: str | None = None
    observation_room: str | None = None
    purity: float | None = None
    contamination_class: str = "unknown"
    transition_zone: bool = False

    @property
    def purity_weight(self) -> float:
        """Weight used for aggregation; legacy/unclassified views stay usable."""
        return 1.0 if self.purity is None else max(0.0, min(1.0, self.purity))


@dataclass
class SemanticNode:
    """A semantic map node that may hold several observations.

    Superset of :class:`Waypoint`: ``to_waypoint`` collapses it back to the
    exact single-embedding representation the ROS stack uses, so offline
    experiments and the deployed system share one data model.
    """

    node_id: str
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    observations: list[Observation] = field(default_factory=list)
    room_id: str | None = None
    scene_id: str = "default"
    navigation_position: tuple[float, float, float] | None = None
    navigation_orientation: tuple[float, float, float, float] | None = None
    neighbors: list[str] = field(default_factory=list)
    creation_timestamp: float = field(
        default_factory=lambda: datetime.now(timezone.utc).timestamp()
    )
    configuration_hash: str = ""

    def embeddings(self) -> list[np.ndarray]:
        """Per-view embeddings, skipping observations without one."""
        return [
            o.embedding for o in self.observations
            if o.embedding is not None and np.asarray(o.embedding).size > 0
        ]

    def object_labels(self) -> list[str]:
        """Unique object labels across all observations (first-seen order)."""
        seen: set[str] = set()
        labels: list[str] = []
        for obs in self.observations:
            for obj in obs.objects:
                if obj.label not in seen:
                    seen.add(obj.label)
                    labels.append(obj.label)
        return labels

    def relations(self) -> list[SpatialRelation]:
        """All relation hypotheses across observations."""
        return [rel for obs in self.observations for rel in obs.relations]

    def to_waypoint(self, embedding: np.ndarray | None = None) -> Waypoint:
        """Collapse to the single-embedding :class:`Waypoint` used by ROS.

        Args:
            embedding: Aggregated embedding to use; defaults to the first
                view's embedding (the current online behaviour: one capture,
                one embedding).
        """
        if embedding is None:
            views = self.embeddings()
            embedding = views[0] if views else np.array([], dtype=np.float32)
        return Waypoint(
            node_id=self.node_id,
            position=self.position,
            orientation=self.orientation,
            embedding=embedding,
            objects=self.object_labels(),
        )


@dataclass
class RankedNode:
    """A node paired with its score and the per-component breakdown."""

    node: SemanticNode
    score: float
    components: dict[str, float] = field(default_factory=dict)
