"""Explicit schemas for offline datasets and ROS 2 campaign analysis.

The schemas in this module are deliberately independent of ROS messages.  They
separate dataset, scene, campaign, run, retrieval, navigation and graph-quality
concepts while retaining the semantic core's existing node/query contracts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from semantic_navigation_core.types import SpatialRelation


class SchemaValidationError(ValueError):
    """Validation failure carrying file, field, value and corrective action."""

    def __init__(
        self,
        source: str,
        field_name: str,
        received: Any,
        expected: str,
        action: str,
    ) -> None:
        self.source = source
        self.field_name = field_name
        self.received = received
        self.expected = expected
        self.action = action
        super().__init__(
            f"{source}: field '{field_name}' received {received!r}; expected "
            f"{expected}. Action: {action}"
        )


def _required(data: Mapping[str, Any], key: str, source: str) -> Any:
    value = data.get(key)
    if value is None or value == "":
        raise SchemaValidationError(
            source, key, value, "a non-empty value", f"set '{key}' in {source}"
        )
    return value


@dataclass(frozen=True)
class OfflineSplitSpec:
    name: str
    scan_ids: tuple[str, ...] = ()
    split_file: str | None = None
    max_images_per_room: int | None = None
    view_angles_degrees: tuple[int, ...] = (0, 90, 180, 270)
    connectivity_dir: str = "connectivity"
    building_splits: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "OfflineSplitSpec":
        raw = dict(data or {})
        return cls(
            name=str(raw.pop("name", "all")),
            scan_ids=tuple(str(v) for v in raw.pop("scan_ids", ())),
            split_file=raw.pop("split_file", None),
            max_images_per_room=raw.pop("max_images_per_room", None),
            view_angles_degrees=tuple(
                int(v) for v in raw.pop("view_angles_degrees", (0, 90, 180, 270))
            ),
            connectivity_dir=str(raw.pop("connectivity_dir", "connectivity")),
            building_splits=dict(raw.pop("building_splits", {})),
        )


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    adapter: str
    root: str
    queries_file: str
    enabled: bool = True
    split: OfflineSplitSpec = field(default_factory=lambda: OfflineSplitSpec("all"))
    r2r_root: str | None = None
    annotations: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    config_path: str = ""

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, Any], source: str = "<dataset config>"
    ) -> "DatasetSpec":
        return cls(
            dataset_id=str(_required(data, "dataset_id", source)),
            adapter=str(_required(data, "adapter", source)),
            root=str(_required(data, "root", source)),
            queries_file=str(_required(data, "queries_file", source)),
            enabled=bool(data.get("enabled", True)),
            split=OfflineSplitSpec.from_mapping(data.get("split")),
            r2r_root=(str(data["r2r_root"]) if data.get("r2r_root") else None),
            annotations=dict(data.get("annotations") or {}),
            metadata=dict(data.get("metadata") or {}),
            config_path=str(source),
        )


@dataclass(frozen=True)
class SimulationSceneSpec:
    scene_id: str
    graph_db: str
    queries_file: str
    enabled: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)
    config_path: str = ""

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, Any], source: str = "<scene config>"
    ) -> "SimulationSceneSpec":
        return cls(
            scene_id=str(_required(data, "scene_id", source)),
            graph_db=str(_required(data, "graph_db", source)),
            queries_file=str(_required(data, "queries_file", source)),
            enabled=bool(data.get("enabled", True)),
            metadata=dict(data.get("metadata") or {}),
            config_path=str(source),
        )


@dataclass
class QuerySpec:
    query_id: str
    text: str
    language: str = "es"
    query_type: str = "object"
    dataset_id: str | None = None
    scene_id: str | None = None
    valid_node_ids: list[str] = field(default_factory=list)
    expected_room: str | None = None
    expected_objects: list[str] = field(default_factory=list)
    expected_relations: list[SpatialRelation] = field(default_factory=list)
    is_negative: bool = False
    target_visible: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, Any], source: str = "<query config>"
    ) -> "QuerySpec":
        language = str(data.get("language", "es"))
        if language not in {"es", "en"}:
            raise SchemaValidationError(
                source, "language", language, "'es' or 'en'", "correct the query language"
            )
        dataset_id = data.get("dataset_id")
        scene_id = data.get("scene_id")
        if not dataset_id and not scene_id:
            raise SchemaValidationError(
                source,
                "dataset_id/scene_id",
                None,
                "one dataset_id or scene_id",
                "associate the query with exactly one experimental source",
            )
        relations = list(data.get("expected_relations") or [])
        for index, relation in enumerate(relations):
            for key in ("subject", "predicate", "obj"):
                if not relation.get(key):
                    raise SchemaValidationError(
                        source,
                        f"expected_relations[{index}].{key}",
                        relation.get(key),
                        "a non-empty string",
                        "complete the relation annotation",
                    )
        return cls(
            query_id=str(_required(data, "query_id", source)),
            text=str(_required(data, "text", source)),
            language=language,
            query_type=str(data.get("query_type", "object")),
            dataset_id=str(dataset_id) if dataset_id else None,
            scene_id=str(scene_id) if scene_id else None,
            valid_node_ids=[str(v) for v in data.get("valid_node_ids", [])],
            expected_room=(str(data["expected_room"]) if data.get("expected_room") else None),
            expected_objects=[str(v) for v in data.get("expected_objects", [])],
            expected_relations=[
                SpatialRelation(
                    subject=str(value["subject"]),
                    predicate=str(value["predicate"]),
                    obj=str(value["obj"]),
                    confidence=float(value.get("confidence", 1.0)),
                )
                for value in relations
            ],
            is_negative=bool(data.get("is_negative", False)),
            target_visible=bool(
                data.get("target_visible", not bool(data.get("is_negative", False)))
            ),
            metadata=dict(data.get("metadata") or {}),
        )


ALLOWED_METHODS = {
    "random_baseline",
    "nearest_node_baseline",
    "room_label_baseline",
    "single_view_siglip",
    "multiview_siglip",
    "siglip_with_objects",
    "siglip_with_objects_and_relations",
    "hybrid_semantic_retrieval",
}


@dataclass(frozen=True)
class CampaignSpec:
    campaign_id: str
    scene_id: str
    run_id: str
    seed: int
    method: str
    start_pose_id: str
    query_suite_id: str
    frozen_config_hash: str
    git_commit: str
    timestamp: str
    status: str
    success_semantics: str | None = None

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, Any], source: str = "<campaign config>"
    ) -> "CampaignSpec":
        method = str(_required(data, "method", source))
        if method not in ALLOWED_METHODS:
            raise SchemaValidationError(
                source,
                "method",
                method,
                f"one of {sorted(ALLOWED_METHODS)}",
                "replace opaque or legacy method identifiers",
            )
        try:
            seed = int(_required(data, "seed", source))
        except (TypeError, ValueError) as exc:
            raise SchemaValidationError(
                source, "seed", data.get("seed"), "an integer", "set a reproducible seed"
            ) from exc
        return cls(
            campaign_id=str(_required(data, "campaign_id", source)),
            scene_id=str(_required(data, "scene_id", source)),
            run_id=str(_required(data, "run_id", source)),
            seed=seed,
            method=method,
            start_pose_id=str(_required(data, "start_pose_id", source)),
            query_suite_id=str(_required(data, "query_suite_id", source)),
            frozen_config_hash=str(_required(data, "frozen_config_hash", source)),
            git_commit=str(_required(data, "git_commit", source)),
            timestamp=str(_required(data, "timestamp", source)),
            status=str(_required(data, "status", source)),
            success_semantics=(
                str(data["success_semantics"]) if data.get("success_semantics") else None
            ),
        )


@dataclass(frozen=True)
class RunSpec:
    campaign: CampaignSpec
    root: Path
    evaluation_file: Path
    manifest_file: Path


@dataclass
class RetrievalResult:
    query_id: str
    predicted_node_id: str | None
    valid_node_ids: list[str]
    rank_first_valid: int | None
    global_similarity: float | None = None
    object_match_score: float | None = None
    crop_similarity: float | None = None
    relation_match_score: float | None = None
    room_match_score: float | None = None
    hybrid_score: float | None = None
    retrieval_latency_ms: float | None = None


@dataclass
class NavigationResult:
    navigation_success: bool | None = None
    navigation_time_s: float | None = None
    path_length_m: float | None = None
    optimal_path_length_m: float | None = None
    final_distance_m: float | None = None
    failure_type: str | None = None


@dataclass
class GraphQualityResult:
    scene_id: str
    waypoint_count: int = 0
    object_count: int = 0
    room_count: int = 0
    observation_count: int = 0
    connected_components: int = 0
    isolated_nodes: int = 0
    mean_degree: float = 0.0
    invalid_edges: int = 0
    objects_without_waypoint: int = 0
    waypoints_without_room: int = 0
    nodes_without_embeddings: int = 0
    inconsistent_embedding_dimensions: bool = False
    empty_or_nonfinite_embeddings: int = 0
    duplicate_nodes: int = 0
    probable_duplicate_objects: int = 0
