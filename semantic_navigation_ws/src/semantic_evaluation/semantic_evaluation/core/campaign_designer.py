"""Data model and persistence helpers for the graphical campaign designer.

This module deliberately has no Qt or ROS imports.  The operator GUI can use
it at runtime while the coordinate conversion and YAML round-trips remain
cheap to test in isolation.
"""

from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from semantic_navigation_core.graph_store import load_rooms, load_semantic_nodes
from semantic_navigation_core.rooms import Room
from semantic_navigation_core.types import SemanticNode


@dataclass(frozen=True)
class OccupancyMap:
    """Metadata required to place world coordinates over an occupancy image."""

    yaml_path: str
    image_path: str
    resolution: float
    origin_x: float
    origin_y: float
    origin_yaw: float = 0.0

    def world_to_pixel(
        self, x: float, y: float, image_height: int
    ) -> tuple[float, float]:
        """Convert a point in the ROS map frame to image pixel coordinates."""
        dx = float(x) - self.origin_x
        dy = float(y) - self.origin_y
        cosine = math.cos(self.origin_yaw)
        sine = math.sin(self.origin_yaw)
        local_x = cosine * dx + sine * dy
        local_y = -sine * dx + cosine * dy
        return local_x / self.resolution, image_height - local_y / self.resolution


@dataclass
class CampaignWorkspace:
    """Editable files and read-only semantic graph for one simulation scene."""

    scene_id: str
    graph_database: str
    map_metadata: OccupancyMap
    queries_file: str
    ground_truth_file: str
    nodes: list[SemanticNode]
    rooms: list[Room]
    queries: dict[str, Any]
    ground_truth: dict[str, Any]


def expanded_path(path: str) -> str:
    """Expand environment variables and ``~`` in a user-facing path."""
    return os.path.abspath(os.path.expandvars(os.path.expanduser(path)))


def load_occupancy_map(path: str) -> OccupancyMap:
    """Load a Nav2 map YAML and resolve its image relative to that file."""
    yaml_path = expanded_path(path)
    with open(yaml_path, encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"map file '{yaml_path}' must contain a mapping")
    image = str(data.get("image", "")).strip()
    if not image:
        raise ValueError(f"map file '{yaml_path}' does not define 'image'")
    image_path = expanded_path(
        image if os.path.isabs(image) else os.path.join(os.path.dirname(yaml_path), image)
    )
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"occupancy image not found: {image_path}")
    resolution = float(data.get("resolution", 0.0))
    if resolution <= 0.0:
        raise ValueError(f"map file '{yaml_path}' requires a positive resolution")
    origin = data.get("origin", [0.0, 0.0, 0.0])
    if not isinstance(origin, (list, tuple)) or len(origin) < 3:
        raise ValueError(f"map file '{yaml_path}' requires origin [x, y, yaw]")
    return OccupancyMap(
        yaml_path=yaml_path,
        image_path=image_path,
        resolution=resolution,
        origin_x=float(origin[0]),
        origin_y=float(origin[1]),
        origin_yaw=float(origin[2]),
    )


def load_yaml_mapping(path: str, defaults: Mapping[str, Any]) -> dict[str, Any]:
    """Load an editable YAML mapping, using defaults for a missing file."""
    resolved = expanded_path(path)
    if not os.path.isfile(resolved):
        return dict(defaults)
    with open(resolved, encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"'{resolved}' must contain a YAML mapping")
    return dict(data)


def load_workspace(
    *,
    scene_id: str,
    graph_database: str,
    map_file: str,
    queries_file: str,
    ground_truth_file: str,
) -> CampaignWorkspace:
    """Load the current graph, map and editable evaluation annotations."""
    graph_path = expanded_path(graph_database)
    nodes = load_semantic_nodes(graph_path, read_only=True)
    queries_path = expanded_path(queries_file)
    ground_truth_path = expanded_path(ground_truth_file)
    queries = load_yaml_mapping(
        queries_path,
        {"suite_id": f"{scene_id}_semantic_queries_v1", "scene_id": scene_id, "cases": []},
    )
    ground_truth = load_yaml_mapping(
        ground_truth_path,
        {
            "scene_id": scene_id,
            "annotation_version": 1,
            "rooms": [],
            "objects": [],
            "relations": [],
            "negative_queries": [],
        },
    )
    queries.setdefault("suite_id", f"{scene_id}_semantic_queries_v1")
    queries.setdefault("scene_id", scene_id)
    queries.setdefault("cases", [])
    ground_truth.setdefault("scene_id", scene_id)
    ground_truth.setdefault("annotation_version", 1)
    for key in ("rooms", "objects", "relations", "negative_queries"):
        ground_truth.setdefault(key, [])
    return CampaignWorkspace(
        scene_id=scene_id,
        graph_database=graph_path,
        map_metadata=load_occupancy_map(map_file),
        queries_file=queries_path,
        ground_truth_file=ground_truth_path,
        nodes=nodes,
        rooms=load_rooms(graph_path),
        queries=queries,
        ground_truth=ground_truth,
    )


def validate_workspace(workspace: CampaignWorkspace) -> None:
    """Reject common annotation mistakes before replacing either YAML file."""
    node_ids = {node.node_id for node in workspace.nodes}
    cases = workspace.queries.get("cases", [])
    if not isinstance(cases, list):
        raise ValueError("queries.cases must be a list")
    seen: set[str] = set()
    for index, case in enumerate(cases):
        if not isinstance(case, Mapping):
            raise ValueError(f"queries.cases[{index}] must be a mapping")
        case_id = str(case.get("case_id", "")).strip()
        query_id = str(case.get("query_id", "")).strip()
        text = str(case.get("query_text", "")).strip()
        if not case_id or not query_id or not text:
            raise ValueError(
                f"queries.cases[{index}] requires case_id, query_id and query_text"
            )
        if case_id in seen:
            raise ValueError(f"duplicate case_id '{case_id}'")
        seen.add(case_id)
        referenced = set(case.get("exact_valid_nodes", [])) | set(
            case.get("nearby_valid_nodes", [])
        )
        unknown = sorted(str(value) for value in referenced - node_ids)
        if unknown:
            raise ValueError(f"case '{case_id}' references unknown nodes: {unknown}")
    for group in ("rooms", "objects", "relations"):
        entries = workspace.ground_truth.get(group, [])
        if not isinstance(entries, list):
            raise ValueError(f"ground_truth.{group} must be a list")
        for index, entry in enumerate(entries):
            if not isinstance(entry, Mapping):
                raise ValueError(f"ground_truth.{group}[{index}] must be a mapping")
            keys = ("valid_nodes",) if group != "objects" else (
                "exact_valid_nodes", "nearby_valid_nodes"
            )
            referenced: set[str] = set()
            for key in keys:
                referenced.update(str(value) for value in entry.get(key, []))
            unknown = sorted(referenced - node_ids)
            if unknown:
                raise ValueError(
                    f"ground_truth.{group}[{index}] references unknown nodes: {unknown}"
                )


def atomic_dump_yaml(path: str, data: Mapping[str, Any]) -> None:
    """Write YAML atomically so an interrupted GUI cannot truncate annotations."""
    destination = Path(expanded_path(path))
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            yaml.safe_dump(
                dict(data), stream, allow_unicode=True, sort_keys=False, width=100
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def save_workspace(workspace: CampaignWorkspace) -> None:
    """Validate and persist both evaluation inputs."""
    validate_workspace(workspace)
    atomic_dump_yaml(workspace.queries_file, workspace.queries)
    atomic_dump_yaml(workspace.ground_truth_file, workspace.ground_truth)
