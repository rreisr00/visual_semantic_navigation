"""Offline dataset adapter: files on disk → ``SemanticNode`` — no rclpy.

Mirror of the ROS-side adapters: where the online system feeds the semantic
core from topics + TF2, this module feeds the *same* core types from files,
so notebooks and simulation share every downstream component.

Scene sources (checked in this order inside :func:`load_scene`):

1. ``graph_db``: a knowledge-graph SQLite file written by the real ROS bridge
   (plus optionally ``images_dir`` with ``<node_id>.png`` capture frames from
   ``teleop_capture``). Multi-view: extra views may be added as
   ``<node_id>__<anything>.png``.
2. ``category_images_dir``: a directory of ``<room>/<image>`` files (e.g.
   ``experiments/siglip/images``); each image becomes a one-view node whose
   ground-truth room is its directory name.

Query YAML schema (``semantic_queries.yaml``)::

    queries:
      - query_id: q001
        text: "ve a la cocina"
        language: es          # es | en
        query_type: room      # room | object | attribute | relation | ...
        scene_id: scene_01
        valid_node_ids: [node_001]   # may be empty + expected_room instead
        expected_objects: []
        expected_relations: []       # {subject, predicate, obj}
        expected_room: cocina
        is_negative: false
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Iterable, Mapping

import numpy as np
import yaml

from semantic_navigation_core import graph_store
from semantic_navigation_core.rooms import Room, load_rooms, room_of_point
from semantic_navigation_core.types import Observation, SemanticNode
from semantic_evaluation.core.experimental_schemas import QuerySpec

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


@dataclass
class SceneSpec:
    """Where a scene's data lives (paths may use ``~``)."""

    scene_id: str
    role: str = "test"                     # development | validation | test
    graph_db: str | None = None
    rooms_file: str | None = None
    images_dir: str | None = None
    category_images_dir: str | None = None
    max_images_per_room: int | None = None

    @classmethod
    def from_dict(cls, data: Mapping) -> "SceneSpec":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class SceneDataset:
    """A loaded scene: nodes + room rectangles + provenance."""

    scene_id: str
    nodes: list[SemanticNode]
    rooms: list[Room] = field(default_factory=list)
    source: str = ""                        # "graph_db" | "category_images"

    def node_ids(self) -> list[str]:
        return [n.node_id for n in self.nodes]

    def room_by_node(self) -> dict[str, str | None]:
        return {n.node_id: n.room_id for n in self.nodes}

    def position_by_node(self) -> dict[str, tuple[float, float]]:
        return {n.node_id: (n.position[0], n.position[1]) for n in self.nodes}


def _expand(path: str | None) -> str | None:
    return os.path.expanduser(path) if path else None


def _extra_views(images_dir: str, node_id: str) -> list[str]:
    """Additional view files following the ``<node_id>__*.png`` convention."""
    prefix = f"{node_id}__"
    return sorted(
        os.path.join(images_dir, f)
        for f in os.listdir(images_dir)
        if f.startswith(prefix) and f.lower().endswith(IMAGE_EXTENSIONS)
    )


def load_scene(spec: SceneSpec) -> SceneDataset:
    """Load a scene according to its spec (see module docstring for sources).

    Raises:
        FileNotFoundError: No usable source found for this spec.
    """
    graph_db = _expand(spec.graph_db)
    if graph_db and os.path.isfile(graph_db):
        images_dir = _expand(spec.images_dir)
        nodes = graph_store.load_semantic_nodes(graph_db, images_dir=images_dir)
        if images_dir and os.path.isdir(images_dir):
            for node in nodes:
                for i, path in enumerate(_extra_views(images_dir, node.node_id)):
                    node.observations.append(Observation(
                        observation_id=f"{node.node_id}__view{i + 1:02d}",
                        image_path=path,
                    ))
        rooms = graph_store.load_rooms(graph_db)
        rooms_file = _expand(spec.rooms_file)
        if not rooms and rooms_file:
            rooms = load_rooms(rooms_file)
        _assign_rooms_by_position(nodes, rooms)
        return SceneDataset(spec.scene_id, nodes, rooms, source="graph_db")

    category_dir = _expand(spec.category_images_dir)
    if category_dir and os.path.isdir(category_dir):
        return SceneDataset(
            spec.scene_id,
            _load_category_nodes(category_dir, spec.max_images_per_room),
            source="category_images",
        )

    raise FileNotFoundError(
        f"Scene '{spec.scene_id}': no graph_db at '{spec.graph_db}' and no "
        f"category_images_dir at '{spec.category_images_dir}'."
    )


def _assign_rooms_by_position(
    nodes: Iterable[SemanticNode], rooms: list[Room]
) -> None:
    """Fill missing ``room_id`` from the room rectangles (graph edges win)."""
    for node in nodes:
        if node.room_id is None and rooms:
            node.room_id = room_of_point(node.position[0], node.position[1], rooms)


def _load_category_nodes(
    category_dir: str, max_per_room: int | None
) -> list[SemanticNode]:
    """One single-view node per image; the directory name is the room."""
    nodes: list[SemanticNode] = []
    for room in sorted(os.listdir(category_dir)):
        room_dir = os.path.join(category_dir, room)
        if not os.path.isdir(room_dir):
            continue
        images = sorted(
            f for f in os.listdir(room_dir)
            if f.lower().endswith(IMAGE_EXTENSIONS)
        )
        if max_per_room is not None:
            images = images[:max_per_room]
        for i, fname in enumerate(images):
            node_id = f"{room}_{i + 1:03d}"
            nodes.append(SemanticNode(
                node_id=node_id,
                room_id=room,
                observations=[Observation(
                    observation_id=f"{node_id}__view01",
                    image_path=os.path.join(room_dir, fname),
                )],
            ))
    return nodes


def validate_scene(dataset: SceneDataset) -> list[str]:
    """Actionable dataset issues (empty list = ready for the experiments)."""
    issues: list[str] = []
    if not dataset.nodes:
        issues.append("scene has no nodes")
    dims = set()
    for node in dataset.nodes:
        has_embedding = bool(node.embeddings())
        has_image = any(
            o.image_path and os.path.isfile(o.image_path)
            for o in node.observations
        )
        if not node.observations:
            issues.append(f"node '{node.node_id}' has no observations")
        elif not has_embedding and not has_image:
            issues.append(
                f"node '{node.node_id}' has neither a stored embedding nor a "
                "readable image (cannot be encoded)"
            )
        for emb in node.embeddings():
            dims.add(int(np.asarray(emb).size))
        for obs in node.observations:
            if obs.image_path and not os.path.isfile(obs.image_path):
                issues.append(
                    f"node '{node.node_id}': missing image '{obs.image_path}'"
                )
    if len(dims) > 1:
        issues.append(f"inconsistent embedding dimensions: {sorted(dims)}")
    return issues


# ── Queries ──────────────────────────────────────────────────────────────── #


ExperimentQuery = QuerySpec


def load_queries(path: str) -> list[ExperimentQuery]:
    """Load ``semantic_queries.yaml`` (missing file → helpful error)."""
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"query file not found: {path}")
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    queries: list[ExperimentQuery] = []
    seen: set[str] = set()
    for index, entry in enumerate(data.get("queries", [])):
        query = QuerySpec.from_mapping(entry, f"{path}:queries[{index}]")
        if query.query_id in seen:
            raise ValueError(f"{path}: duplicate query_id '{query.query_id}'")
        seen.add(query.query_id)
        queries.append(query)
    return queries


def resolve_valid_nodes(
    query: ExperimentQuery, dataset: SceneDataset
) -> list[str]:
    """Ground-truth node ids for a query in this dataset.

    Explicit ``valid_node_ids`` win. When empty and ``expected_room`` is set,
    every node of that room is valid for room queries. Category-image datasets
    also use the category as ground truth for object, attribute and functional
    phrasings: these queries test whether language retrieves the intended image
    category, not whether a particular object instance is localized. Negative
    queries always resolve to [].
    """
    if query.is_negative:
        return []
    if query.valid_node_ids:
        return [v for v in query.valid_node_ids if v in set(dataset.node_ids())]
    metadata = getattr(dataset, "metadata", {}) or {}
    is_category_dataset = (
        getattr(dataset, "source", "") == "category_images"
        or metadata.get("node_semantics") == "independent_image"
    )
    if query.expected_room and (query.query_type == "room" or is_category_dataset):
        expected = query.expected_room.strip().lower()
        return [
            n.node_id for n in dataset.nodes
            if n.room_id and n.room_id.strip().lower() == expected
        ]
    return []


def validate_queries(
    queries: Iterable[ExperimentQuery], dataset: SceneDataset
) -> list[str]:
    """Issues for the queries of ``dataset.scene_id`` (unknown ids, no GT)."""
    node_ids = set(dataset.node_ids())
    issues: list[str] = []
    for q in queries:
        source_id = q.scene_id or q.dataset_id
        if source_id and source_id != dataset.scene_id:
            continue
        unknown = [v for v in q.valid_node_ids if v not in node_ids]
        if unknown:
            issues.append(f"query '{q.query_id}': unknown node ids {unknown}")
        if not q.is_negative and not resolve_valid_nodes(q, dataset):
            issues.append(
                f"query '{q.query_id}': no ground truth resolvable "
                "(annotate valid_node_ids; expected_room is only valid for room queries)"
            )
    return issues


def queries_template(dataset: SceneDataset) -> str:
    """YAML annotation template listing the scene's real node ids.

    Ground truth must be human-annotated — this template only enumerates what
    exists so no ``node_id`` is ever invented.
    """
    lines = [
        f"# Ground-truth annotation template for scene '{dataset.scene_id}'.",
        "# Fill valid_node_ids for each query using ONLY the ids listed below.",
        "#",
        "# Available nodes (id | room | objects):",
    ]
    for n in dataset.nodes:
        objs = ", ".join(n.object_labels()) or "-"
        lines.append(f"#   {n.node_id} | {n.room_id or '-'} | {objs}")
    lines += [
        "queries:",
        "  - query_id: q001",
        '    text: ""',
        "    language: es",
        "    query_type: object",
        f"    scene_id: {dataset.scene_id}",
        "    valid_node_ids: []",
        "    expected_objects: []",
        "    expected_relations: []",
        "    expected_room: null",
        "    is_negative: false",
    ]
    return "\n".join(lines) + "\n"
