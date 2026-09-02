"""Structural quality analysis for persisted semantic graphs."""
from __future__ import annotations

import math
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np

from semantic_evaluation.core.experimental_schemas import GraphQualityResult
from semantic_navigation_core.graph_store import load_records, load_semantic_nodes


def _component_count(nodes: set[str], edges: list[tuple[str, str]]) -> tuple[int, set[str]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for source, target in edges:
        adjacency[source].add(target)
        adjacency[target].add(source)
    unseen = set(nodes)
    components = 0
    while unseen:
        components += 1
        queue = deque([unseen.pop()])
        while queue:
            current = queue.popleft()
            for neighbour in adjacency.get(current, set()) & unseen:
                unseen.remove(neighbour)
                queue.append(neighbour)
    isolated = {node for node in nodes if not adjacency.get(node)}
    return components, isolated


def analyze_graph(scene_id: str, db_path: str | Path) -> tuple[GraphQualityResult, list[dict[str, Any]]]:
    records, raw_edges = load_records(str(db_path))
    semantic_nodes = load_semantic_nodes(str(db_path))
    issues: list[dict[str, Any]] = []
    names = [record.name for record in records]
    name_set = set(names)
    types = Counter(record.type for record in records)
    duplicate_nodes = len(names) - len(name_set)
    valid_edges: list[tuple[str, str]] = []
    invalid_edges = 0
    edge_types = Counter()
    contains_parents: dict[str, list[str]] = defaultdict(list)
    for edge_type, source, target in raw_edges:
        edge_types[edge_type] += 1
        if source not in name_set or target not in name_set:
            invalid_edges += 1
            issues.append({"scene_id": scene_id, "issue": "invalid_edge",
                           "node_id": f"{source}->{target}", "detail": edge_type})
            continue
        valid_edges.append((source, target))
        if edge_type == "CONTAINS":
            contains_parents[target].append(source)
    components, isolated = _component_count(name_set, valid_edges)
    degrees = Counter()
    for source, target in valid_edges:
        degrees[source] += 1
        degrees[target] += 1
    record_by_name = {record.name: record for record in records}
    objects_without_waypoint = 0
    waypoints_without_room = 0
    object_signatures: Counter[tuple[str, str]] = Counter()
    for record in records:
        if record.type == "object":
            parents = [parent for parent in contains_parents.get(record.name, [])
                       if record_by_name[parent].type == "waypoint"]
            if not parents:
                objects_without_waypoint += 1
            label = str(record.properties.get("label", ""))
            for parent in parents or [str(record.properties.get("source_waypoint", ""))]:
                object_signatures[(parent, label)] += 1
        elif record.type == "waypoint":
            room_parents = [parent for parent in contains_parents.get(record.name, [])
                            if record_by_name[parent].type == "room"]
            if not room_parents:
                waypoints_without_room += 1
    dims: set[int] = set()
    nodes_without_embeddings = 0
    empty_or_nonfinite = 0
    observation_count = 0
    for node in semantic_nodes:
        observation_count += len(node.observations)
        embeddings = node.embeddings()
        if not embeddings:
            nodes_without_embeddings += 1
        for embedding in embeddings:
            vector = np.asarray(embedding)
            dims.add(int(vector.size))
            if vector.size == 0 or not np.isfinite(vector).all():
                empty_or_nonfinite += 1
    probable_duplicates = sum(count - 1 for count in object_signatures.values() if count > 1)
    for node_id in sorted(isolated):
        issues.append({"scene_id": scene_id, "issue": "isolated_node",
                       "node_id": node_id, "detail": record_by_name[node_id].type})
    result = GraphQualityResult(
        scene_id=scene_id,
        waypoint_count=types.get("waypoint", 0),
        object_count=types.get("object", 0),
        room_count=types.get("room", 0),
        observation_count=observation_count,
        connected_components=components,
        isolated_nodes=len(isolated),
        mean_degree=(sum(degrees.values()) / len(name_set) if name_set else 0.0),
        invalid_edges=invalid_edges,
        objects_without_waypoint=objects_without_waypoint,
        waypoints_without_room=waypoints_without_room,
        nodes_without_embeddings=nodes_without_embeddings,
        inconsistent_embedding_dimensions=len(dims) > 1,
        empty_or_nonfinite_embeddings=empty_or_nonfinite,
        duplicate_nodes=duplicate_nodes,
        probable_duplicate_objects=probable_duplicates,
    )
    issues.append({"scene_id": scene_id, "issue": "edge_type_counts",
                   "node_id": "", "detail": dict(edge_types)})
    if len(dims) > 1:
        issues.append({"scene_id": scene_id, "issue": "embedding_dimensions",
                       "node_id": "", "detail": sorted(dims)})
    return result, issues


def safe_analyze_graph(
    scene_id: str, db_path: str | Path
) -> tuple[GraphQualityResult, list[dict[str, Any]]]:
    path = Path(db_path)
    if not path.is_file():
        return GraphQualityResult(scene_id=scene_id), [{
            "scene_id": scene_id,
            "issue": "missing_graph",
            "node_id": "",
            "detail": str(path),
        }]
    try:
        return analyze_graph(scene_id, path)
    except (OSError, ValueError, KeyError) as exc:
        return GraphQualityResult(scene_id=scene_id), [{
            "scene_id": scene_id,
            "issue": "invalid_graph",
            "node_id": "",
            "detail": str(exc),
        }]
