"""Offline reader/writer for the knowledge-graph SQLite file — no ROS imports.

The ``knowledge_graph_bridge`` node persists the graph to SQLite with the
schema ``nodes(name, type, properties-JSON)`` / ``edges(type, source_node,
target_node, properties-JSON)``, storing the visual embedding as a JSON string
property. This module reads and writes that exact format so notebooks can
consume graphs produced by the real ROS system and export graphs the bridge
can load back — without duplicating the graph implementation.
"""
from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from semantic_navigation_core.rooms import Room
from semantic_navigation_core.types import (
    Observation,
    ObjectObservation,
    SemanticNode,
    SpatialRelation,
)

NODE_TYPE_WAYPOINT = "waypoint"
NODE_TYPE_OBJECT = "object"
NODE_TYPE_ROOM = "room"
EDGE_CONTAINS = "CONTAINS"


@dataclass
class GraphRecord:
    """One raw graph node row (name, type, decoded properties)."""

    name: str
    type: str
    properties: dict = field(default_factory=dict)


def load_records(
    db_path: str,
) -> tuple[list[GraphRecord], list[tuple[str, str, str]]]:
    """Load raw nodes and edges ``(type, source, target)`` from the SQLite DB.

    Raises:
        FileNotFoundError: The database file does not exist.
    """
    db_path = os.path.expanduser(db_path)
    if not os.path.isfile(db_path):
        raise FileNotFoundError(f"knowledge-graph DB not found: {db_path}")
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT name, type, properties FROM nodes;")
        records = [
            GraphRecord(name, type_, json.loads(props) if props else {})
            for name, type_, props in cur.fetchall()
        ]
        cur.execute("SELECT type, source_node, target_node FROM edges;")
        edges = [tuple(row) for row in cur.fetchall()]
    finally:
        conn.close()
    return records, edges


def _embedding_from_props(props: dict) -> np.ndarray | None:
    """Decode ``visual_embedding``: JSON string (current) or list (legacy)."""
    raw = props.get("visual_embedding")
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = json.loads(raw) if raw else []
    values = np.asarray(raw, dtype=np.float32)
    return values if values.size else None


def load_semantic_nodes(
    db_path: str, images_dir: str | None = None
) -> list[SemanticNode]:
    """Reconstruct :class:`SemanticNode` objects from a bridge-produced DB.

    Waypoints become nodes with a single observation holding the stored
    embedding; object labels come from CONTAINS waypoint→object edges (the
    graph stores no confidences → 1.0); the room comes from the CONTAINS
    room→waypoint edge. When ``images_dir`` contains ``<node_id>.png`` the
    observation is linked to the capture frame saved by ``teleop_capture``.
    """
    records, edges = load_records(db_path)
    by_name = {r.name: r for r in records}
    images_dir = os.path.expanduser(images_dir) if images_dir else None

    nodes: list[SemanticNode] = []
    for rec in records:
        if rec.type != NODE_TYPE_WAYPOINT:
            continue
        p = rec.properties
        image_path = ""
        if images_dir:
            candidate = os.path.join(images_dir, f"{rec.name}.png")
            if os.path.isfile(candidate):
                image_path = candidate
        obs = Observation(
            observation_id=f"{rec.name}__db",
            embedding=_embedding_from_props(p),
            image_path=image_path,
        )
        nodes.append(SemanticNode(
            node_id=rec.name,
            position=(
                float(p.get("pose_x", 0.0)),
                float(p.get("pose_y", 0.0)),
                float(p.get("pose_z", 0.0)),
            ),
            orientation=(
                float(p.get("orient_x", 0.0)),
                float(p.get("orient_y", 0.0)),
                float(p.get("orient_z", 0.0)),
                float(p.get("orient_w", 1.0)),
            ),
            observations=[obs],
        ))

    node_by_id = {n.node_id: n for n in nodes}
    for edge_type, source, target in edges:
        if edge_type != EDGE_CONTAINS:
            continue
        src, tgt = by_name.get(source), by_name.get(target)
        if src is None or tgt is None:
            continue
        if src.type == NODE_TYPE_WAYPOINT and tgt.type == NODE_TYPE_OBJECT:
            label = str(tgt.properties.get("label", "")).strip()
            if label and source in node_by_id:
                node_by_id[source].observations[0].objects.append(
                    ObjectObservation(label=label)
                )
        elif src.type == NODE_TYPE_ROOM and tgt.type == NODE_TYPE_WAYPOINT:
            if target in node_by_id and node_by_id[target].room_id is None:
                node_by_id[target].room_id = source
    return nodes


def load_rooms(db_path: str) -> list[Room]:
    """Room rectangles stored in the graph (same shape as ``rooms.yaml``)."""
    records, _ = load_records(db_path)
    rooms: list[Room] = []
    for rec in records:
        if rec.type != NODE_TYPE_ROOM:
            continue
        p = rec.properties
        rooms.append(Room(
            room_id=rec.name,
            min_x=float(p.get("min_x", 0.0)),
            min_y=float(p.get("min_y", 0.0)),
            max_x=float(p.get("max_x", 0.0)),
            max_y=float(p.get("max_y", 0.0)),
        ).normalized())
    return rooms


# ── Export (bridge-compatible) ───────────────────────────────────────────── #


def _connect_for_write(db_path: str) -> sqlite3.Connection:
    db_path = os.path.expanduser(db_path)
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS nodes (
            name       TEXT PRIMARY KEY,
            type       TEXT NOT NULL,
            properties TEXT
        );
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS edges (
            type        TEXT NOT NULL,
            source_node TEXT NOT NULL,
            target_node TEXT NOT NULL,
            properties  TEXT,
            PRIMARY KEY (type, source_node, target_node),
            FOREIGN KEY (source_node) REFERENCES nodes (name),
            FOREIGN KEY (target_node) REFERENCES nodes (name)
        );
    """)
    return conn


def _upsert_node(cur, name: str, type_: str, props: dict) -> None:
    cur.execute(
        "INSERT INTO nodes(name, type, properties) VALUES(?, ?, ?)"
        " ON CONFLICT(name) DO UPDATE SET type=excluded.type,"
        " properties=excluded.properties;",
        (name, type_, json.dumps(props)),
    )


def _upsert_edge(cur, type_: str, source: str, target: str, props: dict) -> None:
    cur.execute(
        "INSERT INTO edges(type, source_node, target_node, properties)"
        " VALUES(?, ?, ?, ?)"
        " ON CONFLICT(type, source_node, target_node)"
        " DO UPDATE SET properties=excluded.properties;",
        (type_, source, target, json.dumps(props)),
    )


def save_semantic_graph(
    db_path: str,
    nodes: Sequence[SemanticNode],
    rooms: Iterable[Room] = (),
    include_relations: bool = False,
) -> str:
    """Write nodes/objects/rooms (and optionally relations) bridge-style.

    Waypoint properties, the JSON-string embedding, object node naming
    (``<waypoint>_<label>``) and CONTAINS edge directions mirror
    ``knowledge_graph_bridge_node`` exactly, so the exported DB can be loaded
    by the ROS system via its ``db_file_path`` parameter. Multi-view nodes are
    collapsed with ``SemanticNode.to_waypoint()`` (first-view embedding);
    aggregate beforehand if another representation is wanted.

    With ``include_relations`` each relation hypothesis becomes an edge
    ``<predicate>`` between the pair's object nodes with a ``confidence``
    property — additive information the bridge simply loads as extra edges.

    Returns the written path.
    """
    conn = _connect_for_write(db_path)
    try:
        cur = conn.cursor()
        written_rooms: set[str] = set()
        for room in rooms:
            room = room.normalized()
            _upsert_node(cur, room.room_id, NODE_TYPE_ROOM, {
                "min_x": room.min_x, "min_y": room.min_y,
                "max_x": room.max_x, "max_y": room.max_y,
            })
            written_rooms.add(room.room_id)
        # Rooms referenced only via node.room_id (no rectangle known): create a
        # bare room node so the CONTAINS edge never points at a missing node.
        for node in nodes:
            if node.room_id and node.room_id not in written_rooms:
                _upsert_node(cur, node.room_id, NODE_TYPE_ROOM, {})
                written_rooms.add(node.room_id)
        for node in nodes:
            wp = node.to_waypoint()
            props = {
                "pose_x": wp.position[0], "pose_y": wp.position[1],
                "pose_z": wp.position[2],
                "orient_x": wp.orientation[0], "orient_y": wp.orientation[1],
                "orient_z": wp.orientation[2], "orient_w": wp.orientation[3],
                "visual_embedding": json.dumps(
                    [float(v) for v in np.asarray(wp.embedding, dtype=np.float32)]
                ),
            }
            _upsert_node(cur, node.node_id, NODE_TYPE_WAYPOINT, props)
            for label in wp.objects:
                obj_id = f"{node.node_id}_{label}"
                _upsert_node(cur, obj_id, NODE_TYPE_OBJECT, {
                    "label": label, "source_waypoint": node.node_id,
                })
                _upsert_edge(cur, EDGE_CONTAINS, node.node_id, obj_id, {})
            if node.room_id:
                _upsert_edge(cur, EDGE_CONTAINS, node.room_id, node.node_id, {})
            if include_relations:
                for rel in _strongest_relations(node.relations()):
                    src = f"{node.node_id}_{rel.subject}"
                    tgt = f"{node.node_id}_{rel.obj}"
                    _upsert_edge(cur, rel.predicate, src, tgt, {
                        "confidence": rel.confidence,
                    })
        conn.commit()
    finally:
        conn.close()
    return os.path.expanduser(db_path)


def _strongest_relations(
    relations: Sequence[SpatialRelation],
) -> list[SpatialRelation]:
    """Keep the highest-confidence instance per (subject, predicate, obj)."""
    best: dict[tuple[str, str, str], SpatialRelation] = {}
    for rel in relations:
        key = (rel.subject, rel.predicate, rel.obj)
        if key not in best or rel.confidence > best[key].confidence:
            best[key] = rel
    return list(best.values())
