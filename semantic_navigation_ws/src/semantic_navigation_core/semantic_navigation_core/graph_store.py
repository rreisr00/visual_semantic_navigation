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

from semantic_navigation_core.multiview import purity_weighted_node_embedding
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
EDGE_HAS_OBSERVATION = "HAS_OBSERVATION"
EDGE_OBSERVED_IN = "OBSERVED_IN"
EDGE_CONNECTED_TO = "CONNECTED_TO"
SCHEMA_VERSION = 3


@dataclass
class GraphRecord:
    """One raw graph node row (name, type, decoded properties)."""

    name: str
    type: str
    properties: dict = field(default_factory=dict)


def load_records(
    db_path: str,
    read_only: bool = False,
) -> tuple[list[GraphRecord], list[tuple[str, str, str]]]:
    """Load raw nodes and edges ``(type, source, target)`` from the SQLite DB.

    Raises:
        FileNotFoundError: The database file does not exist.
    """
    db_path = os.path.expanduser(db_path)
    if not os.path.isfile(db_path):
        raise FileNotFoundError(f"knowledge-graph DB not found: {db_path}")
    conn = (
        sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        if read_only
        else sqlite3.connect(db_path)
    )
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


def load_metadata(db_path: str) -> dict[str, str]:
    """Load schema/campaign metadata without mutating the database."""
    db_path = os.path.expanduser(db_path)
    if not os.path.isfile(db_path):
        raise FileNotFoundError(f"knowledge-graph DB not found: {db_path}")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='metadata';"
        ).fetchone()
        if not exists:
            return {"schema_version": "1"}
        return {str(key): str(value) for key, value in conn.execute(
            "SELECT key, value FROM metadata;"
        ).fetchall()}
    finally:
        conn.close()


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
    db_path: str, images_dir: str | None = None, read_only: bool = True
) -> list[SemanticNode]:
    """Reconstruct :class:`SemanticNode` objects from a bridge-produced DB.

    Waypoints become nodes with a single observation holding the stored
    embedding; object labels come from CONTAINS waypoint→object edges (the
    graph stores no confidences → 1.0); the room comes from the CONTAINS
    room→waypoint edge. When ``images_dir`` contains ``<node_id>.png`` the
    observation is linked to the capture frame saved by ``teleop_capture``.
    """
    records, edges = load_records(db_path, read_only=read_only)
    by_name = {r.name: r for r in records}
    images_dir = os.path.expanduser(images_dir) if images_dir else None

    nodes: list[SemanticNode] = []
    for rec in records:
        if rec.type != NODE_TYPE_WAYPOINT:
            continue
        p = rec.properties
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
            observations=[],
            scene_id=str(p.get("scene_id", "default")),
            navigation_position=(
                float(p.get("goal_x", p.get("pose_x", 0.0))),
                float(p.get("goal_y", p.get("pose_y", 0.0))),
                float(p.get("goal_z", p.get("pose_z", 0.0))),
            ),
            navigation_orientation=(
                float(p.get("goal_qx", p.get("orient_x", 0.0))),
                float(p.get("goal_qy", p.get("orient_y", 0.0))),
                float(p.get("goal_qz", p.get("orient_z", 0.0))),
                float(p.get("goal_qw", p.get("orient_w", 1.0))),
            ),
            creation_timestamp=float(p.get("creation_timestamp", 0.0)),
            configuration_hash=str(p.get("configuration_hash", "")),
        ))

    node_by_id = {n.node_id: n for n in nodes}
    nodes_with_observations = {
        source for edge_type, source, _target in edges
        if edge_type == EDGE_HAS_OBSERVATION
    }
    for edge_type, source, target in edges:
        if edge_type == "HAS_OBSERVATION" and source in node_by_id:
            observation_record = by_name.get(target)
            if observation_record is None:
                continue
            payload = observation_record.properties.get("payload", "{}")
            payload = json.loads(payload) if isinstance(payload, str) else payload
            objects = []
            for item in payload.get("detections", []):
                box = item.get("bounding_box")
                crop = np.asarray(item.get("crop_embedding", []), dtype=np.float32)
                position_2d = item.get("position_2d")
                position_3d = item.get("position_3d")
                objects.append(ObjectObservation(
                    label=str(item.get("class_name", "")),
                    confidence=float(item.get("confidence", 0.0)),
                    box=tuple(float(v) for v in box) if box and len(box) == 4 else None,
                    embedding=crop if crop.size else None,
                    object_id=str(item.get("object_id", "")),
                    observation_ids=[str(payload.get("observation_id", ""))],
                    associated_node_ids=[source],
                    position_2d=(
                        tuple(float(v) for v in position_2d)
                        if position_2d and len(position_2d) == 2 else None
                    ),
                    position_3d=(
                        tuple(float(v) for v in position_3d)
                        if position_3d and len(position_3d) == 3 else None
                    ),
                    position_3d_frame=str(item.get("position_3d_frame", "")),
                    map_position=(
                        tuple(float(v) for v in (item.get("map_position") or []))
                        if len(item.get("map_position") or []) == 3 else None
                    ),
                    room_id=(str(item["room_id"]) if item.get("room_id") else None),
                ))
            relations = [SpatialRelation(
                subject=str(item.get("subject_id", "")),
                predicate=str(item.get("predicate", "")),
                obj=str(item.get("object_id", "")),
                confidence=float(item.get("confidence", 0.0)),
                subject_id=str(item.get("subject_id", "")),
                object_id=str(item.get("object_id", "")),
                reference_frame=str(item.get("reference_frame", "")),
                source_observation_id=str(item.get("source_observation_id", "")),
                relation_type=str(item.get("relation_type", "visual_2d_hypothesis")),
            ) for item in payload.get("relations", [])]
            stamp = payload.get("timestamp", [0, 0])
            camera_pose = payload.get("camera_pose", [])
            depth_camera_pose = payload.get("depth_camera_pose", [])
            embedding = np.asarray(payload.get("image_embedding", []), dtype=np.float32)
            node_by_id[source].observations.append(Observation(
                observation_id=str(payload.get("observation_id", target)),
                embedding=embedding if embedding.size else None,
                image_path=str(payload.get("image_path", "")),
                objects=objects,
                relations=relations,
                timestamp=(
                    float(stamp[0]) + float(stamp[1]) * 1e-9
                    if len(stamp) == 2 else 0.0
                ),
                camera_frame=str(payload.get("camera_frame", "")),
                camera_position=(
                    tuple(float(v) for v in camera_pose[:3])
                    if len(camera_pose) == 7 else None
                ),
                camera_orientation=(
                    tuple(float(v) for v in camera_pose[3:])
                    if len(camera_pose) == 7 else None
                ),
                depth_camera_frame=str(payload.get("depth_camera_frame", "")),
                depth_camera_position=(
                    tuple(float(v) for v in depth_camera_pose[:3])
                    if len(depth_camera_pose) == 7 else None
                ),
                depth_camera_orientation=(
                    tuple(float(v) for v in depth_camera_pose[3:])
                    if len(depth_camera_pose) == 7 else None
                ),
                requested_yaw=float(payload.get("requested_yaw", 0.0)),
                measured_yaw=float(payload.get("measured_yaw", 0.0)),
                angular_error=float(payload.get("angular_error", 0.0)),
                image_valid=bool(payload.get("image_valid", False)),
                depth_valid=bool(payload.get("depth_valid", False)),
                camera_room=(
                    str(payload["camera_room"]) if payload.get("camera_room") else None
                ),
                observation_room=(
                    str(payload["observation_room"])
                    if payload.get("observation_room") else None
                ),
                purity=(
                    float(payload["purity"]) if payload.get("purity") is not None else None
                ),
                contamination_class=str(payload.get("contamination_class", "unknown")),
                transition_zone=bool(payload.get("transition_zone", False)),
            ))
            continue
        if edge_type != EDGE_CONTAINS:
            if (
                edge_type == "CONNECTED_TO"
                and source in node_by_id
                and target in node_by_id
            ):
                node_by_id[source].neighbors.append(target)
            continue
        src, tgt = by_name.get(source), by_name.get(target)
        if src is None or tgt is None:
            continue
        if src.type == NODE_TYPE_WAYPOINT and tgt.type == NODE_TYPE_OBJECT:
            label = str(tgt.properties.get("label", "")).strip()
            if (
                label
                and source in node_by_id
                and source not in nodes_with_observations
            ):
                if not node_by_id[source].observations:
                    legacy_image = ""
                    if images_dir:
                        candidate = os.path.join(images_dir, f"{source}.png")
                        legacy_image = candidate if os.path.isfile(candidate) else ""
                    node_by_id[source].observations.append(Observation(
                        observation_id=f"{source}__db",
                        embedding=_embedding_from_props(by_name[source].properties),
                        image_path=legacy_image,
                    ))
                # V1 databases encode every detection as a CONTAINS edge.
                # Append all of them to the one synthetic legacy observation.
                node_by_id[source].observations[0].objects.append(
                    ObjectObservation(label=label)
                )
        elif src.type == NODE_TYPE_ROOM and tgt.type == NODE_TYPE_WAYPOINT:
            if target in node_by_id and node_by_id[target].room_id is None:
                node_by_id[target].room_id = source
    for node in nodes:
        if not node.observations:
            node.observations.append(Observation(
                observation_id=f"{node.node_id}__db",
                embedding=_embedding_from_props(by_name[node.node_id].properties),
                image_path=(
                    os.path.join(images_dir, f"{node.node_id}.png")
                    if images_dir and os.path.isfile(
                        os.path.join(images_dir, f"{node.node_id}.png")
                    )
                    else ""
                ),
            ))
    return nodes


def load_rooms(db_path: str) -> list[Room]:
    """Room rectangles stored in the graph (same shape as ``rooms.yaml``)."""
    records, _ = load_records(db_path)
    rooms: list[Room] = []
    for rec in records:
        if rec.type != NODE_TYPE_ROOM:
            continue
        p = rec.properties
        raw_polygon = p.get("polygon", [])
        if isinstance(raw_polygon, str):
            raw_polygon = json.loads(raw_polygon) if raw_polygon else []
        transition = float(p.get("transition_width_m", 0.5))
        if len(raw_polygon) >= 3:
            rooms.append(Room.from_polygon(rec.name, raw_polygon, transition))
        else:
            rooms.append(Room(
                room_id=rec.name,
                min_x=float(p.get("min_x", 0.0)),
                min_y=float(p.get("min_y", 0.0)),
                max_x=float(p.get("max_x", 0.0)),
                max_y=float(p.get("max_y", 0.0)),
                transition_width_m=transition,
            ).normalized())
    return rooms


# ── Export (bridge-compatible) ───────────────────────────────────────────── #


def _connect_for_write(db_path: str) -> sqlite3.Connection:
    db_path = os.path.expanduser(db_path)
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA journal_mode=WAL;")
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
    conn.execute("""
        CREATE TABLE IF NOT EXISTS metadata (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
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
                "polygon": json.dumps([list(point) for point in room.corners()]),
                "transition_width_m": room.transition_width_m,
            })
            written_rooms.add(room.room_id)
        # Rooms referenced only via node.room_id (no rectangle known): create a
        # bare room node so the CONTAINS edge never points at a missing node.
        for node in nodes:
            if node.room_id and node.room_id not in written_rooms:
                _upsert_node(cur, node.room_id, NODE_TYPE_ROOM, {})
                written_rooms.add(node.room_id)
        scene_ids = sorted({node.scene_id for node in nodes})
        config_hashes = sorted({
            node.configuration_hash for node in nodes if node.configuration_hash
        })
        for key, value in (
            ("schema_version", str(SCHEMA_VERSION)),
            ("scene_ids", json.dumps(scene_ids)),
            ("configuration_hashes", json.dumps(config_hashes)),
        ):
            cur.execute(
                "INSERT INTO metadata(key, value) VALUES(?, ?)"
                " ON CONFLICT(key) DO UPDATE SET value=excluded.value;",
                (key, value),
            )

        for node in nodes:
            wp = node.to_waypoint(purity_weighted_node_embedding(node))
            nav_position = node.navigation_position or node.position
            nav_orientation = node.navigation_orientation or node.orientation
            props = {
                "pose_x": wp.position[0], "pose_y": wp.position[1],
                "pose_z": wp.position[2],
                "orient_x": wp.orientation[0], "orient_y": wp.orientation[1],
                "orient_z": wp.orientation[2], "orient_w": wp.orientation[3],
                "goal_x": nav_position[0], "goal_y": nav_position[1],
                "goal_z": nav_position[2],
                "goal_qx": nav_orientation[0], "goal_qy": nav_orientation[1],
                "goal_qz": nav_orientation[2], "goal_qw": nav_orientation[3],
                "scene_id": node.scene_id,
                "creation_timestamp": node.creation_timestamp,
                "configuration_hash": node.configuration_hash,
                "visual_embedding": json.dumps(
                    [float(v) for v in np.asarray(wp.embedding, dtype=np.float32)]
                ),
            }
            _upsert_node(cur, node.node_id, NODE_TYPE_WAYPOINT, props)

            object_ids: dict[str, str] = {}
            label_counts: dict[str, int] = {}
            for observation_index, observation in enumerate(node.observations):
                observation_id = observation.observation_id or (
                    f"{node.node_id}__view_{observation_index:04d}"
                )
                graph_observation_id = f"observation::{observation_id}"
                payload = _observation_payload(node, observation, observation_id)
                _upsert_node(cur, graph_observation_id, "observation", {
                    "observation_id": observation_id,
                    "scene_id": node.scene_id,
                    "timestamp_sec": observation.timestamp,
                    "payload": json.dumps(payload, sort_keys=True),
                })
                _upsert_edge(
                    cur, EDGE_HAS_OBSERVATION,
                    node.node_id, graph_observation_id, {},
                )
                per_view_counts: dict[str, int] = {}
                view_relation_ids: dict[str, str] = {}
                for detected in observation.objects:
                    occurrence = per_view_counts.get(detected.label, 0)
                    per_view_counts[detected.label] = occurrence + 1
                    identity = (
                        detected.object_id
                        if detected.object_id
                        and not detected.object_id.startswith("detection_")
                        else f"{detected.label}::{occurrence}"
                    )
                    if identity in object_ids:
                        object_id = object_ids[identity]
                    else:
                        count = label_counts.get(detected.label, 0)
                        label_counts[detected.label] = count + 1
                        base = f"{node.node_id}_{detected.label}"
                        object_id = base if count == 0 else f"{base}_{count + 1}"
                        object_ids[identity] = object_id
                    if detected.object_id:
                        view_relation_ids[detected.object_id] = object_id
                    view_relation_ids.setdefault(detected.label, object_id)
                    object_props = {
                        "label": detected.label,
                        "source_waypoint": node.node_id,
                        "confidence": detected.confidence,
                        "last_seen": detected.last_seen or observation.timestamp,
                        "bounding_box": json.dumps(
                            list(detected.box) if detected.box is not None else []
                        ),
                        "crop_embedding": json.dumps(
                            [float(v) for v in np.asarray(detected.embedding)]
                            if detected.embedding is not None else []
                        ),
                        "position_2d": json.dumps(detected.position_2d),
                        "position_3d": json.dumps(detected.position_3d),
                        "position_3d_map": json.dumps(detected.map_position),
                        "room_id": detected.room_id or "",
                    }
                    _upsert_node(cur, object_id, NODE_TYPE_OBJECT, object_props)
                    _upsert_edge(cur, EDGE_CONTAINS, node.node_id, object_id, {})
                    _upsert_edge(
                        cur, EDGE_OBSERVED_IN,
                        object_id, graph_observation_id, {},
                    )
                if include_relations:
                    for relation in _strongest_relations(observation.relations):
                        source = (
                            view_relation_ids.get(relation.subject_id)
                            or view_relation_ids.get(relation.subject)
                        )
                        target = (
                            view_relation_ids.get(relation.object_id)
                            or view_relation_ids.get(relation.obj)
                        )
                        if not source or not target:
                            continue
                        _upsert_edge(cur, relation.predicate, source, target, {
                            "confidence": relation.confidence,
                            "reference_frame": relation.reference_frame,
                            "source_observation_id": (
                                relation.source_observation_id or observation_id
                            ),
                            "relation_type": relation.relation_type,
                            "timestamp": relation.timestamp,
                        })
            if node.room_id:
                _upsert_edge(cur, EDGE_CONTAINS, node.room_id, node.node_id, {})
        nodes_by_id = {node.node_id: node for node in nodes}
        for node in nodes:
            for neighbor in node.neighbors:
                if neighbor in nodes_by_id:
                    distance = float(np.linalg.norm(
                        np.asarray(node.position[:2], dtype=np.float32)
                        - np.asarray(nodes_by_id[neighbor].position[:2], dtype=np.float32)
                    ))
                    _upsert_edge(
                        cur, EDGE_CONNECTED_TO, node.node_id, neighbor,
                        {"distance_m": distance},
                    )
        conn.commit()
    finally:
        conn.close()
    return os.path.expanduser(db_path)


def _observation_payload(
    node: SemanticNode,
    observation: Observation,
    observation_id: str,
) -> dict:
    camera_pose = []
    if observation.camera_position is not None and observation.camera_orientation is not None:
        camera_pose = [
            *[float(value) for value in observation.camera_position],
            *[float(value) for value in observation.camera_orientation],
        ]
    depth_camera_pose = []
    if (
        observation.depth_camera_position is not None
        and observation.depth_camera_orientation is not None
    ):
        depth_camera_pose = [
            *[float(value) for value in observation.depth_camera_position],
            *[float(value) for value in observation.depth_camera_orientation],
        ]
    seconds = int(observation.timestamp)
    nanoseconds = int(round((observation.timestamp - seconds) * 1e9))
    return {
        "observation_id": observation_id,
        "node_id": node.node_id,
        "scene_id": node.scene_id,
        "image_path": observation.image_path,
        "timestamp": [seconds, nanoseconds],
        "camera_frame": observation.camera_frame,
        "camera_pose": camera_pose,
        "depth_camera_frame": observation.depth_camera_frame,
        "depth_camera_pose": depth_camera_pose,
        "requested_yaw": observation.requested_yaw,
        "measured_yaw": observation.measured_yaw,
        "angular_error": observation.angular_error,
        "image_valid": observation.image_valid,
        "depth_valid": observation.depth_valid,
        "camera_room": observation.camera_room or "",
        "observation_room": observation.observation_room or "",
        "purity": observation.purity,
        "contamination_class": observation.contamination_class,
        "transition_zone": observation.transition_zone,
        "image_embedding": (
            [float(value) for value in np.asarray(observation.embedding)]
            if observation.embedding is not None else []
        ),
        "detections": [
            {
                "object_id": item.object_id,
                "class_name": item.label,
                "confidence": item.confidence,
                "bounding_box": list(item.box) if item.box is not None else [],
                "crop_embedding": (
                    [float(value) for value in np.asarray(item.embedding)]
                    if item.embedding is not None else []
                ),
                "position_2d": item.position_2d,
                "position_3d": item.position_3d,
                "position_3d_frame": item.position_3d_frame,
                "map_position": item.map_position,
                "room_id": item.room_id or "",
            }
            for item in observation.objects
        ],
        "relations": [
            {
                "subject_id": item.subject_id or item.subject,
                "predicate": item.predicate,
                "object_id": item.object_id or item.obj,
                "confidence": item.confidence,
                "reference_frame": item.reference_frame,
                "source_observation_id": (
                    item.source_observation_id or observation_id
                ),
                "relation_type": item.relation_type,
                "timestamp": [
                    int(item.timestamp),
                    int(round((item.timestamp - int(item.timestamp)) * 1e9)),
                ],
            }
            for item in observation.relations
        ],
    }


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
