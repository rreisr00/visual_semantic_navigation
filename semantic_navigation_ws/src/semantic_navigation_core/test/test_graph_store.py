"""Round-trip tests for the offline knowledge-graph SQLite store."""
import json
import sqlite3

import numpy as np
import pytest

from semantic_navigation_core.graph_store import (
    load_records,
    load_metadata,
    load_rooms,
    load_semantic_nodes,
    save_semantic_graph,
)
from semantic_navigation_core.rooms import Room
from semantic_navigation_core.types import (
    Observation,
    ObjectObservation,
    SemanticNode,
    SpatialRelation,
)


def make_node(node_id="wp_01", room=None):
    obs = Observation(
        observation_id=f"{node_id}__v0",
        embedding=np.array([0.6, 0.8], dtype=np.float32),
        objects=[
            ObjectObservation(label="cup", confidence=0.9, box=(0, 0, 10, 10)),
            ObjectObservation(label="tv", confidence=0.8, box=(20, 0, 40, 10)),
        ],
        relations=[SpatialRelation("cup", "LEFT_OF", "tv", confidence=0.7)],
    )
    return SemanticNode(
        node_id=node_id,
        position=(1.0, 2.0, 0.0),
        orientation=(0.0, 0.0, 0.0, 1.0),
        observations=[obs],
        room_id=room,
    )


def test_round_trip(tmp_path):
    db = str(tmp_path / "graph.db")
    room = Room("cocina", 0.0, 0.0, 5.0, 5.0)
    save_semantic_graph(db, [make_node(room="cocina")], rooms=[room])

    nodes = load_semantic_nodes(db)
    assert len(nodes) == 1
    loaded = nodes[0]
    assert loaded.node_id == "wp_01"
    assert loaded.position == (1.0, 2.0, 0.0)
    assert sorted(loaded.object_labels()) == ["cup", "tv"]
    assert loaded.room_id == "cocina"
    np.testing.assert_allclose(
        loaded.observations[0].embedding, [0.6, 0.8], rtol=1e-6
    )
    rooms = load_rooms(db)
    assert rooms[0].room_id == "cocina" and rooms[0].max_x == 5.0


def test_embedding_stored_as_json_string(tmp_path):
    """Bridge compatibility: visual_embedding must be a JSON string property."""
    db = str(tmp_path / "graph.db")
    save_semantic_graph(db, [make_node()])
    conn = sqlite3.connect(db)
    props = json.loads(conn.execute(
        "SELECT properties FROM nodes WHERE name='wp_01'"
    ).fetchone()[0])
    conn.close()
    assert isinstance(props["visual_embedding"], str)
    assert json.loads(props["visual_embedding"]) == pytest.approx([0.6, 0.8])


def test_relation_edges_optional(tmp_path):
    db = str(tmp_path / "graph.db")
    save_semantic_graph(db, [make_node()], include_relations=True)
    _, edges = load_records(db)
    assert ("LEFT_OF", "wp_01_cup", "wp_01_tv") in edges
    # CONTAINS edges follow the bridge convention: waypoint -> object.
    assert ("CONTAINS", "wp_01", "wp_01_cup") in edges


def test_legacy_list_embedding_accepted(tmp_path):
    db = str(tmp_path / "graph.db")
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE nodes (name TEXT PRIMARY KEY, type TEXT, properties TEXT)"
    )
    conn.execute(
        "CREATE TABLE edges (type TEXT, source_node TEXT, target_node TEXT,"
        " properties TEXT, PRIMARY KEY (type, source_node, target_node))"
    )
    conn.execute(
        "INSERT INTO nodes VALUES (?, ?, ?)",
        ("old", "waypoint", json.dumps({
            "pose_x": 0.0, "pose_y": 0.0,
            "visual_embedding": [0.1, 0.2],  # legacy: raw list, not JSON string
        })),
    )
    conn.commit()
    conn.close()
    nodes = load_semantic_nodes(db)
    np.testing.assert_allclose(nodes[0].observations[0].embedding, [0.1, 0.2])


def test_room_without_rectangle_still_gets_a_node(tmp_path):
    """room_id sin Room explícita → nodo de sala básico (aristas nunca huérfanas)."""
    db = str(tmp_path / "graph.db")
    save_semantic_graph(db, [make_node(room="cocina")])   # rooms=() a propósito
    records, edges = load_records(db)
    by_name = {r.name: r for r in records}
    assert by_name["cocina"].type == "room"
    assert ("CONTAINS", "cocina", "wp_01") in edges


def test_missing_db_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_records(str(tmp_path / "absent.db"))


def test_v2_round_trip_preserves_multiview_metadata_and_topology(tmp_path):
    db = str(tmp_path / "graph.db")
    first = make_node("wp_01")
    first.scene_id = "house"
    first.configuration_hash = "abc123"
    first.navigation_position = (1.2, 2.1, 0.0)
    first.neighbors = ["wp_02"]
    first.observations.append(Observation(
        observation_id="wp_01__v1",
        embedding=np.array([0.8, 0.6], dtype=np.float32),
        timestamp=12.25,
        camera_frame="camera_rgb_optical_frame",
        camera_position=(1.0, 2.0, 0.45),
        camera_orientation=(0.0, 0.0, 0.0, 1.0),
        requested_yaw=1.57,
        measured_yaw=1.55,
        angular_error=0.02,
        objects=[ObjectObservation(
            label="cup",
            object_id="persistent_cup",
            embedding=np.array([1.0, 0.0], dtype=np.float32),
        )],
    ))
    second = make_node("wp_02")
    second.scene_id = "house"
    save_semantic_graph(db, [first, second], include_relations=True)

    loaded = {node.node_id: node for node in load_semantic_nodes(db)}
    assert len(loaded["wp_01"].observations) == 2
    assert loaded["wp_01"].scene_id == "house"
    assert loaded["wp_01"].configuration_hash == "abc123"
    assert loaded["wp_01"].navigation_position == (1.2, 2.1, 0.0)
    assert loaded["wp_01"].neighbors == ["wp_02"]
    assert loaded["wp_01"].observations[1].camera_frame == "camera_rgb_optical_frame"
    assert load_metadata(db)["schema_version"] == "3"


def test_polygon_and_contamination_metadata_round_trip(tmp_path):
    db = str(tmp_path / "graph.db")
    room = Room.from_polygon(
        "office", [(0, 0), (4, 0), (3, 3), (0, 2)], 0.4
    )
    node = make_node(room="office")
    observation = node.observations[0]
    observation.camera_room = "office"
    observation.observation_room = "hall"
    observation.purity = 0.4
    observation.contamination_class = "contaminated"
    observation.transition_zone = True
    observation.objects[0].room_id = "hall"
    observation.objects[0].map_position = (3.5, 1.0, 0.0)
    save_semantic_graph(db, [node], rooms=[room])

    assert load_rooms(db) == [room]
    loaded = load_semantic_nodes(db)[0].observations[0]
    assert loaded.camera_room == "office"
    assert loaded.observation_room == "hall"
    assert loaded.purity == pytest.approx(0.4)
    assert loaded.contamination_class == "contaminated"
    assert loaded.objects[0].room_id == "hall"
