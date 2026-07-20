"""Unit tests for the offline dataset adapter and query loading."""
import numpy as np
import pytest
import yaml

from semantic_evaluation.core.offline_dataset import (
    ExperimentQuery,
    SceneSpec,
    load_queries,
    load_scene,
    queries_template,
    resolve_valid_nodes,
    validate_queries,
    validate_scene,
)
from semantic_navigation_core.graph_store import save_semantic_graph
from semantic_navigation_core.rooms import Room
from semantic_navigation_core.types import Observation, SemanticNode


def _png(path):
    """Minimal 1x1 PNG so image existence checks pass."""
    import struct
    import zlib

    def chunk(tag, data):
        raw = tag + data
        return struct.pack(">I", len(data)) + raw + struct.pack(
            ">I", zlib.crc32(raw)
        )

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"\x00\xff\x00\x00")
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", idat) + chunk(b"IEND", b"")
    )


@pytest.fixture()
def graph_scene(tmp_path):
    db = tmp_path / "kg.db"
    images = tmp_path / "images"
    images.mkdir()
    node = SemanticNode(
        node_id="wp_01",
        position=(1.0, 1.0, 0.0),
        observations=[Observation(
            observation_id="wp_01__v0",
            embedding=np.array([1.0, 0.0], dtype=np.float32),
        )],
    )
    save_semantic_graph(str(db), [node], rooms=[Room("cocina", 0, 0, 5, 5)])
    _png(images / "wp_01.png")
    _png(images / "wp_01__extra.png")   # second view via naming convention
    return SceneSpec(
        scene_id="test_scene", graph_db=str(db), images_dir=str(images)
    )


def test_load_graph_scene_with_extra_views(graph_scene):
    dataset = load_scene(graph_scene)
    assert dataset.source == "graph_db"
    node = dataset.nodes[0]
    assert len(node.observations) == 2          # DB embedding + extra view
    assert node.room_id == "cocina"             # from the room rectangle
    assert validate_scene(dataset) == []


def test_load_category_scene(tmp_path):
    for room in ("kitchen", "bedroom"):
        d = tmp_path / room
        d.mkdir()
        for i in range(3):
            _png(d / f"img_{i}.png")
    spec = SceneSpec(
        scene_id="cats", category_images_dir=str(tmp_path), max_images_per_room=2
    )
    dataset = load_scene(spec)
    assert len(dataset.nodes) == 4              # 2 rooms × max 2 images
    assert {n.room_id for n in dataset.nodes} == {"kitchen", "bedroom"}
    # Images exist but embeddings are pending → still valid (encodable).
    assert validate_scene(dataset) == []


def test_missing_scene_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_scene(SceneSpec(scene_id="ghost", graph_db=str(tmp_path / "no.db")))


def test_queries_yaml_round_trip(tmp_path, graph_scene):
    dataset = load_scene(graph_scene)
    path = tmp_path / "queries.yaml"
    path.write_text(yaml.safe_dump({"queries": [
        {
            "query_id": "q1", "text": "ve a la cocina", "language": "es",
            "query_type": "room", "scene_id": "test_scene",
            "valid_node_ids": [], "expected_room": "cocina",
        },
        {
            "query_id": "q2", "text": "find the pool", "language": "en",
            "query_type": "object", "scene_id": "test_scene",
            "valid_node_ids": [], "is_negative": True,
        },
        {
            "query_id": "q3", "text": "bad ids", "scene_id": "test_scene",
            "valid_node_ids": ["nope"],
        },
    ]}))
    queries = load_queries(str(path))
    assert len(queries) == 3

    # expected_room resolves ground truth without inventing ids.
    assert resolve_valid_nodes(queries[0], dataset) == ["wp_01"]
    assert resolve_valid_nodes(queries[1], dataset) == []   # negative

    issues = validate_queries(queries, dataset)
    assert any("q3" in issue for issue in issues)
    assert not any("q1" in issue for issue in issues)


def test_queries_template_lists_real_ids(graph_scene):
    dataset = load_scene(graph_scene)
    template = queries_template(dataset)
    assert "wp_01" in template
    parsed = yaml.safe_load(template)
    assert parsed["queries"][0]["valid_node_ids"] == []


def test_load_queries_missing_file():
    with pytest.raises(FileNotFoundError):
        load_queries("/nonexistent/queries.yaml")


def test_experiment_query_defaults():
    q = ExperimentQuery(query_id="q", text="t")
    assert q.language == "es" and not q.is_negative
