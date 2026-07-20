import numpy as np

from semantic_evaluation.core.graph_analysis import analyze_graph, safe_analyze_graph
from semantic_navigation_core.graph_store import save_semantic_graph
from semantic_navigation_core.types import Observation, SemanticNode


def test_empty_graph_is_analyzed_without_crashing(tmp_path):
    path = tmp_path / "empty.db"
    save_semantic_graph(str(path), [])
    result, issues = analyze_graph("empty", path)
    assert result.waypoint_count == 0
    assert result.connected_components == 0


def test_inconsistent_embeddings_and_missing_rooms_are_reported(tmp_path):
    path = tmp_path / "graph.db"
    nodes = [
        SemanticNode("a", observations=[Observation("a:v", embedding=np.ones(2))]),
        SemanticNode("b", observations=[Observation("b:v", embedding=np.ones(3))]),
    ]
    save_semantic_graph(str(path), nodes)
    result, issues = analyze_graph("scene", path)
    assert result.inconsistent_embedding_dimensions
    assert result.waypoints_without_room == 2
    assert any(issue["issue"] == "embedding_dimensions" for issue in issues)


def test_missing_graph_returns_structured_issue(tmp_path):
    result, issues = safe_analyze_graph("scene", tmp_path / "missing.db")
    assert result.scene_id == "scene"
    assert issues[0]["issue"] == "missing_graph"
