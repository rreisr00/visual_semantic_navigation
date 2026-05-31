"""Unit tests for the semantic_map_manager package.

These tests run without a live ROS 2 environment – they exercise the pure-Python
logic of the nodes (cosine similarity, knowledge graph adapter, etc.).
"""

import sys
import types


# ---------------------------------------------------------------------------
# Stub out heavy ROS 2 / PyTorch imports so tests run in plain Python
# ---------------------------------------------------------------------------

def _make_stub_module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


def _stub_rclpy() -> None:
    rclpy = _make_stub_module("rclpy")
    rclpy.init = lambda *a, **kw: None  # type: ignore[attr-defined]
    rclpy.spin = lambda *a, **kw: None  # type: ignore[attr-defined]
    rclpy.spin_until_future_complete = lambda *a, **kw: None  # type: ignore[attr-defined]
    rclpy.try_shutdown = lambda *a, **kw: None  # type: ignore[attr-defined]

    node_mod = _make_stub_module("rclpy.node")
    node_mod.Node = object  # type: ignore[attr-defined]

    duration_mod = _make_stub_module("rclpy.duration")
    duration_mod.Duration = lambda **kw: None  # type: ignore[attr-defined]

    time_mod = _make_stub_module("rclpy.time")
    time_mod.Time = lambda: None  # type: ignore[attr-defined]


def _stub_ros_msgs() -> None:
    for pkg in [
        "std_msgs", "std_msgs.msg",
        "sensor_msgs", "sensor_msgs.msg",
        "geometry_msgs", "geometry_msgs.msg",
    ]:
        _make_stub_module(pkg)

    import std_msgs.msg as smsg
    smsg.Empty = object  # type: ignore[attr-defined]
    smsg.String = object  # type: ignore[attr-defined]

    import sensor_msgs.msg as semsg
    semsg.Image = object  # type: ignore[attr-defined]

    import geometry_msgs.msg as gmsg
    gmsg.PoseStamped = object  # type: ignore[attr-defined]
    gmsg.Quaternion = object  # type: ignore[attr-defined]


def _stub_tf2() -> None:
    for pkg in ["tf2_ros"]:
        mod = _make_stub_module(pkg)
        mod.Buffer = object  # type: ignore[attr-defined]
        mod.TransformListener = lambda *a, **kw: None  # type: ignore[attr-defined]
        mod.LookupException = Exception  # type: ignore[attr-defined]
        mod.ConnectivityException = Exception  # type: ignore[attr-defined]
        mod.ExtrapolationException = Exception  # type: ignore[attr-defined]


def _stub_nav2() -> None:
    for pkg in ["nav2_simple_commander", "nav2_simple_commander.robot_navigator"]:
        mod = _make_stub_module(pkg)
    import nav2_simple_commander.robot_navigator as nav
    nav.BasicNavigator = object  # type: ignore[attr-defined]

    class _TaskResult:
        SUCCEEDED = "SUCCEEDED"
        CANCELED = "CANCELED"
        FAILED = "FAILED"

    nav.TaskResult = _TaskResult  # type: ignore[attr-defined]


def _stub_interfaces() -> None:
    for pkg in [
        "semantic_interfaces",
        "semantic_interfaces.srv",
    ]:
        _make_stub_module(pkg)

    import semantic_interfaces.srv as srv
    srv.GetEmbedding = object  # type: ignore[attr-defined]


def _stub_cv_bridge() -> None:
    _make_stub_module("cv_bridge")
    import cv_bridge as cvb
    cvb.CvBridge = object  # type: ignore[attr-defined]


def _stub_knowledge_graph() -> None:
    """Stub the knowledge_graph package so semantic_map_manager can be imported."""
    _make_stub_module("knowledge_graph")

    import knowledge_graph as kg

    class _StubNode:
        def __init__(self, name: str, type_: str):
            self._name = name
            self._type = type_
            self.properties = types.SimpleNamespace(_properties={})

        def get_name(self):
            return self._name

        def get_type(self):
            return self._type

        def set_property(self, key: str, value):
            self.properties._properties[key] = value

    class _StubKnowledgeGraph:
        _instance = None

        @staticmethod
        def get_instance():
            if _StubKnowledgeGraph._instance is None:
                _StubKnowledgeGraph._instance = _StubKnowledgeGraph()
            return _StubKnowledgeGraph._instance

        def __init__(self):
            self._nodes = {}

        def has_node(self, name: str):
            return name in self._nodes

        def get_node(self, name: str):
            return self._nodes[name]

        def create_node(self, name: str, type_: str):
            node = _StubNode(name, type_)
            self._nodes[name] = node
            return node

        def update_node(self, node):
            self._nodes[node.get_name()] = node

        def get_nodes(self):
            return list(self._nodes.values())

    kg.KnowledgeGraph = _StubKnowledgeGraph  # type: ignore[attr-defined]


_stub_rclpy()
_stub_ros_msgs()
_stub_tf2()
_stub_nav2()
_stub_interfaces()
_stub_cv_bridge()
_stub_knowledge_graph()

# ---------------------------------------------------------------------------
# Now we can import the modules under test
# ---------------------------------------------------------------------------

import numpy as np  # noqa: E402

# Import cosine similarity helper from utils directly
import importlib.util
import pathlib  # noqa: E402

_repo = pathlib.Path(__file__).parent.parent
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

_utils_path = _repo / "semantic_map_manager" / "utils.py"
_utils_spec = importlib.util.spec_from_file_location("utils", _utils_path)
_utils_mod = importlib.util.module_from_spec(_utils_spec)  # type: ignore[arg-type]
_utils_spec.loader.exec_module(_utils_mod)  # type: ignore[union-attr]

_cosine_similarity = _utils_mod.cosine_similarity  # type: ignore[attr-defined]

# Import local KnowledgeGraphClient adapter
_kg_path = _repo / "semantic_map_manager" / "knowledge_graph_client.py"
_kg_spec = importlib.util.spec_from_file_location("knowledge_graph_client", _kg_path)
_kg_mod = importlib.util.module_from_spec(_kg_spec)  # type: ignore[arg-type]
_kg_spec.loader.exec_module(_kg_mod)  # type: ignore[union-attr]

KnowledgeGraphClient = _kg_mod.KnowledgeGraphClient  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert abs(_cosine_similarity(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert abs(_cosine_similarity(a, b)) < 1e-6

    def test_opposite_vectors(self):
        v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert _cosine_similarity(v, -v) < -0.99

    def test_zero_vector_does_not_crash(self):
        """Cosine similarity with a zero vector should return a finite number."""
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.zeros(2, dtype=np.float32)
        result = _cosine_similarity(a, b)
        assert np.isfinite(result)

    def test_returns_float(self):
        a = np.array([0.5, 0.5], dtype=np.float32)
        b = np.array([0.5, 0.5], dtype=np.float32)
        assert isinstance(_cosine_similarity(a, b), float)


class TestKnowledgeGraphAdapter:
    def _fresh(self) -> "KnowledgeGraphClient":
        client = KnowledgeGraphClient()
        if hasattr(client, "_graph") and hasattr(client._graph, "_nodes"):
            client._graph._nodes = {}
        return client

    def test_add_and_get_node(self):
        store = self._fresh()
        store.add_node("n1", "waypoint", {"id": "n1", "pose_x": "1.0"})
        nodes = store.get_nodes()
        assert len(nodes) == 1
        assert nodes[0]["node_id"] == "n1"
        assert nodes[0]["class_id"] == "waypoint"
        assert nodes[0]["attributes"]["pose_x"] == "1.0"

    def test_filter_by_class_id(self):
        store = self._fresh()
        store.add_node("n1", "waypoint", {"id": "n1"})
        store.add_node("n2", "other", {"id": "n2"})
        waypoints = store.get_nodes(class_id="waypoint")
        assert len(waypoints) == 1
        assert waypoints[0]["node_id"] == "n1"

    def test_update_existing_node(self):
        store = self._fresh()
        store.add_node("n1", "waypoint", {"id": "n1", "pose_x": "1.0"})
        store.add_node("n1", "waypoint", {"id": "n1", "pose_x": "2.0"})
        nodes = store.get_nodes()
        assert len(nodes) == 1
        assert nodes[0]["attributes"]["pose_x"] == "2.0"

    def test_empty_store_returns_empty_list(self):
        store = self._fresh()
        assert store.get_nodes() == []

    def test_multiple_nodes(self):
        store = self._fresh()
        for i in range(5):
            store.add_node(f"n{i}", "waypoint", {"id": f"n{i}"})
        assert len(store.get_nodes()) == 5
