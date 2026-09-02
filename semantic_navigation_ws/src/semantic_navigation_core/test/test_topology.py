from semantic_navigation_core.topology import NodeCreationPolicy, nearest_node
from semantic_navigation_core.types import SemanticNode


def test_node_creation_requires_time_and_motion():
    policy = NodeCreationPolicy(minimum_translation_m=1.5, minimum_time_s=2.0)
    assert not policy.should_create((0.0, 0.0), 0.0, 1.0, (2.0, 0.0), 0.0, 2.0)
    assert policy.should_create((0.0, 0.0), 0.0, 1.0, (2.0, 0.0), 0.0, 3.0)


def test_nearest_node_is_scene_scoped():
    nodes = [
        SemanticNode("a", position=(0.0, 0.0, 0.0), scene_id="one"),
        SemanticNode("b", position=(0.1, 0.0, 0.0), scene_id="two"),
    ]
    node, distance = nearest_node(nodes, (0.2, 0.0), "one", 0.5)
    assert node.node_id == "a"
    assert distance == 0.2

