from __future__ import annotations

from typing import Any

from semantic_map_manager.utils import create_graph_manager_instance


class KnowledgeGraphClient:
    """Thin adapter around knowledge_graph for semantic waypoint storage."""

    def __init__(self, _node: Any = None) -> None:
        self._graph = create_graph_manager_instance()

    def add_node(self, node_id: str, class_id: str, attributes: dict[str, str]) -> None:
        if self._graph.has_node(node_id):
            node = self._graph.get_node(node_id)
        else:
            node = self._graph.create_node(node_id, class_id)

        for key, value in attributes.items():
            node.set_property(key, str(value))

        self._graph.update_node(node)

    def get_nodes(self, class_id: str | None = None) -> list[dict[str, Any]]:
        nodes = []
        for node in self._graph.get_nodes():
            node_type = node.get_type()
            if class_id is not None and node_type != class_id:
                continue

            attributes = {}
            if hasattr(node, "properties") and hasattr(node.properties, "_properties"):
                attributes = {
                    key: str(value)
                    for key, value in node.properties._properties.items()
                }

            nodes.append(
                {
                    "node_id": node.get_name(),
                    "class_id": node_type,
                    "attributes": attributes,
                }
            )
        return nodes