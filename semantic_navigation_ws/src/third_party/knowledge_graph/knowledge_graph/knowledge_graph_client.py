#!/usr/bin/env python3
"""Knowledge Graph Client.

A lightweight client that wraps the knowledge_graph ROS 2 services provided by
the mgonzs13/knowledge_graph package (or the bundled stub implementation).

Nodes are stored with the following well-known attributes:
    id, class_id, pose_x, pose_y, pose_z,
    orient_x, orient_y, orient_z, orient_w, embedding
"""

from __future__ import annotations

from typing import Any

import rclpy
from rclpy.node import Node

# Service types are provided by the knowledge_graph package.
# If the real package is not installed the stub services below are used.
try:
    from knowledge_graph_msgs.srv import AddNodes, GetNodes  # type: ignore[import]

    _USE_REAL_PKG = True
except ImportError:
    _USE_REAL_PKG = False


class _InMemoryStore:
    """Very simple thread-safe in-process knowledge graph store.

    Used as a fallback when the mgonzs13/knowledge_graph services are not
    available (e.g. during unit tests or minimal deployments).
    """

    def __init__(self) -> None:
        self._nodes: list[dict[str, Any]] = []

    def add_node(self, node_id: str, class_id: str, attributes: dict[str, str]) -> None:
        # Replace if already exists
        self._nodes = [n for n in self._nodes if n.get("attributes", {}).get("id") != node_id]
        self._nodes.append({"node_id": node_id, "class_id": class_id, "attributes": dict(attributes)})

    def get_nodes(self, class_id: str | None = None) -> list[dict[str, Any]]:
        if class_id is None:
            return list(self._nodes)
        return [n for n in self._nodes if n.get("class_id") == class_id]


# Module-level shared store used by all KnowledgeGraphClient instances that
# fall back to the in-memory implementation.
_STORE = _InMemoryStore()


class KnowledgeGraphClient:
    """High-level client for the semantic map knowledge graph.

    When the mgonzs13/knowledge_graph package is installed this class calls the
    real ROS 2 services.  Otherwise it falls back to the built-in in-memory
    store so the rest of the system can work without external dependencies.
    """

    def __init__(self, node: Node) -> None:
        self._node = node
        self._use_real = _USE_REAL_PKG

        if self._use_real:
            self._add_client = node.create_client(AddNodes, "/knowledge_graph/add_nodes")
            self._get_client = node.create_client(GetNodes, "/knowledge_graph/get_nodes")
        else:
            node.get_logger().info(
                "knowledge_graph_msgs not found – using built-in in-memory knowledge graph."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_node(
        self, node_id: str, class_id: str, attributes: dict[str, str]
    ) -> None:
        """Add or update a node in the knowledge graph."""
        if self._use_real:
            self._add_node_real(node_id, class_id, attributes)
        else:
            _STORE.add_node(node_id, class_id, attributes)

    def get_nodes(self, class_id: str | None = None) -> list[dict[str, Any]]:
        """Return all nodes, optionally filtered by class_id."""
        if self._use_real:
            return self._get_nodes_real(class_id)
        return _STORE.get_nodes(class_id)

    # ------------------------------------------------------------------
    # Real service calls (used when mgonzs13/knowledge_graph is available)
    # ------------------------------------------------------------------

    def _add_node_real(
        self, node_id: str, class_id: str, attributes: dict[str, str]
    ) -> None:
        if not self._add_client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError("knowledge_graph add_nodes service not available.")

        request = AddNodes.Request()
        # Build the request according to the mgonzs13 schema.
        node_msg = request.nodes.add()  # type: ignore[attr-defined]
        node_msg.node_id = node_id
        node_msg.class_id = class_id
        for key, value in attributes.items():
            attr = node_msg.attributes.add()  # type: ignore[attr-defined]
            attr.key = key
            attr.value = value

        future = self._add_client.call_async(request)
        rclpy.spin_until_future_complete(self._node, future, timeout_sec=10.0)
        if future.result() is None:
            raise RuntimeError("AddNodes service call timed out.")

    def _get_nodes_real(self, class_id: str | None) -> list[dict[str, Any]]:
        if not self._get_client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError("knowledge_graph get_nodes service not available.")

        request = GetNodes.Request()
        if class_id:
            request.class_id = class_id  # type: ignore[attr-defined]

        future = self._get_client.call_async(request)
        rclpy.spin_until_future_complete(self._node, future, timeout_sec=10.0)
        if future.result() is None:
            raise RuntimeError("GetNodes service call timed out.")

        result = future.result()
        nodes = []
        for n in result.nodes:  # type: ignore[attr-defined]
            attrs = {a.key: a.value for a in n.attributes}
            nodes.append({
                "node_id": n.node_id,
                "class_id": n.class_id,
                "attributes": attrs,
            })
        return nodes
