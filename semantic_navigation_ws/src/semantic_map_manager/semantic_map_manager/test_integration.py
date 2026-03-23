#!/usr/bin/env python3

from __future__ import annotations

import rclpy
from rclpy.node import Node

from semantic_map_manager.utils import create_graph_manager_instance, resolve_graph_manager_class


def _check_siglip_dependencies(node: Node) -> None:
    torch_ok = False
    transformers_ok = False

    try:
        import torch  # noqa: F401

        torch_ok = True
    except ImportError:
        torch_ok = False

    try:
        import transformers  # noqa: F401

        transformers_ok = True
    except ImportError:
        transformers_ok = False

    node.get_logger().info(f"torch disponible: {torch_ok}")
    node.get_logger().info(f"transformers disponible: {transformers_ok}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Node("semantic_map_manager_integration_test")

    try:
        manager_cls = resolve_graph_manager_class()
        node.get_logger().info(f"Clase de grafo resuelta: {manager_cls.__module__}.{manager_cls.__name__}")

        graph = create_graph_manager_instance()
        node.get_logger().info(f"Instancia de grafo creada: {type(graph).__name__}")

        _check_siglip_dependencies(node)
    except Exception as exc:  # noqa: BLE001
        node.get_logger().error(f"Fallo de integración: {exc}")
        raise
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()