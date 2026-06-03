#!/usr/bin/env python3
"""ROS 2 Lifecycle Adapter for the knowledge_graph third-party library.

Lifecycle transitions
---------------------
  configure  – open SQLite, create schema, load existing data, patch graph
  activate   – create /store_waypoint service server
  deactivate – destroy service server
  cleanup    – close SQLite, restore original graph methods
  shutdown   – close SQLite (idempotent)

Service
-------
  /store_waypoint  (semantic_interfaces/srv/StoreWaypoint)
    Request : node_id, pose (PoseStamped), visual_embedding (float32[]),
              detected_objects (string[])
    Response: success (bool), message (string)

SQLite schema
-------------
  nodes(name TEXT PK, type TEXT, properties TEXT)
  edges(type TEXT, source_node TEXT, target_node TEXT, properties TEXT, PK triplet)

The schema and the JSON serialisation format are wire-compatible with the
upstream knowledge_graph_db package so existing databases can be reused.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from typing import List, Optional

import rclpy
import rclpy.executors
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn, State

from knowledge_graph import KnowledgeGraph
from knowledge_graph.graph import Node, Edge
from knowledge_graph_msgs.msg import Node as NodeMsg, Edge as EdgeMsg
from knowledge_graph_msgs.msg import Content, Property

from semantic_interfaces.srv import StoreWaypoint


class KnowledgeGraphBridgeNode(LifecycleNode):
    """Lifecycle adapter: /store_waypoint ROS service → KnowledgeGraph + SQLite."""

    def __init__(self) -> None:
        super().__init__("knowledge_graph_bridge")
        self.declare_parameter(
            "db_file_path",
            os.path.join(
                os.path.expanduser("~"), ".ros", "semantic_maps", "knowledge_graph.db"
            ),
        )

        self._graph: Optional[KnowledgeGraph] = None
        self._conn: Optional[sqlite3.Connection] = None
        self._db_lock = threading.Lock()
        self._store_srv = None

        # Saved originals so on_cleanup can fully restore the graph singleton.
        self._orig_update_node = None
        self._orig_remove_node = None
        self._orig_update_edge = None
        self._orig_remove_edge = None

    # ------------------------------------------------------------------ #
    # Lifecycle callbacks
    # ------------------------------------------------------------------ #

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        raw = self.get_parameter("db_file_path").get_parameter_value().string_value
        db_path = os.path.expanduser(os.path.expandvars(raw))

        try:
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
        except OSError as exc:
            self.get_logger().fatal(f"Cannot create DB directory for '{db_path}': {exc}")
            return TransitionCallbackReturn.FAILURE

        try:
            # check_same_thread=False: the KnowledgeGraph spin thread may call
            # the patched methods from a different thread than this node.
            self._conn = sqlite3.connect(
                db_path, check_same_thread=False, timeout=10.0
            )
            self._conn.execute("PRAGMA journal_mode=WAL;")
        except sqlite3.Error as exc:
            self.get_logger().fatal(f"Cannot open SQLite DB '{db_path}': {exc}")
            return TransitionCallbackReturn.FAILURE

        if not self._create_tables():
            return TransitionCallbackReturn.FAILURE

        try:
            self._graph = KnowledgeGraph.get_instance()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().fatal(f"KnowledgeGraph initialisation failed: {exc}")
            return TransitionCallbackReturn.FAILURE

        self._patch_graph()
        self._load_db()

        self.get_logger().info(f"Configured – DB at '{db_path}'.")
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self._store_srv = self.create_service(
            StoreWaypoint, "store_waypoint", self._handle_store_waypoint
        )
        self.get_logger().info("Activated – /store_waypoint service ready.")
        return super().on_activate(state)

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        if self._store_srv is not None:
            self.destroy_service(self._store_srv)
            self._store_srv = None
        return super().on_deactivate(state)

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        self._restore_graph()
        self._close_db()
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        self._close_db()
        return TransitionCallbackReturn.SUCCESS

    # ------------------------------------------------------------------ #
    # /store_waypoint service callback
    # ------------------------------------------------------------------ #

    def _handle_store_waypoint(
        self,
        request: StoreWaypoint.Request,
        response: StoreWaypoint.Response,
    ) -> StoreWaypoint.Response:
        try:
            self._upsert_waypoint_node(request)
            if request.detected_objects:
                self._upsert_object_nodes_and_edges(
                    request.node_id, list(request.detected_objects)
                )
            response.success = True
            response.message = "OK"
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"StoreWaypoint failed for '{request.node_id}': {exc}")
            response.success = False
            response.message = str(exc)
        return response

    def _upsert_waypoint_node(self, request: StoreWaypoint.Request) -> None:
        pose = request.pose.pose
        if self._graph.has_node(request.node_id):
            node = self._graph.get_node(request.node_id)
        else:
            node = self._graph.create_node(request.node_id, "waypoint")

        node.set_property("pose_x",    float(pose.position.x))
        node.set_property("pose_y",    float(pose.position.y))
        node.set_property("pose_z",    float(pose.position.z))
        node.set_property("orient_x",  float(pose.orientation.x))
        node.set_property("orient_y",  float(pose.orientation.y))
        node.set_property("orient_z",  float(pose.orientation.z))
        node.set_property("orient_w",  float(pose.orientation.w))
        # Store embedding natively as a float vector (VFLOAT) for direct retrieval.
        node.set_property("visual_embedding", list(request.visual_embedding))
        self._graph.update_node(node)

    def _upsert_object_nodes_and_edges(
        self, waypoint_id: str, labels: list[str]
    ) -> None:
        for label in labels:
            obj_id = f"{waypoint_id}_{label}"

            if self._graph.has_node(obj_id):
                obj_node = self._graph.get_node(obj_id)
            else:
                obj_node = self._graph.create_node(obj_id, "object")

            obj_node.set_property("label",            label)
            obj_node.set_property("source_waypoint",  waypoint_id)
            self._graph.update_node(obj_node)

            if not self._graph.has_edge("CONTAINS", waypoint_id, obj_id):
                edge = self._graph.create_edge("CONTAINS", waypoint_id, obj_id)
                self._graph.update_edge(edge)

    # ------------------------------------------------------------------ #
    # Graph method patching
    # Intercepts KnowledgeGraph mutation calls → mirrors them to SQLite.
    # Mirrors the monkey-patching strategy used by knowledge_graph_db.
    # ------------------------------------------------------------------ #

    def _patch_graph(self) -> None:
        self._orig_update_node = self._graph.update_node
        self._orig_remove_node = self._graph.remove_node
        self._orig_update_edge = self._graph.update_edge
        self._orig_remove_edge = self._graph.remove_edge

        self._graph.update_node = self._patched_update_node
        self._graph.remove_node = self._patched_remove_node
        self._graph.update_edge = self._patched_update_edge
        self._graph.remove_edge = self._patched_remove_edge

    def _restore_graph(self) -> None:
        if self._graph is not None and self._orig_update_node is not None:
            self._graph.update_node = self._orig_update_node
            self._graph.remove_node = self._orig_remove_node
            self._graph.update_edge = self._orig_update_edge
            self._graph.remove_edge = self._orig_remove_edge
            self._orig_update_node = None

    def _patched_update_node(self, node: Node) -> None:
        KnowledgeGraph.update_node(self._graph, node)
        try:
            self._persist_node(node.to_msg())
        except sqlite3.Error as exc:
            self.get_logger().error(f"SQLite write failed for node '{node.get_name()}': {exc}")

    def _patched_remove_node(self, node: Node) -> bool:
        edges_snapshot = self._graph.get_edges().copy()
        removed = KnowledgeGraph.remove_node(self._graph, node)
        if removed:
            node_name = node.get_name()
            try:
                with self._db_lock:
                    c = self._conn.cursor()
                    c.execute("DELETE FROM nodes WHERE name = ?;", (node_name,))
                    for e in edges_snapshot:
                        if (
                            e.get_source_node() == node_name
                            or e.get_target_node() == node_name
                        ):
                            msg = e.to_msg()
                            c.execute(
                                "DELETE FROM edges"
                                " WHERE type=? AND source_node=? AND target_node=?;",
                                (msg.type, msg.source_node, msg.target_node),
                            )
                    self._conn.commit()
            except sqlite3.Error as exc:
                self.get_logger().error(
                    f"SQLite delete failed for node '{node_name}': {exc}"
                )
        return removed

    def _patched_update_edge(self, edge: Edge) -> None:
        KnowledgeGraph.update_edge(self._graph, edge)
        try:
            self._persist_edge(edge.to_msg())
        except sqlite3.Error as exc:
            self.get_logger().error(
                f"SQLite write failed for edge '{edge.to_string()}': {exc}"
            )

    def _patched_remove_edge(self, edge: Edge) -> bool:
        removed = KnowledgeGraph.remove_edge(self._graph, edge)
        if removed:
            msg = edge.to_msg()
            try:
                with self._db_lock:
                    self._conn.execute(
                        "DELETE FROM edges"
                        " WHERE type=? AND source_node=? AND target_node=?;",
                        (msg.type, msg.source_node, msg.target_node),
                    )
                    self._conn.commit()
            except sqlite3.Error as exc:
                self.get_logger().error(
                    f"SQLite delete failed for edge '{edge.to_string()}': {exc}"
                )
        return removed

    # ------------------------------------------------------------------ #
    # SQLite helpers
    # ------------------------------------------------------------------ #

    def _create_tables(self) -> bool:
        try:
            with self._db_lock:
                c = self._conn.cursor()
                c.execute("""
                    CREATE TABLE IF NOT EXISTS nodes (
                        name       TEXT PRIMARY KEY,
                        type       TEXT NOT NULL,
                        properties TEXT
                    );
                """)
                c.execute("""
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
                self._conn.commit()
            return True
        except sqlite3.Error as exc:
            self.get_logger().fatal(f"Failed to create SQLite schema: {exc}")
            return False

    def _load_db(self) -> None:
        with self._db_lock:
            c = self._conn.cursor()

            c.execute("SELECT name, type, properties FROM nodes;")
            for name, type_, props_str in c.fetchall():
                node_msg = NodeMsg()
                node_msg.name = name
                node_msg.type = type_
                node_msg.properties = _deserialize_properties(props_str)
                KnowledgeGraph.update_node(self._graph, Node(msg=node_msg))

            c.execute("SELECT type, source_node, target_node, properties FROM edges;")
            for type_, src, tgt, props_str in c.fetchall():
                edge_msg = EdgeMsg()
                edge_msg.type = type_
                edge_msg.source_node = src
                edge_msg.target_node = tgt
                edge_msg.properties = _deserialize_properties(props_str)
                KnowledgeGraph.update_edge(self._graph, Edge(msg=edge_msg))

        self.get_logger().info("Loaded graph from SQLite.")

    def _persist_node(self, msg: NodeMsg) -> None:
        props = _serialize_properties(msg.properties)
        with self._db_lock:
            c = self._conn.cursor()
            c.execute("SELECT count(*) FROM nodes WHERE name = ?;", (msg.name,))
            if c.fetchone()[0]:
                c.execute(
                    "UPDATE nodes SET type=?, properties=? WHERE name=?;",
                    (msg.type, props, msg.name),
                )
            else:
                c.execute(
                    "INSERT INTO nodes(name, type, properties) VALUES(?, ?, ?);",
                    (msg.name, msg.type, props),
                )
            self._conn.commit()

    def _persist_edge(self, msg: EdgeMsg) -> None:
        props = _serialize_properties(msg.properties)
        with self._db_lock:
            c = self._conn.cursor()
            c.execute(
                "SELECT count(*) FROM edges"
                " WHERE type=? AND source_node=? AND target_node=?;",
                (msg.type, msg.source_node, msg.target_node),
            )
            if c.fetchone()[0]:
                c.execute(
                    "UPDATE edges SET properties=?"
                    " WHERE type=? AND source_node=? AND target_node=?;",
                    (props, msg.type, msg.source_node, msg.target_node),
                )
            else:
                c.execute(
                    "INSERT INTO edges(type, source_node, target_node, properties)"
                    " VALUES(?, ?, ?, ?);",
                    (msg.type, msg.source_node, msg.target_node, props),
                )
            self._conn.commit()

    def _close_db(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass
            finally:
                self._conn = None


# ------------------------------------------------------------------ #
# Property (de)serialisation – wire-compatible with knowledge_graph_db
# ------------------------------------------------------------------ #

def _serialize_properties(properties: List[Property]) -> str:
    d: dict = {}
    for p in properties:
        c = p.value
        if   c.type == Content.BOOL:    d[p.key] = bool(c.bool_value)
        elif c.type == Content.INT:     d[p.key] = int(c.int_value)
        elif c.type == Content.FLOAT:   d[p.key] = float(c.float_value)
        elif c.type == Content.DOUBLE:  d[p.key] = float(c.double_value)
        elif c.type == Content.STRING:  d[p.key] = str(c.string_value)
        elif c.type == Content.VBOOL:   d[p.key] = list(c.bool_vector)
        elif c.type == Content.VINT:    d[p.key] = list(c.int_vector)
        elif c.type == Content.VFLOAT:  d[p.key] = list(c.float_vector)
        elif c.type == Content.VDOUBLE: d[p.key] = list(c.double_vector)
        elif c.type == Content.VSTRING: d[p.key] = list(c.string_vector)
    return json.dumps(d)


def _deserialize_properties(props_str: str) -> List[Property]:
    d = json.loads(props_str) if props_str else {}
    result: List[Property] = []
    for key, value in d.items():
        p = Property()
        p.key = key
        if isinstance(value, bool):
            p.value.type = Content.BOOL
            p.value.bool_value = value
        elif isinstance(value, int):
            p.value.type = Content.INT
            p.value.int_value = value
        elif isinstance(value, float):
            p.value.type = Content.FLOAT
            p.value.float_value = value
        elif isinstance(value, str):
            p.value.type = Content.STRING
            p.value.string_value = value
        elif isinstance(value, list) and value:
            first = value[0]
            if isinstance(first, bool):
                p.value.type = Content.VBOOL
                p.value.bool_vector = value
            elif isinstance(first, int):
                p.value.type = Content.VINT
                p.value.int_vector = value
            elif isinstance(first, float):
                p.value.type = Content.VFLOAT
                p.value.float_vector = value
            elif isinstance(first, str):
                p.value.type = Content.VSTRING
                p.value.string_vector = value
        result.append(p)
    return result


# ------------------------------------------------------------------ #
# Entry point
# ------------------------------------------------------------------ #

def main(args=None) -> None:
    rclpy.init(args=args)
    node = KnowledgeGraphBridgeNode()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
