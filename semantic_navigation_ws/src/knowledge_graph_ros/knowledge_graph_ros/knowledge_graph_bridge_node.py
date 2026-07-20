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
  /add_room  (semantic_interfaces/srv/AddRoom)
    Defines a rectangular room zone: creates a type="room" node whose
    rectangle lives in min_x/min_y/max_x/max_y properties, then sweeps
    every existing waypoint — inside → CONTAINS room->waypoint edge,
    outside-but-linked → edge removed (redefinition is self-correcting).
    New waypoints are classified on /store_waypoint.

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
import time
from typing import List, Optional

import rclpy
import rclpy.executors
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn, State

from geometry_msgs.msg import PoseStamped

from knowledge_graph import KnowledgeGraph
from knowledge_graph.graph import Node, Edge
from knowledge_graph_msgs.msg import Node as NodeMsg, Edge as EdgeMsg
from knowledge_graph_msgs.msg import Content, Property

from semantic_interfaces.msg import WaypointInfo, GraphEdge
from semantic_interfaces.srv import (
    AddRoom,
    GetGraphSnapshot,
    GetWaypoints,
    StoreWaypoint,
)

from semantic_navigation_core.rooms import Room, next_instance_name


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
        self._get_waypoints_srv = None
        self._snapshot_srv = None
        self._add_room_srv = None

        # Serialise all graph/SQLite access through one callback group so the
        # store and query services never run concurrently against the graph.
        self._graph_cbg = MutuallyExclusiveCallbackGroup()

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
            StoreWaypoint, "store_waypoint", self._handle_store_waypoint,
            callback_group=self._graph_cbg,
        )
        self._get_waypoints_srv = self.create_service(
            GetWaypoints, "get_waypoints", self._handle_get_waypoints,
            callback_group=self._graph_cbg,
        )
        self._snapshot_srv = self.create_service(
            GetGraphSnapshot, "get_graph_snapshot", self._handle_get_graph_snapshot,
            callback_group=self._graph_cbg,
        )
        self._add_room_srv = self.create_service(
            AddRoom, "add_room", self._handle_add_room,
            callback_group=self._graph_cbg,
        )
        self.get_logger().info(
            "Activated – /store_waypoint, /get_waypoints, /get_graph_snapshot "
            "and /add_room services ready."
        )
        return super().on_activate(state)

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        if self._store_srv is not None:
            self.destroy_service(self._store_srv)
            self._store_srv = None
        if self._get_waypoints_srv is not None:
            self.destroy_service(self._get_waypoints_srv)
            self._get_waypoints_srv = None
        if self._snapshot_srv is not None:
            self.destroy_service(self._snapshot_srv)
            self._snapshot_srv = None
        if self._add_room_srv is not None:
            self.destroy_service(self._add_room_srv)
            self._add_room_srv = None
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
        node_id = request.node_id.strip()
        try:
            if not node_id:
                # No label given: name the waypoint after the room containing
                # its pose ("<room>_<NN>"); timestamped fallback outside rooms.
                node_id = self._auto_name_waypoint(
                    float(request.pose.pose.position.x),
                    float(request.pose.pose.position.y),
                )
            self._upsert_waypoint_node(node_id, request)
            if request.detected_objects:
                self._upsert_object_nodes_and_edges(
                    node_id, list(request.detected_objects)
                )
            self._assign_rooms_for_waypoint(
                node_id,
                float(request.pose.pose.position.x),
                float(request.pose.pose.position.y),
            )
            response.success = True
            response.message = "OK"
            response.node_id = node_id
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"StoreWaypoint failed for '{node_id}': {exc}")
            response.success = False
            response.message = str(exc)
        return response

    def _auto_name_waypoint(self, x: float, y: float) -> str:
        """Room-aware id: "<room>_<NN>" if (x, y) lies inside a room."""
        for node in self._graph.get_nodes():
            if node.get_type() != "room":
                continue
            room = self._room_from_node(node)
            if room.contains(x, y):
                waypoint_names = [
                    n.get_name()
                    for n in self._graph.get_nodes()
                    if n.get_type() == "waypoint"
                ]
                return next_instance_name(room.room_id, waypoint_names)
        return f"waypoint_{time.time_ns()}"

    def _upsert_waypoint_node(
        self, node_id: str, request: StoreWaypoint.Request
    ) -> None:
        pose = request.pose.pose
        if self._graph.has_node(node_id):
            node = self._graph.get_node(node_id)
        else:
            node = self._graph.create_node(node_id, "waypoint")

        node.set_property("pose_x",    float(pose.position.x))
        node.set_property("pose_y",    float(pose.position.y))
        node.set_property("pose_z",    float(pose.position.z))
        node.set_property("orient_x",  float(pose.orientation.x))
        node.set_property("orient_y",  float(pose.orientation.y))
        node.set_property("orient_z",  float(pose.orientation.z))
        node.set_property("orient_w",  float(pose.orientation.w))
        # Store the embedding as a JSON STRING, never as a vector property:
        # the vendored knowledge_graph rejects ROS msg vector fields on the
        # receiving side (array.array is not list), so a VDOUBLE property
        # crashes every other KnowledgeGraph instance (rqt viewer, terminal)
        # on the graph_update topic and the bridge itself on DB reload.
        emb_json = json.dumps([float(v) for v in request.visual_embedding])
        if node.has_property("visual_embedding") and not isinstance(
            node.get_property("visual_embedding"), str
        ):
            # Legacy node with a list-typed property: the library forbids
            # changing a property's type, so rebuild the node (edges are
            # re-created right after by the object/room assignment steps).
            self._graph.remove_node(node)
            node = self._graph.create_node(node_id, "waypoint")
            for k, v in (
                ("pose_x", pose.position.x), ("pose_y", pose.position.y),
                ("pose_z", pose.position.z),
                ("orient_x", pose.orientation.x), ("orient_y", pose.orientation.y),
                ("orient_z", pose.orientation.z), ("orient_w", pose.orientation.w),
            ):
                node.set_property(k, float(v))
        node.set_property("visual_embedding", emb_json)
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
    # Rooms: /add_room service + waypoint classification
    # Edge direction is load-bearing: room is SOURCE, waypoint is TARGET
    # ("CONTAINS" room→waypoint). Object edges are waypoint→object, so
    # _collect_object_labels (outgoing from the waypoint) stays clean and
    # a CONTAINS edge *targeting* a waypoint always comes from a room.
    # ------------------------------------------------------------------ #

    def _handle_add_room(
        self,
        request: AddRoom.Request,
        response: AddRoom.Response,
    ) -> AddRoom.Response:
        room_id = request.room_id.strip()
        if not room_id:
            response.success = False
            response.message = "room_id must not be empty"
            return response
        room = Room(
            room_id, request.min_x, request.min_y, request.max_x, request.max_y
        ).normalized()
        if room.min_x == room.max_x or room.min_y == room.max_y:
            response.success = False
            response.message = "degenerate rectangle (zero width or height)"
            return response

        try:
            # Node must exist (and be persisted) before any edge touches it:
            # create_edge raises on missing endpoints and the SQLite edges
            # table references nodes(name).
            if self._graph.has_node(room_id):
                node = self._graph.get_node(room_id)
            else:
                node = self._graph.create_node(room_id, "room")
            node.set_property("min_x", float(room.min_x))
            node.set_property("min_y", float(room.min_y))
            node.set_property("max_x", float(room.max_x))
            node.set_property("max_y", float(room.max_y))
            self._graph.update_node(node)

            assigned = self._sweep_room(room)
            response.success = True
            response.message = "OK"
            response.waypoints_assigned = assigned
            self.get_logger().info(
                f"Room '{room_id}' [{room.min_x:.2f},{room.min_y:.2f}]–"
                f"[{room.max_x:.2f},{room.max_y:.2f}] → "
                f"{assigned} waypoint(s) assigned."
            )
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"AddRoom failed for '{room_id}': {exc}")
            response.success = False
            response.message = str(exc)
        return response

    def _sweep_room(self, room: Room) -> int:
        """Reconcile CONTAINS room→waypoint edges against the rectangle."""
        assigned = 0
        for node in self._graph.get_nodes():
            if node.get_type() != "waypoint":
                continue
            wp_id = node.get_name()
            inside = room.contains(
                float(_prop(node, "pose_x", 0.0)), float(_prop(node, "pose_y", 0.0))
            )
            has_edge = self._graph.has_edge("CONTAINS", room.room_id, wp_id)
            if inside and not has_edge:
                edge = self._graph.create_edge("CONTAINS", room.room_id, wp_id)
                self._graph.update_edge(edge)
            elif not inside and has_edge:
                # Room was redefined with a new rectangle — drop stale links.
                self._graph.remove_edge(
                    self._graph.get_edge("CONTAINS", room.room_id, wp_id)
                )
            if inside:
                assigned += 1
        return assigned

    def _assign_rooms_for_waypoint(self, waypoint_id: str, x: float, y: float) -> None:
        """Link a (re)stored waypoint to every room whose rectangle holds it."""
        for node in self._graph.get_nodes():
            if node.get_type() != "room":
                continue
            room = self._room_from_node(node)
            has_edge = self._graph.has_edge("CONTAINS", room.room_id, waypoint_id)
            if room.contains(x, y):
                if not has_edge:
                    edge = self._graph.create_edge(
                        "CONTAINS", room.room_id, waypoint_id
                    )
                    self._graph.update_edge(edge)
            elif has_edge:
                # Re-captured waypoint moved out of this room.
                self._graph.remove_edge(
                    self._graph.get_edge("CONTAINS", room.room_id, waypoint_id)
                )

    @staticmethod
    def _room_from_node(node: Node) -> Room:
        return Room(
            room_id=node.get_name(),
            min_x=float(_prop(node, "min_x", 0.0)),
            min_y=float(_prop(node, "min_y", 0.0)),
            max_x=float(_prop(node, "max_x", 0.0)),
            max_y=float(_prop(node, "max_y", 0.0)),
        )

    # ------------------------------------------------------------------ #
    # /get_waypoints service callback
    # ------------------------------------------------------------------ #

    def _handle_get_waypoints(
        self,
        request: GetWaypoints.Request,
        response: GetWaypoints.Response,
    ) -> GetWaypoints.Response:
        class_filter = request.class_filter or "waypoint"
        try:
            for node in self._graph.get_nodes():
                if node.get_type() != class_filter:
                    continue
                response.waypoints.append(self._node_to_waypoint_info(node))
            response.success = True
            response.message = "OK"
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"GetWaypoints failed: {exc}")
            response.waypoints = []
            response.success = False
            response.message = str(exc)
        return response

    # ------------------------------------------------------------------ #
    # /get_graph_snapshot service callback
    # ------------------------------------------------------------------ #

    def _handle_get_graph_snapshot(
        self,
        request: GetGraphSnapshot.Request,
        response: GetGraphSnapshot.Response,
    ) -> GetGraphSnapshot.Response:
        """Read-only snapshot: counts + pose-bearing waypoints + connectivity."""
        try:
            nodes = self._graph.get_nodes()
            edges = self._graph.get_edges()
            response.total_nodes = len(nodes)
            response.total_edges = len(edges)
            for node in nodes:
                if node.get_type() == "waypoint":
                    response.waypoints.append(self._node_to_waypoint_info(node))
            for edge in edges:
                ge = GraphEdge()
                ge.source_node = edge.get_source_node()
                ge.target_node = edge.get_target_node()
                ge.type = edge.get_type()
                response.edges.append(ge)
            response.success = True
            response.message = "OK"
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"GetGraphSnapshot failed: {exc}")
            response.waypoints = []
            response.edges = []
            response.total_nodes = 0
            response.total_edges = 0
            response.success = False
            response.message = str(exc)
        return response

    def _node_to_waypoint_info(self, node: Node) -> WaypointInfo:
        info = WaypointInfo()
        info.node_id = node.get_name()

        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.pose.position.x = float(_prop(node, "pose_x", 0.0))
        pose.pose.position.y = float(_prop(node, "pose_y", 0.0))
        pose.pose.position.z = float(_prop(node, "pose_z", 0.0))
        pose.pose.orientation.x = float(_prop(node, "orient_x", 0.0))
        pose.pose.orientation.y = float(_prop(node, "orient_y", 0.0))
        pose.pose.orientation.z = float(_prop(node, "orient_z", 0.0))
        pose.pose.orientation.w = float(_prop(node, "orient_w", 1.0))
        info.pose = pose

        # visual_embedding is stored as a JSON string (legacy graphs may still
        # hold a float list in memory — accept both).
        embedding = _prop(node, "visual_embedding", "")
        if isinstance(embedding, str):
            embedding = json.loads(embedding) if embedding else []
        info.visual_embedding = [float(v) for v in embedding] if embedding else []

        # Reconstruct detected objects from CONTAINS edges → object nodes' labels.
        info.detected_objects = self._collect_object_labels(node.get_name())
        return info

    def _collect_object_labels(self, waypoint_id: str) -> list[str]:
        labels: list[str] = []
        for edge in self._graph.get_edges_from_node_by_type("CONTAINS", waypoint_id):
            obj_id = edge.get_target_node()
            if not self._graph.has_node(obj_id):
                continue
            label = _prop(self._graph.get_node(obj_id), "label", "")
            if label:
                labels.append(str(label))
        return labels

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
# Graph node helpers
# ------------------------------------------------------------------ #

def _prop(node: Node, key: str, default):
    """Read a node property, returning ``default`` when it is absent."""
    if node.has_property(key):
        return node.get_property(key)
    return default


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
            elif isinstance(first, (int, float)):
                # Migrate legacy numeric vectors (old DBs stored the embedding
                # as a float list) to JSON strings: the vendored knowledge_graph
                # rejects msg vector fields (array.array) when reconstructing
                # nodes, so VINT/VFLOAT/VDOUBLE properties would crash the
                # bridge here on load and every graph_update receiver later.
                p.value.type = Content.STRING
                p.value.string_value = json.dumps(value)
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
