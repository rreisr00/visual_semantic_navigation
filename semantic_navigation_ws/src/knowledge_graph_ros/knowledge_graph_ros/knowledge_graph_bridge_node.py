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
    Defines a polygonal room zone (legacy rectangles remain supported), then
    sweeps
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
from math import hypot, sqrt
from typing import List, Optional

import numpy as np
import rclpy
import rclpy.executors
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn, State

from geometry_msgs.msg import PoseStamped

from knowledge_graph import KnowledgeGraph
from knowledge_graph.graph import Node, Edge
from knowledge_graph_msgs.msg import Node as NodeMsg, Edge as EdgeMsg
from knowledge_graph_msgs.msg import Content, Property

from semantic_interfaces.msg import (
    GraphEdge,
    ObjectDetection,
    ObservationInfo,
    SemanticRelation,
    WaypointInfo,
)
from semantic_interfaces.srv import (
    AddRoom,
    GetGraphSnapshot,
    GetWaypoints,
    StoreWaypoint,
)

from semantic_navigation_core.association import AssociationConfig, match_object
from semantic_navigation_core.contamination import analyze_room_evidence
from semantic_navigation_core.geometry import transform_point
from semantic_navigation_core.rooms import Room, next_waypoint_name, room_of_point
from semantic_navigation_core.types import ObjectObservation as CoreObjectObservation


SCHEMA_VERSION = 3
EDGE_CONNECTED_TO = "CONNECTED_TO"
EDGE_HAS_OBSERVATION = "HAS_OBSERVATION"


class KnowledgeGraphBridgeNode(LifecycleNode):
    """Lifecycle adapter: /store_waypoint ROS service → KnowledgeGraph + SQLite."""

    def __init__(self) -> None:
        super().__init__("knowledge_graph_bridge")
        self.declare_parameter(
            "db_file_path",
            os.path.join(
                os.path.expanduser("~"), ".ros", "semantic_maps", "{scene_id}", "graph.db"
            ),
        )
        self.declare_parameter("scene_id", "aws_small_house")
        self.declare_parameter("duplicate_distance_m", 0.5)
        self.declare_parameter("maximum_edge_distance_m", 4.0)
        self.declare_parameter("configuration_hash", "")
        self.declare_parameter("object_association_crop_similarity", 0.75)

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
        scene_id = self.get_parameter("scene_id").value
        db_path = os.path.expanduser(os.path.expandvars(raw)).format(scene_id=scene_id)

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
            self._conn.execute("PRAGMA foreign_keys=ON;")
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
        scene_id = request.scene_id.strip() or self.get_parameter("scene_id").value
        merged = False
        try:
            if not node_id:
                duplicate = self._nearest_waypoint(
                    float(request.pose.pose.position.x),
                    float(request.pose.pose.position.y),
                    scene_id,
                    float(self.get_parameter("duplicate_distance_m").value),
                )
                if duplicate is not None:
                    node_id = duplicate.get_name()
                    merged = True
                else:
                    node_id = self._auto_name_waypoint(
                        float(request.pose.pose.position.x),
                        float(request.pose.pose.position.y),
                        scene_id,
                    )
            is_new = not self._graph.has_node(node_id)
            self._upsert_waypoint_node(node_id, scene_id, request)
            observation_id = self._upsert_observation(node_id, scene_id, request)
            associations = self._upsert_object_nodes_and_edges(
                node_id, observation_id, request
            )
            self._upsert_relation_edges(
                node_id, observation_id, request, associations
            )
            self._assign_rooms_for_waypoint(
                node_id,
                float(request.pose.pose.position.x),
                float(request.pose.pose.position.y),
            )
            self._refresh_waypoint_embedding(node_id)
            if is_new:
                self._connect_topology(node_id, scene_id)
            response.success = True
            response.message = "OK"
            response.node_id = node_id
            response.merged_with_existing = merged
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"StoreWaypoint failed for '{node_id}': {exc}")
            response.success = False
            response.message = str(exc)
        return response

    def _auto_name_waypoint(self, x: float, y: float, scene_id: str) -> str:
        """Return the next compact graph-wide id (``W1``, ``W2``, …)."""
        del x, y, scene_id
        waypoint_names = [
            node.get_name()
            for node in self._graph.get_nodes()
            if node.get_type() == "waypoint"
        ]
        return next_waypoint_name(waypoint_names)

    def _upsert_waypoint_node(
        self, node_id: str, scene_id: str, request: StoreWaypoint.Request
    ) -> None:
        pose = request.pose.pose
        if self._graph.has_node(node_id):
            node = self._graph.get_node(node_id)
        else:
            node = self._graph.create_node(node_id, "waypoint")

        node.set_property("pose_x", float(pose.position.x))
        node.set_property("pose_y", float(pose.position.y))
        node.set_property("pose_z", float(pose.position.z))
        node.set_property("orient_x", float(pose.orientation.x))
        node.set_property("orient_y", float(pose.orientation.y))
        node.set_property("orient_z", float(pose.orientation.z))
        node.set_property("orient_w", float(pose.orientation.w))
        node.set_property("scene_id", scene_id)
        if not node.has_property("creation_timestamp"):
            node.set_property("creation_timestamp", float(time.time()))
        config_hash = request.configuration_hash.strip() or self.get_parameter(
            "configuration_hash"
        ).value
        node.set_property("configuration_hash", str(config_hash))
        goal = request.navigation_goal_pose.pose
        if request.navigation_goal_pose.header.frame_id:
            for key, value in (
                ("goal_x", goal.position.x), ("goal_y", goal.position.y),
                ("goal_z", goal.position.z), ("goal_qx", goal.orientation.x),
                ("goal_qy", goal.orientation.y), ("goal_qz", goal.orientation.z),
                ("goal_qw", goal.orientation.w),
            ):
                node.set_property(key, float(value))
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
        self,
        waypoint_id: str,
        observation_id: str,
        request: StoreWaypoint.Request,
    ) -> dict[str, str]:
        detections = list(request.observation.detections)
        if not detections:
            detections = []
            for label in request.detected_objects:
                detection = ObjectDetection()
                detection.class_name = label
                detection.confidence = 1.0
                detections.append(detection)
        existing = self._persistent_objects(waypoint_id)
        assigned: set[str] = set()
        associations: dict[str, str] = {}
        association_config = AssociationConfig(
            minimum_crop_similarity=float(
                self.get_parameter("object_association_crop_similarity").value
            )
        )
        for detection in detections:
            label = detection.class_name.strip()
            if not label:
                continue
            crop = [float(value) for value in detection.crop_embedding]
            map_position = _detection_map_position(
                detection,
                request.observation.depth_camera_pose
                if request.observation.depth_camera_pose.header.frame_id
                else request.observation.camera_pose,
            )
            object_room = (
                room_of_point(map_position[0], map_position[1], self._rooms())
                if map_position is not None else None
            )
            transient = CoreObjectObservation(
                label=label,
                embedding=np.asarray(crop, dtype=np.float32) if crop else None,
                position_3d=map_position,
                position_3d_frame="map" if map_position is not None else "",
            )
            match = match_object(
                transient, existing, association_config, excluded_ids=assigned
            )
            if match is None:
                obj_id = self._next_object_id(waypoint_id, label)
                association_confidence = 0.0
            else:
                obj_id = match.object_id
                association_confidence = match.score
            assigned.add(obj_id)
            if detection.object_id:
                associations[detection.object_id] = obj_id
            associations.setdefault(label, obj_id)

            if self._graph.has_node(obj_id):
                obj_node = self._graph.get_node(obj_id)
            else:
                obj_node = self._graph.create_node(obj_id, "object")

            obj_node.set_property("label", label)
            obj_node.set_property("source_waypoint", waypoint_id)
            obj_node.set_property("confidence", float(detection.confidence))
            obj_node.set_property(
                "association_confidence", float(association_confidence)
            )
            obj_node.set_property("last_seen", float(time.time()))
            obj_node.set_property(
                "bounding_box", json.dumps([float(v) for v in detection.bounding_box])
            )
            obj_node.set_property(
                "crop_embedding", json.dumps([float(v) for v in detection.crop_embedding])
            )
            obj_node.set_property(
                "position_2d", json.dumps([float(v) for v in detection.position_2d])
            )
            obj_node.set_property(
                "position_3d",
                json.dumps([float(v) for v in detection.position_3d])
                if detection.position_3d_valid else "[]",
            )
            obj_node.set_property("position_3d_frame", detection.position_3d_frame)
            obj_node.set_property(
                "position_3d_map",
                json.dumps(map_position) if map_position is not None else "[]",
            )
            obj_node.set_property("room_id", object_room or "")
            self._graph.update_node(obj_node)

            if not self._graph.has_edge("CONTAINS", waypoint_id, obj_id):
                edge = self._graph.create_edge("CONTAINS", waypoint_id, obj_id)
                self._graph.update_edge(edge)
            if not self._graph.has_edge("OBSERVED_IN", obj_id, observation_id):
                edge = self._graph.create_edge("OBSERVED_IN", obj_id, observation_id)
                self._graph.update_edge(edge)
            if match is None:
                existing.append(CoreObjectObservation(
                    label=label,
                    object_id=obj_id,
                    embedding=(
                        np.asarray(crop, dtype=np.float32) if crop else None
                    ),
                    position_3d=map_position,
                    position_3d_frame="map" if map_position is not None else "",
                ))
        return associations

    def _persistent_objects(self, waypoint_id: str) -> list[CoreObjectObservation]:
        objects = []
        for edge in self._graph.get_edges_from_node_by_type("CONTAINS", waypoint_id):
            object_id = edge.get_target_node()
            if not self._graph.has_node(object_id):
                continue
            node = self._graph.get_node(object_id)
            if node.get_type() != "object":
                continue
            raw_embedding = str(_prop(node, "crop_embedding", "[]"))
            try:
                embedding = np.asarray(json.loads(raw_embedding), dtype=np.float32)
            except (TypeError, ValueError, json.JSONDecodeError):
                embedding = np.asarray([], dtype=np.float32)
            raw_position = str(_prop(node, "position_3d_map", "[]"))
            try:
                position = json.loads(raw_position)
            except (TypeError, ValueError, json.JSONDecodeError):
                position = []
            objects.append(CoreObjectObservation(
                label=str(_prop(node, "label", "")),
                object_id=object_id,
                embedding=embedding if embedding.size else None,
                position_3d=(
                    tuple(float(value) for value in position)
                    if len(position) == 3 else None
                ),
                position_3d_frame="map" if len(position) == 3 else "",
            ))
        return objects

    def _next_object_id(self, waypoint_id: str, label: str) -> str:
        base = f"{waypoint_id}_{label}"
        if not self._graph.has_node(base):
            return base
        suffix = 2
        while self._graph.has_node(f"{base}_{suffix}"):
            suffix += 1
        return f"{base}_{suffix}"

    def _upsert_observation(
        self, waypoint_id: str, scene_id: str, request: StoreWaypoint.Request
    ) -> str:
        observation = request.observation
        observation_id = observation.observation_id.strip()
        if not observation_id:
            stamp = observation.timestamp
            observation_id = f"{scene_id}_{waypoint_id}_{stamp.sec}_{stamp.nanosec}"
        graph_id = f"observation::{observation_id}"
        node = (
            self._graph.get_node(graph_id)
            if self._graph.has_node(graph_id)
            else self._graph.create_node(graph_id, "observation")
        )
        embedding = list(observation.image_embedding or request.visual_embedding)
        camera_pose_msg = (
            observation.camera_pose
            if observation.camera_pose.header.frame_id else request.pose
        )
        camera_pose = camera_pose_msg.pose
        rooms = self._rooms()
        pose_for_depth = (
            observation.depth_camera_pose
            if observation.depth_camera_pose.header.frame_id
            else camera_pose_msg
        )
        map_positions = []
        for detection in observation.detections:
            map_positions.append(_detection_map_position(detection, pose_for_depth))
        evidence = analyze_room_evidence(
            (
                float(camera_pose.position.x),
                float(camera_pose.position.y),
                float(camera_pose.position.z),
            ),
            [
                (position, float(detection.confidence))
                for detection, position in zip(
                    observation.detections, map_positions
                )
            ],
            rooms,
        )
        detections = [
            _detection_to_dict(detection, position, object_room)
            for detection, position, object_room in zip(
                observation.detections, map_positions, evidence.object_rooms
            )
        ]
        payload = {
            "observation_id": observation_id,
            "node_id": waypoint_id,
            "scene_id": scene_id,
            "image_path": observation.image_path,
            "timestamp": [observation.timestamp.sec, observation.timestamp.nanosec],
            "camera_frame": observation.camera_frame,
            "camera_pose": _pose_to_list(camera_pose),
            "depth_camera_frame": observation.depth_camera_frame,
            "depth_camera_pose": _pose_to_list(
                observation.depth_camera_pose.pose
            ),
            "requested_yaw": float(observation.requested_yaw),
            "measured_yaw": float(observation.measured_yaw),
            "angular_error": float(observation.angular_error),
            "image_valid": bool(observation.image_valid),
            "depth_valid": bool(observation.depth_valid),
            "image_embedding": [float(v) for v in embedding],
            "camera_room": evidence.camera_room or "",
            "observation_room": evidence.observation_room or "",
            "purity": evidence.purity,
            "contamination_class": evidence.contamination_class,
            "transition_zone": evidence.transition_zone,
            "detections": detections,
            "relations": [_relation_to_dict(r) for r in observation.relations],
        }
        node.set_property("observation_id", observation_id)
        node.set_property("scene_id", scene_id)
        node.set_property("timestamp_sec", float(
            observation.timestamp.sec + observation.timestamp.nanosec * 1e-9
        ))
        node.set_property("payload", json.dumps(payload, sort_keys=True))
        self._graph.update_node(node)
        if not self._graph.has_edge(EDGE_HAS_OBSERVATION, waypoint_id, graph_id):
            edge = self._graph.create_edge(EDGE_HAS_OBSERVATION, waypoint_id, graph_id)
            self._graph.update_edge(edge)
        return graph_id

    def _refresh_waypoint_embedding(self, waypoint_id: str) -> None:
        embeddings: list[list[float]] = []
        weights: list[float] = []
        for edge in self._graph.get_edges_from_node_by_type(
            EDGE_HAS_OBSERVATION, waypoint_id
        ):
            observation = self._graph.get_node(edge.get_target_node())
            payload = json.loads(str(_prop(observation, "payload", "{}")))
            values = [float(v) for v in payload.get("image_embedding", [])]
            if values:
                embeddings.append(values)
                purity = payload.get("purity")
                weights.append(
                    1.0 if purity is None else max(0.0, min(1.0, float(purity)))
                )
        if not embeddings or len({len(values) for values in embeddings}) != 1:
            return
        if sum(weights) <= 0.0:
            weights = [1.0] * len(embeddings)
        total_weight = sum(weights)
        mean = [
            sum(values[i] * weight for values, weight in zip(embeddings, weights))
            / total_weight
            for i in range(len(embeddings[0]))
        ]
        norm = sqrt(sum(value * value for value in mean))
        if norm > 0.0:
            mean = [value / norm for value in mean]
        node = self._graph.get_node(waypoint_id)
        node.set_property("visual_embedding", json.dumps(mean))
        self._graph.update_node(node)

    def _upsert_relation_edges(
        self,
        waypoint_id: str,
        observation_id: str,
        request: StoreWaypoint.Request,
        associations: dict[str, str],
    ) -> None:
        objects = [
            edge.get_target_node()
            for edge in self._graph.get_edges_from_node_by_type("CONTAINS", waypoint_id)
            if (
                self._graph.has_node(edge.get_target_node())
                and self._graph.get_node(edge.get_target_node()).get_type() == "object"
            )
        ]

        def resolve(identifier: str) -> str:
            if identifier in associations:
                return associations[identifier]
            for object_id in objects:
                node = self._graph.get_node(object_id)
                if identifier in (object_id, str(_prop(node, "label", ""))):
                    return object_id
            return ""

        for relation in request.observation.relations:
            source = resolve(relation.subject_id)
            target = resolve(relation.object_id)
            if not source or not target or source == target or not relation.predicate:
                continue
            edge = (
                self._graph.get_edge(relation.predicate, source, target)
                if self._graph.has_edge(relation.predicate, source, target)
                else self._graph.create_edge(relation.predicate, source, target)
            )
            edge.set_property("confidence", float(relation.confidence))
            edge.set_property("reference_frame", str(relation.reference_frame))
            edge.set_property("source_observation_id", observation_id)
            edge.set_property("relation_type", str(
                relation.relation_type or "visual_2d_hypothesis"
            ))
            self._graph.update_edge(edge)

    def _nearest_waypoint(
        self, x: float, y: float, scene_id: str, maximum_distance: float,
        exclude: str = "",
    ) -> Optional[Node]:
        nearest = None
        best = float("inf")
        for candidate in self._graph.get_nodes():
            if candidate.get_type() != "waypoint" or candidate.get_name() == exclude:
                continue
            if str(_prop(candidate, "scene_id", "default")) != scene_id:
                continue
            distance = hypot(
                x - float(_prop(candidate, "pose_x", 0.0)),
                y - float(_prop(candidate, "pose_y", 0.0)),
            )
            if distance < best:
                nearest, best = candidate, distance
        return nearest if best <= maximum_distance else None

    def _connect_topology(self, node_id: str, scene_id: str) -> None:
        node = self._graph.get_node(node_id)
        x = float(_prop(node, "pose_x", 0.0))
        y = float(_prop(node, "pose_y", 0.0))
        neighbor = self._nearest_waypoint(
            x, y, scene_id,
            float(self.get_parameter("maximum_edge_distance_m").value),
            exclude=node_id,
        )
        if neighbor is None:
            return
        distance = hypot(
            x - float(_prop(neighbor, "pose_x", 0.0)),
            y - float(_prop(neighbor, "pose_y", 0.0)),
        )
        for source, target in ((node_id, neighbor.get_name()), (neighbor.get_name(), node_id)):
            if self._graph.has_edge(EDGE_CONNECTED_TO, source, target):
                continue
            edge = self._graph.create_edge(EDGE_CONNECTED_TO, source, target)
            edge.set_property("distance_m", float(distance))
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
        transition_width = (
            float(request.transition_width_m)
            if request.transition_width_m > 0.0 else 0.5
        )
        polygon = [(point.x, point.y) for point in request.polygon]
        room = (
            Room.from_polygon(room_id, polygon, transition_width)
            if len(polygon) >= 3
            else Room(
                room_id, request.min_x, request.min_y,
                request.max_x, request.max_y,
                transition_width_m=transition_width,
            ).normalized()
        )
        if (
            len(room.corners()) < 3
            or room.min_x == room.max_x
            or room.min_y == room.max_y
        ):
            response.success = False
            response.message = "degenerate room geometry"
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
            node.set_property("polygon", json.dumps(room.corners()))
            node.set_property("transition_width_m", room.transition_width_m)
            self._graph.update_node(node)

            assigned = self._sweep_room(room)
            response.success = True
            response.message = "OK"
            response.waypoints_assigned = assigned
            self.get_logger().info(
                f"Room '{room_id}' ({len(room.corners())} vertices, "
                f"transition={room.transition_width_m:.2f} m) → "
                f"{assigned} waypoint(s) assigned."
            )
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"AddRoom failed for '{room_id}': {exc}")
            response.success = False
            response.message = str(exc)
        return response

    def _sweep_room(self, room: Room) -> int:
        """Reconcile CONTAINS room→waypoint edges against its geometry."""
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
        """Link a (re)stored waypoint to every room whose geometry holds it."""
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
        raw_polygon = str(_prop(node, "polygon", "[]"))
        try:
            polygon = json.loads(raw_polygon)
        except (TypeError, ValueError, json.JSONDecodeError):
            polygon = []
        transition = float(_prop(node, "transition_width_m", 0.5))
        if len(polygon) >= 3:
            return Room.from_polygon(node.get_name(), polygon, transition)
        return Room(
            room_id=node.get_name(),
            min_x=float(_prop(node, "min_x", 0.0)),
            min_y=float(_prop(node, "min_y", 0.0)),
            max_x=float(_prop(node, "max_x", 0.0)),
            max_y=float(_prop(node, "max_y", 0.0)),
            transition_width_m=transition,
        )

    def _rooms(self) -> list[Room]:
        return [
            self._room_from_node(node)
            for node in self._graph.get_nodes()
            if node.get_type() == "room"
        ]

    # ------------------------------------------------------------------ #
    # /get_waypoints service callback
    # ------------------------------------------------------------------ #

    def _handle_get_waypoints(
        self,
        request: GetWaypoints.Request,
        response: GetWaypoints.Response,
    ) -> GetWaypoints.Response:
        class_filter = request.class_filter or "waypoint"
        scene_filter = request.scene_id.strip()
        try:
            for node in self._graph.get_nodes():
                if node.get_type() != class_filter:
                    continue
                if scene_filter and str(_prop(node, "scene_id", "default")) != scene_filter:
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
        info.scene_id = str(_prop(node, "scene_id", "default"))
        info.configuration_hash = str(_prop(node, "configuration_hash", ""))

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
        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.pose.position.x = float(_prop(node, "goal_x", pose.pose.position.x))
        goal.pose.position.y = float(_prop(node, "goal_y", pose.pose.position.y))
        goal.pose.position.z = float(_prop(node, "goal_z", pose.pose.position.z))
        goal.pose.orientation.x = float(_prop(node, "goal_qx", pose.pose.orientation.x))
        goal.pose.orientation.y = float(_prop(node, "goal_qy", pose.pose.orientation.y))
        goal.pose.orientation.z = float(_prop(node, "goal_qz", pose.pose.orientation.z))
        goal.pose.orientation.w = float(_prop(node, "goal_qw", pose.pose.orientation.w))
        info.navigation_goal_pose = goal
        created = float(_prop(node, "creation_timestamp", 0.0))
        info.creation_timestamp.sec = int(created)
        info.creation_timestamp.nanosec = int((created - int(created)) * 1e9)

        # visual_embedding is stored as a JSON string (legacy graphs may still
        # hold a float list in memory — accept both).
        embedding = _prop(node, "visual_embedding", "")
        if isinstance(embedding, str):
            embedding = json.loads(embedding) if embedding else []
        info.visual_embedding = [float(v) for v in embedding] if embedding else []

        # Reconstruct detected objects from CONTAINS edges → object nodes' labels.
        info.detected_objects = self._collect_object_labels(node.get_name())
        info.observations = self._collect_observations(node.get_name())
        for edge in self._graph.get_edges_to_node_by_type("CONTAINS", node.get_name()):
            source = self._graph.get_node(edge.get_source_node())
            if source.get_type() == "room":
                info.room_label = source.get_name()
                break
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

    def _collect_observations(self, waypoint_id: str) -> list[ObservationInfo]:
        observations: list[ObservationInfo] = []
        for edge in self._graph.get_edges_from_node_by_type(
            EDGE_HAS_OBSERVATION, waypoint_id
        ):
            node = self._graph.get_node(edge.get_target_node())
            payload = json.loads(str(_prop(node, "payload", "{}")))
            msg = ObservationInfo()
            msg.observation_id = str(payload.get("observation_id", ""))
            msg.node_id = waypoint_id
            msg.image_path = str(payload.get("image_path", ""))
            timestamp = payload.get("timestamp", [0, 0])
            msg.timestamp.sec = int(timestamp[0])
            msg.timestamp.nanosec = int(timestamp[1])
            msg.camera_frame = str(payload.get("camera_frame", ""))
            _list_to_pose(payload.get("camera_pose", []), msg.camera_pose)
            msg.depth_camera_frame = str(payload.get("depth_camera_frame", ""))
            _list_to_pose(
                payload.get("depth_camera_pose", []), msg.depth_camera_pose
            )
            msg.requested_yaw = float(payload.get("requested_yaw", 0.0))
            msg.measured_yaw = float(payload.get("measured_yaw", 0.0))
            msg.angular_error = float(payload.get("angular_error", 0.0))
            msg.image_valid = bool(payload.get("image_valid", False))
            msg.depth_valid = bool(payload.get("depth_valid", False))
            msg.camera_room = str(payload.get("camera_room", ""))
            msg.observation_room = str(payload.get("observation_room", ""))
            purity = payload.get("purity")
            msg.purity_valid = purity is not None
            msg.purity = float(purity) if purity is not None else 0.0
            msg.contamination_class = str(
                payload.get("contamination_class", "unknown")
            )
            msg.transition_zone = bool(payload.get("transition_zone", False))
            msg.image_embedding = [float(v) for v in payload.get("image_embedding", [])]
            for detection in payload.get("detections", []):
                item = ObjectDetection()
                item.object_id = str(detection.get("object_id", ""))
                item.class_name = str(detection.get("class_name", ""))
                item.confidence = float(detection.get("confidence", 0.0))
                item.bounding_box = [float(v) for v in detection.get("bounding_box", [0.0] * 4)]
                item.crop_embedding = [float(v) for v in detection.get("crop_embedding", [])]
                item.position_2d = [
                    float(v) for v in detection.get("position_2d", [0.0, 0.0])
                ]
                position_3d = detection.get("position_3d", [])
                if len(position_3d) == 3:
                    item.position_3d = [float(v) for v in position_3d]
                    item.position_3d_valid = True
                    item.position_3d_frame = str(
                        detection.get("position_3d_frame", "")
                    )
                map_position = detection.get("map_position", [])
                if len(map_position) == 3:
                    item.map_position = [float(v) for v in map_position]
                    item.map_position_valid = True
                item.room_id = str(detection.get("room_id", ""))
                msg.detections.append(item)
            for relation in payload.get("relations", []):
                item = SemanticRelation()
                item.subject_id = str(relation.get("subject_id", ""))
                item.predicate = str(relation.get("predicate", ""))
                item.object_id = str(relation.get("object_id", ""))
                item.confidence = float(relation.get("confidence", 0.0))
                item.reference_frame = str(relation.get("reference_frame", ""))
                item.source_observation_id = str(relation.get("source_observation_id", ""))
                item.relation_type = str(relation.get("relation_type", ""))
                stamp = relation.get("timestamp", [0, 0])
                item.timestamp.sec = int(stamp[0])
                item.timestamp.nanosec = int(stamp[1])
                msg.relations.append(item)
            observations.append(msg)
        observations.sort(key=lambda value: (value.timestamp.sec, value.timestamp.nanosec))
        return observations

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
                    # Foreign keys require incident edges to disappear first.
                    c.execute("DELETE FROM nodes WHERE name = ?;", (node_name,))
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
                c.execute("""
                    CREATE TABLE IF NOT EXISTS metadata (
                        key   TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    );
                """)
                c.execute(
                    "INSERT INTO metadata(key, value) VALUES('schema_version', ?)"
                    " ON CONFLICT(key) DO UPDATE SET value=excluded.value;",
                    (str(SCHEMA_VERSION),),
                )
                c.execute(
                    "INSERT INTO metadata(key, value) VALUES('scene_id', ?)"
                    " ON CONFLICT(key) DO UPDATE SET value=excluded.value;",
                    (str(self.get_parameter("scene_id").value),),
                )
                c.execute(
                    "INSERT INTO metadata(key, value) VALUES('configuration_hash', ?)"
                    " ON CONFLICT(key) DO UPDATE SET value=excluded.value;",
                    (str(self.get_parameter("configuration_hash").value),),
                )
                c.execute("CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(type);")
                c.execute(
                    "CREATE INDEX IF NOT EXISTS idx_edges_source_type"
                    " ON edges(source_node, type);"
                )
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


def _pose_to_list(pose) -> list[float]:
    return [
        float(pose.position.x), float(pose.position.y), float(pose.position.z),
        float(pose.orientation.x), float(pose.orientation.y),
        float(pose.orientation.z), float(pose.orientation.w),
    ]


def _list_to_pose(values, pose_stamped: PoseStamped) -> None:
    pose_stamped.header.frame_id = "map"
    if len(values) != 7:
        pose_stamped.pose.orientation.w = 1.0
        return
    pose_stamped.pose.position.x = float(values[0])
    pose_stamped.pose.position.y = float(values[1])
    pose_stamped.pose.position.z = float(values[2])
    pose_stamped.pose.orientation.x = float(values[3])
    pose_stamped.pose.orientation.y = float(values[4])
    pose_stamped.pose.orientation.z = float(values[5])
    pose_stamped.pose.orientation.w = float(values[6])


def _detection_to_dict(
    detection: ObjectDetection,
    map_position: tuple[float, float, float] | None = None,
    room_id: str | None = None,
) -> dict:
    return {
        "object_id": detection.object_id,
        "class_name": detection.class_name,
        "confidence": float(detection.confidence),
        "bounding_box": [float(v) for v in detection.bounding_box],
        "crop_embedding": [float(v) for v in detection.crop_embedding],
        "position_2d": [float(v) for v in detection.position_2d],
        "position_3d": (
            [float(v) for v in detection.position_3d]
            if detection.position_3d_valid else []
        ),
        "position_3d_frame": detection.position_3d_frame,
        "map_position": list(map_position) if map_position is not None else [],
        "room_id": room_id or "",
    }


def _detection_map_position(
    detection: ObjectDetection, camera_pose: PoseStamped
) -> tuple[float, float, float] | None:
    if not detection.position_3d_valid:
        return None
    pose = camera_pose.pose
    return transform_point(
        tuple(float(value) for value in detection.position_3d),
        (pose.position.x, pose.position.y, pose.position.z),
        (
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ),
    )


def _relation_to_dict(relation: SemanticRelation) -> dict:
    return {
        "subject_id": relation.subject_id,
        "predicate": relation.predicate,
        "object_id": relation.object_id,
        "confidence": float(relation.confidence),
        "reference_frame": relation.reference_frame,
        "source_observation_id": relation.source_observation_id,
        "relation_type": relation.relation_type,
        "timestamp": [relation.timestamp.sec, relation.timestamp.nanosec],
    }


# ------------------------------------------------------------------ #
# Property (de)serialisation – wire-compatible with knowledge_graph_db
# ------------------------------------------------------------------ #

def _serialize_properties(properties: List[Property]) -> str:
    d: dict = {}
    for p in properties:
        c = p.value
        if c.type == Content.BOOL:
            d[p.key] = bool(c.bool_value)
        elif c.type == Content.INT:
            d[p.key] = int(c.int_value)
        elif c.type == Content.FLOAT:
            d[p.key] = float(c.float_value)
        elif c.type == Content.DOUBLE:
            d[p.key] = float(c.double_value)
        elif c.type == Content.STRING:
            d[p.key] = str(c.string_value)
        elif c.type == Content.VBOOL:
            d[p.key] = list(c.bool_vector)
        elif c.type == Content.VINT:
            d[p.key] = list(c.int_vector)
        elif c.type == Content.VFLOAT:
            d[p.key] = list(c.float_vector)
        elif c.type == Content.VDOUBLE:
            d[p.key] = list(c.double_vector)
        elif c.type == Content.VSTRING:
            d[p.key] = list(c.string_vector)
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
