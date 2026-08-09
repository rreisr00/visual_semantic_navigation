#!/usr/bin/env python3
"""Semantic Orchestrator node (runtime / retrieval phase).

Exposes a ``NavigateToSemanticGoal`` **action** server
(``/navigate_to_semantic_goal``). For each goal:

  1. Embed the query — text via ``/get_embedding`` or image via
     ``/get_visual_features``.
  2. Pull all waypoints from the knowledge graph via ``/get_waypoints`` and rank
     them with the pure ``semantic_navigation_core.rank_waypoints``
     (cosine, or cosine + Jaccard in ``siglip_yolo`` mode).
  3. Drive the robot to the best match via the Nav2 ``NavigateToPose`` action,
     forwarding ``distance_remaining`` as feedback.

The whole flow is non-blocking with respect to the executor: it runs under a
``MultiThreadedExecutor`` with the action server, service clients and the Nav2
client in separate callback groups, so there is no ``spin_until_future_complete``
inside a callback and no ``while goToPose`` busy-wait.
"""

from __future__ import annotations

import math
import threading
import time

import numpy as np
import rclpy
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import PoseStamped, Quaternion
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Float64, String
from nav2_msgs.action import ComputePathToPose, NavigateToPose
from visualization_msgs.msg import Marker, MarkerArray

from semantic_interfaces.action import NavigateToSemanticGoal
from semantic_interfaces.msg import RetrievalCandidate
from semantic_interfaces.srv import GetEmbedding, GetVisualFeatures, GetWaypoints
from semantic_navigation_core.goal_validation import GridSpec, validate_goal
from semantic_navigation_core.multiview import MultiviewConfig
from semantic_navigation_core.query_semantics import extract_query_semantics
from semantic_navigation_core.ranking import cosine_similarity
from semantic_navigation_core.path_metrics import path_length_2d, spl
from semantic_navigation_core.retrieval import (
    HybridWeights,
    RetrievalConfig,
    SemanticQuery,
    rank_nodes,
)
from semantic_navigation_core.types import (
    ObjectObservation,
    Observation,
    SemanticNode,
    SpatialRelation,
)

SERVICE_TIMEOUT_SEC = 15.0
NAV_TIMEOUT_SEC = 300.0


class SemanticOrchestratorNode(Node):
    """Thin coordinator: query → rank (core) → Nav2."""

    def __init__(self) -> None:
        super().__init__("semantic_orchestrator")

        self.declare_parameter("retrieval_mode", "siglip_yolo")
        self.declare_parameter("hybrid_embedding_weight", 0.7)
        self.declare_parameter("hybrid_object_weight", 0.3)
        self.declare_parameter("retrieval_method", "multiview_siglip")
        self.declare_parameter("multiview_aggregation", "purity_weighted_mean")
        self.declare_parameter("multiview_top_k", 3)
        self.declare_parameter("global_similarity_weight", 0.7)
        self.declare_parameter("object_match_weight", 0.2)
        self.declare_parameter("crop_similarity_weight", 0.1)
        self.declare_parameter("relation_match_weight", 0.0)
        self.declare_parameter("room_match_weight", 0.0)
        self.declare_parameter("room_policy", "strict_filter")
        self.declare_parameter("rejection_threshold", 0.20)
        self.declare_parameter("default_top_k", 5)
        self.declare_parameter("scene_id", "aws_small_house")
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("goal_search_radius_m", 0.75)
        self.declare_parameter("goal_obstacle_margin_m", 0.25)
        self.declare_parameter("occupied_threshold", 65)
        self.declare_parameter("allow_unknown_goals", False)
        self.declare_parameter("navigation_timeout_s", NAV_TIMEOUT_SEC)
        self.declare_parameter("planning_timeout_s", 15.0)
        self.declare_parameter("planner_id", "")
        self.declare_parameter("retrieval_marker_topic", "/semantic_retrieval_markers")
        self.declare_parameter("publish_retrieval_markers", True)

        self._mode = self.get_parameter("retrieval_mode").value
        self._embed_w = self.get_parameter("hybrid_embedding_weight").value
        self._obj_w = self.get_parameter("hybrid_object_weight").value
        self._retrieval_config = RetrievalConfig(
            method=self.get_parameter("retrieval_method").value,
            multiview=MultiviewConfig(
                method=self.get_parameter("multiview_aggregation").value,
                top_k=int(self.get_parameter("multiview_top_k").value),
            ),
            weights=HybridWeights(
                alpha=float(self.get_parameter("global_similarity_weight").value),
                beta=float(self.get_parameter("object_match_weight").value),
                gamma=float(self.get_parameter("crop_similarity_weight").value),
                delta=float(self.get_parameter("relation_match_weight").value),
                epsilon=float(self.get_parameter("room_match_weight").value),
            ),
            room_policy=str(self.get_parameter("room_policy").value),
        )
        self._map_lock = threading.Lock()
        self._map: OccupancyGrid | None = None
        map_qos = QoSProfile(depth=1)
        map_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        map_qos.reliability = ReliabilityPolicy.RELIABLE
        self.create_subscription(
            OccupancyGrid,
            self.get_parameter("map_topic").value,
            self._on_map,
            map_qos,
        )

        # ── Callback groups ───────────────────────────────────────────────── #
        self._action_cbg = ReentrantCallbackGroup()
        self._client_cbg = MutuallyExclusiveCallbackGroup()
        self._nav_cbg = MutuallyExclusiveCallbackGroup()

        # ── Service clients ───────────────────────────────────────────────── #
        self._embed_client = self.create_client(
            GetEmbedding, "get_embedding", callback_group=self._client_cbg
        )
        self._vision_client = self.create_client(
            GetVisualFeatures, "get_visual_features", callback_group=self._client_cbg
        )
        self._waypoints_client = self.create_client(
            GetWaypoints, "get_waypoints", callback_group=self._client_cbg
        )

        # ── Nav2 ──────────────────────────────────────────────────────────── #
        self._nav_client = ActionClient(
            self, NavigateToPose, "navigate_to_pose", callback_group=self._nav_cbg
        )
        self._planner_client = ActionClient(
            self,
            ComputePathToPose,
            "compute_path_to_pose",
            callback_group=self._nav_cbg,
        )

        # ── Evaluation taps ───────────────────────────────────────────────── #
        self._latency_pub = self.create_publisher(Float64, "/retrieval_latency", 10)
        self._result_pub = self.create_publisher(String, "/retrieval_result", 10)
        marker_qos = QoSProfile(depth=1)
        marker_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        marker_qos.reliability = ReliabilityPolicy.RELIABLE
        self._marker_pub = self.create_publisher(
            MarkerArray,
            self.get_parameter("retrieval_marker_topic").value,
            marker_qos,
        )

        # ── Action server ─────────────────────────────────────────────────── #
        self._action_server = ActionServer(
            self,
            NavigateToSemanticGoal,
            "navigate_to_semantic_goal",
            execute_callback=self._execute,
            goal_callback=lambda _g: GoalResponse.ACCEPT,
            cancel_callback=lambda _g: CancelResponse.ACCEPT,
            callback_group=self._action_cbg,
        )

        self.get_logger().info(
            f"Semantic orchestrator ready (mode={self._mode})."
        )

    # ── Action execute ────────────────────────────────────────────────────── #

    def _execute(self, goal_handle) -> NavigateToSemanticGoal.Result:
        result = NavigateToSemanticGoal.Result()
        # Phase latencies default to NaN so an early abort reports "not measured"
        # rather than a misleading 0.0.
        result.visual_extraction_s = math.nan
        result.retrieval_s = math.nan
        result.navigation_s = math.nan
        result.path_length_m = math.nan
        result.optimal_path_length_m = math.nan
        result.spl = math.nan
        result.final_distance_m = math.nan
        result.adjustment_distance_m = math.nan
        goal = goal_handle.request

        # Step 1 – embedding (visual_extraction phase)
        self._feedback(goal_handle, "embedding")
        t_embed = time.perf_counter()
        query_embedding, query_objects = self._embed_query(goal)
        result.visual_extraction_s = time.perf_counter() - t_embed
        if query_embedding is None:
            return self._abort(goal_handle, result, "query embedding failed")

        # Step 2 – retrieve + rank (retrieval phase)
        self._feedback(goal_handle, "ranking")
        t_retrieval = time.perf_counter()
        nodes = self._fetch_waypoints(goal.scene_id.strip())
        if nodes is None:
            return self._abort(goal_handle, result, "get_waypoints failed")
        if not nodes:
            return self._abort(goal_handle, result, "no waypoints in graph")

        query_hints = extract_query_semantics(
            goal.query_text,
            (label for node in nodes for label in node.object_labels()),
            (node.room_id for node in nodes if node.room_id),
        )
        query_objects = list(dict.fromkeys([*query_objects, *query_hints.objects]))

        semantic_query = SemanticQuery(
            text=goal.query_text,
            embedding=query_embedding,
            objects=query_objects,
            relations=query_hints.relations,
            room=query_hints.room,
        )
        ranked = rank_nodes(
            semantic_query,
            nodes,
            self._retrieval_config,
        )
        if not ranked:
            return self._abort(goal_handle, result, "no rankable waypoints")
        result.retrieval_s = time.perf_counter() - t_retrieval
        result.retrieval_latency_ms = result.retrieval_s * 1000.0
        best = ranked[0]
        result.matched_node_id = best.node.node_id
        result.score = float(best.score)
        result.predicted_pose = _node_to_pose(best.node, self.get_clock().now().to_msg())
        result.original_node_pose = result.predicted_pose
        top_k = goal.top_k if goal.top_k > 0 else int(self.get_parameter("default_top_k").value)
        result.top_k_candidates = [
            _candidate_message(
                item, self.get_clock().now().to_msg(), semantic_query
            )
            for item in ranked[:top_k]
        ]
        selected_candidates = ranked[:top_k]
        self._publish_retrieval_markers(selected_candidates)
        self.get_logger().info(
            f"Best waypoint: {best.node.node_id} (score={best.score:.4f})"
        )

        threshold = float(self.get_parameter("rejection_threshold").value)
        if best.score < threshold:
            return self._reject(
                goal_handle, result,
                f"best score {best.score:.4f} below rejection threshold {threshold:.4f}",
            )
        result.accepted = True

        # Evaluation taps: total decision latency + chosen node.
        self._latency_pub.publish(
            Float64(data=result.visual_extraction_s + result.retrieval_s)
        )
        self._result_pub.publish(String(data=best.node.node_id))

        # Decision-only mode: stop after ranking. navigation_s stays NaN.
        if goal.decision_only or not goal.navigate:
            goal_handle.succeed()
            result.success = True
            result.message = "OK (decision only)"
            return result

        # Step 3 – navigate (navigation phase)
        self._feedback(goal_handle, "navigating")
        t_nav = time.perf_counter()
        validated_pose, validation_status, adjustment_distance = (
            self._validated_goal_pose(best.node)
        )
        result.goal_validation_status = validation_status
        result.adjustment_distance_m = adjustment_distance
        if validated_pose is None:
            result.failure_type = "invalid_goal_pose"
            result.nav2_error_message = validation_status
            return self._abort(goal_handle, result, f"invalid goal pose: {validation_status}")
        result.predicted_pose = validated_pose
        result.adjusted_goal_pose = validated_pose
        self._publish_retrieval_markers(selected_candidates, validated_pose)
        planned_length, planning_error, planning_error_code = self._compute_path(
            validated_pose
        )
        if planned_length is None:
            result.failure_type = (
                "no_path" if planning_error_code == 208 else "planner_failure"
            )
            result.nav2_error_code = planning_error_code
            result.nav2_error_message = planning_error
            return self._abort(
                goal_handle, result, f"path validation failed: {planning_error}"
            )
        result.optimal_path_length_m = planned_length
        (
            ok,
            nav_error,
            nav_error_code,
            recoveries,
            path_length,
            final_distance,
        ) = self._navigate(goal_handle, validated_pose)
        result.navigation_s = time.perf_counter() - t_nav
        result.nav2_error_code = nav_error_code
        result.number_of_recoveries = recoveries
        result.path_length_m = path_length
        result.spl = spl(ok, planned_length, path_length)
        result.final_distance_m = final_distance
        if not ok:
            result.success = False
            result.message = "navigation failed"
            result.navigation_success = False
            result.failure_type = _navigation_failure_type(nav_error)
            result.nav2_error_message = nav_error
            if nav_error != "cancelled":
                goal_handle.abort()
            return result

        goal_handle.succeed()
        result.success = True
        result.navigation_success = True
        result.message = "OK"
        return result

    def _compute_path(
        self, target_pose: PoseStamped
    ) -> tuple[float | None, str, int]:
        """Ask the active Nav2 planner to validate and measure the target path."""
        timeout = float(self.get_parameter("planning_timeout_s").value)
        if not self._planner_client.wait_for_server(timeout_sec=min(timeout, 5.0)):
            return None, "compute_path_to_pose action unavailable", 0
        goal = ComputePathToPose.Goal()
        goal.goal = target_pose
        goal.planner_id = str(self.get_parameter("planner_id").value)
        goal.use_start = False
        handle = self._wait(self._planner_client.send_goal_async(goal), timeout)
        if handle is None or not handle.accepted:
            return None, "planner goal rejected", 0
        wrapper = self._wait(handle.get_result_async(), timeout)
        if wrapper is None:
            self._wait(handle.cancel_goal_async(), 2.0)
            return None, "planner timeout", 207
        error_code = int(wrapper.result.error_code)
        if wrapper.status != 4 or error_code != 0:
            message = str(wrapper.result.error_msg) or (
                f"planner_status_{wrapper.status}_error_{error_code}"
            )
            return None, message, error_code
        poses = wrapper.result.path.poses
        if not poses:
            return None, "planner returned an empty path", 208
        length = path_length_2d(
            (pose.pose.position.x, pose.pose.position.y) for pose in poses
        )
        return length, "path_valid", 0

    # ── Step 1: embedding ──────────────────────────────────────────────────── #

    def _embed_query(self, goal) -> tuple[np.ndarray | None, list[str]]:
        if goal.use_image:
            req = GetVisualFeatures.Request()
            req.image = goal.query_image
            res = self._call_service(self._vision_client, req)
            if res is None or not res.success:
                return None, []
            return np.array(res.visual_embedding, dtype=np.float32), list(
                res.detected_objects
            )

        text = goal.query_text.strip()
        if not text:
            return None, []
        req = GetEmbedding.Request()
        req.text = text
        req.use_image = False
        res = self._call_service(self._embed_client, req)
        if res is None or not res.success:
            return None, []
        return np.array(res.embedding, dtype=np.float32), []

    # ── Step 2: retrieval ──────────────────────────────────────────────────── #

    def _fetch_waypoints(self, scene_id: str = "") -> list[SemanticNode] | None:
        req = GetWaypoints.Request()
        req.class_filter = "waypoint"
        req.scene_id = scene_id or self.get_parameter("scene_id").value
        res = self._call_service(self._waypoints_client, req)
        if res is None or not res.success:
            return None
        return [_to_core_node(w) for w in res.waypoints]

    # ── Step 3: navigation ─────────────────────────────────────────────────── #

    def _on_map(self, message: OccupancyGrid) -> None:
        with self._map_lock:
            self._map = message

    def _validated_goal_pose(
        self, node: SemanticNode
    ) -> tuple[PoseStamped | None, str, float]:
        pose = _node_to_pose(node, self.get_clock().now().to_msg())
        with self._map_lock:
            grid_message = self._map
        if grid_message is None:
            return None, "map_unavailable", math.nan
        if grid_message.header.frame_id != self.get_parameter("map_frame").value:
            return None, f"map_frame_mismatch:{grid_message.header.frame_id}", math.nan
        info = grid_message.info
        grid = GridSpec(
            width=int(info.width),
            height=int(info.height),
            resolution=float(info.resolution),
            origin_x=float(info.origin.position.x),
            origin_y=float(info.origin.position.y),
            occupied_threshold=int(self.get_parameter("occupied_threshold").value),
            allow_unknown=bool(self.get_parameter("allow_unknown_goals").value),
        )
        validation = validate_goal(
            pose.pose.position.x,
            pose.pose.position.y,
            grid_message.data,
            grid,
            search_radius_m=float(self.get_parameter("goal_search_radius_m").value),
            obstacle_margin_m=float(self.get_parameter("goal_obstacle_margin_m").value),
        )
        if not validation.valid:
            return None, validation.status, validation.adjustment_distance_m
        pose.pose.position.x = validation.x
        pose.pose.position.y = validation.y
        self.get_logger().info(
            f"Goal validation={validation.status}, adjustment="
            f"{validation.adjustment_distance_m:.3f} m"
        )
        return pose, validation.status, validation.adjustment_distance_m

    def _navigate(self, goal_handle, target_pose: PoseStamped):
        if not self._nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Nav2 navigate_to_pose action not available.")
            return False, "navigate_to_pose action unavailable", 0, 0, 0.0, math.nan

        nav_goal = NavigateToPose.Goal()
        nav_goal.pose = target_pose

        trace = {
            "previous": None,
            "path_length": 0.0,
            "recoveries": 0,
            "final_distance": math.nan,
        }

        def _on_feedback(fb_msg) -> None:
            feedback = fb_msg.feedback
            position = feedback.current_pose.pose.position
            current = (float(position.x), float(position.y))
            if trace["previous"] is not None:
                trace["path_length"] += math.hypot(
                    current[0] - trace["previous"][0],
                    current[1] - trace["previous"][1],
                )
            trace["previous"] = current
            trace["recoveries"] = int(feedback.number_of_recoveries)
            trace["final_distance"] = float(feedback.distance_remaining)
            self._feedback(
                goal_handle, "navigating",
                distance=float(feedback.distance_remaining),
            )

        send_future = self._nav_client.send_goal_async(
            nav_goal, feedback_callback=_on_feedback
        )
        timeout = float(self.get_parameter("navigation_timeout_s").value)
        nav_handle = self._wait(send_future, timeout)
        if nav_handle is None or not nav_handle.accepted:
            self.get_logger().error("Nav2 goal rejected.")
            return False, "goal_rejected", 0, 0, 0.0, math.nan

        result_future = nav_handle.get_result_async()
        deadline = time.monotonic() + timeout
        while not result_future.done() and time.monotonic() < deadline:
            if goal_handle.is_cancel_requested:
                self._wait(nav_handle.cancel_goal_async(), 5.0)
                goal_handle.canceled()
                return (
                    False, "cancelled", 0, trace["recoveries"],
                    trace["path_length"], trace["final_distance"],
                )
            time.sleep(0.05)
        result = result_future.result() if result_future.done() else None
        if result is None:
            self._wait(nav_handle.cancel_goal_async(), 5.0)
            self.get_logger().error("Nav2 navigation timed out.")
            return (
                False, "timeout", 0, trace["recoveries"],
                trace["path_length"], trace["final_distance"],
            )
        # status == 4 → SUCCEEDED (action_msgs/GoalStatus.STATUS_SUCCEEDED)
        error_code = int(result.result.error_code)
        error_message = str(result.result.error_msg)
        succeeded = result.status == 4 and error_code == 0
        detail = "" if succeeded else (
            error_message or f"nav2_status_{result.status}_error_{error_code}"
        )
        return (
            succeeded,
            detail,
            error_code,
            trace["recoveries"],
            trace["path_length"],
            trace["final_distance"],
        )

    # ── Shared async helpers ──────────────────────────────────────────────── #

    def _call_service(self, client, request, timeout: float = SERVICE_TIMEOUT_SEC):
        """Async service call blocking only the action thread (no deadlock)."""
        if not client.service_is_ready():
            self.get_logger().warn(f"Service {client.srv_name} not available.")
            return None
        return self._wait(client.call_async(request), timeout)

    def _wait(self, future, timeout: float):
        """Block the calling (action) thread until ``future`` completes."""
        done = threading.Event()
        future.add_done_callback(lambda _f: done.set())
        if not done.wait(timeout):
            return None
        try:
            return future.result()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Future raised: {exc}")
            return None

    # ── Feedback / result helpers ──────────────────────────────────────────── #

    def _feedback(self, goal_handle, stage: str, distance: float = 0.0) -> None:
        fb = NavigateToSemanticGoal.Feedback()
        fb.stage = stage
        fb.distance_remaining = distance
        goal_handle.publish_feedback(fb)

    def _publish_retrieval_markers(
        self, candidates, navigation_goal: PoseStamped | None = None
    ) -> None:
        if not bool(self.get_parameter("publish_retrieval_markers").value):
            return
        output = MarkerArray()
        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        delete_all.header.frame_id = self.get_parameter("map_frame").value
        output.markers.append(delete_all)
        stamp = self.get_clock().now().to_msg()
        for index, ranked in enumerate(candidates):
            pose = _node_to_pose(ranked.node, stamp)
            marker = Marker()
            marker.header = pose.header
            marker.ns = "retrieval_candidates"
            marker.id = index
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose = pose.pose
            marker.scale.x = marker.scale.y = marker.scale.z = (
                0.38 if index == 0 else 0.24
            )
            marker.color.r = 1.0
            marker.color.g = 0.2 if index == 0 else 0.75
            marker.color.b = 0.1
            marker.color.a = 0.9
            output.markers.append(marker)
            label = Marker()
            label.header = pose.header
            label.ns = "retrieval_candidates"
            label.id = 1000 + index
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = pose.pose.position.x
            label.pose.position.y = pose.pose.position.y
            label.pose.position.z = pose.pose.position.z + 0.45
            label.pose.orientation.w = 1.0
            label.scale.z = 0.22
            label.color.r = label.color.g = label.color.b = label.color.a = 1.0
            label.text = f"{index + 1}: {ranked.node.node_id} ({ranked.score:.3f})"
            output.markers.append(label)
        if navigation_goal is not None:
            goal_marker = Marker()
            goal_marker.header = navigation_goal.header
            goal_marker.ns = "navigation_goal"
            goal_marker.id = 0
            goal_marker.type = Marker.ARROW
            goal_marker.action = Marker.ADD
            goal_marker.pose = navigation_goal.pose
            goal_marker.scale.x = 0.6
            goal_marker.scale.y = 0.12
            goal_marker.scale.z = 0.12
            goal_marker.color.g = 1.0
            goal_marker.color.a = 1.0
            output.markers.append(goal_marker)
        self._marker_pub.publish(output)

    def _abort(self, goal_handle, result, reason: str) -> NavigateToSemanticGoal.Result:
        self.get_logger().error(f"Retrieval aborted: {reason}")
        goal_handle.abort()
        result.success = False
        result.message = reason
        return result

    def _reject(self, goal_handle, result, reason: str) -> NavigateToSemanticGoal.Result:
        self.get_logger().warn(f"Semantic query rejected: {reason}")
        goal_handle.succeed()
        result.success = False
        result.accepted = False
        result.rejection_reason = reason
        result.failure_type = "semantic_rejection_error"
        result.message = "query rejected"
        return result


# ── Module helpers ──────────────────────────────────────────────────────────── #

def _to_core_node(w) -> SemanticNode:
    p = w.pose.pose
    observations = []
    for source in w.observations:
        labels_by_detection_id = {
            item.object_id: item.class_name for item in source.detections
            if item.object_id
        }
        objects = [ObjectObservation(
            label=item.class_name,
            confidence=float(item.confidence),
            box=tuple(float(value) for value in item.bounding_box),
            embedding=(
                np.asarray(item.crop_embedding, dtype=np.float32)
                if item.crop_embedding else None
            ),
            object_id=item.object_id,
            position_2d=tuple(float(value) for value in item.position_2d),
            position_3d=(
                tuple(float(value) for value in item.position_3d)
                if item.position_3d_valid else None
            ),
            position_3d_frame=item.position_3d_frame,
            map_position=(
                tuple(float(value) for value in item.map_position)
                if item.map_position_valid else None
            ),
            room_id=item.room_id or None,
        ) for item in source.detections]
        relations = [SpatialRelation(
            subject=labels_by_detection_id.get(item.subject_id, item.subject_id),
            predicate=item.predicate,
            obj=labels_by_detection_id.get(item.object_id, item.object_id),
            confidence=float(item.confidence),
            subject_id=item.subject_id,
            object_id=item.object_id,
            reference_frame=item.reference_frame,
            source_observation_id=item.source_observation_id,
            relation_type=item.relation_type,
        ) for item in source.relations]
        observations.append(Observation(
            observation_id=source.observation_id,
            embedding=np.asarray(source.image_embedding, dtype=np.float32),
            image_path=source.image_path,
            objects=objects,
            relations=relations,
            timestamp=float(source.timestamp.sec) + float(source.timestamp.nanosec) * 1e-9,
            camera_frame=source.camera_frame,
            camera_position=(
                source.camera_pose.pose.position.x,
                source.camera_pose.pose.position.y,
                source.camera_pose.pose.position.z,
            ) if source.camera_pose.header.frame_id else None,
            camera_orientation=(
                source.camera_pose.pose.orientation.x,
                source.camera_pose.pose.orientation.y,
                source.camera_pose.pose.orientation.z,
                source.camera_pose.pose.orientation.w,
            ) if source.camera_pose.header.frame_id else None,
            depth_camera_frame=source.depth_camera_frame,
            depth_camera_position=(
                source.depth_camera_pose.pose.position.x,
                source.depth_camera_pose.pose.position.y,
                source.depth_camera_pose.pose.position.z,
            ) if source.depth_camera_pose.header.frame_id else None,
            depth_camera_orientation=(
                source.depth_camera_pose.pose.orientation.x,
                source.depth_camera_pose.pose.orientation.y,
                source.depth_camera_pose.pose.orientation.z,
                source.depth_camera_pose.pose.orientation.w,
            ) if source.depth_camera_pose.header.frame_id else None,
            requested_yaw=float(source.requested_yaw),
            measured_yaw=float(source.measured_yaw),
            angular_error=float(source.angular_error),
            image_valid=bool(source.image_valid),
            depth_valid=bool(source.depth_valid),
            camera_room=source.camera_room or None,
            observation_room=source.observation_room or None,
            purity=float(source.purity) if source.purity_valid else None,
            contamination_class=source.contamination_class or "unknown",
            transition_zone=bool(source.transition_zone),
        ))
    if not observations:
        observations.append(Observation(
            observation_id=f"{w.node_id}__legacy",
            embedding=np.asarray(w.visual_embedding, dtype=np.float32),
            objects=[ObjectObservation(label=label) for label in w.detected_objects],
        ))
    nav = w.navigation_goal_pose.pose
    return SemanticNode(
        node_id=w.node_id,
        position=(p.position.x, p.position.y, p.position.z),
        orientation=(p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w),
        observations=observations,
        room_id=w.room_label or None,
        scene_id=w.scene_id or "default",
        navigation_position=(nav.position.x, nav.position.y, nav.position.z),
        navigation_orientation=(
            nav.orientation.x, nav.orientation.y, nav.orientation.z, nav.orientation.w
        ),
        configuration_hash=w.configuration_hash,
    )


def _node_to_pose(node: SemanticNode, stamp) -> PoseStamped:
    pose = PoseStamped()
    pose.header.frame_id = "map"
    pose.header.stamp = stamp
    position = node.navigation_position or node.position
    orientation = node.navigation_orientation or node.orientation
    pose.pose.position.x = float(position[0])
    pose.pose.position.y = float(position[1])
    pose.pose.position.z = float(position[2])
    pose.pose.orientation = Quaternion(
        x=float(orientation[0]),
        y=float(orientation[1]),
        z=float(orientation[2]),
        w=float(orientation[3]),
    )
    return pose


def _candidate_message(
    ranked, stamp, query: SemanticQuery
) -> RetrievalCandidate:
    candidate = RetrievalCandidate()
    candidate.node_id = ranked.node.node_id
    candidate.pose = _node_to_pose(ranked.node, stamp)
    candidate.score = float(ranked.score)
    candidate.global_similarity = float(
        ranked.components.get("global_similarity", 0.0)
    )
    candidate.object_match_score = float(
        ranked.components.get("object_match_score", 0.0)
    )
    candidate.crop_similarity = float(
        ranked.components.get("crop_similarity", 0.0)
    )
    candidate.relation_match_score = float(
        ranked.components.get("relation_match_score", 0.0)
    )
    candidate.room_match_score = float(
        ranked.components.get("room_match_score", 0.0)
    )
    query_labels = {label.strip() for label in query.objects}
    if candidate.object_match_score > 0.0:
        for observation in ranked.node.observations:
            for detected in observation.objects:
                if detected.label.strip() not in query_labels:
                    continue
                if (
                    detected.object_id
                    and detected.object_id not in candidate.matched_object_ids
                ):
                    candidate.matched_object_ids.append(detected.object_id)
                if detected.label not in candidate.matched_object_labels:
                    candidate.matched_object_labels.append(detected.label)
    if candidate.crop_similarity > 0.0 and query.embedding is not None:
        best_score = float("-inf")
        for observation in ranked.node.observations:
            for detected in observation.objects:
                if detected.embedding is None:
                    continue
                score = cosine_similarity(query.embedding, detected.embedding)
                if score > best_score:
                    best_score = score
                    candidate.best_crop_object_id = detected.object_id
                    candidate.best_crop_object_label = detected.label
    return candidate


def _navigation_failure_type(message: str) -> str:
    normalized = message.lower()
    if normalized == "cancelled":
        return "cancelled"
    if normalized == "timeout":
        return "timeout"
    if "control" in normalized:
        return "controller_failure"
    if "path" in normalized or "planning" in normalized:
        return "no_path"
    return "planner_failure"


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SemanticOrchestratorNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            executor.shutdown(timeout_sec=2.0)
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        finally:
            rclpy.try_shutdown()


if __name__ == "__main__":
    main()
