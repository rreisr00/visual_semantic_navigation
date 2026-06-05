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

import threading
import time

import numpy as np
import rclpy
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, Quaternion
from std_msgs.msg import Float64, String
from nav2_msgs.action import NavigateToPose

from semantic_interfaces.action import NavigateToSemanticGoal
from semantic_interfaces.srv import GetEmbedding, GetVisualFeatures, GetWaypoints
from semantic_navigation_core import rank_waypoints
from semantic_navigation_core.types import Waypoint

SERVICE_TIMEOUT_SEC = 15.0
NAV_TIMEOUT_SEC = 300.0


class SemanticOrchestratorNode(Node):
    """Thin coordinator: query → rank (core) → Nav2."""

    def __init__(self) -> None:
        super().__init__("semantic_orchestrator")

        self.declare_parameter("retrieval_mode", "siglip_yolo")
        self.declare_parameter("hybrid_embedding_weight", 0.7)
        self.declare_parameter("hybrid_object_weight", 0.3)

        self._mode = self.get_parameter("retrieval_mode").value
        self._embed_w = self.get_parameter("hybrid_embedding_weight").value
        self._obj_w = self.get_parameter("hybrid_object_weight").value

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

        # ── Evaluation taps ───────────────────────────────────────────────── #
        self._latency_pub = self.create_publisher(Float64, "/retrieval_latency", 10)
        self._result_pub = self.create_publisher(String, "/retrieval_result", 10)

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
        goal = goal_handle.request
        t_start = time.perf_counter()

        # Step 1 – embedding
        self._feedback(goal_handle, "embedding")
        query_embedding, query_objects = self._embed_query(goal)
        if query_embedding is None:
            return self._abort(goal_handle, result, "query embedding failed")

        # Step 2 – retrieve + rank
        self._feedback(goal_handle, "ranking")
        waypoints = self._fetch_waypoints()
        if waypoints is None:
            return self._abort(goal_handle, result, "get_waypoints failed")
        if not waypoints:
            return self._abort(goal_handle, result, "no waypoints in graph")

        ranked = rank_waypoints(
            query_embedding, query_objects, waypoints,
            mode=self._mode, embed_weight=self._embed_w, object_weight=self._obj_w,
        )
        if not ranked:
            return self._abort(goal_handle, result, "no rankable waypoints")
        best = ranked[0]
        self.get_logger().info(
            f"Best waypoint: {best.waypoint.node_id} (score={best.score:.4f})"
        )

        # Evaluation taps: total retrieval latency + chosen node.
        self._latency_pub.publish(Float64(data=time.perf_counter() - t_start))
        self._result_pub.publish(String(data=best.waypoint.node_id))

        # Step 3 – navigate
        self._feedback(goal_handle, "navigating")
        ok = self._navigate(goal_handle, best.waypoint)
        if not ok:
            result.success = False
            result.matched_node_id = best.waypoint.node_id
            result.score = float(best.score)
            result.message = "navigation failed"
            goal_handle.abort()
            return result

        goal_handle.succeed()
        result.success = True
        result.matched_node_id = best.waypoint.node_id
        result.score = float(best.score)
        result.message = "OK"
        return result

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

    def _fetch_waypoints(self) -> list[Waypoint] | None:
        req = GetWaypoints.Request()
        req.class_filter = "waypoint"
        res = self._call_service(self._waypoints_client, req)
        if res is None or not res.success:
            return None
        return [_to_core_waypoint(w) for w in res.waypoints]

    # ── Step 3: navigation ─────────────────────────────────────────────────── #

    def _navigate(self, goal_handle, waypoint: Waypoint) -> bool:
        if not self._nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Nav2 navigate_to_pose action not available.")
            return False

        nav_goal = NavigateToPose.Goal()
        nav_goal.pose = _waypoint_to_pose(waypoint, self.get_clock().now().to_msg())

        def _on_feedback(fb_msg) -> None:
            self._feedback(
                goal_handle, "navigating",
                distance=float(fb_msg.feedback.distance_remaining),
            )

        send_future = self._nav_client.send_goal_async(
            nav_goal, feedback_callback=_on_feedback
        )
        nav_handle = self._wait(send_future, NAV_TIMEOUT_SEC)
        if nav_handle is None or not nav_handle.accepted:
            self.get_logger().error("Nav2 goal rejected.")
            return False

        result_future = nav_handle.get_result_async()
        result = self._wait(result_future, NAV_TIMEOUT_SEC)
        if result is None:
            self.get_logger().error("Nav2 navigation timed out.")
            return False
        # status == 4 → SUCCEEDED (action_msgs/GoalStatus.STATUS_SUCCEEDED)
        return result.status == 4

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

    def _abort(self, goal_handle, result, reason: str) -> NavigateToSemanticGoal.Result:
        self.get_logger().error(f"Retrieval aborted: {reason}")
        goal_handle.abort()
        result.success = False
        result.message = reason
        return result


# ── Module helpers ──────────────────────────────────────────────────────────── #

def _to_core_waypoint(w) -> Waypoint:
    p = w.pose.pose
    return Waypoint(
        node_id=w.node_id,
        position=(p.position.x, p.position.y, p.position.z),
        orientation=(p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w),
        embedding=np.array(w.visual_embedding, dtype=np.float32),
        objects=list(w.detected_objects),
    )


def _waypoint_to_pose(waypoint: Waypoint, stamp) -> PoseStamped:
    pose = PoseStamped()
    pose.header.frame_id = "map"
    pose.header.stamp = stamp
    pose.pose.position.x = float(waypoint.position[0])
    pose.pose.position.y = float(waypoint.position[1])
    pose.pose.position.z = float(waypoint.position[2])
    pose.pose.orientation = Quaternion(
        x=float(waypoint.orientation[0]),
        y=float(waypoint.orientation[1]),
        z=float(waypoint.orientation[2]),
        w=float(waypoint.orientation[3]),
    )
    return pose


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
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
