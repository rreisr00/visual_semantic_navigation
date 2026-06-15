#!/usr/bin/env python3
"""Waypoint Capture Manager node (teaching phase).

Exposes a ``CaptureWaypoint`` **action** server (``/capture_waypoint``). On each
goal it runs the capture pipeline, driven by the pure
``semantic_navigation_core.CaptureStateMachine``:

  1. Snapshot the latest camera frame from a thread-safe buffer.
  2. Read the current map→base_link transform (non-blocking, latest cached).
  3. Call ``/get_visual_features`` with the buffered image.
  4. Call ``/store_waypoint`` with the pose, embedding and detected objects.
  5. Publish per-stage feedback and a final result.

Concurrency
-----------
Runs under a ``MultiThreadedExecutor``. The action server lives in a
``ReentrantCallbackGroup`` while the service clients live in a separate
``MutuallyExclusiveCallbackGroup``, so the execute thread can block on a service
future while another executor thread delivers the response — no deadlock.

A camera watchdog publishes ``diagnostic_msgs`` and warns when frames go stale.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import rclpy
import rclpy.time
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image

import tf2_ros
from diagnostic_updater import Updater, DiagnosticStatusWrapper

from semantic_interfaces.action import CaptureWaypoint
from semantic_interfaces.srv import GetVisualFeatures, StoreWaypoint
from semantic_navigation_core import CaptureState, CaptureStateMachine
from semantic_navigation_core.capture_state_machine import (
    STAGE_ENCODED,
    STAGE_GOT_IMAGE,
    STAGE_GOT_POSE,
    STAGE_STORED,
)

# A frame older than this (wall seconds) marks the camera as stale.
CAMERA_STALE_SEC = 2.0
# Per service call upper bound.
SERVICE_TIMEOUT_SEC = 15.0


class WaypointCaptureNode(Node):
    """Captures pose + visual features and persists them as semantic waypoints."""

    def __init__(self) -> None:
        super().__init__("kg_manager")

        # ── Thread-safe image buffer ──────────────────────────────────────── #
        self._image_lock = threading.Lock()
        self._latest_image: Optional[Image] = None
        self._latest_image_mono: float = 0.0

        self.create_subscription(
            Image, "/camera/image_raw", self._image_callback, 10
        )

        # ── TF2 (runs in the background via the node's executor) ──────────── #
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # ── Callback groups ───────────────────────────────────────────────── #
        self._action_cbg = ReentrantCallbackGroup()
        self._client_cbg = MutuallyExclusiveCallbackGroup()

        # ── Service clients ───────────────────────────────────────────────── #
        self._vision_client = self.create_client(
            GetVisualFeatures, "get_visual_features", callback_group=self._client_cbg
        )
        self._store_client = self.create_client(
            StoreWaypoint, "store_waypoint", callback_group=self._client_cbg
        )

        # ── Action server ─────────────────────────────────────────────────── #
        self._action_server = ActionServer(
            self,
            CaptureWaypoint,
            "capture_waypoint",
            execute_callback=self._execute_capture,
            goal_callback=lambda _g: GoalResponse.ACCEPT,
            cancel_callback=lambda _g: CancelResponse.ACCEPT,
            callback_group=self._action_cbg,
        )

        # ── Camera watchdog / diagnostics ─────────────────────────────────── #
        self._diag = Updater(self)
        self._diag.setHardwareID("kg_manager")
        self._diag.add("camera", self._camera_diagnostic)
        self.create_timer(1.0, self._diag.update)

        self.get_logger().info("WaypointCaptureNode ready (CaptureWaypoint action).")

    # ── Image buffer ──────────────────────────────────────────────────────── #

    def _image_callback(self, msg: Image) -> None:
        with self._image_lock:
            self._latest_image = msg
            self._latest_image_mono = time.monotonic()

    def _snapshot_image(self) -> tuple[Optional[Image], float]:
        with self._image_lock:
            return self._latest_image, self._latest_image_mono

    # ── Camera watchdog ───────────────────────────────────────────────────── #

    def _camera_age(self) -> Optional[float]:
        _, mono = self._snapshot_image()
        if mono == 0.0:
            return None
        return time.monotonic() - mono

    def _camera_diagnostic(
        self, stat: DiagnosticStatusWrapper
    ) -> DiagnosticStatusWrapper:
        age = self._camera_age()
        if age is None:
            stat.summary(DiagnosticStatusWrapper.WARN, "No camera frame received yet")
        elif age > CAMERA_STALE_SEC:
            stat.summary(
                DiagnosticStatusWrapper.ERROR, f"Camera stale ({age:.1f}s old)"
            )
        else:
            stat.summary(DiagnosticStatusWrapper.OK, "Camera streaming")
        stat.add("age_sec", f"{age:.2f}" if age is not None else "n/a")
        return stat

    # ── Capture action ────────────────────────────────────────────────────── #

    def _execute_capture(self, goal_handle) -> CaptureWaypoint.Result:
        sm = CaptureStateMachine()
        result = CaptureWaypoint.Result()

        # Step 1+2 – snapshot image and TF pose
        image, _ = self._snapshot_image()
        pose = self._lookup_pose() if image is not None else None
        sm.start(has_image=image is not None, has_pose=pose is not None)
        if sm.state == CaptureState.FAILED:
            return self._abort(goal_handle, result, sm.reason)

        self._publish_feedback(goal_handle, STAGE_GOT_IMAGE)
        self._publish_feedback(goal_handle, STAGE_GOT_POSE)

        if goal_handle.is_cancel_requested:
            sm.cancel()
            return self._cancelled(goal_handle, result, sm.reason)

        # Step 3 – visual features
        features = self._call_service(
            self._vision_client, self._build_features_req(image)
        )
        if features is None or not features.success:
            msg = "service unavailable/timeout" if features is None else features.message
            sm.features_failed(msg)
            return self._abort(goal_handle, result, sm.reason)
        sm.features_ok()
        self._publish_feedback(goal_handle, STAGE_ENCODED)

        if goal_handle.is_cancel_requested:
            sm.cancel()
            return self._cancelled(goal_handle, result, sm.reason)

        # Step 4 – store
        # Use the human label as the node_id when provided (e.g. "cocina_01"),
        # so downstream room-level metrics can derive the room via room_key().
        # Fall back to a unique timestamped id when no label is given.
        label = goal_handle.request.label.strip()
        waypoint_id = label if label else f"waypoint_{time.time_ns()}"
        store_req = self._build_store_req(waypoint_id, pose, features)
        store = self._call_service(self._store_client, store_req)
        if store is None or not store.success:
            msg = "service unavailable/timeout" if store is None else store.message
            sm.store_failed(msg)
            return self._abort(goal_handle, result, sm.reason)
        sm.store_ok()
        self._publish_feedback(goal_handle, STAGE_STORED)

        # Done
        goal_handle.succeed()
        result.success = True
        result.node_id = waypoint_id
        result.message = "OK"
        x, y = pose.pose.position.x, pose.pose.position.y
        self.get_logger().info(
            f"Waypoint '{waypoint_id}' stored at ({x:.2f}, {y:.2f})."
        )
        return result

    # ── Helpers ───────────────────────────────────────────────────────────── #

    def _lookup_pose(self) -> Optional[PoseStamped]:
        """Non-blocking TF lookup of the latest map→base_link transform."""
        try:
            transform = self._tf_buffer.lookup_transform(
                "map", "base_link", rclpy.time.Time()
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as exc:
            self.get_logger().warn(f"TF2 lookup failed: {exc}")
            return None
        return _transform_to_pose_stamped(transform)

    def _call_service(self, client, request, timeout: float = SERVICE_TIMEOUT_SEC):
        """Async service call that blocks the *action* thread only.

        Safe under MultiThreadedExecutor: the response future is completed by a
        different executor thread (clients live in their own callback group).
        """
        if not client.service_is_ready():
            self.get_logger().warn(f"Service {client.srv_name} not available.")
            return None
        future = client.call_async(request)
        done = threading.Event()
        future.add_done_callback(lambda _f: done.set())
        if not done.wait(timeout):
            self.get_logger().error(f"Service {client.srv_name} timed out.")
            return None
        try:
            return future.result()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Service {client.srv_name} raised: {exc}")
            return None

    @staticmethod
    def _build_features_req(image: Image) -> GetVisualFeatures.Request:
        req = GetVisualFeatures.Request()
        req.image = image
        return req

    @staticmethod
    def _build_store_req(
        waypoint_id: str, pose: PoseStamped, features: GetVisualFeatures.Response
    ) -> StoreWaypoint.Request:
        req = StoreWaypoint.Request()
        req.node_id = waypoint_id
        req.pose = pose
        req.visual_embedding = list(features.visual_embedding)
        req.detected_objects = list(features.detected_objects)
        return req

    def _publish_feedback(self, goal_handle, stage: str) -> None:
        fb = CaptureWaypoint.Feedback()
        fb.stage = stage
        goal_handle.publish_feedback(fb)

    def _abort(self, goal_handle, result, reason: str) -> CaptureWaypoint.Result:
        self.get_logger().error(f"Capture aborted: {reason}")
        goal_handle.abort()
        result.success = False
        result.message = reason
        return result

    def _cancelled(self, goal_handle, result, reason: str) -> CaptureWaypoint.Result:
        self.get_logger().warn(f"Capture cancelled: {reason}")
        goal_handle.canceled()
        result.success = False
        result.message = reason
        return result


# ── Module helpers ──────────────────────────────────────────────────────────── #

def _transform_to_pose_stamped(transform) -> PoseStamped:
    pose = PoseStamped()
    pose.header = transform.header
    pose.pose.position.x = transform.transform.translation.x
    pose.pose.position.y = transform.transform.translation.y
    pose.pose.position.z = transform.transform.translation.z
    pose.pose.orientation = transform.transform.rotation
    return pose


def main(args=None) -> None:
    rclpy.init(args=args)
    node = WaypointCaptureNode()
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
