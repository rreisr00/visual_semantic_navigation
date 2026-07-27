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

import os
import threading
import time
from math import atan2, copysign, cos, pi, sin
from typing import Optional

import rclpy
import rclpy.time
from rclpy.duration import Duration
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import PoseStamped, TwistStamped
from sensor_msgs.msg import CameraInfo, Image

import tf2_ros
from message_filters import ApproximateTimeSynchronizer, Subscriber
from diagnostic_updater import Updater, DiagnosticStatusWrapper

try:
    import cv2
    from cv_bridge import CvBridge

    _IMAGE_SAVE_AVAILABLE = True
except ImportError:
    cv2 = None
    CvBridge = None
    _IMAGE_SAVE_AVAILABLE = False

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

        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("camera_frame", "camera_rgb_frame")
        self.declare_parameter("use_depth", True)
        self.declare_parameter("depth_topic", "/camera/depth/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/depth/camera_info")
        self.declare_parameter("sync_queue_size", 10)
        self.declare_parameter("sync_slop_s", 0.08)
        self.declare_parameter("depth_bundle_max_age_s", 0.5)
        self.declare_parameter("tf_timeout_s", 0.25)
        self.declare_parameter("camera_stale_s", CAMERA_STALE_SEC)
        self.declare_parameter("scene_id", "aws_small_house")
        self.declare_parameter("configuration_hash", "")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("capture_angular_speed_rad_s", 0.6)
        self.declare_parameter("capture_angular_tolerance_rad", 0.05)
        self.declare_parameter("capture_rotation_timeout_s", 20.0)
        self.declare_parameter("capture_stabilization_s", 0.75)
        self.declare_parameter(
            "observations_dir", "~/.ros/semantic_maps/{scene_id}/images"
        )

        # ── Thread-safe image buffer ──────────────────────────────────────── #
        self._image_lock = threading.Lock()
        self._latest_image: Optional[Image] = None
        self._latest_depth: Optional[Image] = None
        self._latest_camera_info: Optional[CameraInfo] = None
        self._latest_image_mono: float = 0.0
        self._latest_rgb_fallback: Optional[Image] = None
        self._latest_rgb_fallback_mono: float = 0.0
        self._image_bridge = CvBridge() if _IMAGE_SAVE_AVAILABLE else None

        self._sensor_subscriptions = []
        self._sensor_sync = None
        self._use_depth = bool(self.get_parameter("use_depth").value)
        if self._use_depth:
            self.create_subscription(
                Image,
                self.get_parameter("image_topic").value,
                self._image_callback,
                qos_profile_sensor_data,
            )
            self._sensor_subscriptions = [
                Subscriber(
                    self, Image, self.get_parameter("image_topic").value,
                    qos_profile=qos_profile_sensor_data,
                ),
                Subscriber(
                    self, Image, self.get_parameter("depth_topic").value,
                    qos_profile=qos_profile_sensor_data,
                ),
                Subscriber(
                    self, CameraInfo,
                    self.get_parameter("camera_info_topic").value,
                    qos_profile=qos_profile_sensor_data,
                ),
            ]
            self._sensor_sync = ApproximateTimeSynchronizer(
                self._sensor_subscriptions,
                queue_size=int(self.get_parameter("sync_queue_size").value),
                slop=float(self.get_parameter("sync_slop_s").value),
            )
            self._sensor_sync.registerCallback(self._synchronized_sensor_callback)
        else:
            self.create_subscription(
                Image,
                self.get_parameter("image_topic").value,
                self._image_callback,
                qos_profile_sensor_data,
            )

        # ── TF2 (runs in the background via the node's executor) ──────────── #
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # ── Callback groups ───────────────────────────────────────────────── #
        self._action_cbg = MutuallyExclusiveCallbackGroup()
        self._client_cbg = MutuallyExclusiveCallbackGroup()
        self._cmd_vel_pub = self.create_publisher(
            TwistStamped, self.get_parameter("cmd_vel_topic").value, 10
        )

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
            received_at = time.monotonic()
            self._latest_rgb_fallback = msg
            self._latest_rgb_fallback_mono = received_at
            if not self._use_depth:
                self._latest_image = msg
                self._latest_depth = None
                self._latest_camera_info = None
                self._latest_image_mono = received_at

    def _synchronized_sensor_callback(
        self, image: Image, depth: Image, camera_info: CameraInfo
    ) -> None:
        with self._image_lock:
            self._latest_image = image
            self._latest_depth = depth
            self._latest_camera_info = camera_info
            self._latest_image_mono = time.monotonic()

    def _snapshot_image(self) -> tuple[Optional[Image], float]:
        image, _, _, received_at = self._snapshot_sensor_bundle()
        return image, received_at

    def _snapshot_sensor_bundle(self):
        with self._image_lock:
            maximum_depth_age = float(
                self.get_parameter("depth_bundle_max_age_s").value
            )
            synchronized_is_current = (
                self._latest_image is not None
                and time.monotonic() - self._latest_image_mono <= maximum_depth_age
            )
            if self._use_depth and not synchronized_is_current:
                return (
                    self._latest_rgb_fallback,
                    None,
                    None,
                    self._latest_rgb_fallback_mono,
                )
            return (
                self._latest_image,
                self._latest_depth,
                self._latest_camera_info,
                self._latest_image_mono,
            )

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
        elif age > float(self.get_parameter("camera_stale_s").value):
            stat.summary(
                DiagnosticStatusWrapper.ERROR, f"Camera stale ({age:.1f}s old)"
            )
        else:
            stat.summary(DiagnosticStatusWrapper.OK, "Camera streaming")
        stat.add("age_sec", f"{age:.2f}" if age is not None else "n/a")
        return stat

    # ── Capture action ────────────────────────────────────────────────────── #

    def _execute_capture(self, goal_handle) -> CaptureWaypoint.Result:
        result = CaptureWaypoint.Result()
        request = goal_handle.request
        relative_views = list(request.relative_view_yaws_deg)
        targets = [float(request.requested_yaw)]
        if relative_views:
            current = self._lookup_current_pose(
                self.get_parameter("base_frame").value
            )
            if current is None:
                return self._abort(goal_handle, result, "base transform unavailable")
            origin_yaw = _yaw_from_pose(current)
            targets = [
                _wrap_angle(origin_yaw + float(value) * pi / 180.0)
                for value in relative_views
            ]

        waypoint_id = request.label.strip()
        merged = False
        total = len(targets)
        for index, requested_yaw in enumerate(targets, start=1):
            if goal_handle.is_cancel_requested:
                self._publish_stop()
                return self._cancelled(goal_handle, result, "cancel requested")
            if relative_views and request.rotate_robot:
                ok, reason = self._rotate_to_yaw(
                    goal_handle, requested_yaw, index, total
                )
                if not ok:
                    if reason == "cancelled":
                        return self._cancelled(goal_handle, result, reason)
                    return self._abort(goal_handle, result, reason)
                not_before = time.monotonic()
                if not self._wait_stabilization(goal_handle):
                    return self._cancelled(goal_handle, result, "cancel requested")
                if not self._wait_for_image_after(not_before):
                    return self._abort(
                        goal_handle, result, "no fresh image after stabilization"
                    )

            store, pose, observation_id, error = self._capture_one(
                goal_handle,
                waypoint_id,
                requested_yaw,
                index,
                total,
            )
            if error:
                if goal_handle.is_cancel_requested:
                    return self._cancelled(goal_handle, result, error)
                return self._abort(goal_handle, result, error)
            waypoint_id = store.node_id or waypoint_id
            merged = merged or bool(store.merged_with_existing)
            result.observation_ids.append(observation_id)
            result.captured_views = index
            x, y = pose.pose.position.x, pose.pose.position.y
            self.get_logger().info(
                f"Waypoint '{waypoint_id}' view {index}/{total} stored "
                f"at ({x:.2f}, {y:.2f})."
            )

        goal_handle.succeed()
        result.success = True
        result.node_id = waypoint_id
        result.observation_id = result.observation_ids[-1]
        result.merged_with_existing = merged
        result.message = "OK"
        return result

    def _capture_one(
        self,
        goal_handle,
        waypoint_id: str,
        requested_yaw: float,
        current_view: int,
        total_views: int,
    ):
        sm = CaptureStateMachine()

        # Step 1+2 – snapshot image and TF pose
        image, depth, camera_info, image_mono = self._snapshot_sensor_bundle()
        stale_limit = float(self.get_parameter("camera_stale_s").value)
        image_is_fresh = image is not None and time.monotonic() - image_mono <= stale_limit
        pose = (
            self._lookup_pose(image, self.get_parameter("base_frame").value)
            if image_is_fresh else None
        )
        sm.start(has_image=image is not None, has_pose=pose is not None)
        if sm.state == CaptureState.FAILED:
            return None, pose, "", sm.reason
        if not image_is_fresh:
            return None, pose, "", "camera frame is stale"
        camera_pose = self._lookup_pose(
            image,
            image.header.frame_id or self.get_parameter("camera_frame").value,
        )
        if camera_pose is None:
            return None, pose, "", "camera transform unavailable at image timestamp"
        depth_camera_pose = None
        if depth is not None:
            depth_camera_pose = self._lookup_pose(
                image, depth.header.frame_id
            )
            if depth_camera_pose is None:
                self.get_logger().warn(
                    "Depth transform unavailable; storing RGB-only observation"
                )
                depth = None
                camera_info = None

        measured_yaw = _yaw_from_pose(pose)
        self._publish_feedback(
            goal_handle, STAGE_GOT_IMAGE, current_view, total_views,
            requested_yaw, measured_yaw,
        )
        self._publish_feedback(
            goal_handle, STAGE_GOT_POSE, current_view, total_views,
            requested_yaw, measured_yaw,
        )

        if goal_handle.is_cancel_requested:
            sm.cancel()
            return None, pose, "", sm.reason

        # Step 3 – visual features
        features = self._call_service(
            self._vision_client,
            self._build_features_req(image, depth, camera_info),
        )
        if features is None or not features.success:
            msg = "service unavailable/timeout" if features is None else features.message
            sm.features_failed(msg)
            return None, pose, "", sm.reason
        sm.features_ok()
        self._publish_feedback(
            goal_handle, STAGE_ENCODED, current_view, total_views,
            requested_yaw, measured_yaw,
        )

        if goal_handle.is_cancel_requested:
            sm.cancel()
            return None, pose, "", sm.reason

        # Step 4 – store
        # Use the human label as the node_id when provided (e.g. "cocina_01").
        # With an empty label the BRIDGE names the waypoint after the room
        # containing its pose ("<room>_<NN>", timestamped fallback outside
        # rooms) and returns the final id in the response.
        store_req = self._build_store_req(
            waypoint_id, pose, camera_pose, image, features,
            goal_handle.request, requested_yaw, depth, depth_camera_pose,
        )
        store = self._call_service(self._store_client, store_req)
        if store is None or not store.success:
            msg = "service unavailable/timeout" if store is None else store.message
            sm.store_failed(msg)
            return None, pose, store_req.observation.observation_id, sm.reason
        sm.store_ok()
        self._publish_feedback(
            goal_handle, STAGE_STORED, current_view, total_views,
            requested_yaw, measured_yaw,
        )
        return store, pose, store_req.observation.observation_id, ""

    # ── Helpers ───────────────────────────────────────────────────────────── #

    def _lookup_pose(self, image: Image, source_frame: str) -> Optional[PoseStamped]:
        """Look up a map-frame pose at the exact sensor timestamp."""
        stamp = rclpy.time.Time.from_msg(image.header.stamp)
        if stamp.nanoseconds == 0:
            self.get_logger().warn("Image timestamp is zero; using latest TF as fallback.")
            stamp = rclpy.time.Time()
        try:
            transform = self._tf_buffer.lookup_transform(
                self.get_parameter("map_frame").value,
                source_frame,
                stamp,
                timeout=Duration(seconds=float(self.get_parameter("tf_timeout_s").value)),
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as exc:
            self.get_logger().warn(f"TF2 lookup failed: {exc}")
            return None
        return _transform_to_pose_stamped(transform)

    def _lookup_current_pose(self, source_frame: str) -> Optional[PoseStamped]:
        try:
            transform = self._tf_buffer.lookup_transform(
                self.get_parameter("map_frame").value,
                source_frame,
                rclpy.time.Time(),
                timeout=Duration(
                    seconds=float(self.get_parameter("tf_timeout_s").value)
                ),
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as exc:
            self.get_logger().warn(f"Current TF2 lookup failed: {exc}")
            return None
        return _transform_to_pose_stamped(transform)

    def _rotate_to_yaw(
        self,
        goal_handle,
        requested_yaw: float,
        current_view: int,
        total_views: int,
    ) -> tuple[bool, str]:
        deadline = time.monotonic() + float(
            self.get_parameter("capture_rotation_timeout_s").value
        )
        tolerance = float(
            self.get_parameter("capture_angular_tolerance_rad").value
        )
        maximum_speed = float(
            self.get_parameter("capture_angular_speed_rad_s").value
        )
        try:
            while time.monotonic() < deadline:
                if goal_handle.is_cancel_requested:
                    return False, "cancelled"
                pose = self._lookup_current_pose(
                    self.get_parameter("base_frame").value
                )
                if pose is None:
                    time.sleep(0.05)
                    continue
                measured_yaw = _yaw_from_pose(pose)
                error = _wrap_angle(requested_yaw - measured_yaw)
                self._publish_feedback(
                    goal_handle, "rotating", current_view, total_views,
                    requested_yaw, measured_yaw,
                )
                if abs(error) <= tolerance:
                    return True, ""
                speed = min(maximum_speed, max(0.12, abs(error) * 1.5))
                command = TwistStamped()
                command.header.stamp = self.get_clock().now().to_msg()
                command.header.frame_id = self.get_parameter("base_frame").value
                command.twist.angular.z = copysign(speed, error)
                self._cmd_vel_pub.publish(command)
                time.sleep(0.05)
            return False, "rotation timeout"
        finally:
            self._publish_stop()

    def _publish_stop(self) -> None:
        command = TwistStamped()
        command.header.stamp = self.get_clock().now().to_msg()
        command.header.frame_id = self.get_parameter("base_frame").value
        self._cmd_vel_pub.publish(command)

    def _wait_stabilization(self, goal_handle) -> bool:
        deadline = time.monotonic() + float(
            self.get_parameter("capture_stabilization_s").value
        )
        while time.monotonic() < deadline:
            if goal_handle.is_cancel_requested:
                return False
            time.sleep(0.025)
        return True

    def _wait_for_image_after(self, not_before: float) -> bool:
        deadline = time.monotonic() + float(
            self.get_parameter("camera_stale_s").value
        )
        while time.monotonic() < deadline:
            image, received_at = self._snapshot_image()
            if image is not None and received_at > not_before:
                return True
            time.sleep(0.025)
        return False

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
    def _build_features_req(
        image: Image,
        depth: Image | None,
        camera_info: CameraInfo | None,
    ) -> GetVisualFeatures.Request:
        req = GetVisualFeatures.Request()
        req.image = image
        if depth is not None and camera_info is not None:
            req.depth_image = depth
            req.camera_info = camera_info
            req.use_depth = True
        return req

    def _build_store_req(
        self,
        waypoint_id: str,
        pose: PoseStamped,
        camera_pose: PoseStamped,
        image: Image,
        features: GetVisualFeatures.Response,
        goal,
        requested_yaw: float,
        depth: Image | None,
        depth_camera_pose: PoseStamped | None,
    ) -> StoreWaypoint.Request:
        req = StoreWaypoint.Request()
        req.node_id = waypoint_id
        req.scene_id = goal.scene_id.strip() or self.get_parameter("scene_id").value
        req.pose = pose
        req.navigation_goal_pose = pose
        req.visual_embedding = list(features.visual_embedding)
        req.detected_objects = list(features.detected_objects)
        req.configuration_hash = self.get_parameter("configuration_hash").value
        stamp = image.header.stamp
        req.observation.observation_id = (
            f"{req.scene_id}_{stamp.sec}_{stamp.nanosec}"
        )
        req.observation.image_path = self._save_observation_image(
            image, req.scene_id, req.observation.observation_id
        )
        req.observation.node_id = waypoint_id
        req.observation.timestamp = stamp
        req.observation.camera_frame = image.header.frame_id or self.get_parameter(
            "camera_frame"
        ).value
        req.observation.camera_pose = camera_pose
        if depth is not None and depth_camera_pose is not None:
            req.observation.depth_camera_frame = depth.header.frame_id
            req.observation.depth_camera_pose = depth_camera_pose
        req.observation.requested_yaw = float(requested_yaw)
        req.observation.measured_yaw = float(_yaw_from_pose(pose))
        req.observation.angular_error = float(
            atan2(
                sin(req.observation.measured_yaw - requested_yaw),
                cos(req.observation.measured_yaw - requested_yaw),
            )
        )
        req.observation.image_valid = True
        req.observation.depth_valid = depth is not None and depth_camera_pose is not None
        req.observation.image_embedding = list(features.visual_embedding)
        req.observation.detections = list(features.detections)
        req.observation.relations = list(features.relations)
        for relation in req.observation.relations:
            relation.source_observation_id = req.observation.observation_id
        return req

    def _save_observation_image(
        self, image: Image, scene_id: str, observation_id: str
    ) -> str:
        if self._image_bridge is None:
            self.get_logger().warn(
                "cv_bridge/OpenCV unavailable; observation image not persisted"
            )
            return ""
        root = os.path.expanduser(os.path.expandvars(
            self.get_parameter("observations_dir").value
        )).format(scene_id=scene_id)
        try:
            os.makedirs(root, exist_ok=True)
            path = os.path.join(root, f"{observation_id}.png")
            frame = self._image_bridge.imgmsg_to_cv2(
                image, desired_encoding="bgr8"
            )
            if not cv2.imwrite(path, frame):
                raise OSError("cv2.imwrite returned false")
            return path
        except (OSError, ValueError, RuntimeError) as exc:
            self.get_logger().error(f"Observation image write failed: {exc}")
            return ""

    def _publish_feedback(
        self,
        goal_handle,
        stage: str,
        current_view: int = 0,
        total_views: int = 0,
        requested_yaw: float = 0.0,
        measured_yaw: float = 0.0,
    ) -> None:
        fb = CaptureWaypoint.Feedback()
        fb.stage = stage
        fb.current_view = current_view
        fb.total_views = total_views
        fb.requested_yaw = requested_yaw
        fb.measured_yaw = measured_yaw
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


def _yaw_from_pose(pose: PoseStamped) -> float:
    q = pose.pose.orientation
    return atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _wrap_angle(angle: float) -> float:
    return atan2(sin(angle), cos(angle))


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
        try:
            executor.shutdown(timeout_sec=2.0)
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        finally:
            rclpy.try_shutdown()


if __name__ == "__main__":
    main()
