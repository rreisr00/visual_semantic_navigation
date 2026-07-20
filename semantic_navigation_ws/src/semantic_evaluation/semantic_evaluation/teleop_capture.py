#!/usr/bin/env python3
"""Keyboard teleop + waypoint capture (thin ROS 2 wrapper).

Drives the robot from the keyboard, buffers the latest camera frame, and on the
capture key (1) saves the frame to ``dataset_dir/<label>_<ts>.png`` and (2) fires
the real ``CaptureWaypoint`` action on kg_manager — fully decoupled from how the
graph stores it.

Threading
---------
``rclpy`` spins in a background **daemon** thread; the blocking ``termios``
keyboard read runs in the **main** thread.

Camera frames arrive with ``qos_profile_sensor_data`` (BEST_EFFORT / KEEP_LAST)
to avoid saturating the network.

Topic / action names, dataset path, speeds and the capture label are ROS
parameters — nothing hardcoded. (Key bindings are UI constants.)

Controls:  w/s forward/back · a/d turn · space stop · c capture · q quit
"""
from __future__ import annotations

import os
import sys
import threading
import time
from typing import Optional

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import Twist, TwistStamped
from sensor_msgs.msg import Image

from semantic_interfaces.action import CaptureWaypoint

try:
    import termios
    import tty

    _HAS_TERMIOS = True
except ImportError:  # pragma: no cover - non-POSIX
    termios = None  # type: ignore[assignment]
    tty = None  # type: ignore[assignment]
    _HAS_TERMIOS = False

try:
    import cv2
    from cv_bridge import CvBridge

    _HAS_CV = True
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]
    CvBridge = None  # type: ignore[assignment]
    _HAS_CV = False

# Key bindings (UI constants, not configuration).
KEY_FORWARD = "w"
KEY_BACK = "s"
KEY_LEFT = "a"
KEY_RIGHT = "d"
KEY_STOP = " "
KEY_CAPTURE = "c"
KEY_QUIT = "q"


class TeleopCaptureNode(Node):
    """Publishes cmd_vel, buffers camera frames, triggers CaptureWaypoint."""

    def __init__(self) -> None:
        super().__init__("teleop_capture")

        self.declare_parameter("cmd_vel_topic", "cmd_vel")
        self.declare_parameter("camera_topic", "/camera/image_raw")
        self.declare_parameter("capture_action_name", "capture_waypoint")
        self.declare_parameter("dataset_dir", "~/semantic_dataset")
        # Empty label → the knowledge-graph bridge auto-names the waypoint
        # after the room containing the robot's pose ("<room>_<NN>").
        self.declare_parameter("capture_label", "")
        self.declare_parameter("linear_speed", 0.4)
        self.declare_parameter("angular_speed", 0.8)
        self.declare_parameter("server_wait_timeout_s", 5.0)
        # Jazzy convention: the gz bridge and Nav2 (enable_stamped_cmd_vel)
        # expect geometry_msgs/TwistStamped on cmd_vel. A plain Twist publisher
        # is type-incompatible and the robot silently ignores the teleop.
        self.declare_parameter("stamped_cmd_vel", True)

        self._linear = float(self.get_parameter("linear_speed").value)
        self._angular = float(self.get_parameter("angular_speed").value)
        self._dataset_dir = os.path.expanduser(
            self.get_parameter("dataset_dir").value
        )
        self._label = self.get_parameter("capture_label").value
        self._server_wait = float(self.get_parameter("server_wait_timeout_s").value)

        self._stamped = bool(self.get_parameter("stamped_cmd_vel").value)
        self._cmd_pub = self.create_publisher(
            TwistStamped if self._stamped else Twist,
            self.get_parameter("cmd_vel_topic").value,
            10,
        )
        self.create_subscription(
            Image,
            self.get_parameter("camera_topic").value,
            self._image_callback,
            qos_profile_sensor_data,
        )
        self._capture_client = ActionClient(
            self, CaptureWaypoint,
            self.get_parameter("capture_action_name").value,
        )

        self._image_lock = threading.Lock()
        self._latest_image: Optional[Image] = None
        self._bridge = CvBridge() if _HAS_CV else None

        os.makedirs(self._dataset_dir, exist_ok=True)
        self.get_logger().info(
            f"Teleop ready. dataset='{self._dataset_dir}', label='{self._label}'."
        )

    # ── Camera buffer ─────────────────────────────────────────────────────── #

    def _image_callback(self, msg: Image) -> None:
        with self._image_lock:
            self._latest_image = msg

    def _snapshot(self) -> Optional[Image]:
        with self._image_lock:
            return self._latest_image

    # ── Motion ────────────────────────────────────────────────────────────── #

    def publish_twist(self, linear: float, angular: float) -> None:
        twist = Twist()
        twist.linear.x = linear
        twist.angular.z = angular
        if self._stamped:
            msg = TwistStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "base_link"
            msg.twist = twist
            self._cmd_pub.publish(msg)
        else:
            self._cmd_pub.publish(twist)

    def stop(self) -> None:
        self.publish_twist(0.0, 0.0)

    def handle_key(self, key: str) -> None:
        if key == KEY_FORWARD:
            self.publish_twist(self._linear, 0.0)
        elif key == KEY_BACK:
            self.publish_twist(-self._linear, 0.0)
        elif key == KEY_LEFT:
            self.publish_twist(0.0, self._angular)
        elif key == KEY_RIGHT:
            self.publish_twist(0.0, -self._angular)
        elif key == KEY_STOP:
            self.stop()
        elif key == KEY_CAPTURE:
            self.capture()

    # ── Capture ───────────────────────────────────────────────────────────── #

    def capture(self) -> None:
        image = self._snapshot()
        if image is None:
            self.get_logger().warn("No camera frame yet; capture skipped.")
            return

        ts = time.strftime("%Y%m%d_%H%M%S")
        self._save_frame(image, ts)
        # Empty label lets the bridge name the waypoint after its room.
        self._trigger_capture_action(f"{self._label}_{ts}" if self._label else "")

    def _save_frame(self, image: Image, ts: str) -> None:
        if not _HAS_CV:
            self.get_logger().warn("cv_bridge/opencv missing; frame not saved.")
            return
        try:
            frame = self._bridge.imgmsg_to_cv2(image, desired_encoding="bgr8")
            prefix = self._label or "frame"
            path = os.path.join(self._dataset_dir, f"{prefix}_{ts}.png")
            cv2.imwrite(path, frame)
            self.get_logger().info(f"Saved frame → '{path}'.")
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Failed to save frame: {exc}")

    def _trigger_capture_action(self, label: str) -> None:
        if not self._capture_client.wait_for_server(timeout_sec=self._server_wait):
            self.get_logger().error("CaptureWaypoint action server unavailable.")
            return
        goal = CaptureWaypoint.Goal()
        goal.label = label
        future = self._capture_client.send_goal_async(goal)
        future.add_done_callback(self._on_capture_goal)
        self.get_logger().info(f"Capture goal sent (label='{label}').")

    def _on_capture_goal(self, future) -> None:
        try:
            handle = future.result()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Capture goal failed: {exc}")
            return
        if not handle.accepted:
            self.get_logger().error("Capture goal rejected.")
            return
        handle.get_result_async().add_done_callback(self._on_capture_result)

    def _on_capture_result(self, future) -> None:
        try:
            result = future.result().result
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Capture result failed: {exc}")
            return
        if result.success:
            self.get_logger().info(f"Captured waypoint '{result.node_id}'.")
        else:
            self.get_logger().error(f"Capture failed: {result.message}")


# ── Keyboard loop (main thread) ───────────────────────────────────────────── #

def _read_key() -> str:
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _keyboard_loop(node: TeleopCaptureNode) -> None:
    print(
        "Teleop: w/s = fwd/back, a/d = turn, space = stop, "
        "c = capture, q = quit"
    )
    while rclpy.ok():
        key = _read_key()
        if key == KEY_QUIT or key == "\x03":  # q or Ctrl-C
            node.stop()
            break
        node.handle_key(key)


def main(args=None) -> None:
    if not _HAS_TERMIOS:
        print("teleop_capture requires a POSIX terminal (termios).", file=sys.stderr)
        return

    rclpy.init(args=args)
    node = TeleopCaptureNode()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        _keyboard_loop(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        # Shut the context down first so rclpy.spin() in the daemon returns,
        # then join before destroying the node.
        rclpy.try_shutdown()
        spin_thread.join(timeout=2.0)
        node.destroy_node()


if __name__ == "__main__":
    main()
