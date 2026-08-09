#!/usr/bin/env python3
"""Interactive room-zone manager (thin ROS 2 wrapper).

Defines rectangular "rooms" in the map frame from RViz clicks and registers
them in the knowledge graph as parent nodes:

  1. Type the room label in this terminal.
  2. Click the two OPPOSITE corners of the rectangle in RViz with the
     *Publish Point* tool (publishes /clicked_point).
  3. The room is sent to the bridge's /add_room service (which creates the
     type="room" node and links every waypoint inside via CONTAINS
     room->waypoint edges), appended to rooms.yaml, and drawn as a latched
     rectangle + label marker.

On startup every room already in rooms.yaml is re-added (idempotent), which
restores rooms after a wiped database and re-sweeps existing waypoints.

Threading follows teleop_capture: rclpy spins in a background daemon thread;
the blocking stdin loop runs in the main thread — run with ``ros2 run`` in
its own terminal (never from a launch file: no tty).
"""
from __future__ import annotations

import queue
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile

from geometry_msgs.msg import Point, PointStamped
from visualization_msgs.msg import Marker, MarkerArray

from semantic_interfaces.srv import AddRoom

from semantic_navigation_core.rooms import Room, load_rooms, save_rooms


class RoomManagerNode(Node):
    """Collects clicked corners, calls /add_room, persists and draws rooms."""

    def __init__(self) -> None:
        super().__init__("room_manager")

        self.declare_parameter("rooms_file", "~/.ros/semantic_maps/rooms.yaml")
        self.declare_parameter("clicked_point_topic", "/clicked_point")
        self.declare_parameter("add_room_service", "add_room")
        self.declare_parameter("marker_topic", "room_markers")
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("line_width", 0.06)
        self.declare_parameter("text_scale", 0.35)
        self.declare_parameter("rect_color", [0.9, 0.6, 0.1, 1.0])
        self.declare_parameter("text_color", [1.0, 1.0, 1.0, 1.0])
        self.declare_parameter("service_wait_timeout_s", 10.0)
        self.declare_parameter("click_timeout_s", 120.0)

        self.rooms_file = self.get_parameter("rooms_file").value
        self._frame_id = self.get_parameter("frame_id").value
        self._service_wait = float(self.get_parameter("service_wait_timeout_s").value)
        self.click_timeout = float(self.get_parameter("click_timeout_s").value)

        self._clicks: queue.Queue[PointStamped] = queue.Queue()
        self.create_subscription(
            PointStamped,
            self.get_parameter("clicked_point_topic").value,
            self._clicks.put,
            10,
        )
        self._add_room_client = self.create_client(
            AddRoom, self.get_parameter("add_room_service").value
        )
        # Latched so RViz clients that connect later still get the rectangles.
        self._marker_pub = self.create_publisher(
            MarkerArray,
            self.get_parameter("marker_topic").value,
            QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL),
        )

        self.rooms: list[Room] = load_rooms(self.rooms_file)
        self.get_logger().info(
            f"Room manager ready — {len(self.rooms)} room(s) in '{self.rooms_file}'."
        )

    # ── Service call (executor spins in another thread) ─────────────────── #

    def call_add_room(self, room: Room) -> tuple[bool, str, int]:
        if not self._add_room_client.wait_for_service(timeout_sec=self._service_wait):
            return False, "add_room service unavailable", 0
        req = AddRoom.Request()
        req.room_id = room.room_id
        req.min_x = float(room.min_x)
        req.min_y = float(room.min_y)
        req.max_x = float(room.max_x)
        req.max_y = float(room.max_y)
        req.transition_width_m = float(room.transition_width_m)

        done = threading.Event()
        future = self._add_room_client.call_async(req)
        future.add_done_callback(lambda _f: done.set())
        if not done.wait(timeout=self._service_wait):
            return False, "add_room call timed out", 0
        try:
            res = future.result()
        except Exception as exc:  # noqa: BLE001
            return False, str(exc), 0
        return res.success, res.message, res.waypoints_assigned

    def register_rooms(self, rooms: list[Room]) -> None:
        """Idempotent re-add (startup: restores DB rooms + re-sweeps)."""
        for room in rooms:
            ok, msg, assigned = self.call_add_room(room)
            if ok:
                self.get_logger().info(
                    f"Room '{room.room_id}' registered ({assigned} waypoint(s))."
                )
            else:
                self.get_logger().error(f"Room '{room.room_id}' failed: {msg}")

    # ── Clicks ───────────────────────────────────────────────────────────── #

    def drain_clicks(self) -> None:
        while True:
            try:
                self._clicks.get_nowait()
            except queue.Empty:
                return

    def next_click(self) -> PointStamped | None:
        try:
            return self._clicks.get(timeout=self.click_timeout)
        except queue.Empty:
            return None

    # ── Persistence + markers ───────────────────────────────────────────── #

    def upsert_room(self, room: Room) -> None:
        self.rooms = [r for r in self.rooms if r.room_id != room.room_id] + [room]
        save_rooms(self.rooms_file, self.rooms)
        self.publish_markers()

    def publish_markers(self) -> None:
        line_width = float(self.get_parameter("line_width").value)
        text_scale = float(self.get_parameter("text_scale").value)
        rect_rgba = [float(v) for v in self.get_parameter("rect_color").value]
        text_rgba = [float(v) for v in self.get_parameter("text_color").value]

        arr = MarkerArray()
        wipe = Marker()
        wipe.action = Marker.DELETEALL
        arr.markers.append(wipe)

        stamp = self.get_clock().now().to_msg()
        for i, room in enumerate(self.rooms):
            rect = Marker()
            rect.header.frame_id = self._frame_id
            rect.header.stamp = stamp
            rect.ns = "rooms/rects"
            rect.id = i
            rect.type = Marker.LINE_STRIP
            rect.action = Marker.ADD
            rect.scale.x = line_width
            rect.color.r, rect.color.g, rect.color.b, rect.color.a = rect_rgba
            corners = room.corners()
            for cx, cy in corners + [corners[0]]:  # closed loop
                p = Point()
                p.x, p.y, p.z = cx, cy, 0.05
                rect.points.append(p)
            arr.markers.append(rect)

            text = Marker()
            text.header.frame_id = self._frame_id
            text.header.stamp = stamp
            text.ns = "rooms/labels"
            text.id = i
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.text = room.room_id
            cx, cy = room.center
            text.pose.position.x = cx
            text.pose.position.y = cy
            text.pose.position.z = 0.3
            text.scale.z = text_scale
            text.color.r, text.color.g, text.color.b, text.color.a = text_rgba
            arr.markers.append(text)

        self._marker_pub.publish(arr)


# ── Interactive loop (main thread) ───────────────────────────────────────── #

def _interactive_loop(node: RoomManagerNode) -> None:
    print("Room manager — define salas con 2 clicks de 'Publish Point' en RViz.")
    print("Escribe el nombre de la sala y pulsa Enter (vacío o 'q' para salir).\n")
    while rclpy.ok():
        try:
            label = input("Nombre de la sala: ").strip()
        except EOFError:
            break
        if not label or label.lower() == "q":
            break

        node.drain_clicks()  # discard stray clicks from before this room
        print(f"  Haz click en 2 esquinas opuestas de '{label}' en RViz…")
        first = node.next_click()
        if first is None:
            print("  Timeout esperando el primer click — sala descartada.")
            continue
        print(f"  Esquina 1: ({first.point.x:.2f}, {first.point.y:.2f})")
        second = node.next_click()
        if second is None:
            print("  Timeout esperando el segundo click — sala descartada.")
            continue
        print(f"  Esquina 2: ({second.point.x:.2f}, {second.point.y:.2f})")

        room = Room(
            room_id=label,
            min_x=first.point.x,
            min_y=first.point.y,
            max_x=second.point.x,
            max_y=second.point.y,
        ).normalized()

        ok, msg, assigned = node.call_add_room(room)
        if ok:
            node.upsert_room(room)
            print(f"  ✓ Sala '{label}' registrada — {assigned} waypoint(s) dentro.\n")
        else:
            print(f"  ✗ Error registrando '{label}': {msg}\n")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RoomManagerNode()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    # Restore/re-sweep rooms from YAML before going interactive.
    node.register_rooms(node.rooms)
    node.publish_markers()

    try:
        _interactive_loop(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.try_shutdown()
        spin_thread.join(timeout=2.0)
        node.destroy_node()


if __name__ == "__main__":
    main()
