#!/usr/bin/env python3
"""Knowledge-graph visualizer (thin ROS 2 wrapper).

Periodically requests a graph snapshot **asynchronously** (never blocking the
executor) and republishes it as a ``visualization_msgs/MarkerArray``:

* ``SPHERE`` per waypoint node,
* ``TEXT_VIEW_FACING`` with the node id,
* ``LINE_LIST`` for edges whose endpoints are both pose-bearing waypoints.

The publisher uses RELIABLE + TRANSIENT_LOCAL (latched) durability so an RViz
client that connects late still receives the most recent graph. Each cycle is
prefixed with a ``DELETEALL`` marker so stale waypoints never linger.

All names, periods, scales and colours are ROS parameters — nothing hardcoded.
"""
from __future__ import annotations

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from builtin_interfaces.msg import Duration
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from semantic_interfaces.srv import GetGraphSnapshot


class GraphVisualizerNode(Node):
    """Renders the knowledge graph as RViz markers."""

    def __init__(self) -> None:
        super().__init__("graph_visualizer")

        # ── Parameters ────────────────────────────────────────────────────── #
        self.declare_parameter("snapshot_service_name", "get_graph_snapshot")
        self.declare_parameter("marker_topic", "semantic_graph_markers")
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("namespace", "semantic_graph")
        self.declare_parameter("publish_period_s", 2.0)
        self.declare_parameter("sphere_scale", 0.3)
        self.declare_parameter("text_scale", 0.25)
        self.declare_parameter("text_z_offset", 0.4)
        self.declare_parameter("line_width", 0.05)
        self.declare_parameter("node_color", [0.1, 0.6, 1.0, 1.0])
        self.declare_parameter("text_color", [1.0, 1.0, 1.0, 1.0])
        self.declare_parameter("edge_color", [0.7, 0.7, 0.7, 0.8])

        self._snapshot_name = self.get_parameter("snapshot_service_name").value
        self._frame_id = self.get_parameter("frame_id").value
        self._ns = self.get_parameter("namespace").value
        self._sphere_scale = float(self.get_parameter("sphere_scale").value)
        self._text_scale = float(self.get_parameter("text_scale").value)
        self._text_z = float(self.get_parameter("text_z_offset").value)
        self._line_width = float(self.get_parameter("line_width").value)
        self._node_color = _to_color(self.get_parameter("node_color").value)
        self._text_color = _to_color(self.get_parameter("text_color").value)
        self._edge_color = _to_color(self.get_parameter("edge_color").value)

        # ── Latched publisher so late RViz subscribers get the last graph ──── #
        latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._marker_pub = self.create_publisher(
            MarkerArray, self.get_parameter("marker_topic").value, latched_qos
        )

        self._client = self.create_client(GetGraphSnapshot, self._snapshot_name)

        period = float(self.get_parameter("publish_period_s").value)
        self.create_timer(period, self._on_timer)

        self.get_logger().info(
            f"Graph visualizer ready (service='{self._snapshot_name}', "
            f"frame='{self._frame_id}')."
        )

    # ── Timer: fire an async snapshot request (non-blocking) ──────────────── #

    def _on_timer(self) -> None:
        if not self._client.service_is_ready():
            self.get_logger().warn(
                f"Snapshot service '{self._snapshot_name}' not ready.",
                throttle_duration_sec=5.0,
            )
            return
        future = self._client.call_async(GetGraphSnapshot.Request())
        future.add_done_callback(self._on_snapshot)

    def _on_snapshot(self, future) -> None:
        try:
            response = future.result()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Snapshot request failed: {exc}")
            return
        if response is None or not response.success:
            self.get_logger().warn("Snapshot response unsuccessful.")
            return
        self._marker_pub.publish(self._build_markers(response))

    # ── Marker construction ───────────────────────────────────────────────── #

    def _build_markers(self, snapshot) -> MarkerArray:
        markers = MarkerArray()

        # Always clear the previous frame first (no ghost waypoints).
        delete_all = Marker()
        delete_all.header.frame_id = self._frame_id
        delete_all.action = Marker.DELETEALL
        markers.markers.append(delete_all)

        positions: dict[str, tuple[float, float, float]] = {}
        marker_id = 0
        stamp = self.get_clock().now().to_msg()

        for wp in snapshot.waypoints:
            p = wp.pose.pose.position
            positions[wp.node_id] = (p.x, p.y, p.z)
            markers.markers.append(
                self._sphere(marker_id, stamp, p.x, p.y, p.z)
            )
            marker_id += 1
            markers.markers.append(
                self._text(marker_id, stamp, wp.node_id, p.x, p.y, p.z)
            )
            marker_id += 1

        edge_marker = self._edges(marker_id, stamp, snapshot.edges, positions)
        if edge_marker is not None:
            markers.markers.append(edge_marker)

        return markers

    def _sphere(self, marker_id, stamp, x, y, z) -> Marker:
        m = self._base(marker_id, stamp, "nodes")
        m.type = Marker.SPHERE
        m.pose.position.x, m.pose.position.y, m.pose.position.z = x, y, z
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = self._sphere_scale
        m.color = self._node_color
        return m

    def _text(self, marker_id, stamp, text, x, y, z) -> Marker:
        m = self._base(marker_id, stamp, "labels")
        m.type = Marker.TEXT_VIEW_FACING
        m.pose.position.x, m.pose.position.y = x, y
        m.pose.position.z = z + self._text_z
        m.pose.orientation.w = 1.0
        m.scale.z = self._text_scale
        m.color = self._text_color
        m.text = text
        return m

    def _edges(self, marker_id, stamp, edges, positions) -> Marker | None:
        m = self._base(marker_id, stamp, "edges")
        m.type = Marker.LINE_LIST
        m.pose.orientation.w = 1.0
        m.scale.x = self._line_width
        m.color = self._edge_color
        for edge in edges:
            src = positions.get(edge.source_node)
            tgt = positions.get(edge.target_node)
            if src is None or tgt is None:
                continue  # endpoint without a pose (e.g. object node) — skip line
            m.points.append(_point(*src))
            m.points.append(_point(*tgt))
        return m if m.points else None

    def _base(self, marker_id, stamp, ns_suffix) -> Marker:
        m = Marker()
        m.header.frame_id = self._frame_id
        m.header.stamp = stamp
        m.ns = f"{self._ns}/{ns_suffix}"
        m.id = marker_id
        m.action = Marker.ADD
        m.lifetime = Duration()  # 0 = forever (refreshed via DELETEALL each cycle)
        return m


# ── Module helpers ────────────────────────────────────────────────────────── #

def _to_color(values) -> ColorRGBA:
    r, g, b, a = (list(values) + [1.0, 1.0, 1.0, 1.0])[:4]
    return ColorRGBA(r=float(r), g=float(g), b=float(b), a=float(a))


def _point(x, y, z):
    from geometry_msgs.msg import Point

    return Point(x=float(x), y=float(y), z=float(z))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GraphVisualizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
