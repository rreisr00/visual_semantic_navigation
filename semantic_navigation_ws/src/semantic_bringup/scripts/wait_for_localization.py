#!/usr/bin/env python3
"""Gate Nav2 startup until timestamped sensor data is transformable in map."""

from __future__ import annotations

import sys
import time

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformListener


class LocalizationGate(Node):
    """Wait for consecutive scans with a complete map-to-base TF chain."""

    def __init__(self) -> None:
        super().__init__('localization_gate')
        self.declare_parameter('target_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('timeout_s', 90.0)
        self.declare_parameter('required_valid_scans', 3)

        self.target_frame = str(self.get_parameter('target_frame').value)
        self.base_frame = str(self.get_parameter('base_frame').value)
        self.required_valid_scans = max(
            1, int(self.get_parameter('required_valid_scans').value)
        )
        timeout_s = max(1.0, float(self.get_parameter('timeout_s').value))
        self.deadline = time.monotonic() + timeout_s
        self.valid_scans = 0
        self.ready = False
        self.timed_out = False

        self.buffer = Buffer(cache_time=Duration(seconds=30.0))
        self.listener = TransformListener(self.buffer, self)
        self.create_subscription(
            LaserScan,
            str(self.get_parameter('scan_topic').value),
            self._on_scan,
            qos_profile_sensor_data,
        )
        self.create_timer(0.5, self._check_timeout)
        self.get_logger().info(
            f'Waiting for {self.target_frame} -> {self.base_frame} at scan timestamps'
        )

    def _on_scan(self, message: LaserScan) -> None:
        stamp = Time.from_msg(message.header.stamp)
        transformable = self.buffer.can_transform(
            self.target_frame,
            self.base_frame,
            stamp,
            timeout=Duration(seconds=0.0),
        )
        self.valid_scans = self.valid_scans + 1 if transformable else 0
        if self.valid_scans >= self.required_valid_scans:
            self.ready = True
            self.get_logger().info(
                f'Localization ready after {self.valid_scans} transformable scans'
            )

    def _check_timeout(self) -> None:
        if time.monotonic() < self.deadline:
            return
        self.timed_out = True
        self.get_logger().error(
            f'Localization timeout: no usable {self.target_frame} -> '
            f'{self.base_frame} transform'
        )


def main(args=None) -> int:
    """Return success only once localization is usable by costmaps."""
    rclpy.init(args=args)
    node = LocalizationGate()
    try:
        while rclpy.ok() and not node.ready and not node.timed_out:
            rclpy.spin_once(node, timeout_sec=0.2)
        return 0 if node.ready else 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())
