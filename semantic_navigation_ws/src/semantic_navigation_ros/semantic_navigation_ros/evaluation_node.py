#!/usr/bin/env python3
"""Evaluation Node (passive metric collector).

Subscribes to latency and result topics to accumulate metrics across the
teaching and retrieval phases, and exposes ``/save_evaluation_results``
(std_srvs/Trigger) which flushes a detailed CSV plus a summary row.

Subscriptions
-------------
/feature_extraction_latency (std_msgs/Float64) – per-call encoder latency
/retrieval_latency           (std_msgs/Float64) – total retrieval latency
/retrieval_result            (std_msgs/String)  – predicted waypoint node_id
/ground_truth_node           (std_msgs/String)  – ground-truth node_id

Graph size is read on demand through the ``/get_waypoints`` service (the
knowledge graph lives in the bridge process, not in-process here).
"""

import csv
import datetime
import pathlib
import threading

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from std_msgs.msg import Float64, String
from std_srvs.srv import Trigger

from semantic_interfaces.srv import GetWaypoints


class EvaluationNode(Node):
    """ROS 2 node for passive evaluation metric collection."""

    def __init__(self) -> None:
        super().__init__("evaluation_node")

        self._lock = threading.Lock()

        self._encoder_latencies: list[float] = []
        self._retrieval_latencies: list[float] = []
        self._predicted_nodes: list[str] = []
        self._ground_truth_nodes: list[str] = []
        self._last_encoder_lat: float = 0.0

        self._client_cbg = MutuallyExclusiveCallbackGroup()
        self._srv_cbg = ReentrantCallbackGroup()

        self._waypoints_client = self.create_client(
            GetWaypoints, "get_waypoints", callback_group=self._client_cbg
        )

        self.create_subscription(
            Float64, "/feature_extraction_latency", self._encoder_lat_callback, 10
        )
        self.create_subscription(
            Float64, "/retrieval_latency", self._retrieval_lat_callback, 10
        )
        self.create_subscription(
            String, "/retrieval_result", self._predicted_callback, 10
        )
        self.create_subscription(
            String, "/ground_truth_node", self._ground_truth_callback, 10
        )

        self.create_service(
            Trigger, "save_evaluation_results", self._handle_save,
            callback_group=self._srv_cbg,
        )

        self.get_logger().info("Evaluation node ready.")

    # ── Callbacks ─────────────────────────────────────────────────────────── #

    def _encoder_lat_callback(self, msg: Float64) -> None:
        with self._lock:
            self._last_encoder_lat = msg.data

    def _retrieval_lat_callback(self, msg: Float64) -> None:
        with self._lock:
            self._retrieval_latencies.append(msg.data)
            self._encoder_latencies.append(self._last_encoder_lat)

    def _predicted_callback(self, msg: String) -> None:
        with self._lock:
            self._predicted_nodes.append(msg.data)

    def _ground_truth_callback(self, msg: String) -> None:
        with self._lock:
            self._ground_truth_nodes.append(msg.data)

    # ── Save service ──────────────────────────────────────────────────────── #

    def _handle_save(
        self, _request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        try:
            path = self._save_results()
            response.success = True
            response.message = str(path)
            self.get_logger().info(f"Evaluation results saved to {path}")
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Save failed: {exc}")
            response.success = False
            response.message = str(exc)
        return response

    def _graph_size(self) -> int:
        """Number of waypoints via /get_waypoints, or -1 if unavailable."""
        if not self._waypoints_client.service_is_ready():
            return -1
        req = GetWaypoints.Request()
        req.class_filter = "waypoint"
        future = self._waypoints_client.call_async(req)
        done = threading.Event()
        future.add_done_callback(lambda _f: done.set())
        if not done.wait(5.0):
            return -1
        res = future.result()
        return len(res.waypoints) if res and res.success else -1

    def _save_results(self) -> pathlib.Path:
        with self._lock:
            enc_lats = list(self._encoder_latencies)
            ret_lats = list(self._retrieval_latencies)
            pred = list(self._predicted_nodes)
            gt = list(self._ground_truth_nodes)

        n_ret = len(ret_lats)
        avg_enc = sum(enc_lats) / len(enc_lats) if enc_lats else 0.0
        avg_ret = sum(ret_lats) / n_ret if n_ret else 0.0
        avg_kg = avg_ret - avg_enc

        n_pairs = min(len(pred), len(gt))
        top1 = (
            sum(1 for p, g in zip(pred[:n_pairs], gt[:n_pairs]) if p == g) / n_pairs
            if n_pairs
            else 0.0
        )
        room_acc = (
            sum(
                1
                for p, g in zip(pred[:n_pairs], gt[:n_pairs])
                if self._same_room(p, g)
            )
            / n_pairs
            if n_pairs
            else 0.0
        )

        n_nodes = self._graph_size()

        out_dir = pathlib.Path.home() / "ros2_evaluation_results"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = out_dir / f"results_{ts}.csv"

        fieldnames = [
            "timestamp",
            "encoder_lat_s",
            "total_lat_s",
            "kg_query_lat_s",
            "predicted_node",
            "ground_truth_node",
        ]
        n_rows = max(n_ret, len(pred), len(gt))

        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(n_rows):
                enc_l = enc_lats[i] if i < len(enc_lats) else ""
                ret_l = ret_lats[i] if i < len(ret_lats) else ""
                kg_l = (
                    (ret_l - enc_l)
                    if isinstance(enc_l, float) and isinstance(ret_l, float)
                    else ""
                )
                writer.writerow(
                    {
                        "timestamp": ts,
                        "encoder_lat_s": enc_l,
                        "total_lat_s": ret_l,
                        "kg_query_lat_s": kg_l,
                        "predicted_node": pred[i] if i < len(pred) else "",
                        "ground_truth_node": gt[i] if i < len(gt) else "",
                    }
                )
            writer.writerow(
                {
                    "timestamp": "SUMMARY",
                    "encoder_lat_s": f"avg={avg_enc:.4f}",
                    "total_lat_s": f"avg={avg_ret:.4f}",
                    "kg_query_lat_s": f"avg={avg_kg:.4f}",
                    "predicted_node": (
                        f"top1={top1:.3f} room_acc={room_acc:.3f} n_nodes={n_nodes}"
                    ),
                    "ground_truth_node": f"n_pairs={n_pairs}",
                }
            )

        self.get_logger().info(
            f"Metrics — avg_enc={avg_enc:.3f}s avg_ret={avg_ret:.3f}s "
            f"avg_kg={avg_kg:.3f}s top1={top1:.3f} room_acc={room_acc:.3f} "
            f"nodes={n_nodes}"
        )
        return csv_path

    # ── Utilities ─────────────────────────────────────────────────────────── #

    @staticmethod
    def _same_room(node_a: str, node_b: str) -> bool:
        prefix_a = node_a.split("_")[0] if "_" in node_a else node_a
        prefix_b = node_b.split("_")[0] if "_" in node_b else node_b
        return prefix_a == prefix_b

    def destroy_node(self) -> None:
        if self._retrieval_latencies or self._predicted_nodes:
            try:
                self._save_results()
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(f"Auto-save on shutdown failed: {exc}")
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = EvaluationNode()
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
