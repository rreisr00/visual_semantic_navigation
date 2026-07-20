#!/usr/bin/env python3
"""Evaluation campaign harness (thin ROS 2 wrapper).

Drives a suite of test cases against the semantic-navigation stack and exports a
CSV. All business logic (accuracy, aggregation, CSV schema, hardware sampling)
lives in ``semantic_evaluation.core`` — this node only orchestrates I/O.

Concurrency (anti-deadlock pattern)
-----------------------------------
* A ``MultiThreadedExecutor`` spins the node in a background daemon thread.
* The campaign loop runs in the **main** thread and waits on each future via a
  ``threading.Event`` armed from ``future.add_done_callback(...)``. We never call
  ``spin_until_future_complete`` on an executor that is already spinning.
* The action client and the snapshot service client live in **separate**
  callback groups so their responses can be delivered concurrently.

Per case
--------
read graph context (service) → send action goal (``decision_only`` configurable)
→ await result → sample hardware → annotate accuracy → accumulate. After the last
case the results are aggregated and written to CSV.

Everything (topic/service/action names, paths, periods, thresholds) is a ROS
parameter loaded from ``config/evaluation_params.yaml`` — nothing is hardcoded.
"""
from __future__ import annotations

import os
import json
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import rclpy
import yaml
from rclpy.action import ActionClient
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from semantic_interfaces.action import NavigateToSemanticGoal
from semantic_interfaces.srv import GetGraphSnapshot

from semantic_evaluation.core import (
    GraphContext,
    HardwareSampler,
    LatencyBreakdown,
    TestCaseResult,
    annotate_accuracy,
    build_room_map,
    write_csv,
)

# Optional: only needed for image-based test cases.
try:
    import cv2
    from cv_bridge import CvBridge

    _HAS_CV = True
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]
    CvBridge = None  # type: ignore[assignment]
    _HAS_CV = False


@dataclass
class _TestCase:
    case_id: str
    expected_node_id: str
    query_text: str = ""
    image_path: str = ""

    @property
    def use_image(self) -> bool:
        return bool(self.image_path)


class EvaluationCollectorNode(Node):
    """Active evaluation harness over the NavigateToSemanticGoal action."""

    def __init__(self) -> None:
        super().__init__("evaluation_collector")

        # ── Parameters (no hardcoded names / paths / periods) ─────────────── #
        self.declare_parameter("action_name", "navigate_to_semantic_goal")
        self.declare_parameter("snapshot_service_name", "get_graph_snapshot")
        self.declare_parameter("test_suite_path", "")
        self.declare_parameter(
            "output_dir", "~/visual_semantic_navigation/experiments/simulation/campaigns"
        )
        self.declare_parameter("campaign_id", "")
        self.declare_parameter("scene_id", "scene_unset")
        self.declare_parameter("run_id", "")
        self.declare_parameter("seed", 42)
        self.declare_parameter("method", "single_view_siglip")
        self.declare_parameter("start_pose_id", "start_pose_unset")
        self.declare_parameter("query_suite_id", "")
        self.declare_parameter("frozen_config_hash", "")
        self.declare_parameter("success_semantics", "")
        self.declare_parameter("decision_only", False)
        self.declare_parameter("room_separator", "_")
        self.declare_parameter("room_strategy", "strip_last")
        # "graph": room of a waypoint = its CONTAINS room->waypoint parent in
        # the knowledge graph (label heuristic as per-node fallback).
        # "label": legacy behavior, room derived from the node_id only.
        self.declare_parameter("room_source", "graph")
        self.declare_parameter("server_wait_timeout_s", 20.0)
        self.declare_parameter("service_timeout_s", 10.0)
        self.declare_parameter("goal_response_timeout_s", 15.0)
        self.declare_parameter("result_timeout_s", 330.0)
        self.declare_parameter("hardware_refresh_period_s", 0.5)

        self._action_name = self.get_parameter("action_name").value
        self._snapshot_name = self.get_parameter("snapshot_service_name").value
        self._test_suite_path = os.path.expanduser(
            self.get_parameter("test_suite_path").value
        )
        self._output_dir = os.path.expanduser(self.get_parameter("output_dir").value)
        self._scene_id = str(self.get_parameter("scene_id").value)
        self._method = str(self.get_parameter("method").value)
        self._campaign_id = str(self.get_parameter("campaign_id").value)
        self._run_id = str(self.get_parameter("run_id").value)
        self._seed = int(self.get_parameter("seed").value)
        self._start_pose_id = str(self.get_parameter("start_pose_id").value)
        self._query_suite_id = str(self.get_parameter("query_suite_id").value)
        self._frozen_config_hash = str(
            self.get_parameter("frozen_config_hash").value
        )
        self._success_semantics = str(
            self.get_parameter("success_semantics").value
        )
        self._decision_only = bool(self.get_parameter("decision_only").value)
        self._separator = self.get_parameter("room_separator").value
        self._strategy = self.get_parameter("room_strategy").value
        self._room_source = self.get_parameter("room_source").value
        # waypoint -> parent room, rebuilt from each graph snapshot.
        self._room_map: dict[str, str] | None = None
        self._server_wait_timeout = float(
            self.get_parameter("server_wait_timeout_s").value
        )
        self._service_timeout = float(self.get_parameter("service_timeout_s").value)
        self._goal_response_timeout = float(
            self.get_parameter("goal_response_timeout_s").value
        )
        self._result_timeout = float(self.get_parameter("result_timeout_s").value)

        # ── Separate callback groups for action vs snapshot service ───────── #
        self._action_cbg = MutuallyExclusiveCallbackGroup()
        self._snapshot_cbg = MutuallyExclusiveCallbackGroup()
        self._timer_cbg = MutuallyExclusiveCallbackGroup()

        self._action_client = ActionClient(
            self, NavigateToSemanticGoal, self._action_name,
            callback_group=self._action_cbg,
        )
        self._snapshot_client = self.create_client(
            GetGraphSnapshot, self._snapshot_name, callback_group=self._snapshot_cbg,
        )

        # ── Hardware sampling: timer keeps the CPU delta warm ─────────────── #
        self._hw = HardwareSampler()
        refresh_period = float(self.get_parameter("hardware_refresh_period_s").value)
        self.create_timer(
            refresh_period, self._hw.refresh, callback_group=self._timer_cbg
        )

        self._bridge = CvBridge() if _HAS_CV else None

        self.get_logger().info(
            f"Evaluation collector ready (action='{self._action_name}', "
            f"decision_only={self._decision_only}, psutil={self._hw.available})."
        )

    # ── Campaign (runs in the MAIN thread) ────────────────────────────────── #

    def run_campaign(self) -> str | None:
        """Execute every case, write the CSV, and return its path (or None)."""
        cases = self._load_test_suite()
        if not cases:
            self.get_logger().error(
                f"No test cases found in '{self._test_suite_path}'. Aborting."
            )
            return None

        if not self._action_client.wait_for_server(
            timeout_sec=self._server_wait_timeout
        ):
            self.get_logger().error(
                f"Action server '{self._action_name}' unavailable. Aborting."
            )
            return None

        results: list[TestCaseResult] = []
        for idx, case in enumerate(cases, start=1):
            self.get_logger().info(
                f"[{idx}/{len(cases)}] case '{case.case_id}' "
                f"(expected={case.expected_node_id})."
            )
            results.append(self._run_case(case))

        return self._export(results)

    def _run_case(self, case: _TestCase) -> TestCaseResult:
        graph = self._read_graph_context()
        goal = self._build_goal(case)
        if goal is None:
            return self._failed_result(case, graph, "could not build goal")

        outcome = self._send_goal_and_wait(goal)
        hardware = self._hw.sample()

        if outcome is None:
            return self._failed_result(case, graph, "no action result", hardware)

        result = TestCaseResult(
            case_id=case.case_id,
            query=case.image_path if case.use_image else case.query_text,
            query_kind="image" if case.use_image else "text",
            expected_node_id=case.expected_node_id,
            predicted_node_id=outcome.matched_node_id,
            success=bool(outcome.success),
            latency=LatencyBreakdown(
                visual_extraction_s=float(outcome.visual_extraction_s),
                retrieval_s=float(outcome.retrieval_s),
                navigation_s=float(outcome.navigation_s),
            ),
            hardware=hardware,
            graph=graph,
            score=float(outcome.score),
        )
        return annotate_accuracy(
            result, self._separator, self._strategy, self._room_map
        )

    # ── Goal construction / dispatch ──────────────────────────────────────── #

    def _build_goal(self, case: _TestCase):
        goal = NavigateToSemanticGoal.Goal()
        goal.decision_only = self._decision_only
        if case.use_image:
            image_msg = self._load_image(case.image_path)
            if image_msg is None:
                return None
            goal.use_image = True
            goal.query_image = image_msg
        else:
            goal.use_image = False
            goal.query_text = case.query_text
        return goal

    def _send_goal_and_wait(self, goal):
        """Send a goal and block the *main* thread on the result future."""
        send_future = self._action_client.send_goal_async(goal)
        goal_handle = self._await(send_future, self._goal_response_timeout)
        if goal_handle is None:
            self.get_logger().error("Goal response timed out.")
            return None
        if not goal_handle.accepted:
            self.get_logger().error("Goal rejected by server.")
            return None

        result_future = goal_handle.get_result_async()
        result_msg = self._await(result_future, self._result_timeout)
        if result_msg is None:
            self.get_logger().error("Result timed out.")
            return None
        return result_msg.result

    def _read_graph_context(self) -> GraphContext:
        if not self._snapshot_client.service_is_ready():
            self.get_logger().warn(
                f"Snapshot service '{self._snapshot_name}' not ready; "
                "graph context will be 0/0."
            )
            return GraphContext()
        future = self._snapshot_client.call_async(GetGraphSnapshot.Request())
        res = self._await(future, self._service_timeout)
        if res is None or not res.success:
            return GraphContext()
        if self._room_source == "graph":
            self._room_map = build_room_map(
                [(e.type, e.source_node, e.target_node) for e in res.edges],
                {w.node_id for w in res.waypoints},
            )
        return GraphContext(total_nodes=res.total_nodes, total_edges=res.total_edges)

    # ── Anti-deadlock future waiter (executor already spinning elsewhere) ──── #

    def _await(self, future, timeout: float):
        done = threading.Event()
        future.add_done_callback(lambda _f: done.set())
        if not done.wait(timeout):
            return None
        try:
            return future.result()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Future raised: {exc}")
            return None

    # ── I/O helpers ───────────────────────────────────────────────────────── #

    def _load_test_suite(self) -> list[_TestCase]:
        if not self._test_suite_path or not os.path.isfile(self._test_suite_path):
            return []
        with open(self._test_suite_path, "r") as fh:
            data = yaml.safe_load(fh) or {}
        raw_cases = data.get("cases", [])
        cases: list[_TestCase] = []
        for i, entry in enumerate(raw_cases):
            case_id = str(entry.get("case_id", f"case_{i:03d}"))
            cases.append(
                _TestCase(
                    case_id=case_id,
                    expected_node_id=str(entry.get("expected_node_id", "")),
                    query_text=str(entry.get("query_text", "")),
                    image_path=os.path.expanduser(str(entry.get("image_path", ""))),
                )
            )
        return cases

    def _load_image(self, path: str):
        if not _HAS_CV:
            self.get_logger().error(
                "Image case requested but cv_bridge/opencv are not installed."
            )
            return None
        if not os.path.isfile(path):
            self.get_logger().error(f"Image not found: '{path}'.")
            return None
        frame = cv2.imread(path)
        if frame is None:
            self.get_logger().error(f"Failed to read image: '{path}'.")
            return None
        return self._bridge.cv2_to_imgmsg(frame, encoding="bgr8")

    def _export(self, results: list[TestCaseResult]) -> str | None:
        timestamp = datetime.now(timezone.utc)
        stamp = timestamp.strftime("%Y%m%dT%H%M%SZ")
        run_id = self._run_id or f"run_{stamp}"
        campaign_id = self._campaign_id or f"{self._scene_id}_{self._method}"
        run_dir = os.path.join(self._output_dir, self._scene_id, run_id)
        try:
            os.makedirs(run_dir, exist_ok=False)
        except OSError as exc:
            self.get_logger().error(f"Cannot create output dir: {exc}")
            return None
        path = os.path.join(run_dir, "evaluation.csv")
        # room_map must reach write_csv: aggregate() re-annotates defensively
        # and would otherwise clobber graph-based room_correct values.
        write_csv(path, results, self._separator, self._strategy, self._room_map)
        query_suite_id = self._query_suite_id or os.path.basename(
            self._test_suite_path
        )
        metadata = {
            "campaign_id": campaign_id,
            "scene_id": self._scene_id,
            "run_id": run_id,
            "seed": self._seed,
            "method": self._method,
            "start_pose_id": self._start_pose_id,
            "query_suite_id": query_suite_id,
            "frozen_config_hash": self._frozen_config_hash,
            "git_commit": self._git_commit(),
            "timestamp": timestamp.isoformat(),
            "status": "complete",
        }
        if self._success_semantics:
            metadata["success_semantics"] = self._success_semantics
        with open(os.path.join(run_dir, "campaign.yaml"), "w", encoding="utf-8") as handle:
            yaml.safe_dump(metadata, handle, sort_keys=False)
        manifest = {
            **metadata,
            "decision_only": self._decision_only,
            "test_suite_path": self._test_suite_path,
            "n_cases": len(results),
            "room_source": self._room_source,
        }
        with open(os.path.join(run_dir, "manifest.json"), "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
        self.get_logger().info(f"Wrote {len(results)} results to '{path}'.")
        return path

    @staticmethod
    def _git_commit() -> str:
        try:
            completed = subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                timeout=5, check=False,
            )
            return completed.stdout.strip() if completed.returncode == 0 else "unknown"
        except (OSError, subprocess.TimeoutExpired):
            return "unknown"

    def _failed_result(
        self, case: _TestCase, graph: GraphContext, reason: str, hardware=None
    ) -> TestCaseResult:
        self.get_logger().warn(f"Case '{case.case_id}' failed: {reason}.")
        result = TestCaseResult(
            case_id=case.case_id,
            query=case.image_path if case.use_image else case.query_text,
            query_kind="image" if case.use_image else "text",
            expected_node_id=case.expected_node_id,
            predicted_node_id="",
            success=False,
            hardware=hardware if hardware is not None else self._hw.sample(),
            graph=graph,
        )
        return annotate_accuracy(
            result, self._separator, self._strategy, self._room_map
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = EvaluationCollectorNode()

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        node.run_campaign()
    except KeyboardInterrupt:
        pass
    finally:
        # Stop the background spin and let the daemon thread unwind *before* the
        # node/context are torn down, so rclpy does not abort mid-spin.
        executor.shutdown()
        spin_thread.join(timeout=5.0)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
