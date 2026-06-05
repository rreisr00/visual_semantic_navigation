#!/usr/bin/env python3
"""Lifecycle manager / supervisor for the semantic navigation stack.

Drives a configurable set of lifecycle nodes (``visual_encoder``,
``knowledge_graph_bridge`` by default) toward the ``active`` state and keeps them
there. A periodic reconciliation loop reads each node's current state and issues
the next transition needed:

    unconfigured → (configure) → inactive → (activate) → active

Because it reconciles on every tick, this also recovers nodes that drop back to
``inactive`` (e.g. a ``visual_encoder`` that deactivated itself after repeated
CUDA OOM): the next tick re-activates — or, if the node went unconfigured, fully
reconfigures it.

This is a deliberately small alternative to ``nav2_lifecycle_manager`` to keep
the stack self-contained.
"""

from __future__ import annotations

import threading

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from lifecycle_msgs.msg import State, Transition
from lifecycle_msgs.srv import ChangeState, GetState

SERVICE_TIMEOUT_SEC = 5.0


class LifecycleManagerNode(Node):
    """Periodically reconciles managed lifecycle nodes to the active state."""

    def __init__(self) -> None:
        super().__init__("lifecycle_manager")

        self.declare_parameter(
            "managed_nodes", ["visual_encoder", "knowledge_graph_bridge"]
        )
        self.declare_parameter("reconcile_period_sec", 2.0)

        self._managed = list(self.get_parameter("managed_nodes").value)
        period = float(self.get_parameter("reconcile_period_sec").value)

        self._client_cbg = MutuallyExclusiveCallbackGroup()
        self._timer_cbg = ReentrantCallbackGroup()

        # One (get_state, change_state) client pair per managed node.
        self._get_clients = {}
        self._change_clients = {}
        for name in self._managed:
            self._get_clients[name] = self.create_client(
                GetState, f"{name}/get_state", callback_group=self._client_cbg
            )
            self._change_clients[name] = self.create_client(
                ChangeState, f"{name}/change_state", callback_group=self._client_cbg
            )

        self.create_timer(period, self._reconcile, callback_group=self._timer_cbg)
        self.get_logger().info(
            f"Lifecycle manager supervising: {', '.join(self._managed)}"
        )

    # ── Reconciliation ────────────────────────────────────────────────────── #

    def _reconcile(self) -> None:
        for name in self._managed:
            try:
                self._reconcile_node(name)
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(f"Reconcile of '{name}' failed: {exc}")

    def _reconcile_node(self, name: str) -> None:
        state = self._get_state(name)
        if state is None:
            return  # node not up yet; try again next tick

        if state == State.PRIMARY_STATE_UNCONFIGURED:
            self.get_logger().info(f"'{name}' unconfigured → configure")
            self._change_state(name, Transition.TRANSITION_CONFIGURE)
        elif state == State.PRIMARY_STATE_INACTIVE:
            self.get_logger().info(f"'{name}' inactive → activate")
            self._change_state(name, Transition.TRANSITION_ACTIVATE)
        elif state == State.PRIMARY_STATE_ACTIVE:
            pass  # desired state
        elif state == State.PRIMARY_STATE_FINALIZED:
            self.get_logger().warn(f"'{name}' finalized — cannot recover.")

    # ── Lifecycle service calls ───────────────────────────────────────────── #

    def _get_state(self, name: str) -> int | None:
        client = self._get_clients[name]
        if not client.service_is_ready():
            return None
        res = self._wait(client.call_async(GetState.Request()))
        return res.current_state.id if res is not None else None

    def _change_state(self, name: str, transition_id: int) -> bool:
        client = self._change_clients[name]
        if not client.service_is_ready():
            return False
        req = ChangeState.Request()
        req.transition.id = transition_id
        res = self._wait(client.call_async(req))
        return bool(res and res.success)

    def _wait(self, future, timeout: float = SERVICE_TIMEOUT_SEC):
        done = threading.Event()
        future.add_done_callback(lambda _f: done.set())
        if not done.wait(timeout):
            return None
        try:
            return future.result()
        except Exception:  # noqa: BLE001
            return None


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LifecycleManagerNode()
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
