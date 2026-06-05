"""Pure capture state machine for the waypoint-teaching workflow.

The ROS node owns the I/O (camera buffer, TF, service calls); this class owns
only the *sequencing* and partial-failure logic so it can be unit-tested with no
ROS graph running.

Happy path::

    IDLE --start--> WAITING_FEATURES --features_ok--> WAITING_STORE
         --store_ok--> DONE

Any ``*_failed`` event, or starting without an image/pose, transitions to
FAILED with a human-readable reason. The machine is single-shot: build one per
capture goal.
"""
from __future__ import annotations

from enum import Enum


class CaptureState(Enum):
    IDLE = "idle"
    WAITING_FEATURES = "waiting_features"
    WAITING_STORE = "waiting_store"
    DONE = "done"
    FAILED = "failed"


# Feedback stage strings (mirror the CaptureWaypoint.action feedback contract).
STAGE_GOT_IMAGE = "got_image"
STAGE_GOT_POSE = "got_pose"
STAGE_ENCODED = "encoded"
STAGE_STORED = "stored"


class CaptureStateMachine:
    """Single-shot sequencer for one waypoint capture."""

    def __init__(self) -> None:
        self._state = CaptureState.IDLE
        self._reason: str = ""

    # ── inspection ──────────────────────────────────────────────────────── #

    @property
    def state(self) -> CaptureState:
        return self._state

    @property
    def reason(self) -> str:
        return self._reason

    @property
    def is_terminal(self) -> bool:
        return self._state in (CaptureState.DONE, CaptureState.FAILED)

    @property
    def succeeded(self) -> bool:
        return self._state == CaptureState.DONE

    # ── transitions ─────────────────────────────────────────────────────── #

    def start(self, has_image: bool, has_pose: bool) -> CaptureState:
        """Begin a capture given whether image and pose are available."""
        if self._state != CaptureState.IDLE:
            return self._fail(f"start() called from {self._state.value}")
        if not has_image:
            return self._fail("no image in buffer")
        if not has_pose:
            return self._fail("TF pose unavailable")
        self._state = CaptureState.WAITING_FEATURES
        return self._state

    def features_ok(self) -> CaptureState:
        """Visual encoder returned an embedding + objects."""
        if self._state != CaptureState.WAITING_FEATURES:
            return self._fail(f"features_ok() from {self._state.value}")
        self._state = CaptureState.WAITING_STORE
        return self._state

    def features_failed(self, message: str) -> CaptureState:
        return self._fail(f"feature extraction failed: {message}")

    def store_ok(self) -> CaptureState:
        """Knowledge-graph bridge persisted the waypoint."""
        if self._state != CaptureState.WAITING_STORE:
            return self._fail(f"store_ok() from {self._state.value}")
        self._state = CaptureState.DONE
        return self._state

    def store_failed(self, message: str) -> CaptureState:
        return self._fail(f"store failed: {message}")

    def cancel(self) -> CaptureState:
        """External cancellation request."""
        if not self.is_terminal:
            self._fail("cancelled")
        return self._state

    # ── helpers ─────────────────────────────────────────────────────────── #

    def _fail(self, reason: str) -> CaptureState:
        self._state = CaptureState.FAILED
        self._reason = reason
        return self._state
