#!/usr/bin/env python3
"""Voice command node (thin ROS 2 wrapper over semantic_voice.core).

Listens for spoken English commands, transcribes them with faster-whisper,
parses them with the rules+regex intent parser, and dispatches:

  MoveToPosition -> nav2_msgs/NavigateToPose        (map-frame coordinates)
  SaveWaypoint   -> semantic_interfaces/CaptureWaypoint (label = node_id)
  SemanticGoal   -> semantic_interfaces/NavigateToSemanticGoal (query_text)

Activation modes (``activation_mode`` parameter):
  text          type commands on stdin — no mic/GPU needed, ideal for testing
  push_to_talk  press the PTT key to start recording, press again to stop
                (raw terminals cannot see key-release, so it is toggle-to-talk)
  vad           hands-free: RMS speech segmenter chops the mic stream into
                utterances

Threading follows teleop_capture: ``rclpy`` spins in a background daemon
thread; the blocking interaction loop (stdin / mic) runs in the main thread.
One command is in flight at a time — utterances heard while busy are dropped.

faster-whisper/sounddevice live in the project ML venv; run via
``ros2 launch semantic_voice voice.launch.py`` (injects PYTHONPATH) or prefix
PYTHONPATH manually when using ``ros2 run`` (needed for stdin modes).
"""
from __future__ import annotations

import sys
import threading

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose

from semantic_interfaces.action import CaptureWaypoint, NavigateToSemanticGoal

from semantic_voice.core import (
    MoveToPosition,
    NoIntent,
    SaveWaypoint,
    SemanticGoal,
    SpeechSegmenter,
    parse,
)

try:
    import termios
    import tty

    _HAS_TERMIOS = True
except ImportError:  # pragma: no cover - non-POSIX
    termios = None  # type: ignore[assignment]
    tty = None  # type: ignore[assignment]
    _HAS_TERMIOS = False

KEY_QUIT = "q"


class VoiceCommandNode(Node):
    """Transcribes voice commands and dispatches them as action goals."""

    def __init__(self) -> None:
        super().__init__("voice_command")

        self.declare_parameter("activation_mode", "push_to_talk")
        self.declare_parameter("model_size", "small")
        self.declare_parameter("device", "cuda")
        self.declare_parameter("compute_type", "int8_float16")
        self.declare_parameter("language", "en")
        self.declare_parameter("mic_device", "")
        self.declare_parameter("sample_rate", 16000)
        self.declare_parameter("vad_rms_threshold", 0.015)
        self.declare_parameter("vad_hangover_s", 0.8)
        self.declare_parameter("vad_max_utterance_s", 12.0)
        self.declare_parameter("ptt_key", "r")
        self.declare_parameter("goal_frame_id", "map")
        self.declare_parameter("navigate_action_name", "navigate_to_pose")
        self.declare_parameter("capture_action_name", "capture_waypoint")
        self.declare_parameter("semantic_action_name", "navigate_to_semantic_goal")
        self.declare_parameter("server_wait_timeout_s", 5.0)

        self.activation_mode = self.get_parameter("activation_mode").value
        self.sample_rate = int(self.get_parameter("sample_rate").value)
        self.ptt_key = str(self.get_parameter("ptt_key").value)[:1] or "r"
        self._frame_id = self.get_parameter("goal_frame_id").value
        self._server_wait = float(self.get_parameter("server_wait_timeout_s").value)

        self._nav_client = ActionClient(
            self, NavigateToPose,
            self.get_parameter("navigate_action_name").value,
        )
        self._capture_client = ActionClient(
            self, CaptureWaypoint,
            self.get_parameter("capture_action_name").value,
        )
        self._semantic_client = ActionClient(
            self, NavigateToSemanticGoal,
            self.get_parameter("semantic_action_name").value,
        )

        self._busy = threading.Event()

        # Audio/ASR are only needed for the mic modes: text mode must work
        # without the venv, portaudio or a GPU.
        self._transcriber = None
        self._mic = None
        self._segmenter = None
        if self.activation_mode in ("push_to_talk", "vad"):
            self._init_audio()

        self.get_logger().info(
            f"Voice command ready (mode='{self.activation_mode}')."
        )

    def _init_audio(self) -> None:
        from semantic_voice.core.audio_capture import MicrophoneCapture
        from semantic_voice.core.transcriber import WhisperTranscriber

        self._transcriber = WhisperTranscriber(
            model_size=self.get_parameter("model_size").value,
            device=self.get_parameter("device").value,
            compute_type=self.get_parameter("compute_type").value,
            language=self.get_parameter("language").value,
        )
        self._mic = MicrophoneCapture(
            device=self.get_parameter("mic_device").value,
            sample_rate=self.sample_rate,
        )
        self._segmenter = SpeechSegmenter(
            sample_rate=self.sample_rate,
            rms_threshold=float(self.get_parameter("vad_rms_threshold").value),
            hangover_s=float(self.get_parameter("vad_hangover_s").value),
            max_utterance_s=float(self.get_parameter("vad_max_utterance_s").value),
        )
        self.get_logger().info(
            f"Loading Whisper '{self.get_parameter('model_size').value}' on "
            f"{self.get_parameter('device').value}… (first run downloads it)"
        )
        self._transcriber.load()
        self.get_logger().info(
            f"Whisper ready on {self._transcriber.device}."
        )

    # ── Command pipeline ─────────────────────────────────────────────────── #

    def handle_utterance(self, audio) -> None:
        if self._busy.is_set():
            self.get_logger().warn("Command in progress; utterance dropped.")
            return
        text = self._transcriber.transcribe(audio, self.sample_rate)
        if not text:
            self.get_logger().info("Heard nothing intelligible.")
            return
        self.handle_text(text)

    def handle_text(self, text: str) -> None:
        if self._busy.is_set():
            self.get_logger().warn("Command in progress; input dropped.")
            return
        intent = parse(text)
        self.get_logger().info(f"Heard: \"{text}\" → {intent}")

        if isinstance(intent, NoIntent):
            self.get_logger().warn(f"No intent: {intent.reason}")
            return
        if isinstance(intent, MoveToPosition):
            self._dispatch_move(intent)
        elif isinstance(intent, SaveWaypoint):
            self._dispatch_capture(intent)
        elif isinstance(intent, SemanticGoal):
            self._dispatch_semantic(intent)

    # ── Dispatchers ──────────────────────────────────────────────────────── #

    def _send(self, client: ActionClient, name: str, goal, feedback_cb=None) -> None:
        if not client.wait_for_server(timeout_sec=self._server_wait):
            self.get_logger().error(f"Action server '{name}' unavailable.")
            return
        self._busy.set()
        future = client.send_goal_async(goal, feedback_callback=feedback_cb)
        future.add_done_callback(self._on_goal_response)

    def _on_goal_response(self, future) -> None:
        try:
            handle = future.result()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Goal failed: {exc}")
            self._busy.clear()
            return
        if not handle.accepted:
            self.get_logger().error("Goal rejected.")
            self._busy.clear()
            return
        handle.get_result_async().add_done_callback(self._on_result)

    def _on_result(self, future) -> None:
        try:
            result = future.result().result
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Result failed: {exc}")
            self._busy.clear()
            return
        self._busy.clear()

        if isinstance(result, NavigateToSemanticGoal.Result):
            if result.success:
                self.get_logger().info(
                    f"Reached '{result.matched_node_id}' "
                    f"(score {result.score:.3f})."
                )
            else:
                self.get_logger().error(f"Semantic goal failed: {result.message}")
        elif isinstance(result, CaptureWaypoint.Result):
            if result.success:
                self.get_logger().info(f"Saved waypoint '{result.node_id}'.")
            else:
                self.get_logger().error(f"Capture failed: {result.message}")
        else:  # NavigateToPose.Result carries no status payload
            self.get_logger().info("Navigation goal finished.")

    def _dispatch_move(self, intent: MoveToPosition) -> None:
        goal = NavigateToPose.Goal()
        pose = PoseStamped()
        pose.header.frame_id = self._frame_id
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = intent.x
        pose.pose.position.y = intent.y
        pose.pose.orientation.w = 1.0
        goal.pose = pose
        self.get_logger().info(f"Navigating to ({intent.x}, {intent.y})…")
        self._send(self._nav_client, "navigate_to_pose", goal)

    def _dispatch_capture(self, intent: SaveWaypoint) -> None:
        goal = CaptureWaypoint.Goal()
        goal.label = intent.label
        self.get_logger().info(f"Capturing waypoint (label='{intent.label}')…")
        self._send(self._capture_client, "capture_waypoint", goal)

    def _dispatch_semantic(self, intent: SemanticGoal) -> None:
        goal = NavigateToSemanticGoal.Goal()
        goal.query_text = intent.query
        goal.use_image = False
        goal.decision_only = False
        self.get_logger().info(f"Semantic goal: \"{intent.query}\"…")
        self._send(self._semantic_client, "navigate_to_semantic_goal", goal,
                   feedback_cb=self._on_semantic_feedback)

    def _on_semantic_feedback(self, msg) -> None:
        fb = msg.feedback
        if fb.stage == "navigating":
            self.get_logger().info(
                f"[{fb.stage}] {fb.distance_remaining:.2f} m remaining",
                throttle_duration_sec=2.0,
            )
        else:
            self.get_logger().info(f"[{fb.stage}]")


# ── Interaction loops (main thread) ──────────────────────────────────────── #

def _read_key() -> str:
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _text_loop(node: VoiceCommandNode) -> None:
    print("Text mode: type a command and press Enter ('q' to quit).")
    print("  e.g.  move to 1.0 0.5 | save waypoint kitchen | go to the sofa")
    while rclpy.ok():
        try:
            line = input("> ").strip()
        except EOFError:
            break
        if line.lower() == KEY_QUIT:
            break
        if line:
            node.handle_text(line)


def _ptt_loop(node: VoiceCommandNode) -> None:
    key = node.ptt_key
    print(f"Push-to-talk: '{key}' = start/stop recording, '{KEY_QUIT}' = quit.")
    while rclpy.ok():
        k = _read_key()
        if k == KEY_QUIT or k == "\x03":
            break
        if k != key:
            continue
        stop = threading.Event()
        take: list = []
        rec = threading.Thread(
            target=lambda: take.append(node._mic.record_toggle(stop)),
            daemon=True,
        )
        rec.start()
        print("● recording — press again to stop")
        while True:
            k = _read_key()
            if k in (key, "\x03", KEY_QUIT):
                break
        stop.set()
        rec.join(timeout=2.0)
        if take and take[0].size:
            print("… transcribing")
            node.handle_utterance(take[0])
        if k in ("\x03", KEY_QUIT):
            break


def _vad_loop(node: VoiceCommandNode) -> None:
    print("VAD mode: hands-free listening (Ctrl-C to quit).")
    stop = threading.Event()
    try:
        for chunk in node._mic.stream_chunks(stop):
            utterance = node._segmenter.push(chunk)
            if utterance is not None:
                node.handle_utterance(utterance)
    except KeyboardInterrupt:
        stop.set()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VoiceCommandNode()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    mode = node.activation_mode
    try:
        if mode == "text":
            _text_loop(node)
        elif mode == "push_to_talk":
            if not _HAS_TERMIOS:
                print("push_to_talk requires a POSIX terminal.", file=sys.stderr)
            else:
                _ptt_loop(node)
        elif mode == "vad":
            _vad_loop(node)
        else:
            node.get_logger().error(
                f"Unknown activation_mode '{mode}' "
                "(expected text | push_to_talk | vad)."
            )
    except KeyboardInterrupt:
        pass
    finally:
        # Shut the context down first so rclpy.spin() in the daemon returns,
        # then join before destroying the node.
        rclpy.try_shutdown()
        spin_thread.join(timeout=2.0)
        node.destroy_node()


if __name__ == "__main__":
    main()
