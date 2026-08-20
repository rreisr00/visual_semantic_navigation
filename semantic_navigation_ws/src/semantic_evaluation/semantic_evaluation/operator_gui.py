#!/usr/bin/env python3
"""Graphical console for semantic mapping and voice navigation."""

from __future__ import annotations

import math
import os
import queue
import signal
import sys
import threading
import time
from typing import Any

from action_msgs.msg import GoalStatus
from cv_bridge import CvBridge
from geometry_msgs.msg import (
    Point,
    Point32,
    PointStamped,
    PoseStamped,
    PoseWithCovarianceStamped,
    Twist,
    TwistStamped,
)
from nav2_msgs.action import NavigateToPose
from python_qt_binding.QtCore import QEvent, QObject, Qt, QTimer
from python_qt_binding.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPixmap
from python_qt_binding.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGraphicsEllipseItem,
    QGraphicsScene,
    QGraphicsView,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
import rclpy
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    qos_profile_sensor_data,
    QoSDurabilityPolicy,
    QoSProfile,
)
from rclpy.time import Time
from ros_gz_interfaces.msg import Entity
from ros_gz_interfaces.srv import SetEntityPose
from semantic_evaluation.core.campaign_designer import load_occupancy_map
from semantic_evaluation.core.operator_gui_logic import (
    motion_from_directions,
    normalized_room_bounds,
    parse_view_angles,
)
from semantic_evaluation.campaign_designer_widget import CampaignDesignerWidget
from semantic_evaluation.retrieval_evaluation_widget import (
    RetrievalEvaluationWidget,
)
from semantic_interfaces.action import CaptureWaypoint, NavigateToSemanticGoal
from semantic_interfaces.srv import AddRoom
from semantic_navigation_core.rooms import load_rooms, Room, save_rooms
from semantic_voice.core import (
    MoveToPosition,
    NoIntent,
    parse,
    SaveWaypoint,
    SemanticGoal,
)
from sensor_msgs.msg import Image
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

# The PyPI OpenCV wheel bundled in the ML virtual environment overrides Qt's
# platform-plugin path when cv_bridge imports cv2.  That xcb plugin belongs to
# OpenCV's private Qt build and makes python_qt_binding abort at QApplication
# startup.  Preserve explicitly configured Qt paths, but discard OpenCV's
# injected defaults so ROS uses the system Qt installation.
for _qt_env_name in ('QT_QPA_PLATFORM_PLUGIN_PATH', 'QT_QPA_FONTDIR'):
    if '/cv2/qt/' in os.environ.get(_qt_env_name, ''):
        os.environ.pop(_qt_env_name, None)


_OPERATOR_STYLE = """
QMainWindow, QWidget { background-color: #101722; color: #e6edf3; }
QTabWidget::pane { border: 1px solid #31445a; background: #111a26; }
QTabBar::tab {
    background: #1b2938; color: #b8c7d9; border: 1px solid #31445a;
    padding: 9px 16px; margin-right: 2px;
}
QTabBar::tab:selected {
    background: #176b87; color: white; border-bottom: 3px solid #5ee1ff;
}
QTabBar::tab:hover { background: #24506b; }
QGroupBox {
    background: #172231; border: 1px solid #36516a; border-radius: 7px;
    margin-top: 13px; padding: 12px 8px 8px 8px; font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin; left: 10px; padding: 0 6px; color: #6dd5ed;
}
QLineEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox, QListWidget,
QTableWidget, QGraphicsView {
    background: #0d141e; color: #e6edf3; border: 1px solid #344b61;
    border-radius: 4px; selection-background-color: #197a9e;
    selection-color: white;
}
QLineEdit:focus, QTextEdit:focus, QComboBox:focus, QSpinBox:focus,
QDoubleSpinBox:focus, QListWidget:focus, QTableWidget:focus {
    border: 1px solid #54c8e8;
}
QHeaderView::section {
    background: #21354a; color: #dcebf5; border: 0;
    border-right: 1px solid #3a5268; padding: 6px; font-weight: 600;
}
QTableWidget { alternate-background-color: #152131; gridline-color: #293d50; }
QPushButton {
    background: #26445c; color: #f3f8fb; border: 1px solid #3b6582;
    border-radius: 5px; padding: 6px 10px; min-height: 20px;
}
QPushButton:hover { background: #32617f; border-color: #62c4e2; }
QPushButton:pressed { background: #16364a; }
QPushButton:disabled {
    background: #252d36; color: #697784; border-color: #343d46;
}
QPushButton#accentButton { background: #087f5b; border-color: #20c997; }
QPushButton#accentButton:hover { background: #099268; }
QPushButton#dangerButton { background: #9d3544; border-color: #e16473; }
QPushButton#dangerButton:hover { background: #b53f50; }
QPushButton#voicePttButton {
    background: #6441a5; border: 2px solid #9b7bd1;
    font-weight: 700; min-height: 34px;
}
QPushButton#voicePttButton:pressed {
    background: #b23a48; border-color: #ff7b89;
}
QProgressBar {
    background: #0d141e; border: 1px solid #344b61;
    border-radius: 4px; text-align: center;
}
QProgressBar::chunk { background: #168aad; border-radius: 3px; }
QSplitter::handle { background: #2b5871; }
QSplitter::handle:hover { background: #50b9d7; }
QLabel#sectionHint {
    background: #152536; border-left: 4px solid #2fb7d4;
    padding: 8px; color: #cfe8f2;
}
QLabel#shortcutBadge {
    background: #2d2147; border: 1px solid #7957b3; border-radius: 5px;
    padding: 6px; color: #e6d9ff; font-weight: 600;
}
"""


class _SceneTeleportView(QGraphicsView):
    """Clickable occupancy map with separate robot and destination markers."""

    def __init__(self, selected_callback) -> None:
        self._scene = QGraphicsScene()
        super().__init__(self._scene)
        self._selected_callback = selected_callback
        self._map_size = (0, 0)
        self._destination: QGraphicsEllipseItem | None = None
        self._robot: QGraphicsEllipseItem | None = None
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setMinimumSize(360, 260)
        self.setToolTip(
            'Clic izquierdo: elegir destino. Rueda: zoom. '
            'Use las barras para desplazarse.'
        )

    @property
    def map_size(self) -> tuple[int, int]:
        return self._map_size

    def set_map(self, pixmap: QPixmap) -> None:
        """Replace the occupancy image and fit it in the current viewport."""
        self._scene.clear()
        self._destination = None
        self._robot = None
        self._map_size = (pixmap.width(), pixmap.height())
        self._scene.addPixmap(pixmap).setZValue(-10.0)
        self._scene.setSceneRect(0.0, 0.0, pixmap.width(), pixmap.height())
        QTimer.singleShot(0, self.fit_map)

    def fit_map(self) -> None:
        if self._map_size != (0, 0):
            self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    def set_destination(self, x: float, y: float, *, warning: bool) -> None:
        if self._destination is not None:
            self._scene.removeItem(self._destination)
        colour = QColor('#ff9f43' if warning else '#34d399')
        self._destination = self._scene.addEllipse(
            x - 7.0, y - 7.0, 14.0, 14.0,
            QPen(QColor('white'), 2.0), QBrush(colour),
        )
        self._destination.setZValue(5.0)
        self._destination.setToolTip('Destino de teletransporte')

    def set_robot(self, x: float, y: float) -> None:
        if self._robot is None:
            self._robot = self._scene.addEllipse(
                -6.0, -6.0, 12.0, 12.0,
                QPen(QColor('white'), 1.5), QBrush(QColor('#3b82f6')),
            )
            self._robot.setZValue(4.0)
            self._robot.setToolTip('Posición actual del robot')
        self._robot.setRect(x - 6.0, y - 6.0, 12.0, 12.0)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton and self._map_size != (0, 0):
            point = self.mapToScene(event.pos())
            width, height = self._map_size
            if 0.0 <= point.x() < width and 0.0 <= point.y() < height:
                self._selected_callback(point.x(), point.y())
                event.accept()
                return
        super().mousePressEvent(event)

    def wheelEvent(self, event) -> None:  # noqa: N802
        factor = 1.18 if event.angleDelta().y() > 0 else 1.0 / 1.18
        self.scale(factor, factor)
        event.accept()


class SemanticOperatorNode(Node):
    """ROS-facing backend used by the Qt operator window."""

    def __init__(self) -> None:
        super().__init__('semantic_operator_gui')
        self.declare_parameter('camera_topic', '/camera/image_raw')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('stamped_cmd_vel', True)
        self.declare_parameter('capture_action_name', '/capture_waypoint')
        self.declare_parameter(
            'semantic_action_name', '/navigate_to_semantic_goal'
        )
        self.declare_parameter('navigate_action_name', '/navigate_to_pose')
        self.declare_parameter('add_room_service', '/add_room')
        self.declare_parameter('clicked_point_topic', '/clicked_point')
        self.declare_parameter('room_marker_topic', '/room_markers')
        self.declare_parameter('scene_id', 'aws_small_house')
        self.declare_parameter('rooms_file', '')
        self.declare_parameter('map_file', '')
        self.declare_parameter('graph_database', '')
        self.declare_parameter('queries_file', '')
        self.declare_parameter('ground_truth_file', '')
        self.declare_parameter('start_poses_file', '')
        self.declare_parameter('campaign_output_dir', '')
        self.declare_parameter('robot_entity_name', 'semantic_robot')
        self.declare_parameter('world_name', 'default')
        self.declare_parameter('initial_pose_topic', '/initialpose')
        self.declare_parameter('teleport_height', 0.01)
        self.declare_parameter('frozen_config_hash', '')
        self.declare_parameter('frozen_config_path', '')
        self.declare_parameter(
            'retrieval_method', 'hybrid_semantic_retrieval'
        )
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('linear_speed', 0.25)
        self.declare_parameter('angular_speed', 0.7)
        self.declare_parameter('voice_model_size', 'small')
        self.declare_parameter('voice_device', 'cpu')
        self.declare_parameter('voice_compute_type', 'int8')
        self.declare_parameter('voice_language', 'es')
        self.declare_parameter('voice_mic_device', '')
        self.declare_parameter('voice_sample_rate', 16000)
        self.declare_parameter('voice_top_k', 5)

        self.scene_id = str(self.get_parameter('scene_id').value)
        self.camera_topic = str(self.get_parameter('camera_topic').value)
        self.linear_speed = float(self.get_parameter('linear_speed').value)
        self.angular_speed = float(self.get_parameter('angular_speed').value)
        self.voice_language = str(self.get_parameter('voice_language').value)
        self._voice_model_size = str(
            self.get_parameter('voice_model_size').value
        )
        self._voice_device = str(self.get_parameter('voice_device').value)
        self._voice_compute_type = str(
            self.get_parameter('voice_compute_type').value
        )
        self._voice_mic_device = str(
            self.get_parameter('voice_mic_device').value
        )
        self._voice_sample_rate = int(
            self.get_parameter('voice_sample_rate').value
        )
        self._map_frame = str(self.get_parameter('map_frame').value)
        self._base_frame = str(self.get_parameter('base_frame').value)
        self.map_file = os.path.expanduser(
            str(self.get_parameter('map_file').value).strip()
        )
        graph_database = str(
            self.get_parameter('graph_database').value
        ).strip()
        self.graph_database = os.path.expanduser(
            graph_database
            or f'~/.ros/semantic_maps/{self.scene_id}/graph.db'
        )
        self.queries_file = os.path.expanduser(
            str(self.get_parameter('queries_file').value).strip()
        )
        self.ground_truth_file = os.path.expanduser(
            str(self.get_parameter('ground_truth_file').value).strip()
        )
        self.start_poses_file = os.path.expanduser(
            str(self.get_parameter('start_poses_file').value).strip()
        )
        self.campaign_output_dir = os.path.expanduser(
            str(self.get_parameter('campaign_output_dir').value).strip()
        )
        self.robot_entity_name = str(
            self.get_parameter('robot_entity_name').value
        )
        self.world_name = str(self.get_parameter('world_name').value)
        self.teleport_height = float(
            self.get_parameter('teleport_height').value
        )
        self.frozen_config_hash = str(
            self.get_parameter('frozen_config_hash').value
        )
        self.frozen_config_path = os.path.expanduser(
            str(self.get_parameter('frozen_config_path').value).strip()
        )
        self.retrieval_method = str(
            self.get_parameter('retrieval_method').value
        )
        rooms_file = str(self.get_parameter('rooms_file').value).strip()
        self.rooms_file = os.path.expanduser(
            rooms_file
            or f'~/.ros/semantic_maps/{self.scene_id}/rooms.yaml'
        )

        self._events: queue.SimpleQueue[tuple[str, dict[str, Any]]] = (
            queue.SimpleQueue()
        )
        self._frame_lock = threading.Lock()
        self._latest_image: Image | None = None
        self._frame_received_at: float | None = None
        self._clicked_lock = threading.Lock()
        self._latest_clicked_point: tuple[float, float] | None = None
        self._capture_lock = threading.Lock()
        self._capture_active = False
        self._capture_goal_handle = None
        self._navigation_lock = threading.Lock()
        self._navigation_active = False
        self._navigation_goal_handle = None
        self._voice_lock = threading.Lock()
        self._voice_loading = False
        self._voice_ready = False
        self._voice_recording = False
        self._voice_stop_event: threading.Event | None = None
        self._voice_transcriber = None
        self._voice_microphone = None
        self._rooms_lock = threading.Lock()
        self._teleport_lock = threading.Lock()
        self._teleport_pending = False

        stamped = bool(self.get_parameter('stamped_cmd_vel').value)
        self._stamped_cmd_vel = stamped
        self._cmd_pub = self.create_publisher(
            TwistStamped if stamped else Twist,
            str(self.get_parameter('cmd_vel_topic').value),
            10,
        )
        self.create_subscription(
            Image,
            self.camera_topic,
            self._on_image,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            PointStamped,
            str(self.get_parameter('clicked_point_topic').value),
            self._on_clicked_point,
            10,
        )
        self._capture_client = ActionClient(
            self,
            CaptureWaypoint,
            str(self.get_parameter('capture_action_name').value),
        )
        self._semantic_client = ActionClient(
            self,
            NavigateToSemanticGoal,
            str(self.get_parameter('semantic_action_name').value),
        )
        self._navigate_client = ActionClient(
            self,
            NavigateToPose,
            str(self.get_parameter('navigate_action_name').value),
        )
        self._room_client = self.create_client(
            AddRoom,
            str(self.get_parameter('add_room_service').value),
        )
        self._teleport_client = self.create_client(
            SetEntityPose,
            f'/world/{self.world_name}/set_pose',
        )
        initial_pose_qos = QoSProfile(depth=10)
        self._initial_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped,
            str(self.get_parameter('initial_pose_topic').value),
            initial_pose_qos,
        )
        marker_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._room_marker_pub = self.create_publisher(
            MarkerArray,
            str(self.get_parameter('room_marker_topic').value),
            marker_qos,
        )
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        try:
            self._rooms = load_rooms(self.rooms_file)
        except Exception as exc:  # noqa: BLE001
            self._rooms = []
            self.push_event('error', message=f'No se pudieron cargar las salas: {exc}')
        self.publish_room_markers()
        self.get_logger().info(
            f"Operator GUI backend ready for scene '{self.scene_id}'."
        )

    def push_event(self, kind: str, **payload: Any) -> None:
        self._events.put((kind, payload))

    def pop_events(self) -> list[tuple[str, dict[str, Any]]]:
        events: list[tuple[str, dict[str, Any]]] = []
        while True:
            try:
                events.append(self._events.get_nowait())
            except queue.Empty:
                return events

    def _on_image(self, message: Image) -> None:
        with self._frame_lock:
            self._latest_image = message
            self._frame_received_at = time.monotonic()

    def latest_image(self) -> Image | None:
        with self._frame_lock:
            return self._latest_image

    def camera_age(self) -> float | None:
        with self._frame_lock:
            received_at = self._frame_received_at
        return None if received_at is None else time.monotonic() - received_at

    def _on_clicked_point(self, message: PointStamped) -> None:
        if message.header.frame_id and message.header.frame_id != self._map_frame:
            self.push_event(
                'error',
                message=(
                    f"Punto RViz ignorado: frame '{message.header.frame_id}', "
                    f"se esperaba '{self._map_frame}'."
                ),
            )
            return
        with self._clicked_lock:
            self._latest_clicked_point = (
                float(message.point.x),
                float(message.point.y),
            )
        self.push_event(
            'info',
            message=(
                f'Punto RViz recibido: ({message.point.x:.2f}, '
                f'{message.point.y:.2f}).'
            ),
        )

    def latest_clicked_position(self) -> tuple[float, float] | None:
        with self._clicked_lock:
            return self._latest_clicked_point

    def current_position(self) -> tuple[float, float] | None:
        pose = self.current_pose()
        return None if pose is None else pose[:2]

    def current_pose(self) -> tuple[float, float, float] | None:
        """Return map-frame x, y and yaw for the current robot transform."""
        try:
            transform = self._tf_buffer.lookup_transform(
                self._map_frame,
                self._base_frame,
                Time(),
            )
        except TransformException:
            return None
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        yaw = math.atan2(
            2.0 * (rotation.w * rotation.z + rotation.x * rotation.y),
            1.0 - 2.0 * (rotation.y * rotation.y + rotation.z * rotation.z),
        )
        return float(translation.x), float(translation.y), yaw

    @property
    def teleport_pending(self) -> bool:
        with self._teleport_lock:
            return self._teleport_pending

    def teleport_robot(self, x: float, y: float, yaw: float) -> bool:
        """Move the Gazebo model and publish the matching AMCL initial pose."""
        if self.capture_active:
            self.push_event(
                'error',
                message='Cancela la captura antes de teletransportar el robot.',
            )
            return False
        with self._teleport_lock:
            if self._teleport_pending:
                self.push_event(
                    'error', message='Ya hay un teletransporte en curso.'
                )
                return False
            if not self._teleport_client.service_is_ready():
                self.push_event(
                    'error',
                    message=(
                        f'El servicio /world/{self.world_name}/set_pose '
                        'no está disponible.'
                    ),
                )
                return False
            self._teleport_pending = True

        request = SetEntityPose.Request()
        request.entity.name = self.robot_entity_name
        request.entity.type = Entity.MODEL
        request.pose.position.x = float(x)
        request.pose.position.y = float(y)
        request.pose.position.z = self.teleport_height
        request.pose.orientation.z = math.sin(float(yaw) / 2.0)
        request.pose.orientation.w = math.cos(float(yaw) / 2.0)
        self.stop()
        self.push_event(
            'teleport_started',
            message=f'Teletransportando a ({x:.2f}, {y:.2f})…',
        )
        future = self._teleport_client.call_async(request)
        future.add_done_callback(
            lambda completed: self._on_teleport_result(completed, request)
        )
        return True

    def _on_teleport_result(self, future, request: SetEntityPose.Request) -> None:
        with self._teleport_lock:
            self._teleport_pending = False
        try:
            response = future.result()
        except Exception as exc:  # noqa: BLE001
            self.push_event(
                'teleport_result', success=False,
                message=f'Falló el teletransporte: {exc}',
            )
            return
        if response is None or not response.success:
            self.push_event(
                'teleport_result', success=False,
                message='Gazebo rechazó el teletransporte.',
            )
            return

        initial = PoseWithCovarianceStamped()
        initial.header.frame_id = self._map_frame
        initial.header.stamp = self.get_clock().now().to_msg()
        initial.pose.pose = request.pose
        initial.pose.covariance[0] = 0.25
        initial.pose.covariance[7] = 0.25
        initial.pose.covariance[35] = 0.0685
        # AMCL can miss a single initial-pose sample while its lifecycle node
        # is transitioning.  Repeating it is cheap and mirrors the campaign
        # runner's reset behaviour without blocking the ROS executor thread.
        for _ in range(3):
            self._initial_pose_pub.publish(initial)
        yaw = 2.0 * math.atan2(
            request.pose.orientation.z, request.pose.orientation.w
        )
        self.push_event(
            'teleport_result',
            success=True,
            x=float(request.pose.position.x),
            y=float(request.pose.position.y),
            yaw=yaw,
            message=(
                f'Robot teletransportado a ({request.pose.position.x:.2f}, '
                f'{request.pose.position.y:.2f}); AMCL resincronizado.'
            ),
        )

    def publish_motion(self, linear: float, angular: float) -> None:
        twist = Twist()
        twist.linear.x = float(linear)
        twist.angular.z = float(angular)
        if self._stamped_cmd_vel:
            message = TwistStamped()
            message.header.stamp = self.get_clock().now().to_msg()
            message.header.frame_id = self._base_frame
            message.twist = twist
            self._cmd_pub.publish(message)
        else:
            self._cmd_pub.publish(twist)

    def stop(self) -> None:
        self.publish_motion(0.0, 0.0)

    @property
    def capture_active(self) -> bool:
        with self._capture_lock:
            return self._capture_active

    @property
    def navigation_active(self) -> bool:
        with self._navigation_lock:
            return self._navigation_active

    def voice_state(self) -> tuple[bool, bool, bool]:
        """Return loading, ready and recording flags for the Qt thread."""
        with self._voice_lock:
            return (
                self._voice_loading,
                self._voice_ready,
                self._voice_recording,
            )

    def capture_server_ready(self) -> bool:
        return self._capture_client.server_is_ready()

    def room_service_ready(self) -> bool:
        return self._room_client.service_is_ready()

    def semantic_server_ready(self) -> bool:
        return self._semantic_client.server_is_ready()

    def prepare_voice(self, language: str) -> bool:
        """Load Whisper asynchronously so the Qt event loop remains responsive."""
        with self._voice_lock:
            if self._voice_ready:
                self.push_event('voice_ready', language=self.voice_language)
                return True
            if self._voice_loading:
                return False
            self._voice_loading = True
            self.voice_language = language.strip() or 'es'
        self.push_event(
            'voice_loading',
            message='Cargando Whisper; la primera carga puede tardar…',
        )
        threading.Thread(
            target=self._prepare_voice_worker,
            daemon=True,
        ).start()
        return True

    def _prepare_voice_worker(self) -> None:
        try:
            from semantic_voice.core.audio_capture import MicrophoneCapture
            from semantic_voice.core.transcriber import WhisperTranscriber

            transcriber = WhisperTranscriber(
                model_size=self._voice_model_size,
                device=self._voice_device,
                compute_type=self._voice_compute_type,
                language=self.voice_language,
            )
            microphone = MicrophoneCapture(
                device=self._voice_mic_device,
                sample_rate=self._voice_sample_rate,
            )
            transcriber.load()
        except Exception as exc:  # noqa: BLE001
            with self._voice_lock:
                self._voice_loading = False
            self.push_event(
                'voice_error',
                message=f'No se pudo preparar el reconocimiento de voz: {exc}',
            )
            return
        with self._voice_lock:
            self._voice_transcriber = transcriber
            self._voice_microphone = microphone
            self._voice_loading = False
            self._voice_ready = True
        self.push_event(
            'voice_ready',
            language=self.voice_language,
            device=transcriber.device,
            message=(
                f'Whisper preparado en {transcriber.device}; '
                'mantén pulsado el botón para hablar.'
            ),
        )

    def start_voice_recording(self) -> bool:
        with self._voice_lock:
            if not self._voice_ready:
                self.push_event(
                    'voice_error',
                    message='Carga primero el reconocimiento de voz.',
                )
                return False
            if self._voice_recording:
                return False
            self._voice_recording = True
            self._voice_stop_event = threading.Event()
            stop_event = self._voice_stop_event
            microphone = self._voice_microphone
            transcriber = self._voice_transcriber
        self.push_event('voice_recording', message='Escuchando…')
        threading.Thread(
            target=self._voice_record_worker,
            args=(microphone, transcriber, stop_event),
            daemon=True,
        ).start()
        return True

    def stop_voice_recording(self) -> bool:
        with self._voice_lock:
            stop_event = self._voice_stop_event
            recording = self._voice_recording
        if not recording or stop_event is None:
            return False
        stop_event.set()
        return True

    def _voice_record_worker(self, microphone, transcriber, stop_event) -> None:
        try:
            audio = microphone.record_toggle(stop_event)
            if not audio.size:
                self.push_event(
                    'voice_idle',
                    message='No se recibió audio del micrófono.',
                )
                return
            self.push_event('voice_transcribing', message='Transcribiendo…')
            transcript = transcriber.transcribe(
                audio,
                self._voice_sample_rate,
            )
            if not transcript:
                self.push_event(
                    'voice_idle',
                    message='No se reconoció ninguna frase.',
                )
                return
            self.push_event('voice_transcript', transcript=transcript)
        except Exception as exc:  # noqa: BLE001
            self.push_event(
                'voice_error',
                message=f'Error capturando o transcribiendo audio: {exc}',
            )
        finally:
            with self._voice_lock:
                self._voice_recording = False
                self._voice_stop_event = None

    def shutdown_voice(self) -> None:
        with self._voice_lock:
            stop_event = self._voice_stop_event
        if stop_event is not None:
            stop_event.set()

    def send_semantic_navigation(
        self,
        query: str,
        language: str,
        decision_only: bool = False,
        *,
        event_kind: str = 'semantic',
        top_k: int | None = None,
    ) -> bool:
        text = query.strip()
        if not text:
            self.push_event(
                'error',
                kind=event_kind,
                message='La consulta semántica está vacía.',
            )
            return False
        goal = NavigateToSemanticGoal.Goal()
        goal.query_text = text
        goal.use_image = False
        goal.decision_only = bool(decision_only)
        goal.language = language.strip() or 'es'
        goal.scene_id = self.scene_id
        goal.current_node_id = ''
        goal.top_k = (
            int(top_k) if top_k is not None
            else int(self.get_parameter('voice_top_k').value)
        )
        goal.navigate = not decision_only
        return self._send_navigation_goal(
            self._semantic_client,
            goal,
            event_kind,
            f'Consulta semántica enviada: «{text}».',
        )

    def send_pose_navigation(self, x: float, y: float) -> bool:
        pose = PoseStamped()
        pose.header.frame_id = self._map_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.orientation.w = 1.0
        goal = NavigateToPose.Goal()
        goal.pose = pose
        return self._send_navigation_goal(
            self._navigate_client,
            goal,
            'pose',
            f'Navegación por voz a ({x:.2f}, {y:.2f}).',
        )

    def _send_navigation_goal(self, client, goal, kind: str, message: str) -> bool:
        if self.capture_active:
            self.push_event(
                'error',
                kind=kind,
                message='Espera a que termine la captura antes de navegar.',
            )
            return False
        with self._navigation_lock:
            if self._navigation_active:
                self.push_event(
                    'error', kind=kind, message='Ya hay una navegación en curso.'
                )
                return False
            if not client.server_is_ready():
                self.push_event(
                    'error',
                    kind=kind,
                    message='El servidor de navegación solicitado no está disponible.',
                )
                return False
            self._navigation_active = True
            self._navigation_goal_handle = None
        self.stop()
        self.push_event('navigation_started', kind=kind, message=message)
        try:
            future = client.send_goal_async(
                goal,
                feedback_callback=(
                    (
                        lambda feedback, navigation_kind=kind:
                        self._on_semantic_navigation_feedback(
                            feedback, navigation_kind
                        )
                    ) if kind in ('semantic', 'evaluation') else None
                ),
            )
        except Exception as exc:  # noqa: BLE001
            self._finish_navigation()
            self.push_event(
                'error', kind=kind, message=f'Error enviando navegación: {exc}'
            )
            return False
        future.add_done_callback(
            lambda done, navigation_kind=kind: self._on_navigation_goal(
                done,
                navigation_kind,
            )
        )
        return True

    def _on_navigation_goal(self, future, kind: str) -> None:
        try:
            handle = future.result()
        except Exception as exc:  # noqa: BLE001
            self._finish_navigation()
            self.push_event(
                'error', kind=kind, message=f'Navegación rechazada: {exc}'
            )
            return
        if not handle.accepted:
            self._finish_navigation()
            self.push_event(
                'error',
                kind=kind,
                message='El objetivo de navegación fue rechazado.',
            )
            return
        with self._navigation_lock:
            self._navigation_goal_handle = handle
        handle.get_result_async().add_done_callback(
            lambda done, navigation_kind=kind: self._on_navigation_result(
                done,
                navigation_kind,
            )
        )

    def _on_semantic_navigation_feedback(
        self, feedback_message, kind: str
    ) -> None:
        feedback = feedback_message.feedback
        self.push_event(
            'navigation_feedback',
            kind=kind,
            stage=feedback.stage,
            distance=float(feedback.distance_remaining),
        )

    def _on_navigation_result(self, future, kind: str) -> None:
        try:
            response = future.result()
            result = response.result
            status = int(response.status)
        except Exception as exc:  # noqa: BLE001
            self._finish_navigation()
            self.push_event(
                'error', kind=kind, message=f'Error recibiendo navegación: {exc}'
            )
            return
        self._finish_navigation()
        if kind in ('semantic', 'evaluation'):
            self.push_event(
                'navigation_result',
                kind=kind,
                success=bool(result.success),
                matched_node=result.matched_node_id,
                score=float(result.score),
                accepted=bool(result.accepted),
                navigation_success=bool(result.navigation_success),
                message=result.message,
                failure_type=result.failure_type,
                rejection_reason=result.rejection_reason,
                candidates=[{
                    'node_id': item.node_id,
                    'score': float(item.score),
                    'global_similarity': float(item.global_similarity),
                    'object_match_score': float(item.object_match_score),
                    'crop_similarity': float(item.crop_similarity),
                    'relation_match_score': float(item.relation_match_score),
                    'room_match_score': float(item.room_match_score),
                    'matched_object_ids': list(item.matched_object_ids),
                    'matched_object_labels': list(item.matched_object_labels),
                    'best_crop_object_id': item.best_crop_object_id,
                    'best_crop_object_label': item.best_crop_object_label,
                } for item in result.top_k_candidates],
            )
            return
        succeeded = status == GoalStatus.STATUS_SUCCEEDED
        self.push_event(
            'navigation_result',
            kind=kind,
            success=succeeded,
            matched_node='',
            score=0.0,
            navigation_success=succeeded,
            message=(
                'Destino por coordenadas alcanzado.'
                if succeeded else f'Navegación finalizada con estado {status}.'
            ),
            failure_type='' if succeeded else 'navigate_to_pose_failed',
        )

    def _finish_navigation(self) -> None:
        with self._navigation_lock:
            self._navigation_active = False
            self._navigation_goal_handle = None

    def cancel_navigation(self) -> bool:
        with self._navigation_lock:
            handle = self._navigation_goal_handle
            active = self._navigation_active
        if handle is None:
            message = (
                'El objetivo todavía está esperando aceptación.'
                if active else 'No hay una navegación activa.'
            )
            self.push_event('info', message=message)
            return False
        handle.cancel_goal_async()
        self.push_event('info', message='Cancelación de navegación solicitada.')
        return True

    def send_capture(
        self,
        label: str,
        relative_views: list[float],
        rotate_robot: bool,
    ) -> bool:
        if self.navigation_active:
            self.push_event(
                'error',
                message='Cancela la navegación antes de iniciar una captura.',
            )
            return False
        with self._capture_lock:
            if self._capture_active:
                self.push_event('error', message='Ya hay una captura en curso.')
                return False
            if not self._capture_client.server_is_ready():
                self.push_event(
                    'error',
                    message='La acción /capture_waypoint no está disponible.',
                )
                return False
            self._capture_active = True

        goal = CaptureWaypoint.Goal()
        goal.label = label.strip()
        goal.scene_id = self.scene_id
        goal.requested_yaw = 0.0
        goal.relative_view_yaws_deg = [float(value) for value in relative_views]
        goal.rotate_robot = bool(rotate_robot)
        self.push_event(
            'capture_started',
            message=(
                f"Captura enviada para '{goal.label or '<nombre automático>'}' "
                f'({max(1, len(relative_views))} vista(s)).'
            ),
        )
        future = self._capture_client.send_goal_async(
            goal,
            feedback_callback=self._on_capture_feedback,
        )
        future.add_done_callback(self._on_capture_goal)
        return True

    def _on_capture_feedback(self, feedback_message) -> None:
        feedback = feedback_message.feedback
        self.push_event(
            'capture_feedback',
            stage=feedback.stage,
            current=int(feedback.current_view),
            total=int(feedback.total_views),
            requested_yaw=float(feedback.requested_yaw),
            measured_yaw=float(feedback.measured_yaw),
        )

    def _finish_capture(self) -> None:
        with self._capture_lock:
            self._capture_active = False
            self._capture_goal_handle = None

    def _on_capture_goal(self, future) -> None:
        try:
            handle = future.result()
        except Exception as exc:  # noqa: BLE001
            self._finish_capture()
            self.push_event('error', message=f'Error enviando captura: {exc}')
            return
        if not handle.accepted:
            self._finish_capture()
            self.push_event('error', message='La captura fue rechazada.')
            return
        with self._capture_lock:
            self._capture_goal_handle = handle
        handle.get_result_async().add_done_callback(self._on_capture_result)

    def _on_capture_result(self, future) -> None:
        try:
            result = future.result().result
        except Exception as exc:  # noqa: BLE001
            self._finish_capture()
            self.push_event('error', message=f'Error recibiendo captura: {exc}')
            return
        self._finish_capture()
        self.push_event(
            'capture_result',
            success=bool(result.success),
            node_id=result.node_id,
            views=int(result.captured_views),
            message=result.message,
            merged=bool(result.merged_with_existing),
        )

    def cancel_capture(self) -> bool:
        with self._capture_lock:
            handle = self._capture_goal_handle
        if handle is None:
            self.push_event('info', message='No hay una captura aceptada que cancelar.')
            return False
        handle.cancel_goal_async()
        self.push_event('info', message='Cancelación de captura solicitada.')
        return True

    def create_room(
        self,
        room_id: str,
        corner_a: tuple[float, float],
        corner_b: tuple[float, float],
        polygon: list[tuple[float, float]] | None = None,
        transition_width_m: float = 0.5,
    ) -> bool:
        label = room_id.strip()
        if not label:
            self.push_event('error', message='La sala necesita un nombre.')
            return False
        if not self._room_client.service_is_ready():
            self.push_event('error', message='El servicio /add_room no está disponible.')
            return False
        try:
            room = (
                Room.from_polygon(label, polygon, transition_width_m)
                if polygon else Room(
                    label, *normalized_room_bounds(corner_a, corner_b),
                    transition_width_m=transition_width_m,
                )
            )
        except ValueError as exc:
            self.push_event('error', message=str(exc))
            return False
        request = AddRoom.Request()
        request.room_id = room.room_id
        request.min_x = room.min_x
        request.min_y = room.min_y
        request.max_x = room.max_x
        request.max_y = room.max_y
        request.polygon = [Point32(x=x, y=y, z=0.0) for x, y in room.corners()]
        request.transition_width_m = room.transition_width_m
        future = self._room_client.call_async(request)
        future.add_done_callback(
            lambda done, requested_room=room: self._on_room_result(
                done,
                requested_room,
            )
        )
        self.push_event('info', message=f"Registrando sala '{room.room_id}'…")
        return True

    def _on_room_result(self, future, room: Room) -> None:
        try:
            result = future.result()
        except Exception as exc:  # noqa: BLE001
            self.push_event('error', message=f'Error creando sala: {exc}')
            return
        if not result.success:
            self.push_event('error', message=f'Sala rechazada: {result.message}')
            return
        with self._rooms_lock:
            self._rooms = [
                existing
                for existing in self._rooms
                if existing.room_id != room.room_id
            ] + [room]
            rooms = list(self._rooms)
        try:
            save_rooms(self.rooms_file, rooms)
        except Exception as exc:  # noqa: BLE001
            self.push_event(
                'error',
                message=f'Sala creada, pero no se pudo guardar el YAML: {exc}',
            )
        self.publish_room_markers()
        self.push_event(
            'room_result',
            message=(
                f"Sala '{room.room_id}' creada; "
                f'{result.waypoints_assigned} waypoint(s) asignados.'
            ),
        )

    def rooms_snapshot(self) -> list[Room]:
        with self._rooms_lock:
            return list(self._rooms)

    def publish_room_markers(self) -> None:
        with self._rooms_lock:
            rooms = list(self._rooms)
        markers = MarkerArray()
        wipe = Marker()
        wipe.action = Marker.DELETEALL
        markers.markers.append(wipe)
        stamp = self.get_clock().now().to_msg()
        for index, room in enumerate(rooms):
            rectangle = Marker()
            rectangle.header.frame_id = self._map_frame
            rectangle.header.stamp = stamp
            rectangle.ns = 'operator_gui/rooms'
            rectangle.id = index
            rectangle.type = Marker.LINE_STRIP
            rectangle.action = Marker.ADD
            rectangle.scale.x = 0.06
            rectangle.color.r = 0.9
            rectangle.color.g = 0.6
            rectangle.color.b = 0.1
            rectangle.color.a = 1.0
            corners = room.corners()
            for x, y in corners + [corners[0]]:
                point = Point()
                point.x, point.y, point.z = x, y, 0.05
                rectangle.points.append(point)
            markers.markers.append(rectangle)

            text = Marker()
            text.header.frame_id = self._map_frame
            text.header.stamp = stamp
            text.ns = 'operator_gui/labels'
            text.id = index
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.text = room.room_id
            text.pose.position.x, text.pose.position.y = room.center
            text.pose.position.z = 0.3
            text.scale.z = 0.35
            text.color.r = 1.0
            text.color.g = 1.0
            text.color.b = 1.0
            text.color.a = 1.0
            markers.markers.append(text)
        self._room_marker_pub.publish(markers)


class ArrowKeyFilter(QObject):
    """Capture teleoperation and push-to-talk keys at application level."""

    KEY_DIRECTIONS = {
        Qt.Key_Up: 'forward',
        Qt.Key_Down: 'back',
        Qt.Key_Left: 'left',
        Qt.Key_Right: 'right',
    }

    def __init__(self, window: 'SemanticOperatorWindow') -> None:
        super().__init__(window)
        self._window = window

    def eventFilter(self, watched, event) -> bool:  # noqa: N802
        if event.type() in (
            QEvent.ApplicationDeactivate,
            QEvent.WindowDeactivate,
        ):
            self._window.release_keyboard_controls()
            return super().eventFilter(watched, event)
        key = getattr(event, 'key', lambda: None)()
        if key == Qt.Key_Space and not event.isAutoRepeat():
            if (
                event.type() == QEvent.KeyRelease
                and self._window.voice_shortcut_active()
            ):
                self._window.set_voice_push_to_talk(False)
                return True
            if self._window.voice_shortcut_enabled():
                if event.type() == QEvent.KeyPress:
                    self._window.set_voice_push_to_talk(True)
                    return True
        direction = self.KEY_DIRECTIONS.get(key)
        if direction is None or event.isAutoRepeat():
            return super().eventFilter(watched, event)
        if not self._window.teleop_keys_enabled():
            return super().eventFilter(watched, event)
        if event.type() == QEvent.KeyPress:
            self._window.set_direction(direction, True)
            return True
        if event.type() == QEvent.KeyRelease:
            self._window.set_direction(direction, False)
            return True
        return super().eventFilter(watched, event)


class SemanticOperatorWindow(QMainWindow):
    """Qt window for camera, mapping, teleoperation and voice navigation."""

    def __init__(self, node: SemanticOperatorNode) -> None:
        super().__init__()
        self._node = node
        self._bridge = CvBridge()
        self._last_image: Image | None = None
        self._directions: set[str] = set()
        self._last_motion = (0.0, 0.0)
        self._current_position: tuple[float, float] | None = None
        self._current_yaw = 0.0
        self._teleport_map_metadata = None
        self._teleport_image = QImage()
        self._selected_teleport: tuple[float, float] | None = None
        self._selected_teleport_warning = False
        self._space_voice_active = False
        self._key_filter = ArrowKeyFilter(self)

        self.setWindowTitle(f'Operador semántico — {node.scene_id}')
        self.resize(1360, 860)
        self.setMinimumSize(820, 560)
        self.setStyleSheet(_OPERATOR_STYLE)
        self._build_ui()
        application = QApplication.instance()
        if application is not None:
            application.installEventFilter(self._key_filter)

        self._update_timer = QTimer(self)
        self._update_timer.timeout.connect(self._update_fast)
        self._update_timer.start(100)
        self._status_timer = QTimer(self)
        self._status_timer.timeout.connect(self._update_connections)
        self._status_timer.start(500)
        self._append_log(
            f'GUI preparada. Cámara: {node.camera_topic}; salas: {node.rooms_file}'
        )
        self._refresh_rooms()

    def teleop_keys_enabled(self) -> bool:
        """Only capture arrow keys while the mapping tab is visible."""
        return not hasattr(self, 'main_tabs') or self.main_tabs.currentIndex() == 0

    def voice_shortcut_enabled(self) -> bool:
        """Accept Space as PTT only on Navigation and outside text editors."""
        if not hasattr(self, 'main_tabs') or self.main_tabs.currentIndex() != 1:
            return False
        if not hasattr(self, 'voice_ptt') or not self.voice_ptt.isEnabled():
            return False
        focus = QApplication.focusWidget()
        return not isinstance(
            focus, (QLineEdit, QTextEdit, QAbstractSpinBox, QComboBox)
        )

    def voice_shortcut_active(self) -> bool:
        """Return whether Space currently owns the microphone gesture."""
        return self._space_voice_active

    def set_voice_push_to_talk(self, pressed: bool) -> None:
        """Mirror a held Space key to the microphone push-to-talk button."""
        if pressed == self._space_voice_active:
            return
        self._space_voice_active = pressed
        self.voice_ptt.setDown(pressed)
        if pressed:
            if not self._start_voice_recording():
                self._space_voice_active = False
                self.voice_ptt.setDown(False)
        else:
            self._node.stop_voice_recording()

    def release_keyboard_controls(self) -> None:
        """Stop motion and recording if the window loses keyboard focus."""
        self._directions.clear()
        self._node.stop()
        if self._space_voice_active:
            self.set_voice_push_to_talk(False)

    def _tab_changed(self, index: int) -> None:
        self.release_keyboard_controls()
        if index == 2:
            self.retrieval_evaluation.reload()
        elif index == 3:
            self.campaign_designer.reload()

    def _build_ui(self) -> None:
        self.main_tabs = QTabWidget()
        self.main_tabs.setDocumentMode(True)
        self.setCentralWidget(self.main_tabs)

        root = QSplitter(Qt.Horizontal)
        self.main_tabs.addTab(root, 'Construcción del mapa semántico')

        camera_panel = QWidget()
        camera_layout = QVBoxLayout(camera_panel)
        construction_views = QTabWidget()
        construction_views.setDocumentMode(True)
        camera_layout.addWidget(construction_views, stretch=1)

        camera_page = QWidget()
        camera_page_layout = QVBoxLayout(camera_page)
        self.camera_label = QLabel(f'Esperando {self._node.camera_topic}…')
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(320, 240)
        self.camera_label.setStyleSheet(
            'background: #17191c; color: #b8bec7; border: 1px solid #3b4048;'
        )
        camera_page_layout.addWidget(self.camera_label, stretch=1)
        construction_views.addTab(camera_page, 'Cámara RGB-D')
        construction_views.addTab(
            self._teleport_map_page(), 'Mapa y teletransporte'
        )
        self.pose_label = QLabel('Pose map → base_link: no disponible')
        camera_layout.addWidget(self.pose_label)
        root.addWidget(camera_panel)

        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_layout.addWidget(self._connection_group())
        controls_layout.addWidget(self._motion_group())
        controls_layout.addWidget(self._capture_group())
        controls_layout.addWidget(self._room_group())
        controls_layout.addStretch(1)
        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setWidget(controls)
        root.addWidget(controls_scroll)
        root.setChildrenCollapsible(False)
        root.setHandleWidth(7)
        root.setStretchFactor(0, 3)
        root.setStretchFactor(1, 2)
        root.setSizes([820, 500])

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(150)
        camera_layout.addWidget(self.log)

        navigation_page = QWidget()
        navigation_layout = QVBoxLayout(navigation_page)
        navigation_intro = QLabel(
            'Introduzca una orden o use el micrófono. Las consultas semánticas '
            'se resolverán mediante el grafo; las coordenadas se enviarán a Nav2.'
        )
        navigation_intro.setWordWrap(True)
        navigation_intro.setObjectName('sectionHint')
        navigation_layout.addWidget(navigation_intro)
        navigation_layout.addWidget(self._voice_group())
        navigation_layout.addStretch(1)
        self.main_tabs.addTab(navigation_page, 'Navegación')

        self.retrieval_evaluation = RetrievalEvaluationWidget(
            scene_id=self._node.scene_id,
            graph_database=self._node.graph_database,
            map_file=self._node.map_file,
            queries_file=self._node.queries_file,
            ground_truth_file=self._node.ground_truth_file,
            submit_callback=self._run_evaluation_query,
            cancel_callback=self._node.cancel_navigation,
            campaign_changed_callback=lambda: (
                self.campaign_designer.reload()
                if hasattr(self, 'campaign_designer') else None
            ),
            log_callback=lambda message, error=False: self._append_log(
                message, error=error
            ),
        )
        self.main_tabs.addTab(self.retrieval_evaluation, 'Evaluación')

        self.campaign_designer = CampaignDesignerWidget(
            scene_id=self._node.scene_id,
            graph_database=self._node.graph_database,
            map_file=self._node.map_file,
            queries_file=self._node.queries_file,
            ground_truth_file=self._node.ground_truth_file,
            start_poses_file=self._node.start_poses_file,
            robot_entity_name=self._node.robot_entity_name,
            world_name=self._node.world_name,
            frozen_config_hash=self._node.frozen_config_hash,
            frozen_config_path=self._node.frozen_config_path,
            retrieval_method=self._node.retrieval_method,
            campaign_output_dir=self._node.campaign_output_dir,
            semantic_action_name=str(
                self._node.get_parameter('semantic_action_name').value
            ),
            log_callback=lambda message, error=False: self._append_log(
                message, error=error
            ),
        )
        self.main_tabs.addTab(
            self.campaign_designer, 'Diseño y evaluación de campañas'
        )
        self.main_tabs.currentChanged.connect(self._tab_changed)

    def _teleport_map_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        intro = QLabel(
            'Seleccione directamente un punto del mapa. Azul: robot; verde: '
            'destino en zona aparentemente libre; naranja: revise el destino.'
        )
        intro.setWordWrap(True)
        intro.setObjectName('sectionHint')
        layout.addWidget(intro)

        self.teleport_map_view = _SceneTeleportView(
            self._map_destination_selected
        )
        layout.addWidget(self.teleport_map_view, stretch=1)
        self.teleport_selection = QLabel('No hay un destino seleccionado')
        self.teleport_selection.setWordWrap(True)
        layout.addWidget(self.teleport_selection)

        controls = QHBoxLayout()
        self.teleport_keep_yaw = QCheckBox('Mantener orientación actual')
        self.teleport_keep_yaw.setChecked(True)
        controls.addWidget(self.teleport_keep_yaw)
        controls.addWidget(QLabel('Yaw'))
        self.teleport_yaw = QDoubleSpinBox()
        self.teleport_yaw.setRange(-180.0, 180.0)
        self.teleport_yaw.setDecimals(1)
        self.teleport_yaw.setSuffix('°')
        self.teleport_yaw.setEnabled(False)
        self.teleport_keep_yaw.toggled.connect(
            lambda checked: self.teleport_yaw.setEnabled(not checked)
        )
        controls.addWidget(self.teleport_yaw)
        fit = QPushButton('Ajustar mapa')
        fit.clicked.connect(self.teleport_map_view.fit_map)
        controls.addWidget(fit)
        self.teleport_button = QPushButton('Teletransportar…')
        self.teleport_button.setObjectName('dangerButton')
        self.teleport_button.setEnabled(False)
        self.teleport_button.clicked.connect(self._confirm_teleport)
        controls.addWidget(self.teleport_button)
        layout.addLayout(controls)

        try:
            metadata = load_occupancy_map(self._node.map_file)
            pixmap = QPixmap(metadata.image_path)
            image = QImage(metadata.image_path)
            if pixmap.isNull() or image.isNull():
                raise ValueError(f'Qt no pudo abrir {metadata.image_path}')
            self._teleport_map_metadata = metadata
            self._teleport_image = image
            self.teleport_map_view.set_map(pixmap)
        except Exception as exc:  # noqa: BLE001
            self.teleport_selection.setText(
                f'No se pudo cargar el mapa de la escena: {exc}'
            )
            self.teleport_map_view.setEnabled(False)
        return page

    def _map_destination_selected(
        self, pixel_x: float, pixel_y: float
    ) -> None:
        metadata = self._teleport_map_metadata
        if metadata is None or self._teleport_image.isNull():
            return
        image_x = min(
            self._teleport_image.width() - 1, max(0, int(pixel_x))
        )
        image_y = min(
            self._teleport_image.height() - 1, max(0, int(pixel_y))
        )
        colour = self._teleport_image.pixelColor(image_x, image_y)
        warning = not metadata.pixel_is_free(
            colour.red(), colour.green(), colour.blue()
        )
        world = metadata.pixel_to_world(
            pixel_x, pixel_y, self._teleport_image.height()
        )
        self._selected_teleport = world
        self._selected_teleport_warning = warning
        self.teleport_map_view.set_destination(
            pixel_x, pixel_y, warning=warning
        )
        suffix = (
            ' · ADVERTENCIA: el píxel no parece espacio libre'
            if warning else ' · zona aparentemente libre'
        )
        self.teleport_selection.setText(
            f'Destino: x={world[0]:.3f}, y={world[1]:.3f}{suffix}'
        )
        self.teleport_button.setEnabled(not self._node.teleport_pending)

    def _confirm_teleport(self) -> None:
        if self._selected_teleport is None:
            return
        x, y = self._selected_teleport
        keep_yaw = self.teleport_keep_yaw.isChecked()
        yaw = (
            self._current_yaw
            if keep_yaw else math.radians(self.teleport_yaw.value())
        )
        warning = ''
        if self._selected_teleport_warning:
            warning = (
                '\n\nAdvertencia: el punto no parece una celda libre. '
                'Podría estar dentro de una pared u obstáculo.'
            )
        response = QMessageBox.question(
            self,
            'Confirmar teletransporte',
            f'¿Teletransportar {self._node.robot_entity_name} a:\n\n'
            f'x={x:.3f} m\ny={y:.3f} m\nyaw={math.degrees(yaw):.1f}°?'
            f'{warning}\n\nSe cancelará cualquier navegación activa.',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if response != QMessageBox.Yes:
            return
        self.emergency_stop()
        if self._node.teleport_robot(x, y, yaw):
            self.teleport_button.setEnabled(False)
            self.teleport_selection.setText(
                f'Teletransporte solicitado a x={x:.3f}, y={y:.3f}…'
            )

    def _run_evaluation_query(
        self, query: str, language: str, navigate: bool
    ) -> bool:
        return self._node.send_semantic_navigation(
            query,
            language,
            decision_only=not navigate,
            event_kind='evaluation',
            top_k=1000,
        )

    def _connection_group(self) -> QGroupBox:
        group = QGroupBox('Conexiones ROS 2')
        layout = QGridLayout(group)
        self.camera_status = QLabel('● cámara')
        self.capture_status = QLabel('● captura')
        self.room_status = QLabel('● salas')
        self.tf_status = QLabel('● TF')
        self.navigation_status = QLabel('● navegación')
        for index, label in enumerate(
            (
                self.camera_status,
                self.capture_status,
                self.room_status,
                self.tf_status,
                self.navigation_status,
            )
        ):
            layout.addWidget(label, index // 2, index % 2)
            self._set_indicator(label, False)
        return group

    def _motion_group(self) -> QGroupBox:
        group = QGroupBox('Mover robot — flechas del teclado o botones')
        layout = QGridLayout(group)
        buttons = {
            'forward': ('↑', 0, 1),
            'left': ('←', 1, 0),
            'back': ('↓', 1, 1),
            'right': ('→', 1, 2),
        }
        for direction, (text, row, column) in buttons.items():
            button = QPushButton(text)
            button.setMinimumHeight(42)
            button.pressed.connect(
                lambda selected=direction: self.set_direction(selected, True)
            )
            button.released.connect(
                lambda selected=direction: self.set_direction(selected, False)
            )
            layout.addWidget(button, row, column)
        stop = QPushButton('PARAR')
        stop.setObjectName('dangerButton')
        stop.setStyleSheet('font-weight: bold;')
        stop.clicked.connect(self.emergency_stop)
        layout.addWidget(stop, 2, 0, 1, 3)
        self.linear_speed = self._number_box(0.0, 1.5, self._node.linear_speed, 0.05)
        self.angular_speed = self._number_box(0.0, 3.0, self._node.angular_speed, 0.1)
        layout.addWidget(QLabel('Velocidad m/s'), 3, 0)
        layout.addWidget(self.linear_speed, 3, 1, 1, 2)
        layout.addWidget(QLabel('Giro rad/s'), 4, 0)
        layout.addWidget(self.angular_speed, 4, 1, 1, 2)
        return group

    def _voice_group(self) -> QGroupBox:
        group = QGroupBox('Navegación por voz')
        layout = QGridLayout(group)
        self.voice_language = QComboBox()
        self.voice_language.addItem('Español', 'es')
        self.voice_language.addItem('English', 'en')
        selected_language = self.voice_language.findData(
            self._node.voice_language
        )
        self.voice_language.setCurrentIndex(max(0, selected_language))
        self.voice_prepare = QPushButton('Cargar Whisper')
        self.voice_prepare.clicked.connect(self._prepare_voice)
        layout.addWidget(QLabel('Idioma'), 0, 0)
        layout.addWidget(self.voice_language, 0, 1)
        layout.addWidget(self.voice_prepare, 0, 2)

        self.voice_query = QLineEdit()
        self.voice_query.setPlaceholderText(
            'ej. ve al sofá o navega al punto 1.0, -2.0'
        )
        self.voice_query.returnPressed.connect(self._dispatch_voice_query)
        layout.addWidget(self.voice_query, 1, 0, 1, 3)

        self.voice_ptt = QPushButton('Mantener para hablar')
        self.voice_ptt.setObjectName('voicePttButton')
        self.voice_ptt.setEnabled(False)
        self.voice_ptt.pressed.connect(self._start_voice_recording)
        self.voice_ptt.released.connect(self._node.stop_voice_recording)
        send = QPushButton('Ejecutar texto')
        send.setObjectName('accentButton')
        send.clicked.connect(self._dispatch_voice_query)
        cancel = QPushButton('Cancelar navegación')
        cancel.setObjectName('dangerButton')
        cancel.clicked.connect(self._node.cancel_navigation)
        layout.addWidget(self.voice_ptt, 2, 0)
        layout.addWidget(send, 2, 1)
        layout.addWidget(cancel, 2, 2)

        self.voice_decision_only = QCheckBox('Solo buscar, sin mover el robot')
        layout.addWidget(self.voice_decision_only, 3, 0, 1, 3)
        shortcut = QLabel('⌨ Mantén pulsado ESPACIO para hablar')
        shortcut.setObjectName('shortcutBadge')
        shortcut.setAlignment(Qt.AlignCenter)
        layout.addWidget(shortcut, 4, 0, 1, 3)
        self.voice_stage = QLabel('Whisper sin cargar')
        self.voice_stage.setWordWrap(True)
        layout.addWidget(self.voice_stage, 5, 0, 1, 3)
        return group

    def _capture_group(self) -> QGroupBox:
        group = QGroupBox('Nodos y observaciones')
        layout = QFormLayout(group)
        self.node_label = QLineEdit()
        self.node_label.setPlaceholderText('vacío = siguiente nombre W1, W2, …')
        self.views = QLineEdit('0, 90, 180, 270')
        layout.addRow('Nombre del nodo', self.node_label)
        layout.addRow('Vistas relativas (°)', self.views)
        buttons = QHBoxLayout()
        single = QPushButton('Crear nodo (1 vista)')
        single.clicked.connect(self._capture_single)
        multiview = QPushButton('Tomar observaciones')
        multiview.clicked.connect(self._capture_multiview)
        cancel = QPushButton('Cancelar')
        cancel.clicked.connect(self._node.cancel_capture)
        buttons.addWidget(single)
        buttons.addWidget(multiview)
        buttons.addWidget(cancel)
        layout.addRow(buttons)
        self.capture_progress = QProgressBar()
        self.capture_progress.setRange(0, 100)
        self.capture_progress.setValue(0)
        self.capture_stage = QLabel('Sin captura activa')
        layout.addRow(self.capture_progress)
        layout.addRow(self.capture_stage)
        return group

    def _room_group(self) -> QGroupBox:
        group = QGroupBox('Crear sala poligonal')
        layout = QGridLayout(group)
        self.room_name = QLineEdit()
        self.room_name.setPlaceholderText('ej. salon, cocina, despacho')
        layout.addWidget(QLabel('Nombre'), 0, 0)
        layout.addWidget(self.room_name, 0, 1, 1, 3)

        self.ax = self._coordinate_box()
        self.ay = self._coordinate_box()
        self.bx = self._coordinate_box()
        self.by = self._coordinate_box()
        layout.addWidget(QLabel('Esquina A'), 1, 0)
        layout.addWidget(self.ax, 1, 1)
        layout.addWidget(self.ay, 1, 2)
        layout.addWidget(QLabel('x / y'), 1, 3)
        layout.addWidget(QLabel('Esquina B'), 2, 0)
        layout.addWidget(self.bx, 2, 1)
        layout.addWidget(self.by, 2, 2)
        layout.addWidget(QLabel('x / y'), 2, 3)

        self.room_polygon = QLineEdit()
        self.room_polygon.setPlaceholderText('opcional: x,y; x,y; x,y; …')
        layout.addWidget(QLabel('Vértices'), 3, 0)
        layout.addWidget(self.room_polygon, 3, 1, 1, 3)
        self.room_transition = self._number_box(0.0, 5.0, 0.5, 0.1)
        layout.addWidget(QLabel('Transición (m)'), 4, 0)
        layout.addWidget(self.room_transition, 4, 1)

        a_robot = QPushButton('A ← pose robot')
        b_robot = QPushButton('B ← pose robot')
        a_click = QPushButton('A ← punto RViz')
        b_click = QPushButton('B ← punto RViz')
        a_robot.clicked.connect(lambda: self._set_corner('a', 'robot'))
        b_robot.clicked.connect(lambda: self._set_corner('b', 'robot'))
        a_click.clicked.connect(lambda: self._set_corner('a', 'rviz'))
        b_click.clicked.connect(lambda: self._set_corner('b', 'rviz'))
        layout.addWidget(a_robot, 5, 0, 1, 2)
        layout.addWidget(b_robot, 5, 2, 1, 2)
        layout.addWidget(a_click, 6, 0, 1, 2)
        layout.addWidget(b_click, 6, 2, 1, 2)

        create = QPushButton('Crear / actualizar sala')
        create.clicked.connect(self._create_room)
        layout.addWidget(create, 7, 0, 1, 4)
        self.room_list = QListWidget()
        self.room_list.setMaximumHeight(85)
        layout.addWidget(self.room_list, 8, 0, 1, 4)
        return group

    @staticmethod
    def _number_box(
        minimum: float,
        maximum: float,
        value: float,
        step: float,
    ) -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setRange(minimum, maximum)
        box.setDecimals(2)
        box.setSingleStep(step)
        box.setValue(value)
        return box

    @classmethod
    def _coordinate_box(cls) -> QDoubleSpinBox:
        box = cls._number_box(-1000.0, 1000.0, 0.0, 0.1)
        box.setDecimals(3)
        return box

    @staticmethod
    def _set_indicator(label: QLabel, connected: bool) -> None:
        color = '#39a85b' if connected else '#b74a4a'
        label.setStyleSheet(f'color: {color}; font-weight: bold;')

    def set_direction(self, direction: str, pressed: bool) -> None:
        if pressed:
            self._directions.add(direction)
        else:
            self._directions.discard(direction)
        self._publish_motion()

    def _publish_motion(self) -> None:
        if self._node.capture_active or self._node.navigation_active:
            if self._last_motion != (0.0, 0.0):
                self._node.stop()
                self._last_motion = (0.0, 0.0)
            return
        motion = motion_from_directions(
            self._directions,
            self.linear_speed.value(),
            self.angular_speed.value(),
        )
        if self._directions or motion != self._last_motion:
            self._node.publish_motion(*motion)
        self._last_motion = motion

    def emergency_stop(self) -> None:
        self._directions.clear()
        self._last_motion = (0.0, 0.0)
        if self._node.navigation_active:
            self._node.cancel_navigation()
        self._node.stop()
        self._append_log('Parada enviada.')

    def _prepare_voice(self) -> None:
        language = str(self.voice_language.currentData())
        self._node.prepare_voice(language)

    def _start_voice_recording(self) -> bool:
        self.emergency_stop()
        return self._node.start_voice_recording()

    def _dispatch_voice_query(self, text: str | None = None) -> None:
        if not isinstance(text, str):
            text = None
        transcript = (text if text is not None else self.voice_query.text()).strip()
        if not transcript:
            QMessageBox.warning(
                self,
                'Comando vacío',
                'Escribe una consulta o graba una orden de voz.',
            )
            return
        self.voice_query.setText(transcript)
        intent = parse(transcript)
        self._append_log(f'Orden interpretada: {intent}')
        if isinstance(intent, NoIntent):
            self.voice_stage.setText(f'Orden no reconocida: {intent.reason}')
            return
        if isinstance(intent, MoveToPosition):
            self._node.send_pose_navigation(intent.x, intent.y)
            return
        if isinstance(intent, SaveWaypoint):
            self.node_label.setText(intent.label)
            self._capture_single()
            return
        if isinstance(intent, SemanticGoal):
            self._node.send_semantic_navigation(
                intent.query,
                str(self.voice_language.currentData()),
                self.voice_decision_only.isChecked(),
            )

    def _capture_single(self) -> None:
        self.emergency_stop()
        self._node.send_capture(self.node_label.text(), [], False)

    def _capture_multiview(self) -> None:
        try:
            views = parse_view_angles(self.views.text())
        except ValueError as exc:
            QMessageBox.warning(self, 'Vistas no válidas', str(exc))
            return
        if not views:
            QMessageBox.warning(
                self,
                'Vistas no válidas',
                'Introduce al menos un ángulo para la captura multivista.',
            )
            return
        self.emergency_stop()
        self._node.send_capture(self.node_label.text(), views, True)

    def _set_corner(self, corner: str, source: str) -> None:
        position = (
            self._current_position
            if source == 'robot'
            else self._node.latest_clicked_position()
        )
        if position is None:
            QMessageBox.warning(
                self,
                'Posición no disponible',
                'Todavía no hay una pose TF o un punto de RViz disponible.',
            )
            return
        x_box, y_box = (self.ax, self.ay) if corner == 'a' else (self.bx, self.by)
        x_box.setValue(position[0])
        y_box.setValue(position[1])

    def _create_room(self) -> None:
        polygon: list[tuple[float, float]] = []
        raw_polygon = self.room_polygon.text().strip()
        if raw_polygon:
            try:
                parsed = [
                    tuple(float(value.strip()) for value in vertex.split(','))
                    for vertex in raw_polygon.split(';') if vertex.strip()
                ]
                if len(parsed) < 3 or any(len(vertex) != 2 for vertex in parsed):
                    raise ValueError
                polygon = [(vertex[0], vertex[1]) for vertex in parsed]
            except (TypeError, ValueError):
                QMessageBox.warning(
                    self, 'Polígono no válido',
                    'Usa al menos tres vértices con formato x,y; x,y; x,y.',
                )
                return
        self._node.create_room(
            self.room_name.text(),
            (self.ax.value(), self.ay.value()),
            (self.bx.value(), self.by.value()),
            polygon=polygon or None,
            transition_width_m=self.room_transition.value(),
        )

    def _update_fast(self) -> None:
        if not rclpy.ok():
            QApplication.quit()
            return
        self._publish_motion()
        self._update_camera()
        for kind, payload in self._node.pop_events():
            self._handle_event(kind, payload)

    def _update_camera(self) -> None:
        message = self._node.latest_image()
        if message is None or message is self._last_image:
            return
        self._last_image = message
        try:
            frame = self._bridge.imgmsg_to_cv2(message, desired_encoding='rgb8')
            height, width = frame.shape[:2]
            bytes_per_line = int(frame.strides[0])
            image = QImage(
                frame.data,
                width,
                height,
                bytes_per_line,
                QImage.Format_RGB888,
            ).copy()
            pixmap = QPixmap.fromImage(image).scaled(
                self.camera_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            self.camera_label.setPixmap(pixmap)
        except Exception as exc:  # noqa: BLE001
            self.camera_label.setText(f'Error convirtiendo imagen: {exc}')

    def _update_connections(self) -> None:
        age = self._node.camera_age()
        self._set_indicator(self.camera_status, age is not None and age < 2.0)
        self._set_indicator(self.capture_status, self._node.capture_server_ready())
        self._set_indicator(self.room_status, self._node.room_service_ready())
        self._set_indicator(
            self.navigation_status,
            self._node.semantic_server_ready(),
        )
        current_pose = self._node.current_pose()
        self._current_position = (
            None if current_pose is None else current_pose[:2]
        )
        self._set_indicator(self.tf_status, current_pose is not None)
        if current_pose is None:
            self.pose_label.setText('Pose map → base_link: no disponible')
        else:
            x, y, self._current_yaw = current_pose
            self.pose_label.setText(
                f'Pose map → base_link: x={x:.3f}, y={y:.3f}, '
                f'yaw={math.degrees(self._current_yaw):.1f}°'
            )
            if (
                self._teleport_map_metadata is not None
                and not self._teleport_image.isNull()
            ):
                pixel = self._teleport_map_metadata.world_to_pixel(
                    x, y, self._teleport_image.height()
                )
                self.teleport_map_view.set_robot(*pixel)

    def _handle_event(self, kind: str, payload: dict[str, Any]) -> None:
        if payload.get('kind') == 'evaluation' and kind in (
            'navigation_started',
            'navigation_feedback',
            'navigation_result',
            'error',
        ):
            self.retrieval_evaluation.handle_event(kind, payload)
            message = payload.get('message')
            if message:
                self._append_log(str(message), error=kind == 'error')
            return
        if kind == 'teleport_started':
            self.teleport_button.setEnabled(False)
            self.teleport_selection.setText(str(payload.get('message', '')))
        elif kind == 'teleport_result':
            self.teleport_button.setEnabled(
                self._selected_teleport is not None
            )
            self.teleport_selection.setText(str(payload.get('message', '')))
        elif kind == 'voice_loading':
            self.voice_prepare.setEnabled(False)
            self.voice_language.setEnabled(False)
            self.voice_ptt.setEnabled(False)
            self.voice_stage.setText('Cargando Whisper…')
        elif kind == 'voice_ready':
            self.voice_prepare.setEnabled(False)
            self.voice_language.setEnabled(False)
            self.voice_ptt.setEnabled(True)
            device = payload.get('device', 'preparado')
            self.voice_stage.setText(f'Whisper listo ({device})')
        elif kind == 'voice_recording':
            self.voice_ptt.setText('● Escuchando; suelta para enviar')
            self.voice_ptt.setStyleSheet(
                'font-weight: bold; background: #9e2f2f; color: white;'
            )
            self.voice_stage.setText('Grabando…')
        elif kind == 'voice_transcribing':
            self.voice_ptt.setEnabled(False)
            self.voice_ptt.setText('Transcribiendo…')
            self.voice_stage.setText('Transcribiendo audio…')
        elif kind == 'voice_transcript':
            transcript = str(payload['transcript'])
            self._reset_voice_button()
            self.voice_stage.setText(f'Reconocido: «{transcript}»')
            self._append_log(f'Transcripción: «{transcript}»')
            self._dispatch_voice_query(transcript)
        elif kind in ('voice_idle', 'voice_error'):
            self._reset_voice_button()
            self.voice_stage.setText(str(payload.get('message', 'Voz inactiva')))
        elif kind == 'navigation_started':
            self.voice_stage.setText('Objetivo enviado; esperando aceptación…')
        elif kind == 'navigation_feedback':
            stage = str(payload['stage'])
            distance = float(payload['distance'])
            if stage == 'navigating':
                self.voice_stage.setText(
                    f'Navegando — distancia restante: {distance:.2f} m'
                )
            else:
                self.voice_stage.setText(f'Navegación semántica: {stage}')
        elif kind == 'navigation_result':
            if payload['success']:
                matched = str(payload.get('matched_node', ''))
                suffix = f' — nodo {matched}' if matched else ''
                self.voice_stage.setText(f'Objetivo completado{suffix}')
            else:
                failure = payload.get('failure_type') or payload.get('message')
                self.voice_stage.setText(f'Navegación fallida: {failure}')
        elif kind == 'capture_started':
            self.capture_progress.setValue(0)
            self.capture_stage.setText('Enviando objetivo…')
        elif kind == 'capture_feedback':
            current = payload['current']
            total = max(1, payload['total'])
            self.capture_progress.setValue(int(100 * current / total))
            self.capture_stage.setText(
                f"{payload['stage']} — vista {current}/{total}"
            )
        elif kind == 'capture_result':
            if payload['success']:
                self.capture_progress.setValue(100)
                suffix = ' (fusionado)' if payload['merged'] else ''
                self.capture_stage.setText(
                    f"Nodo {payload['node_id']}: {payload['views']} vista(s){suffix}"
                )
            else:
                self.capture_progress.setValue(0)
                self.capture_stage.setText(f"Error: {payload['message']}")
        elif kind == 'room_result':
            self._refresh_rooms()
        message = payload.get('message')
        if message:
            self._append_log(
                str(message),
                error=(
                    kind in ('error', 'voice_error')
                    or (kind == 'teleport_result' and not payload.get('success'))
                ),
            )

    def _reset_voice_button(self) -> None:
        loading, ready, _recording = self._node.voice_state()
        self._space_voice_active = False
        self.voice_ptt.setDown(False)
        self.voice_ptt.setText('Mantener para hablar')
        self.voice_ptt.setStyleSheet('')
        self.voice_ptt.setEnabled(ready)
        self.voice_prepare.setEnabled(not ready and not loading)
        self.voice_language.setEnabled(not ready and not loading)

    def _refresh_rooms(self) -> None:
        self.room_list.clear()
        for room in sorted(self._node.rooms_snapshot(), key=lambda value: value.room_id):
            self.room_list.addItem(
                f'{room.room_id}: {len(room.corners())} vértices · '
                f'transición {room.transition_width_m:.2f} m'
            )

    def _append_log(self, message: str, *, error: bool = False) -> None:
        timestamp = time.strftime('%H:%M:%S')
        color = '#d45b5b' if error else '#c7cbd1'
        self.log.append(f"<span style='color:{color}'>[{timestamp}] {message}</span>")

    def closeEvent(self, event) -> None:  # noqa: N802
        self.emergency_stop()
        self._node.shutdown_voice()
        application = QApplication.instance()
        if application is not None:
            application.removeEventFilter(self._key_filter)
        super().closeEvent(event)


def main(args=None) -> None:
    rclpy.init(args=args)
    application = QApplication.instance() or QApplication([sys.argv[0]])
    node = SemanticOperatorNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    window = SemanticOperatorWindow(node)
    window.show()
    signal.signal(signal.SIGINT, lambda _signum, _frame: application.quit())
    try:
        application.exec_()
    finally:
        node.shutdown_voice()
        node.stop()
        executor.shutdown(timeout_sec=2.0)
        spin_thread.join(timeout=2.0)
        executor.remove_node(node)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
