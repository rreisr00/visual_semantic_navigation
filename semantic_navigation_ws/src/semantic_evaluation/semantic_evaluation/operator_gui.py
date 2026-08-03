#!/usr/bin/env python3
"""Graphical operator console for semantic mapping campaigns."""

from __future__ import annotations

import os
import queue
import signal
import sys
import threading
import time
from typing import Any

from cv_bridge import CvBridge
from geometry_msgs.msg import Point, PointStamped, Twist, TwistStamped
from python_qt_binding.QtCore import QEvent, QObject, Qt, QTimer
from python_qt_binding.QtGui import QImage, QPixmap
from python_qt_binding.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QFormLayout,
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
    QSplitter,
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
from semantic_evaluation.core.operator_gui_logic import (
    motion_from_directions,
    normalized_room_bounds,
    parse_view_angles,
)
from semantic_interfaces.action import CaptureWaypoint
from semantic_interfaces.srv import AddRoom
from semantic_navigation_core.rooms import load_rooms, Room, save_rooms
from sensor_msgs.msg import Image
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray


class SemanticOperatorNode(Node):
    """ROS-facing backend used by the Qt operator window."""

    def __init__(self) -> None:
        super().__init__('semantic_operator_gui')
        self.declare_parameter('camera_topic', '/camera/image_raw')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('stamped_cmd_vel', True)
        self.declare_parameter('capture_action_name', '/capture_waypoint')
        self.declare_parameter('add_room_service', '/add_room')
        self.declare_parameter('clicked_point_topic', '/clicked_point')
        self.declare_parameter('room_marker_topic', '/room_markers')
        self.declare_parameter('scene_id', 'aws_small_house')
        self.declare_parameter('rooms_file', '')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('linear_speed', 0.25)
        self.declare_parameter('angular_speed', 0.7)

        self.scene_id = str(self.get_parameter('scene_id').value)
        self.camera_topic = str(self.get_parameter('camera_topic').value)
        self.linear_speed = float(self.get_parameter('linear_speed').value)
        self.angular_speed = float(self.get_parameter('angular_speed').value)
        self._map_frame = str(self.get_parameter('map_frame').value)
        self._base_frame = str(self.get_parameter('base_frame').value)
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
        self._rooms_lock = threading.Lock()

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
        self._room_client = self.create_client(
            AddRoom,
            str(self.get_parameter('add_room_service').value),
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
        try:
            transform = self._tf_buffer.lookup_transform(
                self._map_frame,
                self._base_frame,
                Time(),
            )
        except TransformException:
            return None
        translation = transform.transform.translation
        return float(translation.x), float(translation.y)

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

    def capture_server_ready(self) -> bool:
        return self._capture_client.server_is_ready()

    def room_service_ready(self) -> bool:
        return self._room_client.service_is_ready()

    def send_capture(
        self,
        label: str,
        relative_views: list[float],
        rotate_robot: bool,
    ) -> bool:
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
    ) -> bool:
        label = room_id.strip()
        if not label:
            self.push_event('error', message='La sala necesita un nombre.')
            return False
        if not self._room_client.service_is_ready():
            self.push_event('error', message='El servicio /add_room no está disponible.')
            return False
        try:
            bounds = normalized_room_bounds(corner_a, corner_b)
        except ValueError as exc:
            self.push_event('error', message=str(exc))
            return False
        room = Room(label, *bounds)
        request = AddRoom.Request()
        request.room_id = room.room_id
        request.min_x = room.min_x
        request.min_y = room.min_y
        request.max_x = room.max_x
        request.max_y = room.max_y
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
    """Capture arrow key press/release events regardless of focused widget."""

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
        direction = self.KEY_DIRECTIONS.get(getattr(event, 'key', lambda: None)())
        if direction is None or event.isAutoRepeat():
            return super().eventFilter(watched, event)
        if event.type() == QEvent.KeyPress:
            self._window.set_direction(direction, True)
            return True
        if event.type() == QEvent.KeyRelease:
            self._window.set_direction(direction, False)
            return True
        return super().eventFilter(watched, event)


class SemanticOperatorWindow(QMainWindow):
    """Qt window for camera, teleoperation, capture and room definition."""

    def __init__(self, node: SemanticOperatorNode) -> None:
        super().__init__()
        self._node = node
        self._bridge = CvBridge()
        self._last_image: Image | None = None
        self._directions: set[str] = set()
        self._last_motion = (0.0, 0.0)
        self._current_position: tuple[float, float] | None = None
        self._key_filter = ArrowKeyFilter(self)

        self.setWindowTitle(f'Operador semántico — {node.scene_id}')
        self.resize(1180, 760)
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

    def _build_ui(self) -> None:
        root = QSplitter(Qt.Horizontal)
        self.setCentralWidget(root)

        camera_panel = QWidget()
        camera_layout = QVBoxLayout(camera_panel)
        self.camera_label = QLabel(f'Esperando {self._node.camera_topic}…')
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setStyleSheet(
            'background: #17191c; color: #b8bec7; border: 1px solid #3b4048;'
        )
        camera_layout.addWidget(self.camera_label, stretch=1)
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
        root.addWidget(controls)
        root.setStretchFactor(0, 3)
        root.setStretchFactor(1, 2)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(150)
        camera_layout.addWidget(self.log)

    def _connection_group(self) -> QGroupBox:
        group = QGroupBox('Conexiones ROS 2')
        layout = QGridLayout(group)
        self.camera_status = QLabel('● cámara')
        self.capture_status = QLabel('● captura')
        self.room_status = QLabel('● salas')
        self.tf_status = QLabel('● TF')
        for index, label in enumerate(
            (self.camera_status, self.capture_status, self.room_status, self.tf_status)
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
        stop.setStyleSheet('font-weight: bold; background: #9e2f2f; color: white;')
        stop.clicked.connect(self.emergency_stop)
        layout.addWidget(stop, 2, 0, 1, 3)
        self.linear_speed = self._number_box(0.0, 1.5, self._node.linear_speed, 0.05)
        self.angular_speed = self._number_box(0.0, 3.0, self._node.angular_speed, 0.1)
        layout.addWidget(QLabel('Velocidad m/s'), 3, 0)
        layout.addWidget(self.linear_speed, 3, 1, 1, 2)
        layout.addWidget(QLabel('Giro rad/s'), 4, 0)
        layout.addWidget(self.angular_speed, 4, 1, 1, 2)
        return group

    def _capture_group(self) -> QGroupBox:
        group = QGroupBox('Nodos y observaciones')
        layout = QFormLayout(group)
        self.node_label = QLineEdit()
        self.node_label.setPlaceholderText('vacío = nombre automático según sala')
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
        group = QGroupBox('Crear sala rectangular')
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

        a_robot = QPushButton('A ← pose robot')
        b_robot = QPushButton('B ← pose robot')
        a_click = QPushButton('A ← punto RViz')
        b_click = QPushButton('B ← punto RViz')
        a_robot.clicked.connect(lambda: self._set_corner('a', 'robot'))
        b_robot.clicked.connect(lambda: self._set_corner('b', 'robot'))
        a_click.clicked.connect(lambda: self._set_corner('a', 'rviz'))
        b_click.clicked.connect(lambda: self._set_corner('b', 'rviz'))
        layout.addWidget(a_robot, 3, 0, 1, 2)
        layout.addWidget(b_robot, 3, 2, 1, 2)
        layout.addWidget(a_click, 4, 0, 1, 2)
        layout.addWidget(b_click, 4, 2, 1, 2)

        create = QPushButton('Crear / actualizar sala')
        create.clicked.connect(self._create_room)
        layout.addWidget(create, 5, 0, 1, 4)
        self.room_list = QListWidget()
        self.room_list.setMaximumHeight(85)
        layout.addWidget(self.room_list, 6, 0, 1, 4)
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
        if self._node.capture_active:
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
        self._node.stop()
        self._append_log('Parada enviada.')

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
        self._node.create_room(
            self.room_name.text(),
            (self.ax.value(), self.ay.value()),
            (self.bx.value(), self.by.value()),
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
        self._current_position = self._node.current_position()
        self._set_indicator(self.tf_status, self._current_position is not None)
        if self._current_position is None:
            self.pose_label.setText('Pose map → base_link: no disponible')
        else:
            x, y = self._current_position
            self.pose_label.setText(f'Pose map → base_link: x={x:.3f}, y={y:.3f}')

    def _handle_event(self, kind: str, payload: dict[str, Any]) -> None:
        if kind == 'capture_started':
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
            self._append_log(str(message), error=kind == 'error')

    def _refresh_rooms(self) -> None:
        self.room_list.clear()
        for room in sorted(self._node.rooms_snapshot(), key=lambda value: value.room_id):
            self.room_list.addItem(
                f'{room.room_id}: ({room.min_x:.2f}, {room.min_y:.2f}) → '
                f'({room.max_x:.2f}, {room.max_y:.2f})'
            )

    def _append_log(self, message: str, *, error: bool = False) -> None:
        timestamp = time.strftime('%H:%M:%S')
        color = '#d45b5b' if error else '#c7cbd1'
        self.log.append(f"<span style='color:{color}'>[{timestamp}] {message}</span>")

    def closeEvent(self, event) -> None:  # noqa: N802
        self.emergency_stop()
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
        node.stop()
        executor.shutdown(timeout_sec=2.0)
        spin_thread.join(timeout=2.0)
        executor.remove_node(node)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
