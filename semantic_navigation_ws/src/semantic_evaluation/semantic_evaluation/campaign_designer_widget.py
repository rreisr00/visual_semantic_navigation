"""Interactive Qt editor for scene queries and semantic ground truth."""

from __future__ import annotations

import math
import os
from collections.abc import Callable

from python_qt_binding.QtCore import QPointF, Qt
from python_qt_binding.QtGui import (
    QBrush,
    QColor,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
)
from python_qt_binding.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGraphicsEllipseItem,
    QGraphicsScene,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from semantic_evaluation.campaign_runner_widget import CampaignRunnerWidget
from semantic_evaluation.core.campaign_designer import (
    CampaignWorkspace,
    load_workspace,
    save_workspace,
)
from semantic_evaluation.core.relation_layout import (
    non_overlapping_label_rects,
    radial_node_positions,
    relation_lane_offsets,
)


def _csv_values(text: str) -> list[str]:
    """Return stable, unique comma-separated identifiers."""
    values: list[str] = []
    for raw in text.replace(";", ",").split(","):
        value = raw.strip()
        if value and value not in values:
            values.append(value)
    return values


def _set_csv(widget: QLineEdit, values: list[str]) -> None:
    widget.setText(", ".join(str(value) for value in values))


class _NodeItem(QGraphicsEllipseItem):
    """Selectable map marker carrying a semantic node identifier."""

    def __init__(self, node_id: str, x: float, y: float) -> None:
        radius = 7.0
        super().__init__(x - radius, y - radius, radius * 2.0, radius * 2.0)
        self.node_id = node_id
        self.setFlag(QGraphicsEllipseItem.ItemIsSelectable, True)
        self.setBrush(QBrush(QColor("#2e86de")))
        self.setPen(QPen(QColor("white"), 1.5))
        self.setToolTip(node_id)
        self.setZValue(3.0)


class _MapView(QGraphicsView):
    """Graphics view with mouse-centred wheel zoom and hand panning."""

    def __init__(self, scene: QGraphicsScene) -> None:
        super().__init__(scene)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setDragMode(QGraphicsView.ScrollHandDrag)

    def wheelEvent(self, event) -> None:  # noqa: N802
        factor = 1.18 if event.angleDelta().y() > 0 else 1.0 / 1.18
        self.scale(factor, factor)
        event.accept()


class CampaignDesignerWidget(QWidget):
    """Map-centred workspace for inspecting and annotating one scene graph."""

    def __init__(
        self,
        *,
        scene_id: str,
        graph_database: str,
        map_file: str,
        queries_file: str,
        ground_truth_file: str,
        start_poses_file: str = "",
        robot_entity_name: str = "semantic_robot",
        world_name: str = "default",
        frozen_config_hash: str = "",
        frozen_config_path: str = "",
        retrieval_method: str = "hybrid_semantic_retrieval",
        campaign_output_dir: str = "",
        semantic_action_name: str = "/navigate_to_semantic_goal",
        log_callback: Callable[[str, bool], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._scene_id = scene_id
        self._graph_database = graph_database
        self._map_file = map_file
        self._queries_file = queries_file
        self._ground_truth_file = ground_truth_file
        self._start_poses_file = start_poses_file
        self._robot_entity_name = robot_entity_name
        self._world_name = world_name
        self._frozen_config_hash = frozen_config_hash
        self._frozen_config_path = frozen_config_path
        self._retrieval_method = retrieval_method
        self._campaign_output_dir = campaign_output_dir
        self._semantic_action_name = semantic_action_name
        self._log_callback = log_callback
        self._workspace: CampaignWorkspace | None = None
        self._selected_node_id = ""
        self._selected_case_id = ""
        self._node_items: dict[str, _NodeItem] = {}
        self._build_ui()
        self.reload()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        toolbar = QHBoxLayout()
        self.scene_label = QLabel(f"Escena: {self._scene_id}")
        self.scene_label.setStyleSheet("font-weight: bold")
        toolbar.addWidget(self.scene_label)
        toolbar.addStretch(1)
        reload_button = QPushButton("Recargar grafo")
        reload_button.clicked.connect(self.reload)
        toolbar.addWidget(reload_button)
        save_button = QPushButton("Guardar campaña y ground truth")
        save_button.setObjectName('accentButton')
        save_button.clicked.connect(self.save)
        toolbar.addWidget(save_button)
        root.addLayout(toolbar)

        self.path_label = QLabel()
        self.path_label.setWordWrap(True)
        root.addWidget(self.path_label)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter, stretch=1)

        map_panel = QWidget()
        map_layout = QVBoxLayout(map_panel)
        self.map_scene = QGraphicsScene(self)
        self.map_scene.selectionChanged.connect(self._map_selection_changed)
        self.map_view = _MapView(self.map_scene)
        self.map_view.setRenderHint(QPainter.Antialiasing, True)
        map_layout.addWidget(self.map_view, stretch=1)
        self.node_label = QLabel("Seleccione un nodo en el mapa")
        map_layout.addWidget(self.node_label)
        splitter.addWidget(map_panel)

        self.inspector = QTabWidget()
        self.inspector.addTab(self._observations_tab(), "Observaciones")
        self.inspector.addTab(self._ground_truth_tab(), "Ground truth")
        self.inspector.addTab(self._queries_tab(), "Queries de escena")
        self.campaign_runner = CampaignRunnerWidget(
            scene_id=self._scene_id,
            graph_database=self._graph_database,
            map_file=self._map_file,
            queries_file=self._queries_file,
            ground_truth_file=self._ground_truth_file,
            start_poses_file=self._start_poses_file,
            robot_entity_name=self._robot_entity_name,
            world_name=self._world_name,
            frozen_config_hash=self._frozen_config_hash,
            frozen_config_path=self._frozen_config_path,
            method=self._retrieval_method,
            output_dir=self._campaign_output_dir,
            action_name=self._semantic_action_name,
            pre_run_callback=self._save_for_campaign,
            log_callback=self._log,
        )
        self.inspector.addTab(
            self.campaign_runner, "Validación y ejecución"
        )
        splitter.addWidget(self.inspector)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(7)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([760, 600])

    def _observations_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        self.observation_combo = QComboBox()
        self.observation_combo.currentIndexChanged.connect(
            self._show_selected_observation
        )
        layout.addWidget(self.observation_combo)
        self.observation_meta = QLabel("Sin nodo seleccionado")
        self.observation_meta.setWordWrap(True)
        layout.addWidget(self.observation_meta)

        vertical = QSplitter(Qt.Vertical)
        vertical.setChildrenCollapsible(False)
        vertical.setHandleWidth(7)

        self.observation_image = QLabel("Sin imagen")
        self.observation_image.setAlignment(Qt.AlignCenter)
        self.observation_image.setMinimumHeight(150)
        self.observation_image.setStyleSheet(
            "background: #17191c; color: #b8bec7; border: 1px solid #3b4048;"
        )
        vertical.addWidget(self.observation_image)

        details = QSplitter(Qt.Horizontal)
        details.setChildrenCollapsible(False)
        details.setHandleWidth(7)
        objects_panel = QWidget()
        objects_layout = QVBoxLayout(objects_panel)
        objects_layout.setContentsMargins(0, 0, 0, 0)
        objects_layout.addWidget(QLabel("Objetos detectados"))
        self.objects_table = QTableWidget(0, 4)
        self.objects_table.setHorizontalHeaderLabels(
            ["Objeto / instancia", "Clase", "Confianza", "Room"]
        )
        self.objects_table.setAlternatingRowColors(True)
        self.objects_table.horizontalHeader().setStretchLastSection(True)
        objects_layout.addWidget(self.objects_table)
        details.addWidget(objects_panel)

        relations_panel = QWidget()
        relations_layout = QVBoxLayout(relations_panel)
        relations_layout.setContentsMargins(0, 0, 0, 0)
        relations_layout.addWidget(QLabel("Relaciones entre objetos"))
        relation_hint = QLabel(
            'Cada relación tiene un color y número propios. Use la rueda para '
            'hacer zoom y arrastre para desplazarse.'
        )
        relation_hint.setWordWrap(True)
        relation_hint.setObjectName('sectionHint')
        relations_layout.addWidget(relation_hint)
        self.relation_scene = QGraphicsScene(self)
        self.relation_view = _MapView(self.relation_scene)
        self.relation_view.setRenderHint(QPainter.Antialiasing, True)
        self.relation_view.setMinimumSize(260, 170)
        relations_layout.addWidget(self.relation_view)
        details.addWidget(relations_panel)
        details.setSizes([360, 520])
        vertical.addWidget(details)
        vertical.setSizes([300, 330])
        layout.addWidget(vertical, stretch=1)
        return page

    def _ground_truth_tab(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        contents = QWidget()
        layout = QVBoxLayout(contents)

        room_group = QGroupBox("Correspondencia nodo → room")
        room_layout = QVBoxLayout(room_group)
        self.room_combo = QComboBox()
        self.room_combo.setEditable(True)
        room_layout.addWidget(self.room_combo)
        room_button = QPushButton("Asignar nodo a room")
        room_button.clicked.connect(self._assign_room)
        room_layout.addWidget(room_button)
        self.room_membership = QLabel("Sin nodo seleccionado")
        self.room_membership.setWordWrap(True)
        room_layout.addWidget(self.room_membership)
        layout.addWidget(room_group)

        object_group = QGroupBox("Objetos válidos para el nodo")
        object_layout = QVBoxLayout(object_group)
        self.gt_objects = QListWidget()
        object_layout.addWidget(self.gt_objects)
        object_form = QFormLayout()
        self.object_category = QLineEdit()
        self.object_room = QLineEdit()
        self.object_level = QComboBox()
        self.object_level.addItems(["exact", "nearby"])
        object_form.addRow("Categoría", self.object_category)
        object_form.addRow("Room", self.object_room)
        object_form.addRow("Validez", self.object_level)
        object_layout.addLayout(object_form)
        object_buttons = QHBoxLayout()
        add_object = QPushButton("Añadir / actualizar")
        add_object.clicked.connect(self._add_gt_object)
        remove_object = QPushButton("Quitar del nodo")
        remove_object.clicked.connect(self._remove_gt_object)
        object_buttons.addWidget(add_object)
        object_buttons.addWidget(remove_object)
        object_layout.addLayout(object_buttons)
        layout.addWidget(object_group)

        relation_group = QGroupBox("Relaciones válidas para el nodo")
        relation_layout = QVBoxLayout(relation_group)
        self.gt_relations = QListWidget()
        relation_layout.addWidget(self.gt_relations)
        relation_form = QFormLayout()
        self.relation_subject = QLineEdit()
        self.relation_predicate = QComboBox()
        self.relation_predicate.setEditable(True)
        self.relation_predicate.addItems(
            ["LEFT_OF", "RIGHT_OF", "ABOVE", "BELOW", "INSIDE", "CONTAINS",
             "POSSIBLY_ON_TOP_OF", "NEAR"]
        )
        self.relation_object = QLineEdit()
        self.relation_type = QComboBox()
        self.relation_type.setEditable(True)
        self.relation_type.addItems(
            ["visual_2d_hypothesis", "depth_3d", "manual_ground_truth"]
        )
        relation_form.addRow("Sujeto", self.relation_subject)
        relation_form.addRow("Predicado", self.relation_predicate)
        relation_form.addRow("Objeto", self.relation_object)
        relation_form.addRow("Tipo", self.relation_type)
        relation_layout.addLayout(relation_form)
        relation_buttons = QHBoxLayout()
        add_relation = QPushButton("Añadir / actualizar")
        add_relation.clicked.connect(self._add_gt_relation)
        remove_relation = QPushButton("Quitar del nodo")
        remove_relation.clicked.connect(self._remove_gt_relation)
        relation_buttons.addWidget(add_relation)
        relation_buttons.addWidget(remove_relation)
        relation_layout.addLayout(relation_buttons)
        layout.addWidget(relation_group)
        layout.addStretch(1)
        scroll.setWidget(contents)
        outer.addWidget(scroll)
        return page

    def _queries_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        suite_form = QFormLayout()
        self.suite_id = QLineEdit()
        suite_form.addRow("Suite / campaña", self.suite_id)
        layout.addLayout(suite_form)

        query_splitter = QSplitter(Qt.Vertical)
        query_splitter.setChildrenCollapsible(False)
        query_splitter.setHandleWidth(7)
        layout.addWidget(query_splitter, stretch=1)
        self.query_list = QListWidget()
        self.query_list.currentItemChanged.connect(self._query_selected)
        query_splitter.addWidget(self.query_list)

        editor_scroll = QScrollArea()
        editor_scroll.setWidgetResizable(True)
        editor = QWidget()
        editor_layout = QVBoxLayout(editor)

        form = QFormLayout()
        self.case_id = QLineEdit()
        self.query_id = QLineEdit()
        self.query_text = QLineEdit()
        self.query_type = QComboBox()
        self.query_type.setEditable(True)
        self.query_type.addItems(
            ["object", "object_room", "object_relation", "room", "negative"]
        )
        self.query_language = QComboBox()
        self.query_language.addItems(["es", "en"])
        self.query_start_pose = QLineEdit("origin")
        self.query_timeout = QLineEdit("300.0")
        self.query_negative = QCheckBox("La respuesta correcta es rechazar")
        self.query_target_visible = QCheckBox("El objetivo es visible en la escena")
        self.query_target_visible.setChecked(True)
        self.query_negative.toggled.connect(
            lambda checked: self.query_target_visible.setChecked(False)
            if checked else None
        )
        self.query_exact = QLineEdit()
        self.query_nearby = QLineEdit()
        form.addRow("case_id", self.case_id)
        form.addRow("query_id", self.query_id)
        form.addRow("Texto", self.query_text)
        form.addRow("Tipo", self.query_type)
        form.addRow("Idioma", self.query_language)
        form.addRow("Pose inicial", self.query_start_pose)
        form.addRow("Timeout [s]", self.query_timeout)
        form.addRow("Criterio exacto", self.query_exact)
        form.addRow("Criterio cercano", self.query_nearby)
        form.addRow("Query negativa", self.query_negative)
        form.addRow("Visibilidad GT", self.query_target_visible)
        editor_layout.addLayout(form)

        node_buttons = QHBoxLayout()
        exact_button = QPushButton("Añadir nodo seleccionado a exactos")
        exact_button.clicked.connect(lambda: self._append_selected_node(self.query_exact))
        nearby_button = QPushButton("Añadir nodo seleccionado a cercanos")
        nearby_button.clicked.connect(
            lambda: self._append_selected_node(self.query_nearby)
        )
        node_buttons.addWidget(exact_button)
        node_buttons.addWidget(nearby_button)
        editor_layout.addLayout(node_buttons)

        buttons = QHBoxLayout()
        new_button = QPushButton("Nueva query")
        new_button.clicked.connect(self._new_query)
        update_button = QPushButton("Añadir / actualizar query")
        update_button.setObjectName('accentButton')
        update_button.clicked.connect(self._upsert_query)
        delete_button = QPushButton("Eliminar query")
        delete_button.setObjectName('dangerButton')
        delete_button.clicked.connect(self._delete_query)
        buttons.addWidget(new_button)
        buttons.addWidget(update_button)
        buttons.addWidget(delete_button)
        editor_layout.addLayout(buttons)
        editor_layout.addStretch(1)
        editor_scroll.setWidget(editor)
        query_splitter.addWidget(editor_scroll)
        query_splitter.setSizes([190, 470])
        return page

    def reload(self) -> None:
        try:
            self._workspace = load_workspace(
                scene_id=self._scene_id,
                graph_database=self._graph_database,
                map_file=self._map_file,
                queries_file=self._queries_file,
                ground_truth_file=self._ground_truth_file,
            )
        except Exception as exc:  # noqa: BLE001
            self._workspace = None
            self.map_scene.clear()
            self.node_label.setText(f"No se pudo cargar la campaña: {exc}")
            self._log(str(exc), error=True)
            return
        self.path_label.setText(
            f"Grafo: {self._workspace.graph_database} · "
            f"Queries: {self._workspace.queries_file} · "
            f"Ground truth: {self._workspace.ground_truth_file}"
        )
        self.suite_id.setText(str(self._workspace.queries.get("suite_id", "")))
        self._draw_map()
        self._refresh_query_list()
        if self._selected_node_id not in self._node_items:
            self._selected_node_id = ""
        if self._selected_node_id:
            self._select_node(self._selected_node_id)
        self._log(
            f"Campaña recargada: {len(self._workspace.nodes)} nodo(s)."
        )
        if hasattr(self, "campaign_runner"):
            self.campaign_runner.reload()

    def save(self) -> None:
        if self._workspace is None:
            return
        self._workspace.queries["suite_id"] = self.suite_id.text().strip()
        try:
            save_workspace(self._workspace)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "No se pudo guardar", str(exc))
            self._log(f"No se pudo guardar: {exc}", error=True)
            return
        QMessageBox.information(
            self,
            "Campaña guardada",
            "Se guardaron las queries y el ground truth de la escena.",
        )
        self._log("Queries y ground truth guardados.")
        self.campaign_runner.reload()

    def _save_for_campaign(self) -> bool:
        if self._workspace is None:
            return False
        self._workspace.queries["suite_id"] = self.suite_id.text().strip()
        try:
            save_workspace(self._workspace)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Campaña no guardada", str(exc))
            return False
        self.campaign_runner.reload()
        return True

    def _draw_map(self) -> None:
        assert self._workspace is not None
        self.map_scene.clear()
        self._node_items.clear()
        pixmap = QPixmap(self._workspace.map_metadata.image_path)
        if pixmap.isNull():
            raise ValueError(
                f"Qt no pudo abrir {self._workspace.map_metadata.image_path}"
            )
        self.map_scene.addPixmap(pixmap).setZValue(-10.0)
        height = pixmap.height()
        map_meta = self._workspace.map_metadata
        positions = {
            node.node_id: map_meta.world_to_pixel(
                node.position[0], node.position[1], height
            )
            for node in self._workspace.nodes
        }
        room_pen = QPen(QColor(142, 68, 173, 190), 2.0)
        room_brush = QBrush(QColor(142, 68, 173, 35))
        for room in self._workspace.rooms:
            pixels = [
                map_meta.world_to_pixel(x, y, height) for x, y in room.corners()
            ]
            polygon = QPolygonF([QPointF(x, y) for x, y in pixels])
            transition_pen = QPen(QColor(142, 68, 173, 45))
            transition_pen.setWidthF(max(
                2.0, 2.0 * room.transition_width_m / map_meta.resolution
            ))
            self.map_scene.addPolygon(
                polygon, transition_pen, QBrush(Qt.NoBrush)
            ).setZValue(-1.0)
            self.map_scene.addPolygon(polygon, room_pen, room_brush)
            x = min(point[0] for point in pixels)
            y = min(point[1] for point in pixels)
            label = self.map_scene.addText(room.room_id)
            label.setDefaultTextColor(QColor("#8e44ad"))
            label.setPos(x + 3.0, y + 3.0)
            label.setZValue(1.0)
        edge_pen = QPen(QColor(90, 98, 108, 170), 2.0)
        drawn: set[tuple[str, str]] = set()
        for node in self._workspace.nodes:
            for neighbor in node.neighbors:
                key = tuple(sorted((node.node_id, neighbor)))
                if key in drawn or neighbor not in positions:
                    continue
                drawn.add(key)
                start, end = positions[node.node_id], positions[neighbor]
                self.map_scene.addLine(
                    start[0], start[1], end[0], end[1], edge_pen
                ).setZValue(1.0)
        for node in self._workspace.nodes:
            x, y = positions[node.node_id]
            item = _NodeItem(node.node_id, x, y)
            self.map_scene.addItem(item)
            text = self.map_scene.addText(node.node_id)
            text.setPos(x + 8.0, y - 18.0)
            text.setDefaultTextColor(QColor("#c0392b"))
            text.setZValue(4.0)
            self._node_items[node.node_id] = item
        self.map_scene.setSceneRect(
            0.0, 0.0, float(pixmap.width()), float(pixmap.height())
        )
        self.map_view.fitInView(self.map_scene.sceneRect(), Qt.KeepAspectRatio)

    def _map_selection_changed(self) -> None:
        for item in self.map_scene.selectedItems():
            if isinstance(item, _NodeItem):
                self._select_node(item.node_id)
                return

    def _select_node(self, node_id: str) -> None:
        if self._workspace is None:
            return
        node = next(
            (value for value in self._workspace.nodes if value.node_id == node_id),
            None,
        )
        if node is None:
            return
        self._selected_node_id = node_id
        self.node_label.setText(
            f"Nodo: {node_id} · room del grafo: {node.room_id or 'sin asignar'} · "
            f"pose: ({node.position[0]:.2f}, {node.position[1]:.2f}) · "
            f"{len(node.observations)} observación(es)"
        )
        marker = self._node_items.get(node_id)
        if marker is not None and not marker.isSelected():
            marker.setSelected(True)
            self.map_view.centerOn(marker)
        self._refresh_observations(node)
        self._refresh_ground_truth()

    def _refresh_observations(self, node) -> None:
        self.observation_combo.blockSignals(True)
        self.observation_combo.clear()
        for index, observation in enumerate(node.observations):
            label = observation.observation_id or f"vista {index + 1}"
            self.observation_combo.addItem(label, index)
        self.observation_combo.blockSignals(False)
        self._show_selected_observation()

    def _show_selected_observation(self) -> None:
        observation = self._current_observation()
        if observation is None:
            self.observation_meta.setText("No hay observaciones para este nodo")
            self.observation_image.setText("Sin imagen")
            self.observation_image.setPixmap(QPixmap())
            self.objects_table.setRowCount(0)
            self.relation_scene.clear()
            return
        purity_text = (
            f"{observation.purity:.3f} ({observation.contamination_class})"
            if observation.purity is not None else "unknown"
        )
        self.observation_meta.setText(
            f"{len(observation.objects)} objeto(s), "
            f"{len(observation.relations)} relación(es) · "
            f"yaw solicitado/medido: {observation.requested_yaw:.1f}° / "
            f"{observation.measured_yaw:.1f}° · "
            f"RGB {'válido' if observation.image_valid else 'inválido'}, "
            f"depth {'válido' if observation.depth_valid else 'no disponible'} · "
            f"camera_room={observation.camera_room or 'unknown'}, "
            f"observation_room={observation.observation_room or 'unknown'} · "
            f"purity={purity_text} · "
            f"transición={'sí' if observation.transition_zone else 'no'}"
        )
        self._show_observation_image(observation)
        self.objects_table.setRowCount(len(observation.objects))
        for row, detected in enumerate(observation.objects):
            identifier = detected.object_id or f"objeto_{row + 1}"
            self.objects_table.setItem(row, 0, QTableWidgetItem(identifier))
            self.objects_table.setItem(row, 1, QTableWidgetItem(detected.label))
            self.objects_table.setItem(
                row, 2, QTableWidgetItem(f"{detected.confidence:.3f}")
            )
            self.objects_table.setItem(
                row, 3, QTableWidgetItem(detected.room_id or "unknown")
            )
        self._draw_relation_graph(observation)

    def _show_observation_image(self, observation) -> None:
        path = os.path.expanduser(observation.image_path)
        pixmap = QPixmap(path) if path and os.path.isfile(path) else QPixmap()
        if pixmap.isNull():
            self.observation_image.setPixmap(QPixmap())
            self.observation_image.setText(
                f"Imagen no disponible\n{observation.image_path or '(sin ruta)'}"
            )
            return
        painter = QPainter(pixmap)
        painter.setPen(QPen(QColor("#00e676"), 3.0))
        for detected in observation.objects:
            if detected.box is None:
                continue
            x1, y1, x2, y2 = detected.box
            painter.drawRect(int(x1), int(y1), int(x2 - x1), int(y2 - y1))
            painter.drawText(int(x1), max(12, int(y1) - 3), detected.label)
        painter.end()
        self.observation_image.setText("")
        self.observation_image.setPixmap(
            pixmap.scaled(
                620, 320, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

    def _draw_relation_graph(self, observation) -> None:
        self.relation_scene.clear()
        objects: dict[str, str] = {}
        for index, detected in enumerate(observation.objects):
            identifier = detected.object_id or detected.label or f"objeto_{index + 1}"
            objects[identifier] = detected.label or identifier
        for relation in observation.relations:
            subject = relation.subject_id or relation.subject
            obj = relation.object_id or relation.obj
            if subject:
                objects.setdefault(subject, relation.subject or subject)
            if obj:
                objects.setdefault(obj, relation.obj or obj)
        if not objects:
            self.relation_scene.addText("No hay objetos ni relaciones")
            return

        positions = radial_node_positions(list(objects))
        relations = [
            relation for relation in observation.relations
            if (relation.subject_id or relation.subject) in positions
            and (relation.object_id or relation.obj) in positions
        ]
        pairs = [
            (
                relation.subject_id or relation.subject,
                relation.object_id or relation.obj,
            )
            for relation in relations
        ]
        lane_offsets = relation_lane_offsets(pairs)
        palette = (
            "#ff9f43", "#55efc4", "#74b9ff", "#fd79a8",
            "#a29bfe", "#feca57", "#00cec9", "#e17055",
        )
        desired_labels: list[tuple[float, float]] = []
        label_sizes: list[tuple[float, float]] = []
        label_texts: list[str] = []
        paths: list[tuple[QPainterPath, QColor, tuple[float, float], float]] = []
        for index, (relation, lane) in enumerate(zip(relations, lane_offsets)):
            subject = relation.subject_id or relation.subject
            obj = relation.object_id or relation.obj
            start, end = positions[subject], positions[obj]
            colour = QColor(palette[index % len(palette)])
            path, control, arrow_tip, angle = self._relation_path(
                start, end, lane
            )
            paths.append((path, colour, arrow_tip, angle))
            desired_labels.append(control)
            description = (
                f"R{index + 1} · {relation.predicate} "
                f"({relation.confidence:.2f})"
            )
            label_texts.append(description)
            label_sizes.append((max(116.0, 7.2 * len(description) + 16.0), 27.0))

        node_rects = [
            (position[0] - 65.0, position[1] - 28.0, 130.0, 56.0)
            for position in positions.values()
        ]
        label_rects = non_overlapping_label_rects(
            desired_labels, label_sizes, occupied=node_rects
        )

        for index, (path, colour, end, angle) in enumerate(paths):
            edge = self.relation_scene.addPath(path, QPen(colour, 2.5))
            edge.setZValue(1.0)
            arrow = self._arrow_head(end, angle)
            arrow_item = self.relation_scene.addPolygon(
                arrow, QPen(colour, 1.0), QBrush(colour)
            )
            arrow_item.setZValue(1.5)
            rectangle = label_rects[index]
            background = self.relation_scene.addRect(
                *rectangle,
                QPen(colour, 1.5),
                QBrush(QColor(20, 29, 40, 238)),
            )
            background.setZValue(3.0)
            label = self.relation_scene.addText(label_texts[index])
            label.setDefaultTextColor(colour.lighter(135))
            label.setPos(rectangle[0] + 6.0, rectangle[1] + 2.0)
            label.setZValue(4.0)
            label.setToolTip(
                f"{pairs[index][0]} —{relations[index].predicate}→ "
                f"{pairs[index][1]}"
            )

        for identifier, label_text in objects.items():
            x, y = positions[identifier]
            node = self.relation_scene.addEllipse(
                x - 58.0,
                y - 25.0,
                116.0,
                50.0,
                QPen(QColor("#69c7ff"), 2.2),
                QBrush(QColor("#16354b")),
            )
            node.setZValue(5.0)
            text = self.relation_scene.addText(label_text)
            text.setDefaultTextColor(QColor("#e9f7ff"))
            text.setPos(x - min(52.0, 3.5 * len(label_text)), y - 13.0)
            text.setToolTip(identifier)
            text.setZValue(6.0)

        self.relation_scene.setSceneRect(
            self.relation_scene.itemsBoundingRect().adjusted(-35, -35, 35, 35)
        )
        self.relation_view.fitInView(
            self.relation_scene.sceneRect(), Qt.KeepAspectRatio
        )

    @staticmethod
    def _relation_path(
        start: tuple[float, float],
        end: tuple[float, float],
        lane: float,
    ) -> tuple[
        QPainterPath,
        tuple[float, float],
        tuple[float, float],
        float,
    ]:
        """Create a curved edge, its label anchor and terminal angle."""
        path = QPainterPath()
        if start == end:
            path.moveTo(start[0] + 58.0, start[1])
            path.cubicTo(
                start[0] + 105.0,
                start[1] - 90.0,
                start[0] - 75.0,
                start[1] - 105.0,
                start[0] - 58.0,
                start[1],
            )
            arrow_tip = (start[0] - 58.0, start[1])
            angle = math.atan2(105.0, 17.0)
            return path, (start[0], start[1] - 105.0), arrow_tip, angle
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = max(1.0, math.hypot(dx, dy))
        unit_x, unit_y = dx / length, dy / length
        normal_x, normal_y = -unit_y, unit_x
        curve = lane if lane else 18.0
        control = (
            (start[0] + end[0]) / 2.0 + normal_x * curve,
            (start[1] + end[1]) / 2.0 + normal_y * curve,
        )
        radius = 1.0 / math.sqrt(
            (unit_x / 58.0) ** 2 + (unit_y / 25.0) ** 2
        )
        edge_start = (start[0] + unit_x * radius, start[1] + unit_y * radius)
        edge_end = (end[0] - unit_x * radius, end[1] - unit_y * radius)
        path.moveTo(*edge_start)
        path.quadTo(QPointF(*control), QPointF(*edge_end))
        angle = math.atan2(edge_end[1] - control[1], edge_end[0] - control[0])
        return path, control, edge_end, angle

    @staticmethod
    def _arrow_head(
        end: tuple[float, float], angle: float
    ) -> QPolygonF:
        """Return a compact arrow head pointing into a relation target."""
        tip = QPointF(*end)
        back_x = tip.x() - math.cos(angle) * 13.0
        back_y = tip.y() - math.sin(angle) * 13.0
        normal_x, normal_y = -math.sin(angle) * 6.0, math.cos(angle) * 6.0
        return QPolygonF([
            tip,
            QPointF(back_x + normal_x, back_y + normal_y),
            QPointF(back_x - normal_x, back_y - normal_y),
        ])

    def _current_observation(self):
        if self._workspace is None or not self._selected_node_id:
            return None
        node = next(
            (value for value in self._workspace.nodes
             if value.node_id == self._selected_node_id),
            None,
        )
        index = self.observation_combo.currentIndex()
        if node is None or index < 0 or index >= len(node.observations):
            return None
        return node.observations[index]

    def _refresh_ground_truth(self) -> None:
        if self._workspace is None:
            return
        ground_truth = self._workspace.ground_truth
        rooms = [
            str(entry.get("room_id", ""))
            for entry in ground_truth.get("rooms", [])
            if entry.get("room_id")
        ]
        for room in self._workspace.rooms:
            if room.room_id not in rooms:
                rooms.append(room.room_id)
        current_room = next(
            (
                str(entry.get("room_id", ""))
                for entry in ground_truth.get("rooms", [])
                if self._selected_node_id in entry.get("valid_nodes", [])
            ),
            "",
        )
        self.room_combo.clear()
        self.room_combo.addItems(rooms)
        self.room_combo.setCurrentText(current_room)
        self.room_membership.setText(
            f"Asignación de evaluación: {current_room or 'sin asignar'}"
        )
        self.gt_objects.clear()
        for index, entry in enumerate(ground_truth.get("objects", [])):
            level = ""
            if self._selected_node_id in entry.get("exact_valid_nodes", []):
                level = "exact"
            elif self._selected_node_id in entry.get("nearby_valid_nodes", []):
                level = "nearby"
            if not level:
                continue
            item = QListWidgetItem(
                f"{entry.get('category', '')} · {entry.get('room_id', '')} · {level}"
            )
            item.setData(Qt.UserRole, index)
            self.gt_objects.addItem(item)
        self.gt_relations.clear()
        for index, entry in enumerate(ground_truth.get("relations", [])):
            if self._selected_node_id not in entry.get("valid_nodes", []):
                continue
            item = QListWidgetItem(
                f"{entry.get('subject', '')} —{entry.get('predicate', '')}→ "
                f"{entry.get('object', '')}"
            )
            item.setData(Qt.UserRole, index)
            self.gt_relations.addItem(item)

    def _assign_room(self) -> None:
        if self._workspace is None or not self._require_node():
            return
        room_id = self.room_combo.currentText().strip()
        if not room_id:
            QMessageBox.warning(self, "Room requerida", "Escriba o seleccione una room.")
            return
        entries = self._workspace.ground_truth["rooms"]
        selected = None
        for entry in entries:
            valid_nodes = entry.setdefault("valid_nodes", [])
            if self._selected_node_id in valid_nodes:
                valid_nodes.remove(self._selected_node_id)
            if str(entry.get("room_id", "")) == room_id:
                selected = entry
        if selected is None:
            selected = {"room_id": room_id, "valid_nodes": []}
            entries.append(selected)
        selected.setdefault("valid_nodes", []).append(self._selected_node_id)
        self._refresh_ground_truth()

    def _add_gt_object(self) -> None:
        if self._workspace is None or not self._require_node():
            return
        category = self.object_category.text().strip()
        room_id = self.object_room.text().strip()
        if not category:
            QMessageBox.warning(self, "Categoría requerida", "Escriba una categoría.")
            return
        entries = self._workspace.ground_truth["objects"]
        selected = None
        for entry in entries:
            if all((
                str(entry.get("category", "")) == category,
                str(entry.get("room_id", "")) == room_id,
            )):
                selected = entry
                break
        if selected is None:
            selected = {
                "category": category,
                "room_id": room_id,
                "exact_valid_nodes": [],
                "nearby_valid_nodes": [],
            }
            entries.append(selected)
        for key in ("exact_valid_nodes", "nearby_valid_nodes"):
            values = selected.setdefault(key, [])
            if self._selected_node_id in values:
                values.remove(self._selected_node_id)
        key = f"{self.object_level.currentText()}_valid_nodes"
        selected[key].append(self._selected_node_id)
        self._refresh_ground_truth()

    def _remove_gt_object(self) -> None:
        if self._workspace is None:
            return
        item = self.gt_objects.currentItem()
        if item is None:
            return
        entries = self._workspace.ground_truth["objects"]
        index = int(item.data(Qt.UserRole))
        if index >= len(entries):
            return
        entry = entries[index]
        for key in ("exact_valid_nodes", "nearby_valid_nodes"):
            values = entry.setdefault(key, [])
            if self._selected_node_id in values:
                values.remove(self._selected_node_id)
        if not entry["exact_valid_nodes"] and not entry["nearby_valid_nodes"]:
            entries.pop(index)
        self._refresh_ground_truth()

    def _add_gt_relation(self) -> None:
        if self._workspace is None or not self._require_node():
            return
        subject = self.relation_subject.text().strip()
        predicate = self.relation_predicate.currentText().strip()
        obj = self.relation_object.text().strip()
        relation_type = self.relation_type.currentText().strip()
        if not subject or not predicate or not obj:
            QMessageBox.warning(
                self, "Relación incompleta", "Complete sujeto, predicado y objeto."
            )
            return
        entries = self._workspace.ground_truth["relations"]
        selected = next(
            (
                entry for entry in entries
                if all((
                    str(entry.get("subject", "")) == subject,
                    str(entry.get("predicate", "")) == predicate,
                    str(entry.get("object", "")) == obj,
                    str(entry.get("relation_type", "")) == relation_type,
                ))
            ),
            None,
        )
        if selected is None:
            selected = {
                "subject": subject,
                "predicate": predicate,
                "object": obj,
                "relation_type": relation_type,
                "valid_nodes": [],
            }
            entries.append(selected)
        if self._selected_node_id not in selected["valid_nodes"]:
            selected["valid_nodes"].append(self._selected_node_id)
        self._refresh_ground_truth()

    def _remove_gt_relation(self) -> None:
        if self._workspace is None:
            return
        item = self.gt_relations.currentItem()
        if item is None:
            return
        entries = self._workspace.ground_truth["relations"]
        index = int(item.data(Qt.UserRole))
        if index >= len(entries):
            return
        valid_nodes = entries[index].setdefault("valid_nodes", [])
        if self._selected_node_id in valid_nodes:
            valid_nodes.remove(self._selected_node_id)
        if not valid_nodes:
            entries.pop(index)
        self._refresh_ground_truth()

    def _refresh_query_list(self) -> None:
        self.query_list.clear()
        if self._workspace is None:
            return
        for case in self._workspace.queries.get("cases", []):
            item = QListWidgetItem(
                f"{case.get('case_id', '')} · {case.get('query_text', '')}"
            )
            item.setData(Qt.UserRole, str(case.get("case_id", "")))
            self.query_list.addItem(item)

    def _query_selected(self, current, _previous) -> None:
        if current is None or self._workspace is None:
            return
        case_id = str(current.data(Qt.UserRole))
        case = next(
            (
                value for value in self._workspace.queries.get("cases", [])
                if str(value.get("case_id", "")) == case_id
            ),
            None,
        )
        if case is None:
            return
        self._selected_case_id = case_id
        self.case_id.setText(case_id)
        self.query_id.setText(str(case.get("query_id", "")))
        self.query_text.setText(str(case.get("query_text", "")))
        self.query_type.setCurrentText(str(case.get("query_type", "object")))
        self.query_language.setCurrentText(str(case.get("language", "es")))
        self.query_start_pose.setText(str(case.get("start_pose_id", "origin")))
        self.query_timeout.setText(str(case.get("timeout_s", 300.0)))
        is_negative = bool(case.get("is_negative", False))
        self.query_negative.setChecked(is_negative)
        self.query_target_visible.setChecked(bool(
            case.get("target_visible", not is_negative)
        ))
        _set_csv(self.query_exact, list(case.get("exact_valid_nodes", [])))
        _set_csv(self.query_nearby, list(case.get("nearby_valid_nodes", [])))

    def _new_query(self) -> None:
        self._selected_case_id = ""
        for widget in (self.case_id, self.query_id, self.query_text,
                       self.query_exact, self.query_nearby):
            widget.clear()
        self.query_type.setCurrentText("object")
        self.query_language.setCurrentText("es")
        self.query_start_pose.setText("origin")
        self.query_timeout.setText("300.0")
        self.query_negative.setChecked(False)
        self.query_target_visible.setChecked(True)
        self.query_list.clearSelection()

    def _upsert_query(self) -> None:
        if self._workspace is None:
            return
        case_id = self.case_id.text().strip()
        query_id = self.query_id.text().strip()
        text = self.query_text.text().strip()
        if not case_id or not query_id or not text:
            QMessageBox.warning(
                self,
                "Query incompleta",
                "case_id, query_id y texto son obligatorios.",
            )
            return
        try:
            timeout = float(self.query_timeout.text())
        except ValueError:
            QMessageBox.warning(self, "Timeout inválido", "El timeout debe ser numérico.")
            return
        cases = self._workspace.queries["cases"]
        existing = next(
            (
                case for case in cases
                if str(case.get("case_id", "")) == self._selected_case_id
            ),
            None,
        )
        if existing is None:
            duplicate = any(str(case.get("case_id", "")) == case_id for case in cases)
            if duplicate:
                QMessageBox.warning(self, "case_id duplicado", case_id)
                return
            existing = {}
            cases.append(existing)
        previous_query_id = str(existing.get("query_id", ""))
        existing.clear()
        existing.update({
            "case_id": case_id,
            "query_id": query_id,
            "query_text": text,
            "query_type": self.query_type.currentText().strip(),
            "language": self.query_language.currentText(),
            "start_pose_id": self.query_start_pose.text().strip() or "origin",
            "exact_valid_nodes": _csv_values(self.query_exact.text()),
            "nearby_valid_nodes": _csv_values(self.query_nearby.text()),
            "is_negative": self.query_negative.isChecked(),
            "target_visible": self.query_target_visible.isChecked(),
            "timeout_s": timeout,
        })
        negative_queries = self._workspace.ground_truth["negative_queries"]
        for candidate in (previous_query_id, query_id):
            if candidate in negative_queries:
                negative_queries.remove(candidate)
        if existing["is_negative"] and query_id not in negative_queries:
            negative_queries.append(query_id)
        self._selected_case_id = case_id
        self._refresh_query_list()

    def _delete_query(self) -> None:
        if self._workspace is None or not self._selected_case_id:
            return
        cases = self._workspace.queries["cases"]
        case = next(
            (
                value for value in cases
                if str(value.get("case_id", "")) == self._selected_case_id
            ),
            None,
        )
        if case is not None:
            query_id = str(case.get("query_id", ""))
            cases.remove(case)
            negatives = self._workspace.ground_truth["negative_queries"]
            if query_id in negatives:
                negatives.remove(query_id)
        self._new_query()
        self._refresh_query_list()

    def _append_selected_node(self, target: QLineEdit) -> None:
        if not self._require_node():
            return
        values = _csv_values(target.text())
        if self._selected_node_id not in values:
            values.append(self._selected_node_id)
        _set_csv(target, values)

    def _require_node(self) -> bool:
        if self._selected_node_id:
            return True
        QMessageBox.warning(self, "Seleccione un nodo", "Pulse primero un nodo del mapa.")
        return False

    def _log(self, message: str, error: bool = False) -> None:
        if self._log_callback is not None:
            self._log_callback(message, error)
