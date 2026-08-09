"""Interactive inspection of a live semantic-navigation ranking."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from typing import Any

from python_qt_binding.QtCore import QPointF, Qt
from python_qt_binding.QtGui import (
    QBrush,
    QColor,
    QPainter,
    QPen,
    QPixmap,
    QPolygonF,
)
from python_qt_binding.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGraphicsEllipseItem,
    QGraphicsScene,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from semantic_evaluation.core.campaign_designer import (
    CampaignWorkspace,
    load_workspace,
    save_workspace,
)
from semantic_evaluation.core.campaign_authoring import (
    confirm_object_ground_truth,
    identifier_from_query,
    unique_case_id,
    upsert_query_case,
)
from semantic_evaluation.core.retrieval_visualization import object_evidence


class _EvaluationNodeItem(QGraphicsEllipseItem):
    """Map marker whose colour can represent its retrieval state."""

    def __init__(self, node_id: str, x: float, y: float) -> None:
        radius = 8.0
        super().__init__(x - radius, y - radius, radius * 2.0, radius * 2.0)
        self.node_id = node_id
        self.setFlag(QGraphicsEllipseItem.ItemIsSelectable, True)
        self.setPen(QPen(QColor("white"), 1.5))
        self.setZValue(3.0)
        self.set_state("neutral")

    def set_state(self, state: str, tooltip: str = "") -> None:
        colours = {
            "neutral": "#2e86de",
            "best": "#00a65a",
            "ranked": "#f39c12",
            "unrelated": "#c0392b",
        }
        self.setBrush(QBrush(QColor(colours.get(state, colours["neutral"]))))
        self.setToolTip(tooltip or self.node_id)

    def set_annotation(self, validity: str) -> None:
        """Draw a separate border for human-confirmed validity."""
        if validity == 'exact':
            self.setPen(QPen(QColor('#00ff88'), 4.0))
        elif validity == 'nearby':
            self.setPen(QPen(QColor('#ffe066'), 4.0))
        else:
            self.setPen(QPen(QColor('white'), 1.5))


class _EvaluationMapView(QGraphicsView):
    def __init__(self, scene: QGraphicsScene) -> None:
        super().__init__(scene)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setDragMode(QGraphicsView.ScrollHandDrag)

    def wheelEvent(self, event) -> None:  # noqa: N802
        factor = 1.18 if event.angleDelta().y() > 0 else 1.0 / 1.18
        self.scale(factor, factor)
        event.accept()


class RetrievalEvaluationWidget(QWidget):
    """Run one query and explain the returned node and object evidence."""

    def __init__(
        self,
        *,
        scene_id: str,
        graph_database: str,
        map_file: str,
        queries_file: str,
        ground_truth_file: str,
        submit_callback: Callable[[str, str, bool], bool],
        cancel_callback: Callable[[], bool],
        campaign_changed_callback: Callable[[], None] | None = None,
        log_callback: Callable[[str, bool], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._scene_id = scene_id
        self._graph_database = graph_database
        self._map_file = map_file
        self._queries_file = queries_file
        self._ground_truth_file = ground_truth_file
        self._submit_callback = submit_callback
        self._cancel_callback = cancel_callback
        self._campaign_changed_callback = campaign_changed_callback
        self._log_callback = log_callback
        self._workspace: CampaignWorkspace | None = None
        self._node_items: dict[str, _EvaluationNodeItem] = {}
        self._candidate_by_node: dict[str, dict[str, Any]] = {}
        self._rank_by_node: dict[str, int] = {}
        self._selected_node_id = ""
        self._query_active = False
        self._has_result = False
        self._exact_nodes: set[str] = set()
        self._nearby_nodes: set[str] = set()
        self._last_prefilled_query = ''
        self._build_ui()
        self.reload()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        query_row = QHBoxLayout()
        self.query = QLineEdit()
        self.query.setPlaceholderText(
            "Consulta semántica, p. ej. «ve al sofá del salón»"
        )
        self.query.returnPressed.connect(self.evaluate)
        query_row.addWidget(self.query, stretch=1)
        self.language = QComboBox()
        self.language.addItem("Español", "es")
        self.language.addItem("English", "en")
        query_row.addWidget(self.language)
        self.navigate = QCheckBox("Navegar realmente al mejor nodo")
        self.navigate.setToolTip(
            "Desmarcado: calcula el ranking sin enviar un objetivo a Nav2."
        )
        query_row.addWidget(self.navigate)
        self.evaluate_button = QPushButton("Evaluar")
        self.evaluate_button.clicked.connect(self.evaluate)
        query_row.addWidget(self.evaluate_button)
        cancel = QPushButton("Cancelar")
        cancel.clicked.connect(self._cancel_callback)
        query_row.addWidget(cancel)
        reload_button = QPushButton("Recargar grafo")
        reload_button.clicked.connect(self.reload)
        query_row.addWidget(reload_button)
        root.addLayout(query_row)

        self.status = QLabel(
            "Escriba una consulta. Azul: sin evaluar; verde: mejor nodo; "
            "naranja: candidato; rojo: fuera del ranking devuelto."
        )
        self.status.setWordWrap(True)
        root.addWidget(self.status)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(7)
        root.addWidget(splitter, stretch=1)
        map_panel = QWidget()
        map_layout = QVBoxLayout(map_panel)
        self.map_scene = QGraphicsScene(self)
        self.map_scene.selectionChanged.connect(self._map_selection_changed)
        self.map_view = _EvaluationMapView(self.map_scene)
        self.map_view.setRenderHint(QPainter.Antialiasing, True)
        map_layout.addWidget(self.map_view, stretch=1)
        self.node_summary = QLabel(
            "Seleccione un nodo en el mapa o en el ranking"
        )
        self.node_summary.setWordWrap(True)
        map_layout.addWidget(self.node_summary)
        splitter.addWidget(map_panel)

        inspector = QWidget()
        inspector_layout = QVBoxLayout(inspector)
        inspector_layout.setContentsMargins(0, 0, 0, 0)
        inspector_splitter = QSplitter(Qt.Vertical)
        inspector_splitter.setChildrenCollapsible(False)
        inspector_splitter.setHandleWidth(7)
        inspector_layout.addWidget(inspector_splitter)

        ranking_panel = QWidget()
        ranking_layout = QVBoxLayout(ranking_panel)
        ranking_layout.addWidget(QLabel("Ranking y componentes"))
        self.ranking = QTableWidget(0, 8)
        self.ranking.setHorizontalHeaderLabels(
            [
                "#", "Nodo", "Total", "Global", "Objetos", "Crops",
                "Relaciones", "Room",
            ]
        )
        self.ranking.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.ranking.setSelectionMode(QAbstractItemView.SingleSelection)
        self.ranking.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.ranking.setAlternatingRowColors(True)
        self.ranking.itemSelectionChanged.connect(
            self._ranking_selection_changed
        )
        self.ranking.horizontalHeader().setStretchLastSection(True)
        ranking_layout.addWidget(self.ranking, stretch=1)
        self.component_summary = QLabel("Todavía no hay un ranking")
        self.component_summary.setWordWrap(True)
        ranking_layout.addWidget(self.component_summary)
        inspector_splitter.addWidget(ranking_panel)

        details = QSplitter(Qt.Horizontal)
        details.setChildrenCollapsible(False)
        details.setHandleWidth(7)
        authoring_scroll = QScrollArea()
        authoring_scroll.setWidgetResizable(True)
        authoring_container = QWidget()
        authoring_layout = QVBoxLayout(authoring_container)
        authoring_layout.addWidget(self._authoring_group())
        authoring_layout.addStretch(1)
        authoring_scroll.setWidget(authoring_container)
        details.addWidget(authoring_scroll)

        evidence_panel = QWidget()
        evidence_layout = QVBoxLayout(evidence_panel)
        evidence_layout.addWidget(QLabel("Observación e influencia de objetos"))
        self.observation_combo = QComboBox()
        self.observation_combo.currentIndexChanged.connect(
            self._show_selected_observation
        )
        evidence_layout.addWidget(self.observation_combo)
        self.observation_meta = QLabel("Sin nodo seleccionado")
        self.observation_meta.setWordWrap(True)
        evidence_layout.addWidget(self.observation_meta)
        self.observation_image = QLabel("Sin imagen")
        self.observation_image.setAlignment(Qt.AlignCenter)
        self.observation_image.setMinimumHeight(150)
        self.observation_image.setStyleSheet(
            "background: #17191c; color: #b8bec7; border: 1px solid #3b4048;"
        )
        evidence_layout.addWidget(self.observation_image, stretch=2)
        self.objects = QTableWidget(0, 4)
        self.objects.setHorizontalHeaderLabels(
            ["Instancia", "Clase", "Confianza", "Influencia en ranking"]
        )
        self.objects.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.objects.setAlternatingRowColors(True)
        self.objects.horizontalHeader().setStretchLastSection(True)
        evidence_layout.addWidget(self.objects, stretch=1)
        details.addWidget(evidence_panel)
        details.setSizes([330, 470])
        inspector_splitter.addWidget(details)
        inspector_splitter.setSizes([310, 500])
        splitter.addWidget(inspector)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([620, 740])

    def _authoring_group(self) -> QGroupBox:
        group = QGroupBox('Convertir en caso confirmado de evaluación')
        layout = QVBoxLayout(group)
        form = QFormLayout()
        self.case_id = QLineEdit()
        self.query_id = QLineEdit()
        self.query_type = QComboBox()
        self.query_type.setEditable(True)
        self.query_type.addItems([
            'object', 'object_room', 'object_relation', 'room', 'negative'
        ])
        self.start_pose = QLineEdit('origin')
        self.case_timeout = QDoubleSpinBox()
        self.case_timeout.setRange(1.0, 3600.0)
        self.case_timeout.setValue(300.0)
        self.case_timeout.setSuffix(' s')
        form.addRow('case_id', self.case_id)
        form.addRow('query_id', self.query_id)
        form.addRow('Tipo', self.query_type)
        form.addRow('Pose inicial', self.start_pose)
        form.addRow('Timeout', self.case_timeout)
        layout.addLayout(form)
        self.negative_query = QCheckBox(
            'Consulta negativa: la respuesta correcta es rechazar'
        )
        self.negative_query.toggled.connect(self._negative_toggled)
        layout.addWidget(self.negative_query)
        self.confirm_objects = QCheckBox(
            'Confirmar también como GT los objetos verdes de los nodos marcados'
        )
        self.confirm_objects.setToolTip(
            'Solo se guardan al pulsar el botón; el modelo nunca los confirma solo.'
        )
        layout.addWidget(self.confirm_objects)
        buttons = QHBoxLayout()
        exact = QPushButton('Nodo seleccionado → exacto')
        exact.clicked.connect(lambda: self._mark_selected_node('exact'))
        nearby = QPushButton('Nodo seleccionado → cercano')
        nearby.clicked.connect(lambda: self._mark_selected_node('nearby'))
        remove = QPushButton('Quitar criterio')
        remove.clicked.connect(lambda: self._mark_selected_node('none'))
        buttons.addWidget(exact)
        buttons.addWidget(nearby)
        buttons.addWidget(remove)
        layout.addLayout(buttons)
        self.annotation_summary = QLabel(
            'Exactos: — · Cercanos: —. El ranking nunca se confirma automáticamente.'
        )
        self.annotation_summary.setWordWrap(True)
        layout.addWidget(self.annotation_summary)
        save = QPushButton('Guardar scene_query y ground truth confirmado')
        save.setObjectName('accentButton')
        save.clicked.connect(self._save_confirmed_case)
        layout.addWidget(save)
        return group

    def reload(self) -> None:
        try:
            self._workspace = load_workspace(
                scene_id=self._scene_id,
                graph_database=self._graph_database,
                map_file=self._map_file,
                queries_file=self._queries_file,
                ground_truth_file=self._ground_truth_file,
            )
            self._draw_map()
        except Exception as exc:  # noqa: BLE001
            self._workspace = None
            self.map_scene.clear()
            self.status.setText(f"No se pudo cargar la escena: {exc}")
            self._log(str(exc), error=True)
            return
        self._candidate_by_node.clear()
        self._rank_by_node.clear()
        self._has_result = False
        self.ranking.setRowCount(0)
        self.status.setText(
            f"Escena {self._scene_id}: {len(self._workspace.nodes)} nodo(s). "
            "Introduzca una consulta para obtener el ranking."
        )
        self._clear_annotations()

    def evaluate(self) -> None:
        query = self.query.text().strip()
        if not query:
            QMessageBox.warning(self, "Consulta vacía", "Escriba una consulta.")
            return
        navigate = self.navigate.isChecked()
        if self._submit_callback(
            query, str(self.language.currentData()), navigate
        ):
            self._query_active = True
            self._has_result = False
            self.evaluate_button.setEnabled(False)
            self.status.setText(
                "Consulta enviada; calculando ranking"
                + (" y navegación…" if navigate else "…")
            )
            self._reset_marker_states()

    def handle_event(self, kind: str, payload: Mapping[str, Any]) -> None:
        if payload.get("kind") != "evaluation":
            return
        if kind == "navigation_started":
            self.status.setText(
                str(payload.get("message", "Consulta enviada…"))
            )
        elif kind == "navigation_feedback":
            stage = str(payload.get("stage", "procesando"))
            distance = float(payload.get("distance", 0.0))
            if stage == "navigating":
                self.status.setText(
                    "Ranking calculado; navegando "
                    f"({distance:.2f} m restantes)…"
                )
            else:
                self.status.setText(f"Evaluación: {stage}…")
        elif kind == "navigation_result":
            self._query_active = False
            self.evaluate_button.setEnabled(True)
            self._apply_result(payload)
        elif kind == "error":
            self._query_active = False
            self.evaluate_button.setEnabled(True)
            self.status.setText(
                str(payload.get("message", "Error de evaluación"))
            )

    def _apply_result(self, payload: Mapping[str, Any]) -> None:
        candidates = [
            dict(value)
            for value in payload.get("candidates", [])
            if isinstance(value, Mapping)
        ]
        self._candidate_by_node = {
            str(value.get("node_id", "")): value
            for value in candidates
            if value.get("node_id")
        }
        self._rank_by_node = {
            str(value.get("node_id", "")): index
            for index, value in enumerate(candidates, start=1)
        }
        self._has_result = True
        self._prefill_case_fields()
        best_node = str(payload.get("matched_node", ""))
        for node_id, marker in self._node_items.items():
            candidate = self._candidate_by_node.get(node_id)
            rank = self._rank_by_node.get(node_id)
            if node_id == best_node:
                state = "best"
            elif candidate is not None:
                state = "ranked"
            else:
                state = "unrelated"
            tooltip = node_id
            if candidate is not None:
                tooltip += (
                    f" · #{rank} · score={float(candidate['score']):.4f}"
                )
            marker.set_state(state, tooltip)
        self._fill_ranking(candidates)
        accepted = bool(payload.get("accepted", False))
        navigation_success = bool(payload.get("navigation_success", False))
        success = bool(payload.get("success", False))
        if accepted:
            movement = (
                "navegación completada" if navigation_success
                else "ranking completado sin movimiento"
            )
            if self.navigate.isChecked() and not navigation_success:
                movement = (
                    "ranking completado; la navegación no terminó "
                    "correctamente"
                )
            self.status.setText(
                f"Mejor nodo: {best_node} · "
                f"score={float(payload.get('score', 0.0)):.4f} · "
                f"{movement}."
            )
        else:
            reason = payload.get("rejection_reason") or payload.get("message")
            self.status.setText(
                "Consulta rechazada por umbral. Mejor candidato: "
                f"{best_node or 'ninguno'}. {reason}"
            )
        if not success and accepted:
            failure = payload.get("failure_type") or payload.get("message")
            self.status.setText(f"{self.status.text()} Error: {failure}.")
        if best_node:
            self._select_node(best_node)
        self._log(self.status.text(), error=not success and accepted)

    def _prefill_case_fields(self) -> None:
        if self._workspace is None:
            return
        text = self.query.text().strip()
        if not text or text == self._last_prefilled_query:
            return
        base = unique_case_id(
            self._workspace, identifier_from_query(text)
        )
        language = str(self.language.currentData()) or 'es'
        self.case_id.setText(base)
        self.query_id.setText(f'{base}_{language}')
        self._last_prefilled_query = text
        self._clear_annotations()

    def _negative_toggled(self, checked: bool) -> None:
        if checked:
            self.query_type.setCurrentText('negative')
            self._clear_annotations()

    def _mark_selected_node(self, validity: str) -> None:
        node_id = self._selected_node_id
        if not node_id:
            QMessageBox.warning(
                self, 'Nodo requerido', 'Seleccione primero un nodo del mapa.'
            )
            return
        if self.negative_query.isChecked() and validity != 'none':
            QMessageBox.warning(
                self,
                'Consulta negativa',
                'Una consulta negativa no puede tener nodos válidos.',
            )
            return
        self._exact_nodes.discard(node_id)
        self._nearby_nodes.discard(node_id)
        if validity == 'exact':
            self._exact_nodes.add(node_id)
        elif validity == 'nearby':
            self._nearby_nodes.add(node_id)
        self._refresh_annotation_markers()

    def _clear_annotations(self) -> None:
        self._exact_nodes.clear()
        self._nearby_nodes.clear()
        if hasattr(self, 'annotation_summary'):
            self._refresh_annotation_markers()

    def _refresh_annotation_markers(self) -> None:
        for node_id, marker in self._node_items.items():
            validity = (
                'exact' if node_id in self._exact_nodes
                else 'nearby' if node_id in self._nearby_nodes
                else 'none'
            )
            marker.set_annotation(validity)
        exact = ', '.join(sorted(self._exact_nodes)) or '—'
        nearby = ', '.join(sorted(self._nearby_nodes)) or '—'
        self.annotation_summary.setText(
            f'Exactos: {exact} · Cercanos: {nearby}. '
            'Borde verde = exacto; borde amarillo = cercano.'
        )

    def _save_confirmed_case(self) -> None:
        if self._workspace is None:
            return
        try:
            upsert_query_case(
                self._workspace,
                case_id=self.case_id.text(),
                query_id=self.query_id.text(),
                query_text=self.query.text(),
                query_type=self.query_type.currentText(),
                language=str(self.language.currentData()),
                start_pose_id=self.start_pose.text(),
                exact_valid_nodes=sorted(self._exact_nodes),
                nearby_valid_nodes=sorted(self._nearby_nodes),
                is_negative=self.negative_query.isChecked(),
                target_visible=not self.negative_query.isChecked(),
                timeout_s=self.case_timeout.value(),
            )
            confirmed_objects = 0
            if self.confirm_objects.isChecked():
                evidence_by_node: dict[str, list[str]] = {}
                for node_id in self._exact_nodes | self._nearby_nodes:
                    candidate = self._candidate_by_node.get(node_id, {})
                    labels = list(candidate.get('matched_object_labels', []))
                    crop_label = str(
                        candidate.get('best_crop_object_label', '')
                    ).strip()
                    if crop_label:
                        labels.append(crop_label)
                    evidence_by_node[node_id] = labels
                confirmed_objects = confirm_object_ground_truth(
                    self._workspace,
                    evidence_by_node,
                    exact_nodes=self._exact_nodes,
                    nearby_nodes=self._nearby_nodes,
                )
            save_workspace(self._workspace)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, 'No se pudo guardar', str(exc))
            return
        message = (
            f'Caso {self.case_id.text().strip()} guardado en scene_queries y '
            f'ground truth; {confirmed_objects} asignación(es) de objeto '
            'confirmada(s).'
        )
        self.status.setText(message)
        self._log(message)
        if self._campaign_changed_callback is not None:
            self._campaign_changed_callback()

    def _fill_ranking(self, candidates: list[dict[str, Any]]) -> None:
        component_keys = (
            "score",
            "global_similarity",
            "object_match_score",
            "crop_similarity",
            "relation_match_score",
            "room_match_score",
        )
        self.ranking.blockSignals(True)
        self.ranking.setRowCount(len(candidates))
        for row, candidate in enumerate(candidates):
            values = [str(row + 1), str(candidate.get("node_id", ""))]
            values.extend(
                f"{float(candidate.get(key, 0.0)):.4f}" for key in component_keys
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.UserRole, str(candidate.get("node_id", "")))
                if row == 0:
                    item.setBackground(QBrush(QColor(19, 109, 76, 145)))
                    item.setForeground(QBrush(QColor('#d8ffed')))
                else:
                    item.setForeground(QBrush(QColor('#ffd08a')))
                self.ranking.setItem(row, column, item)
        self.ranking.resizeColumnsToContents()
        self.ranking.blockSignals(False)

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
        metadata = self._workspace.map_metadata
        positions = {
            node.node_id: metadata.world_to_pixel(
                node.position[0], node.position[1], height
            )
            for node in self._workspace.nodes
        }
        for room in self._workspace.rooms:
            pixels = [
                metadata.world_to_pixel(x, y, height) for x, y in room.corners()
            ]
            if not pixels:
                continue
            polygon = QPolygonF([QPointF(x, y) for x, y in pixels])
            self.map_scene.addPolygon(
                polygon,
                QPen(QColor(142, 68, 173, 190), 2.0),
                QBrush(QColor(142, 68, 173, 35)),
            )
            label = self.map_scene.addText(room.room_id)
            label.setDefaultTextColor(QColor("#8e44ad"))
            label.setPos(
                min(point[0] for point in pixels) + 3.0,
                min(point[1] for point in pixels) + 3.0,
            )
        drawn: set[tuple[str, str]] = set()
        edge_pen = QPen(QColor(90, 98, 108, 170), 2.0)
        for node in self._workspace.nodes:
            for neighbour in node.neighbors:
                key = tuple(sorted((node.node_id, neighbour)))
                if key in drawn or neighbour not in positions:
                    continue
                drawn.add(key)
                start, end = positions[node.node_id], positions[neighbour]
                self.map_scene.addLine(
                    start[0], start[1], end[0], end[1], edge_pen
                ).setZValue(1.0)
        for node in self._workspace.nodes:
            x, y = positions[node.node_id]
            marker = _EvaluationNodeItem(node.node_id, x, y)
            self.map_scene.addItem(marker)
            label = self.map_scene.addText(node.node_id)
            label.setDefaultTextColor(QColor("#e9f5ff"))
            label.setPos(x + 9.0, y - 19.0)
            label.setZValue(4.0)
            self._node_items[node.node_id] = marker
        self.map_scene.setSceneRect(
            0.0, 0.0, float(pixmap.width()), float(pixmap.height())
        )
        self.map_view.fitInView(self.map_scene.sceneRect(), Qt.KeepAspectRatio)

    def _reset_marker_states(self) -> None:
        self._candidate_by_node.clear()
        self._rank_by_node.clear()
        self.ranking.setRowCount(0)
        for marker in self._node_items.values():
            marker.set_state("neutral")

    def _map_selection_changed(self) -> None:
        for item in self.map_scene.selectedItems():
            if isinstance(item, _EvaluationNodeItem):
                self._select_node(item.node_id)
                return

    def _ranking_selection_changed(self) -> None:
        selected = self.ranking.selectedItems()
        if selected:
            self._select_node(str(selected[0].data(Qt.UserRole)))

    def _select_node(self, node_id: str) -> None:
        if self._workspace is None:
            return
        node = next(
            (
                value for value in self._workspace.nodes
                if value.node_id == node_id
            ),
            None,
        )
        if node is None:
            return
        self._selected_node_id = node_id
        candidate = self._candidate_by_node.get(node_id)
        rank = self._rank_by_node.get(node_id)
        if candidate is None:
            ranking_text = (
                "fuera del ranking"
                if self._has_result else "todavía sin evaluar"
            )
            self.component_summary.setText(ranking_text)
        else:
            ranking_text = (
                f"posición #{rank}, score={float(candidate['score']):.4f}"
            )
            self.component_summary.setText(
                f"Total {float(candidate['score']):.4f} · "
                f"global {float(candidate['global_similarity']):.4f} · "
                f"objetos {float(candidate['object_match_score']):.4f} · "
                f"crops {float(candidate['crop_similarity']):.4f} · "
                f"relaciones {float(candidate['relation_match_score']):.4f} · "
                f"room {float(candidate['room_match_score']):.4f}"
            )
        self.node_summary.setText(
            f"Nodo {node_id} · {ranking_text} · "
            f"room={node.room_id or 'unknown'} · "
            f"pose=({node.position[0]:.2f}, {node.position[1]:.2f}) · "
            f"{len(node.observations)} observación(es)"
        )
        marker = self._node_items.get(node_id)
        if marker is not None and not marker.isSelected():
            marker.setSelected(True)
            self.map_view.centerOn(marker)
        if candidate is not None:
            row = rank - 1
            if self.ranking.currentRow() != row:
                self.ranking.selectRow(row)
        self.observation_combo.blockSignals(True)
        self.observation_combo.clear()
        for index, observation in enumerate(node.observations):
            self.observation_combo.addItem(
                observation.observation_id or f"vista {index + 1}", index
            )
        self.observation_combo.blockSignals(False)
        self._show_selected_observation()

    def _show_selected_observation(self) -> None:
        observation = self._current_observation()
        if observation is None:
            self.observation_meta.setText("El nodo no tiene observaciones")
            self.observation_image.setPixmap(QPixmap())
            self.observation_image.setText("Sin imagen")
            self.objects.setRowCount(0)
            return
        candidate = self._candidate_by_node.get(self._selected_node_id)
        self.observation_meta.setText(
            f"{len(observation.objects)} objeto(s) · "
            "verde: evidencia explícita de la consulta · "
            "rojo: sin evidencia explícita · "
            f"imagen={observation.image_path or 'no disponible'}"
        )
        self.objects.setRowCount(len(observation.objects))
        evidence_by_index: dict[int, tuple[str, ...]] = {}
        for row, detected in enumerate(observation.objects):
            identifier = detected.object_id or f"objeto_{row + 1}"
            evidence = object_evidence(identifier, detected.label, candidate)
            evidence_by_index[row] = evidence
            values = (
                identifier,
                detected.label,
                f"{detected.confidence:.3f}",
                " + ".join(evidence) if evidence else "no usado",
            )
            colour = QColor(
                "#006b3c" if evidence else "#a93226"
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setForeground(colour)
                self.objects.setItem(row, column, item)
        self.objects.resizeColumnsToContents()
        self._draw_observation_image(observation, evidence_by_index)

    def _draw_observation_image(
        self, observation, evidence_by_index: Mapping[int, tuple[str, ...]]
    ) -> None:
        path = os.path.expanduser(observation.image_path)
        source = QPixmap(path) if path and os.path.isfile(path) else QPixmap()
        if source.isNull():
            self.observation_image.setPixmap(QPixmap())
            self.observation_image.setText(
                f"Imagen no disponible\n{observation.image_path or '(sin ruta)'}"
            )
            return
        pixmap = source.copy()
        painter = QPainter(pixmap)
        for index, detected in enumerate(observation.objects):
            if detected.box is None:
                continue
            evidence = evidence_by_index.get(index, ())
            colour = QColor("#00e676" if evidence else "#ff3b30")
            painter.setPen(QPen(colour, 4.0))
            x1, y1, x2, y2 = detected.box
            painter.drawRect(int(x1), int(y1), int(x2 - x1), int(y2 - y1))
            suffix = (
                f" [{'+'.join(evidence)}]" if evidence else " [no usado]"
            )
            painter.drawText(
                int(x1), max(14, int(y1) - 4), f"{detected.label}{suffix}"
            )
        painter.end()
        self.observation_image.setText("")
        self.observation_image.setPixmap(
            pixmap.scaled(
                680, 360, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

    def _current_observation(self):
        if self._workspace is None or not self._selected_node_id:
            return None
        node = next(
            (
                value for value in self._workspace.nodes
                if value.node_id == self._selected_node_id
            ),
            None,
        )
        index = self.observation_combo.currentIndex()
        if node is None or index < 0 or index >= len(node.observations):
            return None
        return node.observations[index]

    def _log(self, message: str, *, error: bool = False) -> None:
        if self._log_callback is not None:
            self._log_callback(message, error)
