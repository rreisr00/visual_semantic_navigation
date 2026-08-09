"""Qt front-end for the existing reproducible ROS campaign collector."""

from __future__ import annotations

import csv
import os
import re
import signal
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from python_qt_binding.QtCore import QProcess, Qt, QTimer
from python_qt_binding.QtGui import QBrush, QColor
from python_qt_binding.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from semantic_evaluation.core.campaign_authoring import campaign_issues
from semantic_evaluation.core.campaign_designer import load_workspace


class CampaignRunnerWidget(QWidget):
    """Validate, execute and review one scene's campaign."""

    def __init__(
        self,
        *,
        scene_id: str,
        graph_database: str,
        map_file: str,
        queries_file: str,
        ground_truth_file: str,
        start_poses_file: str = '',
        robot_entity_name: str = 'semantic_robot',
        world_name: str = 'default',
        frozen_config_hash: str = '',
        frozen_config_path: str = '',
        method: str = 'hybrid_semantic_retrieval',
        output_dir: str = '',
        action_name: str = '/navigate_to_semantic_goal',
        pre_run_callback: Callable[[], bool] | None = None,
        log_callback: Callable[[str, bool], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._scene_id = scene_id
        self._graph_database = graph_database
        self._map_file = map_file
        self._queries_file = queries_file
        self._ground_truth_file = ground_truth_file
        self._start_poses_file = os.path.expanduser(start_poses_file)
        self._robot_entity_name = robot_entity_name
        self._world_name = world_name
        self._frozen_config_hash = frozen_config_hash
        self._frozen_config_path = os.path.expanduser(frozen_config_path)
        self._method = method
        self._action_name = action_name
        self._pre_run_callback = pre_run_callback
        self._log_callback = log_callback
        self._process: QProcess | None = None
        self._run_dir: Path | None = None
        default_output = (
            output_dir
            or '~/visual_semantic_navigation/experiments/simulation/campaigns'
        )
        self._default_output_dir = os.path.expanduser(default_output)
        self._build_ui()
        self.reload()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        toolbar = QHBoxLayout()
        validate = QPushButton('Validar campaña')
        validate.clicked.connect(self.reload)
        toolbar.addWidget(validate)
        toolbar.addStretch(1)
        self.validation_summary = QLabel('Sin validar')
        toolbar.addWidget(self.validation_summary)
        layout.addLayout(toolbar)

        sections = QSplitter(Qt.Vertical)
        sections.setChildrenCollapsible(False)
        sections.setHandleWidth(7)
        layout.addWidget(sections, stretch=1)

        validation_panel = QWidget()
        validation_layout = QVBoxLayout(validation_panel)
        validation_layout.setContentsMargins(0, 0, 0, 0)
        self.issues = QTableWidget(0, 3)
        self.issues.setHorizontalHeaderLabels(
            ['Severidad', 'Referencia', 'Problema']
        )
        self.issues.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.issues.setAlternatingRowColors(True)
        self.issues.horizontalHeader().setStretchLastSection(True)
        validation_layout.addWidget(self.issues)
        sections.addWidget(validation_panel)

        execution_panel = QWidget()
        execution_layout = QVBoxLayout(execution_panel)
        execution_layout.setContentsMargins(0, 0, 0, 0)
        form = QFormLayout()
        self.mode = QComboBox()
        self.mode.addItem('Solo recuperación (recomendado)', True)
        self.mode.addItem('Recuperación + navegación real', False)
        self.seed = QSpinBox()
        self.seed.setRange(0, 2_147_483_647)
        self.seed.setValue(42)
        self.method = QLineEdit(self._method)
        self.output_dir = QLineEdit(self._default_output_dir)
        form.addRow('Modo', self.mode)
        form.addRow('Semilla', self.seed)
        form.addRow('Método registrado', self.method)
        form.addRow('Directorio de resultados', self.output_dir)
        execution_layout.addLayout(form)

        buttons = QHBoxLayout()
        self.run_button = QPushButton('Ejecutar campaña completa')
        self.run_button.setObjectName('accentButton')
        self.run_button.clicked.connect(self.run)
        self.cancel_button = QPushButton('Cancelar campaña')
        self.cancel_button.setObjectName('dangerButton')
        self.cancel_button.clicked.connect(self.cancel)
        self.cancel_button.setEnabled(False)
        buttons.addWidget(self.run_button)
        buttons.addWidget(self.cancel_button)
        execution_layout.addLayout(buttons)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        execution_layout.addWidget(self.progress)
        self.run_status = QLabel('No hay una ejecución activa')
        self.run_status.setWordWrap(True)
        execution_layout.addWidget(self.run_status)
        self.process_log = QTextEdit()
        self.process_log.setReadOnly(True)
        execution_layout.addWidget(self.process_log)
        sections.addWidget(execution_panel)

        results_panel = QWidget()
        results_layout = QVBoxLayout(results_panel)
        results_layout.setContentsMargins(0, 0, 0, 0)
        self.metrics = QLabel('Sin resultados cargados')
        self.metrics.setWordWrap(True)
        results_layout.addWidget(self.metrics)
        self.results = QTableWidget(0, 8)
        self.results.setHorizontalHeaderLabels([
            'Caso', 'Query', 'Predicción', 'Exacto', 'Cercano',
            'Navegación', 'Rank', 'Fallo',
        ])
        self.results.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.results.setAlternatingRowColors(True)
        self.results.horizontalHeader().setStretchLastSection(True)
        results_layout.addWidget(self.results)
        sections.addWidget(results_panel)
        sections.setSizes([180, 260, 320])

    def reload(self) -> None:
        try:
            workspace = load_workspace(
                scene_id=self._scene_id,
                graph_database=self._graph_database,
                map_file=self._map_file,
                queries_file=self._queries_file,
                ground_truth_file=self._ground_truth_file,
            )
            found = campaign_issues(workspace)
        except Exception as exc:  # noqa: BLE001
            found = []
            self.validation_summary.setText(f'No se pudo validar: {exc}')
            self.run_button.setEnabled(False)
            return
        errors = sum(issue.severity == 'error' for issue in found)
        warnings = sum(issue.severity == 'warning' for issue in found)
        self.validation_summary.setText(
            f'{errors} error(es), {warnings} aviso(s)'
        )
        self.issues.setRowCount(len(found))
        for row, issue in enumerate(found):
            values = (issue.severity, issue.reference, issue.message)
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if issue.severity == 'error':
                    item.setForeground(QBrush(QColor('#ff7b89')))
                elif issue.severity == 'warning':
                    item.setForeground(QBrush(QColor('#ffd166')))
                self.issues.setItem(row, column, item)
        self.issues.resizeColumnsToContents()
        self.run_button.setEnabled(errors == 0 and self._process is None)

    def run(self) -> None:
        if self._pre_run_callback is not None and not self._pre_run_callback():
            return
        self.reload()
        if not self.run_button.isEnabled():
            QMessageBox.warning(
                self,
                'Campaña no ejecutable',
                'Corrija los errores mostrados antes de ejecutar.',
            )
            return
        decision_only = bool(self.mode.currentData())
        if not decision_only and not os.path.isfile(self._start_poses_file):
            QMessageBox.warning(
                self,
                'Faltan poses iniciales',
                'La navegación real requiere start_poses_file para restaurar '
                'el mismo estado antes de cada caso.',
            )
            return
        warning_count = sum(
            self.issues.item(row, 0).text() == 'warning'
            for row in range(self.issues.rowCount())
        )
        if warning_count:
            response = QMessageBox.question(
                self,
                'Campaña con avisos',
                f'Hay {warning_count} aviso(s). ¿Ejecutar de todas formas?',
            )
            if response != QMessageBox.Yes:
                return

        timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        run_id = f'gui_{timestamp}'
        output = os.path.abspath(os.path.expanduser(self.output_dir.text()))
        self._run_dir = Path(output) / self._scene_id / run_id
        workspace = load_workspace(
            scene_id=self._scene_id,
            graph_database=self._graph_database,
            map_file=self._map_file,
            queries_file=self._queries_file,
            ground_truth_file=self._ground_truth_file,
        )
        suite_id = str(workspace.queries.get('suite_id', '')).strip()
        total = len(workspace.queries.get('cases', []))
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(0)
        self.results.setRowCount(0)
        self.metrics.setText('Campaña en curso…')
        self.process_log.clear()

        args = [
            'run', 'semantic_evaluation', 'evaluation_collector',
            '--ros-args',
            '-r', f'__node:=evaluation_collector_{timestamp.lower()}',
            '-p', f'action_name:={self._action_name}',
            '-p', f'test_suite_path:={self._queries_file}',
            '-p', f'ground_truth_path:={self._ground_truth_file}',
            '-p', f'graph_database:={self._graph_database}',
            '-p', f'map_file:={self._map_file}',
            '-p', f'frozen_config_path:={self._frozen_config_path}',
            '-p', f'output_dir:={output}',
            '-p', f'campaign_id:={suite_id or self._scene_id}',
            '-p', f'scene_id:={self._scene_id}',
            '-p', f'run_id:={run_id}',
            '-p', f'seed:={self.seed.value()}',
            '-p', f'method:={self.method.text().strip()}',
            '-p', f'query_suite_id:={suite_id}',
            '-p', f'frozen_config_hash:={self._frozen_config_hash}',
            '-p', f'decision_only:={str(decision_only).lower()}',
            '-p', f'start_poses_path:={self._start_poses_file}',
            '-p', f'reset_pose_service:=/world/{self._world_name}/set_pose',
            '-p', f'robot_entity_name:={self._robot_entity_name}',
        ]
        process = QProcess(self)
        process.setProcessChannelMode(QProcess.MergedChannels)
        process.readyReadStandardOutput.connect(self._read_output)
        process.finished.connect(self._finished)
        process.errorOccurred.connect(self._process_error)
        self._process = process
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.run_status.setText(
            f'Ejecutando {total} caso(s); run_id={run_id}'
        )
        process.start('ros2', args)

    def cancel(self) -> None:
        if self._process is None:
            return
        self.run_status.setText('Cancelando campaña…')
        try:
            os.kill(int(self._process.processId()), signal.SIGINT)
        except (OSError, ValueError):
            self._process.terminate()
        QTimer.singleShot(3000, self._kill_if_running)

    def _kill_if_running(self) -> None:
        if self._process is not None and self._process.state() != QProcess.NotRunning:
            self._process.kill()

    def _read_output(self) -> None:
        if self._process is None:
            return
        text = bytes(self._process.readAllStandardOutput()).decode(
            'utf-8', errors='replace'
        )
        self.process_log.insertPlainText(text)
        self.process_log.ensureCursorVisible()
        for current, total in re.findall(r'\[(\d+)/(\d+)\]\s+case', text):
            self.progress.setRange(0, int(total))
            self.progress.setValue(int(current) - 1)

    def _finished(self, exit_code: int, _exit_status) -> None:
        process = self._process
        self._process = None
        self.cancel_button.setEnabled(False)
        self.run_button.setEnabled(True)
        if process is not None:
            process.deleteLater()
        if exit_code != 0:
            self.run_status.setText(
                f'La campaña terminó con código {exit_code}. Revise el log.'
            )
            self._log(self.run_status.text(), error=True)
            return
        if self._run_dir is None:
            return
        result_file = self._run_dir / 'evaluation.csv'
        if not result_file.is_file():
            self.run_status.setText(
                'El collector terminó sin generar evaluation.csv.'
            )
            self._log(self.run_status.text(), error=True)
            return
        self._load_results(result_file)
        self.progress.setValue(self.progress.maximum())
        self.run_status.setText(f'Resultados guardados en {self._run_dir}')
        self._log(self.run_status.text())

    def _process_error(self, _error) -> None:
        if self._process is None:
            return
        self.run_status.setText(
            f'No se pudo ejecutar el collector: {self._process.errorString()}'
        )
        self._log(self.run_status.text(), error=True)
        if self._process.state() == QProcess.NotRunning:
            process = self._process
            self._process = None
            self.cancel_button.setEnabled(False)
            self.run_button.setEnabled(True)
            process.deleteLater()

    def _load_results(self, path: Path) -> None:
        with path.open(newline='', encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))
        cases = [row for row in rows if row.get('case_id') != '__aggregate__']
        self.results.setRowCount(len(cases))
        columns = (
            'case_id', 'query', 'predicted_node_id', 'semantic_success',
            'nearby_semantic_success', 'navigation_success',
            'rank_first_valid', 'failure_type',
        )
        for row_index, row in enumerate(cases):
            for column, key in enumerate(columns):
                self.results.setItem(
                    row_index, column, QTableWidgetItem(str(row.get(key, '')))
                )
        self.results.resizeColumnsToContents()
        exact = _mean_boolean(cases, 'semantic_success')
        nearby = _mean_boolean(cases, 'nearby_semantic_success')
        navigation = _mean_boolean(cases, 'navigation_success')
        reciprocal = _mean_number(cases, 'reciprocal_rank')
        reciprocal_text = 'n/d' if reciprocal is None else f'{reciprocal:.3f}'
        self.metrics.setText(
            f'{len(cases)} caso(s) · exactitud semántica={_percent(exact)} · '
            f'exactitud cercana={_percent(nearby)} · '
            f'éxito de navegación={_percent(navigation)} · '
            f'MRR={reciprocal_text}'
        )

    def _log(self, message: str, *, error: bool = False) -> None:
        if self._log_callback is not None:
            self._log_callback(message, error)


def _boolean(value: str) -> bool | None:
    normalized = str(value).strip().casefold()
    if normalized in {'true', '1', 'yes'}:
        return True
    if normalized in {'false', '0', 'no'}:
        return False
    return None


def _mean_boolean(rows: list[dict[str, str]], key: str) -> float | None:
    values = [_boolean(row.get(key, '')) for row in rows]
    defined = [value for value in values if value is not None]
    return sum(defined) / len(defined) if defined else None


def _mean_number(rows: list[dict[str, str]], key: str) -> float | None:
    values: list[float] = []
    for row in rows:
        try:
            values.append(float(row.get(key, '')))
        except (TypeError, ValueError):
            continue
    return sum(values) / len(values) if values else None


def _percent(value: float | None) -> str:
    return 'n/d' if value is None else f'{100.0 * value:.1f}%'
