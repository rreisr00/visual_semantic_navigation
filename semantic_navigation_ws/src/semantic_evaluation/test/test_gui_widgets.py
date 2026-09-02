import os
from types import SimpleNamespace

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from python_qt_binding.QtWidgets import QApplication, QGraphicsTextItem, QSplitter

from semantic_evaluation.campaign_designer_widget import CampaignDesignerWidget
from semantic_evaluation.retrieval_evaluation_widget import (
    RetrievalEvaluationWidget,
)
from test_campaign_designer import _workspace


def _application():
    return QApplication.instance() or QApplication([])


def test_evaluation_and_campaign_widgets_initialize_headless(tmp_path):
    application = _application()
    workspace = _workspace(tmp_path)

    evaluation = RetrievalEvaluationWidget(
        scene_id=workspace.scene_id,
        graph_database=workspace.graph_database,
        map_file=workspace.map_metadata.yaml_path,
        queries_file=workspace.queries_file,
        ground_truth_file=workspace.ground_truth_file,
        submit_callback=lambda _query, _language, _navigate: True,
        cancel_callback=lambda: True,
    )
    designer = CampaignDesignerWidget(
        scene_id=workspace.scene_id,
        graph_database=workspace.graph_database,
        map_file=workspace.map_metadata.yaml_path,
        queries_file=workspace.queries_file,
        ground_truth_file=workspace.ground_truth_file,
    )

    assert evaluation.map_scene.items()
    assert designer.campaign_runner.validation_summary.text()
    assert len(evaluation.findChildren(QSplitter)) >= 3
    assert len(designer.findChildren(QSplitter)) >= 4
    assert application is QApplication.instance()


def test_relation_diagram_separates_parallel_labels(tmp_path):
    application = _application()
    workspace = _workspace(tmp_path)
    designer = CampaignDesignerWidget(
        scene_id=workspace.scene_id,
        graph_database=workspace.graph_database,
        map_file=workspace.map_metadata.yaml_path,
        queries_file=workspace.queries_file,
        ground_truth_file=workspace.ground_truth_file,
    )
    objects = [
        SimpleNamespace(object_id='chair_1', label='chair'),
        SimpleNamespace(object_id='table_1', label='table'),
    ]
    relations = [
        SimpleNamespace(
            subject_id='chair_1',
            subject='chair',
            object_id='table_1',
            obj='table',
            predicate=predicate,
            confidence=0.9,
        )
        for predicate in ('LEFT_OF', 'NEAR', 'IN_FRONT_OF', 'BELOW')
    ]

    designer._draw_relation_graph(SimpleNamespace(
        objects=objects, relations=relations
    ))

    labels = [
        item for item in designer.relation_scene.items()
        if isinstance(item, QGraphicsTextItem)
        and item.toPlainText().startswith('R')
    ]
    rectangles = [item.sceneBoundingRect() for item in labels]
    assert len(labels) == 4
    assert all(
        not first.intersects(second)
        for index, first in enumerate(rectangles)
        for second in rectangles[index + 1:]
    )
    assert application is QApplication.instance()
