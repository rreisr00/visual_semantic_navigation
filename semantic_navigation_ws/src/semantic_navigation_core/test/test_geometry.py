import numpy as np

from semantic_navigation_core.geometry import (
    infer_3d_relations,
    project_box_center,
    transform_point,
)
from semantic_navigation_core.types import ObjectObservation


def test_project_box_center_uses_metric_optical_convention():
    depth = np.full((10, 10), 2.0, dtype=np.float32)
    point = project_box_center((4, 4, 6, 6), depth, (10.0, 10.0, 5.0, 5.0))
    assert point == (0.0, 0.0, 2.0)


def test_project_rejects_invalid_depth():
    depth = np.full((10, 10), np.nan, dtype=np.float32)
    assert project_box_center((2, 2, 4, 4), depth, (10, 10, 5, 5)) is None


def test_3d_relations_are_typed_geometric_facts():
    cup = ObjectObservation(
        "cup", object_id="cup_1", position_3d=(0.0, -0.2, 0.8)
    )
    table = ObjectObservation(
        "table", object_id="table_1", position_3d=(0.0, 0.1, 1.2)
    )
    relations = infer_3d_relations([cup, table])
    predicates = {item.predicate for item in relations if item.subject_id == "cup_1"}
    assert {"ABOVE", "IN_FRONT_OF", "NEAR"} <= predicates
    assert all(item.relation_type == "geometric_3d_relation" for item in relations)


def test_transform_point_applies_rotation_then_translation():
    half = np.sqrt(0.5)
    transformed = transform_point(
        (1.0, 0.0, 0.0),
        (2.0, 3.0, 0.0),
        (0.0, 0.0, half, half),
    )
    np.testing.assert_allclose(transformed, (2.0, 4.0, 0.0), atol=1e-6)
