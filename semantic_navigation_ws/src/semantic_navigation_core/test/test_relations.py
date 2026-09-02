"""Unit tests for 2D relation inference and matching."""
from semantic_navigation_core.relations import (
    REL_LEFT_OF,
    REL_NEAR,
    REL_OVERLAPS,
    REL_POSSIBLY_ON_TOP_OF,
    REL_RIGHT_OF,
    infer_relations,
    match_relations,
)
from semantic_navigation_core.types import ObjectObservation, SpatialRelation


def obj(label, box, conf=1.0):
    return ObjectObservation(label=label, confidence=conf, box=box)


def predicates(relations, subject, target):
    return {r.predicate for r in relations if r.subject == subject and r.obj == target}


def test_left_right_pair():
    rels = infer_relations([
        obj("cup", (0, 40, 20, 60)),
        obj("tv", (100, 40, 130, 60)),
    ])
    assert REL_LEFT_OF in predicates(rels, "cup", "tv")
    assert REL_RIGHT_OF in predicates(rels, "tv", "cup")


def test_on_top_of_and_overlap():
    # Cup resting exactly on a table top, horizontally inside it.
    table = obj("table", (0, 100, 200, 200))
    cup = obj("cup", (80, 70, 120, 100))
    rels = infer_relations([cup, table])
    assert REL_POSSIBLY_ON_TOP_OF in predicates(rels, "cup", "table")
    overlapping = infer_relations([
        obj("a", (0, 0, 100, 100)), obj("b", (50, 0, 150, 100)),
    ])
    assert any(r.predicate == REL_OVERLAPS for r in overlapping)


def test_near_confidence_decreases_with_distance():
    close = infer_relations([obj("a", (0, 0, 10, 10)), obj("b", (12, 0, 22, 10))])
    far = infer_relations([obj("a", (0, 0, 10, 10)), obj("b", (18, 0, 28, 10))])
    conf_close = max(r.confidence for r in close if r.predicate == REL_NEAR)
    conf_far = max((r.confidence for r in far if r.predicate == REL_NEAR), default=0.0)
    assert conf_close > conf_far


def test_objects_without_box_are_skipped():
    assert infer_relations([ObjectObservation(label="ghost")]) == []


def test_match_relations_direct_and_inverse():
    node_rels = [SpatialRelation("cup", REL_LEFT_OF, "tv", confidence=0.9)]
    direct = SpatialRelation("cup", REL_LEFT_OF, "tv")
    inverse = SpatialRelation("tv", REL_RIGHT_OF, "cup")
    missing = SpatialRelation("cup", REL_NEAR, "sofa")
    assert match_relations([direct], node_rels) == 0.9
    assert match_relations([inverse], node_rels) == 0.9
    assert match_relations([missing], node_rels) == 0.0
    assert match_relations([direct, missing], node_rels) == 0.45
    assert match_relations([], node_rels) == 0.0
