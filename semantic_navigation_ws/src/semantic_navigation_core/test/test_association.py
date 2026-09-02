import numpy as np

from semantic_navigation_core.association import AssociationConfig, match_object
from semantic_navigation_core.types import ObjectObservation


def _object(object_id, embedding=None, position=None):
    return ObjectObservation(
        label="cup",
        object_id=object_id,
        embedding=None if embedding is None else np.asarray(embedding, dtype=np.float32),
        position_3d=position,
    )


def test_metric_position_has_priority_over_crop():
    detection = _object("", [1.0, 0.0], (0.1, 0.0, 0.0))
    candidates = [
        _object("near", [0.0, 1.0], (0.0, 0.0, 0.0)),
        _object("visual", [1.0, 0.0], (2.0, 0.0, 0.0)),
    ]
    match = match_object(detection, candidates)
    assert match.object_id == "near"
    assert match.evidence == "position_3d"


def test_crop_similarity_and_exclusion_distinguish_repeated_instances():
    detection = _object("", [0.0, 1.0])
    candidates = [_object("a", [1.0, 0.0]), _object("b", [0.0, 1.0])]
    assert match_object(detection, candidates).object_id == "b"
    assert match_object(detection, candidates, excluded_ids={"b"}) is None


def test_ambiguous_class_only_detection_is_not_associated():
    detection = _object("")
    candidates = [_object("a"), _object("b")]
    assert match_object(detection, candidates) is None


def test_similarity_below_threshold_is_rejected_without_fallback():
    detection = _object("", [1.0, 0.0])
    candidate = _object("a", [0.0, 1.0])
    config = AssociationConfig(allow_single_class_fallback=False)
    assert match_object(detection, [candidate], config=config) is None
