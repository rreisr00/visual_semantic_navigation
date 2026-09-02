from semantic_evaluation.core.retrieval_visualization import object_evidence


def test_object_evidence_matches_by_identifier_and_label():
    candidate = {
        "matched_object_ids": ["chair_3"],
        "matched_object_labels": ["SOFA"],
        "best_crop_object_id": "lamp_2",
        "best_crop_object_label": "lamp",
    }

    assert object_evidence("chair_3", "chair", candidate) == ("object_match",)
    assert object_evidence("", "sofa", candidate) == ("object_match",)
    assert object_evidence("lamp_2", "lamp", candidate) == (
        "crop_similarity",
    )


def test_object_evidence_can_report_both_components():
    candidate = {
        "matched_object_ids": ["desk_1"],
        "matched_object_labels": [],
        "best_crop_object_id": "desk_1",
        "best_crop_object_label": "desk",
    }

    assert object_evidence("desk_1", "desk", candidate) == (
        "object_match",
        "crop_similarity",
    )


def test_object_evidence_marks_unrelated_objects_as_unused():
    assert object_evidence("plant_1", "plant", None) == ()
    assert object_evidence("plant_1", "plant", {}) == ()
