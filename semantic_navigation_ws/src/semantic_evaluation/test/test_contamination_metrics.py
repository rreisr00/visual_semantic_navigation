from semantic_evaluation.core.contamination_metrics import (
    summarize_observation_contamination,
)
from semantic_navigation_core.types import Observation, ObjectObservation, SemanticNode


def test_campaign_contamination_summary():
    node = SemanticNode("n", observations=[Observation(
        "o", camera_room="a", purity=0.2, contamination_class="contaminated",
        objects=[ObjectObservation("chair", room_id="b")],
    )])
    row = summarize_observation_contamination([node])
    assert row["cross_room_detection_rate"] == 1.0
    assert row["contaminated_observation_rate"] == 1.0
