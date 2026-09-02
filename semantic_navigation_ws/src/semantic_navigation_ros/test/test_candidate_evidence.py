from types import SimpleNamespace

import numpy as np
from builtin_interfaces.msg import Time

from semantic_navigation_core.retrieval import SemanticQuery
from semantic_navigation_core.types import (
    ObjectObservation,
    Observation,
    SemanticNode,
)
from semantic_navigation_ros.semantic_orchestrator_node import (
    _candidate_message,
)


def test_candidate_message_exposes_object_and_crop_evidence():
    node = SemanticNode(
        node_id='salon_01',
        observations=[Observation(
            observation_id='view_1',
            objects=[
                ObjectObservation(
                    label='sofa',
                    object_id='sofa_1',
                    embedding=np.array([1.0, 0.0], dtype=np.float32),
                ),
                ObjectObservation(
                    label='plant',
                    object_id='plant_1',
                    embedding=np.array([0.0, 1.0], dtype=np.float32),
                ),
            ],
        )],
    )
    ranked = SimpleNamespace(
        node=node,
        score=0.8,
        components={
            'global_similarity': 0.7,
            'object_match_score': 0.9,
            'crop_similarity': 1.0,
            'relation_match_score': 0.0,
            'room_match_score': 1.0,
        },
    )
    query = SemanticQuery(
        embedding=np.array([1.0, 0.0], dtype=np.float32),
        objects=['sofa'],
    )

    candidate = _candidate_message(ranked, Time(), query)

    assert candidate.node_id == 'salon_01'
    assert candidate.matched_object_ids == ['sofa_1']
    assert candidate.matched_object_labels == ['sofa']
    assert candidate.best_crop_object_id == 'sofa_1'
    assert candidate.best_crop_object_label == 'sofa'
