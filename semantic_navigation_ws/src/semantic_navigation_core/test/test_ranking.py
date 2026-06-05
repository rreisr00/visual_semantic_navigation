import numpy as np
import pytest

from semantic_navigation_core.ranking import (
    cosine_similarity,
    jaccard,
    rank_waypoints,
    SIGLIP_PURE,
    SIGLIP_YOLO,
)
from semantic_navigation_core.types import Waypoint


def _wp(node_id, emb, objects=None):
    return Waypoint(
        node_id=node_id,
        position=(0.0, 0.0, 0.0),
        orientation=(0.0, 0.0, 0.0, 1.0),
        embedding=np.asarray(emb, dtype=np.float32),
        objects=objects or [],
    )


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_zero_vector_is_safe(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0

    def test_shape_mismatch_is_safe(self):
        assert cosine_similarity([1.0, 2.0], [1.0]) == 0.0


class TestJaccard:
    def test_identical_sets(self):
        assert jaccard(["a", "b"], ["a", "b"]) == pytest.approx(1.0)

    def test_disjoint_sets(self):
        assert jaccard(["a"], ["b"]) == 0.0

    def test_partial_overlap(self):
        assert jaccard(["a", "b"], ["b", "c"]) == pytest.approx(1 / 3)

    def test_both_empty(self):
        assert jaccard([], []) == 0.0


class TestRankWaypoints:
    def test_orders_by_cosine_descending(self):
        query = [1.0, 0.0]
        wps = [
            _wp("far", [0.0, 1.0]),
            _wp("near", [1.0, 0.0]),
            _wp("mid", [1.0, 1.0]),
        ]
        ranked = rank_waypoints(query, [], wps, mode=SIGLIP_PURE)
        assert [r.waypoint.node_id for r in ranked][0] == "near"
        assert ranked[0].score >= ranked[1].score >= ranked[2].score

    def test_skips_empty_embeddings(self):
        wps = [_wp("empty", []), _wp("ok", [1.0, 0.0])]
        ranked = rank_waypoints([1.0, 0.0], [], wps, mode=SIGLIP_PURE)
        assert len(ranked) == 1
        assert ranked[0].waypoint.node_id == "ok"

    def test_hybrid_mode_uses_objects(self):
        # Two waypoints with identical embeddings; objects break the tie.
        query = [1.0, 0.0]
        wps = [
            _wp("no_obj", [1.0, 0.0], objects=["dog"]),
            _wp("match_obj", [1.0, 0.0], objects=["sofa", "tv"]),
        ]
        ranked = rank_waypoints(
            query, ["sofa", "tv"], wps, mode=SIGLIP_YOLO,
            embed_weight=0.5, object_weight=0.5,
        )
        assert ranked[0].waypoint.node_id == "match_obj"

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError):
            rank_waypoints([1.0], [], [], mode="bogus")

    def test_empty_waypoints_returns_empty(self):
        assert rank_waypoints([1.0, 0.0], [], [], mode=SIGLIP_PURE) == []
