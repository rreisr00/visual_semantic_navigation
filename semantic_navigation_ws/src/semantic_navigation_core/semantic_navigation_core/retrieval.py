"""Retrieval methods over :class:`SemanticNode` — pure Python, no ROS.

Every method reuses the same primitives as the online system
(:func:`~semantic_navigation_core.ranking.cosine_similarity`,
:func:`~semantic_navigation_core.ranking.jaccard`): ``siglip_single_view``
is numerically identical to the orchestrator's ``siglip_pure`` path, and the
object blend generalises its ``siglip_yolo`` cosine+Jaccard blend.

Methods
-------
random_baseline                      random node (seeded RNG).
nearest_node_baseline                nearest node (needs ``query.position``).
room_label_baseline                  exact room-label baseline.
single_view_siglip                   cosine on the first view.
multiview_siglip                     per-view cosine plus aggregation.
siglip_with_objects                  global, object and crop evidence.
siglip_with_objects_and_relations    adds spatial-relation evidence.
hybrid_semantic_retrieval            adds room evidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from semantic_navigation_core.multiview import (
    AGG_SINGLE,
    MultiviewConfig,
    score_node_views,
)
from semantic_navigation_core.ranking import cosine_similarity, jaccard
from semantic_navigation_core.relations import match_relations
from semantic_navigation_core.types import (
    RankedNode,
    SemanticNode,
    SpatialRelation,
)

METHOD_RANDOM_BASELINE = "random_baseline"
METHOD_NEAREST_NODE_BASELINE = "nearest_node_baseline"
METHOD_ROOM_LABEL_BASELINE = "room_label_baseline"
METHOD_SINGLE_VIEW_SIGLIP = "single_view_siglip"
METHOD_MULTIVIEW_SIGLIP = "multiview_siglip"
METHOD_SIGLIP_WITH_OBJECTS = "siglip_with_objects"
METHOD_SIGLIP_WITH_OBJECTS_AND_RELATIONS = "siglip_with_objects_and_relations"
METHOD_HYBRID_SEMANTIC_RETRIEVAL = "hybrid_semantic_retrieval"
ROOM_POLICY_NONE = "none"
ROOM_POLICY_ADDITIVE = "additive"
ROOM_POLICY_STRICT_FILTER = "strict_filter"
SUPPORTED_ROOM_POLICIES = (
    ROOM_POLICY_NONE, ROOM_POLICY_ADDITIVE, ROOM_POLICY_STRICT_FILTER,
)

# Source-compatible constant names; their values use the descriptive contract.
METHOD_BASELINE_RANDOM = METHOD_RANDOM_BASELINE
METHOD_BASELINE_NEAREST = METHOD_NEAREST_NODE_BASELINE
METHOD_BASELINE_ROOM = METHOD_ROOM_LABEL_BASELINE
METHOD_SIGLIP_SINGLE = METHOD_SINGLE_VIEW_SIGLIP
METHOD_SIGLIP_MULTIVIEW = METHOD_MULTIVIEW_SIGLIP
METHOD_SIGLIP_OBJECTS = METHOD_SIGLIP_WITH_OBJECTS
METHOD_SIGLIP_RELATIONS = METHOD_SIGLIP_WITH_OBJECTS_AND_RELATIONS
METHOD_HYBRID_FULL = METHOD_HYBRID_SEMANTIC_RETRIEVAL
SUPPORTED_METHODS: tuple[str, ...] = (
    METHOD_RANDOM_BASELINE, METHOD_NEAREST_NODE_BASELINE,
    METHOD_ROOM_LABEL_BASELINE, METHOD_SINGLE_VIEW_SIGLIP,
    METHOD_MULTIVIEW_SIGLIP, METHOD_SIGLIP_WITH_OBJECTS,
    METHOD_SIGLIP_WITH_OBJECTS_AND_RELATIONS,
    METHOD_HYBRID_SEMANTIC_RETRIEVAL,
)


@dataclass
class SemanticQuery:
    """A retrieval query, decoupled from where it came from (YAML, voice, ROS).

    Attributes:
        text: Original query string (informative; scoring uses ``embedding``).
        embedding: L2-normalised SigLIP text/image embedding.
        objects: Object labels the query mentions (YOLO vocabulary).
        relations: Expected spatial relations (2D hypotheses to match).
        room: Expected room id (normalised, e.g. ``"cocina"``), if any.
        position: (x, y) used only by the ``baseline_nearest`` baseline.
    """

    text: str = ""
    embedding: np.ndarray | None = None
    objects: list[str] = field(default_factory=list)
    relations: list[SpatialRelation] = field(default_factory=list)
    room: str | None = None
    position: tuple[float, float] | None = None


@dataclass
class HybridWeights:
    """Weights of the hybrid score. All components live in [~0, 1].

    ``hybrid = alpha·global + beta·objects + gamma·crops
             + delta·relations + epsilon·room``
    """

    alpha: float = 1.0     # global (multi-view SigLIP) similarity
    beta: float = 0.0      # object-label overlap (confidence-weighted Jaccard)
    gamma: float = 0.0     # best crop-embedding similarity
    delta: float = 0.0     # relation-hypothesis match
    epsilon: float = 0.0   # room match


# Presets: the objects/relations methods are the hybrid score with the later
# components switched off. Splits follow the online siglip_yolo blend (0.7/0.3).
SIGLIP_OBJECTS_DEFAULT = HybridWeights(alpha=0.7, beta=0.2, gamma=0.1)
SIGLIP_RELATIONS_DEFAULT = HybridWeights(
    alpha=0.6, beta=0.15, gamma=0.1, delta=0.15
)
HYBRID_FULL_DEFAULT = HybridWeights(
    alpha=0.5, beta=0.15, gamma=0.1, delta=0.1, epsilon=0.15
)


@dataclass
class RetrievalConfig:
    """Everything a retrieval method needs beyond the query and the nodes."""

    method: str = METHOD_SIGLIP_SINGLE
    multiview: MultiviewConfig = field(default_factory=MultiviewConfig)
    weights: HybridWeights = field(default_factory=HybridWeights)
    seed: int = 42
    room_policy: str = ROOM_POLICY_ADDITIVE

    def __post_init__(self) -> None:
        if self.method not in SUPPORTED_METHODS:
            raise ValueError(
                f"method must be one of {SUPPORTED_METHODS}, got {self.method!r}"
            )
        if self.room_policy not in SUPPORTED_ROOM_POLICIES:
            raise ValueError(
                f"room_policy must be one of {SUPPORTED_ROOM_POLICIES}, "
                f"got {self.room_policy!r}"
            )


# ── Component scores ─────────────────────────────────────────────────────── #


def object_score(query_objects: list[str], node: SemanticNode) -> float:
    """Object-label overlap in [0, 1].

    Confidence-weighted coverage of the query labels when detections carry
    confidences; plain :func:`jaccard` (the online siglip_yolo metric) is the
    lower bound behaviour when every confidence is 1.0.
    """
    if not query_objects:
        return 0.0
    best_conf: dict[str, float] = {}
    for obs in node.observations:
        for obj in obs.objects:
            best_conf[obj.label] = max(best_conf.get(obj.label, 0.0), obj.confidence)
    if not best_conf:
        return 0.0
    coverage = sum(best_conf.get(label, 0.0) for label in set(query_objects))
    coverage /= len(set(query_objects))
    # Blend with Jaccard so nodes cluttered with unrelated objects rank lower.
    jac = jaccard(query_objects, best_conf.keys())
    return 0.5 * coverage + 0.5 * jac


def crop_score(query_embedding: np.ndarray | None, node: SemanticNode) -> float:
    """Best cosine similarity between the query and any object-crop embedding."""
    if query_embedding is None:
        return 0.0
    best = 0.0
    for obs in node.observations:
        for obj in obs.objects:
            if obj.embedding is not None:
                best = max(best, cosine_similarity(query_embedding, obj.embedding))
    return best


def relation_score(
    query_relations: list[SpatialRelation], node: SemanticNode
) -> float:
    """Fraction of expected relations supported by the node's hypotheses."""
    return match_relations(query_relations, node.relations())


def room_score(query_room: str | None, node: SemanticNode) -> float:
    """1.0 when the node's room matches the query room (case-insensitive)."""
    if not query_room or not node.room_id:
        return 0.0
    return 1.0 if node.room_id.strip().lower() == query_room.strip().lower() else 0.0


# ── Per-method node scoring ──────────────────────────────────────────────── #


def _global_score(
    query: SemanticQuery, node: SemanticNode, multiview: MultiviewConfig
) -> float:
    if query.embedding is None:
        return 0.0
    return score_node_views(query.embedding, node, multiview)


def _hybrid_components(
    query: SemanticQuery, node: SemanticNode, config: RetrievalConfig
) -> dict[str, float]:
    w = config.weights
    comps = {"global_similarity": _global_score(query, node, config.multiview)}
    # Components with zero weight are skipped (and reported as 0.0) so the
    # cheaper presets do not pay for relations/rooms they ignore.
    comps["object_match_score"] = object_score(query.objects, node) if w.beta else 0.0
    comps["crop_similarity"] = crop_score(query.embedding, node) if w.gamma else 0.0
    comps["relation_match_score"] = relation_score(query.relations, node) if w.delta else 0.0
    comps["room_match_score"] = room_score(query.room, node)
    return comps


def _combine(
    comps: dict[str, float], w: HybridWeights, room_policy: str
) -> float:
    return (
        w.alpha * comps["global_similarity"]
        + w.beta * comps["object_match_score"]
        + w.gamma * comps["crop_similarity"]
        + w.delta * comps["relation_match_score"]
        + (
            w.epsilon * comps["room_match_score"]
            if room_policy == ROOM_POLICY_ADDITIVE else 0.0
        )
    )


# ── Public API ───────────────────────────────────────────────────────────── #


def rank_nodes(
    query: SemanticQuery,
    nodes: list[SemanticNode],
    config: RetrievalConfig,
    rng: np.random.Generator | None = None,
) -> list[RankedNode]:
    """Rank nodes with the configured method, best first.

    Args:
        query: The semantic query (embedding required for every SigLIP-based
            method).
        nodes: Candidate nodes; nodes without any embedding are skipped for
            embedding-based methods.
        config: Method + multiview + hybrid weights.
        rng: RNG for ``baseline_random`` (defaults to ``config.seed``).

    Returns:
        ``RankedNode`` list sorted high→low with per-component breakdowns.
    """
    method = config.method
    candidates = nodes
    if config.room_policy == ROOM_POLICY_STRICT_FILTER and query.room:
        candidates = [node for node in nodes if room_score(query.room, node) == 1.0]

    if method == METHOD_BASELINE_RANDOM:
        rng = rng or np.random.default_rng(config.seed)
        scores = rng.random(len(candidates))
        ranked = [
            RankedNode(node=n, score=float(s), components={"random": float(s)})
            for n, s in zip(candidates, scores)
        ]
    elif method == METHOD_BASELINE_NEAREST:
        if query.position is None:
            raise ValueError("baseline_nearest requires query.position=(x, y)")
        qx, qy = query.position
        ranked = []
        for n in candidates:
            dist = float(np.hypot(n.position[0] - qx, n.position[1] - qy))
            # Negative distance so "higher score = better" holds like elsewhere.
            ranked.append(
                RankedNode(node=n, score=-dist, components={"distance": dist})
            )
    elif method == METHOD_BASELINE_ROOM:
        ranked = [
            RankedNode(
                node=n,
                score=room_score(query.room, n),
                components={"room": room_score(query.room, n)},
            )
            for n in candidates
        ]
    elif method in (METHOD_SIGLIP_SINGLE, METHOD_SIGLIP_MULTIVIEW):
        # siglip_single_view forces the single-view aggregation → parity
        # with the online siglip_pure path (plain cosine on the stored
        # embedding).
        multiview = (
            replace(config.multiview, method=AGG_SINGLE)
            if method == METHOD_SIGLIP_SINGLE
            else config.multiview
        )
        ranked = []
        for n in candidates:
            if not n.embeddings():
                continue
            score = score_node_views(query.embedding, n, multiview)
            ranked.append(
                RankedNode(node=n, score=score, components={"global_similarity": score})
            )
    else:  # objects / relations / full — hybrid with the configured weights
        ranked = []
        for n in candidates:
            if not n.embeddings():
                continue
            comps = _hybrid_components(query, n, config)
            ranked.append(
                RankedNode(
                    node=n,
                    score=_combine(comps, config.weights, config.room_policy),
                    components=comps,
                )
            )

    ranked.sort(key=lambda r: r.score, reverse=True)
    return ranked


def default_weights(method: str) -> HybridWeights:
    """The documented starting weights for each hybrid method."""
    if method == METHOD_SIGLIP_OBJECTS:
        return replace(SIGLIP_OBJECTS_DEFAULT)
    if method == METHOD_SIGLIP_RELATIONS:
        return replace(SIGLIP_RELATIONS_DEFAULT)
    if method == METHOD_HYBRID_FULL:
        return replace(HYBRID_FULL_DEFAULT)
    return HybridWeights()
