"""2D spatial-relation inference between detected objects — pure Python.

The predicates are *visual hypotheses* derived from bounding boxes in a single
2D image (no depth, no physics): ``POSSIBLY_ON_TOP_OF`` means "the subject's
box rests on the object's box in image space", not a confirmed support
relation. Confidences in [0, 1] express how clearly the geometric pattern
holds.

No ROS imports; operates on
:class:`~semantic_navigation_core.types.ObjectObservation` boxes.
"""
from __future__ import annotations

from typing import Iterable, Sequence

from semantic_navigation_core.types import ObjectObservation, SpatialRelation

REL_LEFT_OF = "LEFT_OF"
REL_RIGHT_OF = "RIGHT_OF"
REL_ABOVE = "ABOVE"
REL_BELOW = "BELOW"
REL_NEAR = "NEAR"
REL_OVERLAPS = "OVERLAPS"
REL_POSSIBLY_ON_TOP_OF = "POSSIBLY_ON_TOP_OF"
SUPPORTED_PREDICATES: tuple[str, ...] = (
    REL_LEFT_OF, REL_RIGHT_OF, REL_ABOVE, REL_BELOW,
    REL_NEAR, REL_OVERLAPS, REL_POSSIBLY_ON_TOP_OF,
)

# Symmetric predicate pairs used when matching query relations.
_INVERSE = {
    REL_LEFT_OF: REL_RIGHT_OF,
    REL_RIGHT_OF: REL_LEFT_OF,
    REL_ABOVE: REL_BELOW,
    REL_BELOW: REL_ABOVE,
}


def _center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _size(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return (max(box[2] - box[0], 1e-6), max(box[3] - box[1], 1e-6))


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _h_overlap_ratio(a, b) -> float:
    """Horizontal overlap as a fraction of the narrower box's width."""
    overlap = min(a[2], b[2]) - max(a[0], b[0])
    if overlap <= 0:
        return 0.0
    return overlap / min(_size(a)[0], _size(b)[0])


def infer_relations(
    objects: Sequence[ObjectObservation],
    near_factor: float = 1.5,
    overlap_iou: float = 0.15,
    on_top_gap_factor: float = 0.35,
) -> list[SpatialRelation]:
    """Infer pairwise 2D relations from detection boxes.

    For every ordered pair (subject, obj) with boxes, emits the dominant
    directional relation (LEFT_OF / RIGHT_OF / ABOVE / BELOW), plus NEAR,
    OVERLAPS and POSSIBLY_ON_TOP_OF when their geometric conditions hold.
    Objects without a box are skipped. Confidence is the relative strength
    of the geometric evidence, clipped to [0, 1].

    Args:
        objects: Detections of a single observation (pixel-space boxes).
        near_factor: Two objects are NEAR when the center distance is below
            ``near_factor`` × mean box diagonal.
        overlap_iou: Minimum IoU to emit OVERLAPS.
        on_top_gap_factor: Max vertical gap between subject bottom and object
            top, as a fraction of the object's height, to emit
            POSSIBLY_ON_TOP_OF (with horizontal overlap required).
    """
    boxed = [o for o in objects if o.box is not None]
    relations: list[SpatialRelation] = []

    for i, subj in enumerate(boxed):
        for j, obj in enumerate(boxed):
            if i == j:
                continue
            pair_conf = min(subj.confidence, obj.confidence)
            sc, oc = _center(subj.box), _center(obj.box)
            dx, dy = oc[0] - sc[0], oc[1] - sc[1]
            sw, sh = _size(subj.box)
            ow, oh = _size(obj.box)

            # Dominant directional relation (image coords: +y goes down).
            if abs(dx) >= abs(dy):
                pred = REL_LEFT_OF if dx > 0 else REL_RIGHT_OF
                strength = abs(dx) / (abs(dx) + abs(dy) + 1e-6)
            else:
                pred = REL_ABOVE if dy > 0 else REL_BELOW
                strength = abs(dy) / (abs(dx) + abs(dy) + 1e-6)
            relations.append(SpatialRelation(
                subj.label, pred, obj.label,
                confidence=round(min(1.0, strength) * pair_conf, 4),
                subject_id=subj.object_id or subj.label,
                object_id=obj.object_id or obj.label,
            ))

            # Symmetric predicates: emit once per unordered pair (i < j).
            if i < j:
                diag = ((sw ** 2 + sh ** 2) ** 0.5 + (ow ** 2 + oh ** 2) ** 0.5) / 2.0
                dist = (dx ** 2 + dy ** 2) ** 0.5
                if dist < near_factor * diag:
                    near_conf = 1.0 - dist / (near_factor * diag)
                    relations.append(SpatialRelation(
                        subj.label, REL_NEAR, obj.label,
                        confidence=round(near_conf * pair_conf, 4),
                        subject_id=subj.object_id or subj.label,
                        object_id=obj.object_id or obj.label,
                    ))
                iou = _iou(subj.box, obj.box)
                if iou >= overlap_iou:
                    relations.append(SpatialRelation(
                        subj.label, REL_OVERLAPS, obj.label,
                        confidence=round(min(1.0, iou / 0.5) * pair_conf, 4),
                        subject_id=subj.object_id or subj.label,
                        object_id=obj.object_id or obj.label,
                    ))

            # subject on top of obj: subject bottom close to obj top, with
            # horizontal overlap (a cup ON a table).
            gap = abs(subj.box[3] - obj.box[1])
            if (
                subj.box[3] <= obj.box[1] + on_top_gap_factor * oh
                and gap <= on_top_gap_factor * oh
                and _h_overlap_ratio(subj.box, obj.box) >= 0.5
            ):
                on_conf = 1.0 - gap / max(on_top_gap_factor * oh, 1e-6)
                relations.append(SpatialRelation(
                    subj.label, REL_POSSIBLY_ON_TOP_OF, obj.label,
                    confidence=round(max(0.0, on_conf) * pair_conf, 4),
                    subject_id=subj.object_id or subj.label,
                    object_id=obj.object_id or obj.label,
                ))

    return relations


def _matches(query: SpatialRelation, candidate: SpatialRelation) -> bool:
    """Label + predicate match, accepting the mirrored directional form."""
    if query.subject == candidate.subject and query.obj == candidate.obj:
        if query.predicate == candidate.predicate:
            return True
    if query.subject == candidate.obj and query.obj == candidate.subject:
        # NEAR / OVERLAPS are symmetric; directional predicates invert.
        if query.predicate in (REL_NEAR, REL_OVERLAPS):
            return query.predicate == candidate.predicate
        return _INVERSE.get(candidate.predicate) == query.predicate
    return False


def match_relations(
    query_relations: Iterable[SpatialRelation],
    node_relations: Sequence[SpatialRelation],
) -> float:
    """Fraction of query relations supported by the node, in [0, 1].

    Each query relation contributes the best matching node relation's
    confidence; unmatched relations contribute 0. Returns 0.0 when the query
    has no relations.
    """
    queries = list(query_relations)
    if not queries:
        return 0.0
    total = 0.0
    for q in queries:
        best = 0.0
        for cand in node_relations:
            if _matches(q, cand):
                best = max(best, cand.confidence)
        total += best
    return total / len(queries)
