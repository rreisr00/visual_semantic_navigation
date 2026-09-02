"""Depth projection and conservative geometric relation extraction."""
from __future__ import annotations

from math import dist
from typing import Sequence

import numpy as np

from semantic_navigation_core.types import ObjectObservation, SpatialRelation


def transform_point(
    point: tuple[float, float, float],
    translation: tuple[float, float, float],
    quaternion: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    """Apply a ``target<-source`` rigid transform to a 3D point."""
    vector = np.asarray(point, dtype=np.float64)
    qx, qy, qz, qw = quaternion
    q_vector = np.asarray((qx, qy, qz), dtype=np.float64)
    rotated = (
        vector
        + 2.0 * qw * np.cross(q_vector, vector)
        + 2.0 * np.cross(q_vector, np.cross(q_vector, vector))
    )
    transformed = rotated + np.asarray(translation, dtype=np.float64)
    return tuple(float(value) for value in transformed)


def project_box_center(
    box: tuple[float, float, float, float],
    depth_image: np.ndarray,
    intrinsics: tuple[float, float, float, float],
    depth_scale: float = 1.0,
    minimum_depth_m: float = 0.1,
    maximum_depth_m: float = 10.0,
) -> tuple[float, float, float] | None:
    """Project the robust median depth around a box centre to optical XYZ.

    Intrinsics are ``(fx, fy, cx, cy)``. The returned coordinates follow the
    ROS optical convention: +x right, +y down, +z forward, in metres.
    """
    depth = np.asarray(depth_image)
    if depth.ndim != 2 or depth.size == 0:
        return None
    x1, y1, x2, y2 = box
    center_u = int(round((x1 + x2) * 0.5))
    center_v = int(round((y1 + y2) * 0.5))
    radius = max(1, int(round(min(abs(x2 - x1), abs(y2 - y1)) * 0.08)))
    u0, u1 = max(0, center_u - radius), min(depth.shape[1], center_u + radius + 1)
    v0, v1 = max(0, center_v - radius), min(depth.shape[0], center_v + radius + 1)
    values = depth[v0:v1, u0:u1].astype(np.float32, copy=False) * depth_scale
    values = values[
        np.isfinite(values)
        & (values >= minimum_depth_m)
        & (values <= maximum_depth_m)
    ]
    if not values.size:
        return None
    z = float(np.median(values))
    fx, fy, cx, cy = intrinsics
    if fx <= 0.0 or fy <= 0.0:
        return None
    return (
        (float(center_u) - cx) * z / fx,
        (float(center_v) - cy) * z / fy,
        z,
    )


def infer_3d_relations(
    objects: Sequence[ObjectObservation],
    near_distance_m: float = 1.0,
    axis_separation_m: float = 0.12,
    reference_frame: str = "camera_depth_optical_frame",
) -> list[SpatialRelation]:
    """Infer metric relations only for detections with valid 3D positions."""
    positioned = [item for item in objects if item.position_3d is not None]
    relations: list[SpatialRelation] = []
    for left_index, subject in enumerate(positioned):
        for right_index, target in enumerate(positioned):
            if left_index == right_index:
                continue
            subject_id = subject.object_id or subject.label
            object_id = target.object_id or target.label
            confidence = min(subject.confidence, target.confidence)
            sx, sy, sz = subject.position_3d
            tx, ty, tz = target.position_3d
            if abs(sy - ty) >= axis_separation_m:
                predicate = "ABOVE" if sy < ty else "BELOW"
                relations.append(SpatialRelation(
                    subject.label,
                    predicate,
                    target.label,
                    confidence=confidence,
                    subject_id=subject_id,
                    object_id=object_id,
                    reference_frame=reference_frame,
                    relation_type="geometric_3d_relation",
                ))
            if abs(sz - tz) >= axis_separation_m:
                predicate = "IN_FRONT_OF" if sz < tz else "BEHIND"
                relations.append(SpatialRelation(
                    subject.label,
                    predicate,
                    target.label,
                    confidence=confidence,
                    subject_id=subject_id,
                    object_id=object_id,
                    reference_frame=reference_frame,
                    relation_type="geometric_3d_relation",
                ))
            if left_index < right_index:
                separation = dist(subject.position_3d, target.position_3d)
                if separation <= near_distance_m:
                    relations.append(SpatialRelation(
                        subject.label,
                        "NEAR",
                        target.label,
                        confidence=confidence * (
                            1.0 - separation / max(near_distance_m, 1e-9)
                        ),
                        subject_id=subject_id,
                        object_id=object_id,
                        reference_frame=reference_frame,
                        relation_type="geometric_3d_relation",
                    ))
    return relations
