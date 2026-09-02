"""Reusable detection, relation and uncertainty metrics."""
from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Mapping, Sequence

import numpy as np

from semantic_navigation_core.types import ObjectObservation, SpatialRelation

_PREDICATE_ALIASES = {
    "left of": "LEFT_OF", "to the left of": "LEFT_OF",
    "right of": "RIGHT_OF", "to the right of": "RIGHT_OF",
    "above": "ABOVE", "over": "ABOVE", "below": "BELOW", "under": "BELOW",
    "near": "NEAR", "next to": "NEAR", "beside": "NEAR", "close to": "NEAR",
    "overlaps": "OVERLAPS", "overlapping": "OVERLAPS",
    "on": "POSSIBLY_ON_TOP_OF", "on top of": "POSSIBLY_ON_TOP_OF",
    "sitting on": "POSSIBLY_ON_TOP_OF", "standing on": "POSSIBLY_ON_TOP_OF",
}
SYMMETRIC_PREDICATES = {"NEAR", "OVERLAPS"}


def normalize_label(value: str) -> str:
    return " ".join(value.strip().lower().replace("_", " ").split())


def normalize_predicate(value: str) -> str | None:
    normalized = normalize_label(value)
    canonical = normalized.upper().replace(" ", "_")
    allowed = {
        "LEFT_OF", "RIGHT_OF", "ABOVE", "BELOW", "NEAR", "OVERLAPS",
        "POSSIBLY_ON_TOP_OF",
    }
    if canonical in allowed:
        return canonical
    return _PREDICATE_ALIASES.get(normalized)


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    intersection = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def match_detections(
    predicted: Sequence[ObjectObservation],
    ground_truth: Sequence[ObjectObservation],
    class_mapping: Mapping[str, str],
    iou_threshold: float = 0.5,
) -> tuple[int, int, int, list[str]]:
    """Greedy class-aware one-to-one box matching: TP, FP, FN, unmapped GT."""
    used: set[int] = set()
    true_positive = 0
    unmapped: list[str] = []
    mapped_gt: list[tuple[str, ObjectObservation]] = []
    for item in ground_truth:
        mapped = class_mapping.get(normalize_label(item.label))
        if mapped is None:
            unmapped.append(item.label)
        else:
            mapped_gt.append((normalize_label(mapped), item))
    for detection in sorted(predicted, key=lambda item: item.confidence, reverse=True):
        best_index = None
        best_iou = 0.0
        for index, (label, target) in enumerate(mapped_gt):
            if index in used or normalize_label(detection.label) != label:
                continue
            if detection.box is None or target.box is None:
                continue
            overlap = box_iou(detection.box, target.box)
            if overlap >= iou_threshold and overlap > best_iou:
                best_index, best_iou = index, overlap
        if best_index is not None:
            used.add(best_index)
            true_positive += 1
    false_positive = len(predicted) - true_positive
    false_negative = len(mapped_gt) - true_positive
    return true_positive, false_positive, false_negative, unmapped


def precision_recall_f1(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else math.nan
    recall = tp / (tp + fn) if tp + fn else math.nan
    f1 = (
        2 * precision * recall / (precision + recall)
        if not math.isnan(precision) and not math.isnan(recall) and precision + recall
        else math.nan
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def detection_average_precision(
    predictions: Mapping[str, Sequence[ObjectObservation]],
    ground_truth: Mapping[str, Sequence[ObjectObservation]],
    class_mapping: Mapping[str, str],
    iou_threshold: float = 0.5,
) -> tuple[list[dict[str, float | int | str]], float]:
    """VOC-style interpolated AP per mapped class and macro mAP."""
    mapped_truth: dict[str, dict[str, list[ObjectObservation]]] = {}
    for observation_id, objects in ground_truth.items():
        for item in objects:
            mapped = class_mapping.get(normalize_label(item.label))
            if mapped is not None and item.box is not None:
                mapped_truth.setdefault(normalize_label(mapped), {}).setdefault(
                    observation_id, []
                ).append(item)
    rows: list[dict[str, float | int | str]] = []
    for label, truth_by_image in sorted(mapped_truth.items()):
        ranked = sorted(
            (
                (item.confidence, observation_id, item)
                for observation_id, objects in predictions.items()
                for item in objects
                if normalize_label(item.label) == label and item.box is not None
            ),
            reverse=True,
            key=lambda value: value[0],
        )
        matched: dict[str, set[int]] = {key: set() for key in truth_by_image}
        tp, fp = [], []
        for _, observation_id, detection in ranked:
            targets = truth_by_image.get(observation_id, [])
            best_index, best_overlap = None, 0.0
            for index, target in enumerate(targets):
                if index in matched.get(observation_id, set()):
                    continue
                overlap = box_iou(detection.box, target.box)
                if overlap >= iou_threshold and overlap > best_overlap:
                    best_index, best_overlap = index, overlap
            if best_index is None:
                tp.append(0.0)
                fp.append(1.0)
            else:
                matched[observation_id].add(best_index)
                tp.append(1.0)
                fp.append(0.0)
        n_truth = sum(len(values) for values in truth_by_image.values())
        if ranked:
            cumulative_tp = np.cumsum(tp)
            cumulative_fp = np.cumsum(fp)
            recall = cumulative_tp / max(n_truth, 1)
            precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1e-12)
            recall_points = np.concatenate(([0.0], recall, [1.0]))
            precision_points = np.concatenate(([0.0], precision, [0.0]))
            for index in range(precision_points.size - 2, -1, -1):
                precision_points[index] = max(
                    precision_points[index], precision_points[index + 1]
                )
            changes = np.where(recall_points[1:] != recall_points[:-1])[0]
            average_precision = float(np.sum(
                (recall_points[changes + 1] - recall_points[changes])
                * precision_points[changes + 1]
            ))
        else:
            average_precision = 0.0
        rows.append({"class": label, "average_precision": average_precision,
                     "n_ground_truth": n_truth, "n_predictions": len(ranked)})
    mean_ap = float(np.mean([row["average_precision"] for row in rows])) if rows else math.nan
    return rows, mean_ap


def _relation_key(relation: SpatialRelation) -> tuple[str, str, str] | None:
    predicate = normalize_predicate(relation.predicate)
    if predicate is None:
        return None
    subject, obj = normalize_label(relation.subject), normalize_label(relation.obj)
    if predicate in SYMMETRIC_PREDICATES and subject > obj:
        subject, obj = obj, subject
    return subject, predicate, obj


def relation_metrics(
    predicted: Iterable[SpatialRelation], ground_truth: Iterable[SpatialRelation]
) -> list[dict[str, float | int | str]]:
    pred = Counter(key for relation in predicted if (key := _relation_key(relation)))
    truth = Counter(key for relation in ground_truth if (key := _relation_key(relation)))
    predicates = sorted({key[1] for key in pred} | {key[1] for key in truth})
    rows: list[dict[str, float | int | str]] = []
    for predicate in predicates:
        pred_count = Counter({key: value for key, value in pred.items() if key[1] == predicate})
        truth_count = Counter({key: value for key, value in truth.items() if key[1] == predicate})
        tp = sum((pred_count & truth_count).values())
        fp = sum(pred_count.values()) - tp
        fn = sum(truth_count.values()) - tp
        rows.append({"predicate": predicate, "tp": tp, "fp": fp, "fn": fn,
                     **precision_recall_f1(tp, fp, fn)})
    return rows


def bootstrap_interval(
    values: Sequence[float | bool],
    confidence_level: float = 0.95,
    samples: int = 2000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Percentile bootstrap interval for a mean/proportion."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return math.nan, math.nan, math.nan
    mean = float(array.mean())
    if array.size == 1 or samples < 2:
        return mean, math.nan, math.nan
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=float)
    for index in range(samples):
        estimates[index] = rng.choice(array, size=array.size, replace=True).mean()
    alpha = (1.0 - confidence_level) / 2.0
    low, high = np.quantile(estimates, [alpha, 1.0 - alpha])
    return mean, float(low), float(high)


def calibrate_rejection_threshold(
    positive_scores: Sequence[float], negative_scores: Sequence[float]
) -> tuple[float, dict[str, float]]:
    """Select a threshold maximizing balanced positive acceptance/negative rejection."""
    positives = np.asarray(positive_scores, dtype=float)
    negatives = np.asarray(negative_scores, dtype=float)
    positives = positives[np.isfinite(positives)]
    negatives = negatives[np.isfinite(negatives)]
    if positives.size == 0 or negatives.size == 0:
        raise ValueError("threshold calibration requires positive and negative validation scores")
    values = np.unique(np.concatenate((positives, negatives)))
    candidates = np.concatenate((
        [values[0] - np.finfo(float).eps],
        (values[:-1] + values[1:]) / 2.0,
        [values[-1] + np.finfo(float).eps],
    ))
    rows = []
    for threshold in candidates:
        positive_acceptance = float(np.mean(positives >= threshold))
        negative_rejection = float(np.mean(negatives < threshold))
        balanced_accuracy = (positive_acceptance + negative_rejection) / 2.0
        rows.append((balanced_accuracy, negative_rejection, positive_acceptance, float(threshold)))
    balanced, rejection, acceptance, threshold = max(rows)
    return threshold, {
        "balanced_accuracy": balanced,
        "positive_acceptance": acceptance,
        "negative_rejection": rejection,
        "n_positive": int(positives.size),
        "n_negative": int(negatives.size),
    }
