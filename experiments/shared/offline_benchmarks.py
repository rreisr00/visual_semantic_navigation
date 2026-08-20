"""Reproducible offline benchmarks and publication-ready figures.

The notebooks call the functions in this module so that the experimental
logic can be unit-tested and reused without copying long notebook cells.
Offline scope is deliberately limited to evidence supported by the available
datasets: single-view VLM retrieval, object detection, object-aware retrieval,
and an isolated evaluation of 2-D relation rules.
"""
from __future__ import annotations

import gc
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from PIL import Image

from plotting import PALETTE, apply_plot_style
from semantic_evaluation.core import EmbeddingCache
from semantic_evaluation.core.config_validation import expand_path
from semantic_evaluation.core.dataset_adapters import load_dataset
from semantic_evaluation.core.evaluation_statistics import (
    SYMMETRIC_PREDICATES,
    bootstrap_interval,
    calibrate_rejection_threshold,
    detection_average_precision,
    match_detections,
    normalize_label,
    normalize_predicate,
    precision_recall_f1,
)
from semantic_evaluation.core.experiment_runner import prepare_queries, run_method
from semantic_evaluation.core.offline_dataset import load_queries
from semantic_evaluation.core.offline_encoding import (
    detect_objects,
    embed_texts,
    encode_observations,
)
from semantic_evaluation.core.retrieval_metrics import results_to_rows, summarize
from semantic_navigation_core.relations import infer_relations
from semantic_navigation_core.retrieval import (
    HybridWeights,
    METHOD_RANDOM_BASELINE,
    METHOD_ROOM_LABEL_BASELINE,
    METHOD_SIGLIP_WITH_OBJECTS,
    METHOD_SINGLE_VIEW_SIGLIP,
    RetrievalConfig,
)
from semantic_vision_core import SemanticVisionPipeline


VLM_LABELS = {
    "siglip_v1": "SigLIP",
    "siglip_v2": "SigLIP2",
}
# Bump when preprocessing changes so generated embeddings are never mixed
# with values produced by an earlier, incompatible tokenization protocol.
VLM_CACHE_REVISION = "official_processor_v1"
DETECTOR_LABELS = {
    "yolov8n": "YOLOv8n",
    "yolo26n": "YOLO26n",
}
QUERY_LABELS = {
    "room": "Habitación",
    "object": "Objeto",
    "multi_object": "Varios objetos",
    "functional": "Funcional",
    "negative": "Negativa",
}


def _sync_cuda() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def _release_pipeline(pipeline: SemanticVisionPipeline | None) -> None:
    del pipeline
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _read_rgb(path: str) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def _available_bundle(
    ctx: dict[str, Any], dataset_id: str, *, require_nodes: bool = True
):
    spec = next(item for item in ctx["dataset_specs"] if item.dataset_id == dataset_id)
    bundle = load_dataset(spec, ctx["repo_root"])
    if bundle.skipped or (require_nodes and not bundle.nodes):
        raise RuntimeError(f"{dataset_id} no disponible: {bundle.skip_reason}")
    return spec, bundle


def _prepared_queries(ctx: dict[str, Any], spec, bundle, require_objects: bool = False):
    query_path = Path(ctx["repo_root"]) / spec.queries_file
    queries = load_queries(str(query_path))
    prepared = prepare_queries(queries, bundle)
    return [
        item for item in prepared
        if (item.query.is_negative or item.valid_node_ids)
        and (not require_objects or bool(item.query.expected_objects))
    ]


def _benchmark_vlm(
    pipeline: SemanticVisionPipeline,
    image_paths: Sequence[str],
    texts: Sequence[str],
) -> dict[str, float | int]:
    count = min(len(image_paths), len(texts))
    if count == 0:
        return {}
    image = _read_rgb(image_paths[0])
    pipeline.embed_image(image)
    pipeline.embed_text(texts[0])
    image_times: list[float] = []
    text_times: list[float] = []
    for path, text in zip(image_paths[:count], texts[:count]):
        image = _read_rgb(path)
        _sync_cuda()
        started = time.perf_counter()
        image_embedding = pipeline.embed_image(image)
        _sync_cuda()
        image_times.append((time.perf_counter() - started) * 1000.0)
        started = time.perf_counter()
        pipeline.embed_text(text)
        _sync_cuda()
        text_times.append((time.perf_counter() - started) * 1000.0)
    model = pipeline._model
    parameters = sum(parameter.numel() for parameter in model.parameters())
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )
    return {
        "parameters": int(parameters),
        "parameter_memory_mb": float(parameter_bytes / 1024**2),
        "embedding_dimension": int(image_embedding.shape[-1]),
        "mean_image_encoding_ms": float(np.mean(image_times)),
        "std_image_encoding_ms": float(np.std(image_times)),
        "mean_text_encoding_ms": float(np.mean(text_times)),
        "std_text_encoding_ms": float(np.std(text_times)),
        "benchmark_samples": int(count),
    }


def _summary_with_intervals(cases: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_keys = ["dataset_id", "method", "query_type", "language"]
    for key, group in cases.groupby(group_keys, dropna=False):
        row = dict(zip(group_keys, key))
        positives = group.loc[~group["is_negative"]]
        negatives = group.loc[group["is_negative"]]
        row["n_queries"] = len(group)
        row["n_positive"] = len(positives)
        row["n_negative"] = len(negatives)
        for source, target in (
            ("recall_at_1", "recall_at_1"),
            ("recall_at_3", "recall_at_3"),
            ("recall_at_5", "recall_at_5"),
            ("reciprocal_rank", "mean_reciprocal_rank"),
        ):
            mean, low, high = bootstrap_interval(positives[source].dropna().tolist())
            row[target], row[f"{target}_ci_low"], row[f"{target}_ci_high"] = mean, low, high
        mean, low, high = bootstrap_interval(negatives["rejected"].dropna().tolist())
        row["negative_rejection_rate"] = mean
        row["negative_rejection_rate_ci_low"] = low
        row["negative_rejection_rate_ci_high"] = high
        row["mean_retrieval_latency_ms"] = group["retrieval_latency_ms"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def _paired_vlm_differences(cases: pd.DataFrame) -> pd.DataFrame:
    subset = cases.loc[
        (~cases["is_negative"]) & cases["method"].isin(VLM_LABELS),
        ["dataset_id", "query_id", "query_type", "language", "method",
         "recall_at_1", "reciprocal_rank"],
    ]
    wide = subset.pivot_table(
        index=["dataset_id", "query_id", "query_type", "language"],
        columns="method",
        values=["recall_at_1", "reciprocal_rank"],
        aggfunc="first",
    )
    rows: list[dict[str, Any]] = []
    if not {"siglip_v1", "siglip_v2"}.issubset(wide.columns.get_level_values(1)):
        return pd.DataFrame(rows)
    for dataset_id, group in wide.groupby(level="dataset_id"):
        row: dict[str, Any] = {"dataset_id": dataset_id, "n_paired_queries": len(group)}
        for metric in ("recall_at_1", "reciprocal_rank"):
            delta = group[(metric, "siglip_v2")] - group[(metric, "siglip_v1")]
            mean, low, high = bootstrap_interval(delta.dropna().tolist())
            row[f"delta_{metric}"] = mean
            row[f"delta_{metric}_ci_low"] = low
            row[f"delta_{metric}_ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def _threshold_diagnostics(cases: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset_id, method), group in cases.loc[cases["method"].isin(VLM_LABELS)].groupby(
        ["dataset_id", "method"]
    ):
        positives = group.loc[~group["is_negative"], "hybrid_score"].dropna().tolist()
        negatives = group.loc[group["is_negative"], "hybrid_score"].dropna().tolist()
        if not positives or not negatives:
            continue
        threshold, metrics = calibrate_rejection_threshold(positives, negatives)
        rows.append({
            "dataset_id": dataset_id,
            "method": method,
            "descriptive_threshold": threshold,
            **metrics,
            "warning": "descriptive only; estimated and evaluated on the same queries",
        })
    return pd.DataFrame(rows)


def run_vlm_benchmark(ctx: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Compare SigLIP and SigLIP2 under the same single-view protocol."""
    config = ctx["config"]
    specs_and_bundles = {
        dataset_id: _available_bundle(ctx, dataset_id)
        for dataset_id in ("siglip_rooms", "sunrgbd")
    }
    all_results = []
    threshold = float(config["retrieval"]["rejection"]["initial_threshold"])

    for dataset_id, (spec, bundle) in specs_and_bundles.items():
        prepared = _prepared_queries(ctx, spec, bundle)
        baselines = {
            "random_baseline": RetrievalConfig(
                method=METHOD_RANDOM_BASELINE, seed=int(config["experiment"]["seed"])
            ),
            "room_label_baseline": RetrievalConfig(method=METHOD_ROOM_LABEL_BASELINE),
        }
        for label, method_config in baselines.items():
            all_results.extend(run_method(label, prepared, bundle, method_config, threshold))

    siglip_config = config["models"]["siglip"]
    variants = dict(siglip_config["variants"])
    cache_root = Path(ctx["repo_root"]) / config["paths"]["cache_root"]
    costs: list[dict[str, Any]] = []
    sample_spec, sample_bundle = specs_and_bundles["sunrgbd"]
    sample_prepared = _prepared_queries(ctx, sample_spec, sample_bundle)
    benchmark_count = int(config["evaluation"].get("benchmark_samples", 20))
    sample_images = [
        observation.image_path
        for node in sample_bundle.nodes
        for observation in node.observations
        if observation.image_path
    ][:benchmark_count]
    sample_texts = [item.query.text for item in sample_prepared][:benchmark_count]

    for variant, model_id in variants.items():
        cache_model_id = f"{model_id}::{VLM_CACHE_REVISION}"
        started = time.perf_counter()
        pipeline = SemanticVisionPipeline(
            retrieval_mode="siglip_pure",
            siglip_model_id=model_id,
            device=ctx["device"],
            processor_fast=bool(siglip_config["processor_fast"]),
            local_files_only=bool(siglip_config.get("local_files_only", False)),
        )
        _sync_cuda()
        load_seconds = time.perf_counter() - started
        cost = _benchmark_vlm(pipeline, sample_images, sample_texts)
        costs.append({
            "method": variant,
            "model_id": model_id,
            "model_load_s": load_seconds,
            "device": ctx["device"],
            **cost,
        })
        cache = EmbeddingCache(str(cache_root / f"vlm_{variant}"))
        for dataset_id, (spec, bundle) in specs_and_bundles.items():
            encode_observations(bundle.nodes, pipeline, cache, cache_model_id)
            prepared = _prepared_queries(ctx, spec, bundle)
            embeddings = embed_texts(
                [item.query.text for item in prepared], pipeline, cache, cache_model_id
            )
            for item, embedding in zip(prepared, embeddings):
                item.embedding = embedding
            all_results.extend(run_method(
                variant,
                prepared,
                bundle,
                RetrievalConfig(method=METHOD_SINGLE_VIEW_SIGLIP),
                threshold,
            ))
        _release_pipeline(pipeline)

    cases = pd.DataFrame(results_to_rows(all_results))
    summary = _summary_with_intervals(cases)
    paired = _paired_vlm_differences(cases)
    thresholds = _threshold_diagnostics(cases)
    return {
        "cases": cases,
        "summary": summary,
        "paired_differences": paired,
        "threshold_diagnostics": thresholds,
        "model_costs": pd.DataFrame(costs),
    }


def _relation_key(relation) -> tuple[str, str, str] | None:
    predicate = normalize_predicate(relation.predicate)
    if predicate is None:
        return None
    subject = normalize_label(relation.subject)
    obj = normalize_label(relation.obj)
    if predicate in SYMMETRIC_PREDICATES and subject > obj:
        subject, obj = obj, subject
    return subject, predicate, obj


def _stream_relation_metrics(bundle) -> tuple[pd.DataFrame, pd.DataFrame]:
    predicted: Counter = Counter()
    annotated: Counter = Counter()
    examples: list[dict[str, Any]] = []
    for image_id, truth in bundle.relation_ground_truth.items():
        inferred = infer_relations(bundle.object_ground_truth.get(image_id, []))
        predicted.update(key for item in inferred if (key := _relation_key(item)))
        annotated.update(key for item in truth if (key := _relation_key(item)))
        if len(examples) < 20:
            examples.append({
                "image_id": image_id,
                "predicted": len(inferred),
                "annotated": len(truth),
                "mean_inferred_confidence": (
                    float(np.mean([item.confidence for item in inferred]))
                    if inferred else math.nan
                ),
            })
    rows = []
    predicates = sorted({key[1] for key in predicted} | {key[1] for key in annotated})
    for predicate in predicates:
        pred = Counter({key: value for key, value in predicted.items() if key[1] == predicate})
        truth = Counter({key: value for key, value in annotated.items() if key[1] == predicate})
        tp = sum((pred & truth).values())
        fp = sum(pred.values()) - tp
        fn = sum(truth.values()) - tp
        rows.append({"predicate": predicate, "tp": tp, "fp": fp, "fn": fn,
                     **precision_recall_f1(tp, fp, fn)})
    return pd.DataFrame(rows), pd.DataFrame(examples)


def _detector_latency(pipeline: SemanticVisionPipeline, paths: Sequence[str]) -> dict[str, float]:
    if not paths:
        return {}
    pipeline.detect(_read_rgb(paths[0]))
    times = []
    for path in paths:
        image = _read_rgb(path)
        _sync_cuda()
        started = time.perf_counter()
        pipeline.detect(image)
        _sync_cuda()
        times.append((time.perf_counter() - started) * 1000.0)
    return {
        "mean_detection_ms": float(np.mean(times)),
        "std_detection_ms": float(np.std(times)),
        "benchmark_samples": len(times),
    }


def _hybrid_weights(config: dict[str, Any]) -> HybridWeights:
    values = config["retrieval"]["weights"]["siglip_with_objects"]
    return HybridWeights(
        alpha=float(values["global_similarity"]),
        beta=float(values["object_match"]),
        gamma=float(values["crop_similarity"]),
        delta=float(values["relation_match"]),
        epsilon=float(values["room_match"]),
    )


def run_detector_benchmark(ctx: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Compare YOLOv8n/YOLO26n and their effect on object-aware retrieval."""
    config = ctx["config"]
    sun_spec, _ = _available_bundle(ctx, "sunrgbd")
    _, visual_genome = _available_bundle(ctx, "visual_genome", require_nodes=False)
    siglip_config = config["models"]["siglip"]
    selected_variant = str(siglip_config["object_retrieval_variant"])
    model_id = str(siglip_config["variants"][selected_variant])
    cache_model_id = f"{model_id}::{VLM_CACHE_REVISION}"
    yolo_config = config["models"]["yolo"]
    operational_confidence = float(yolo_config["confidence_threshold"])
    ap_confidence = float(yolo_config.get("ap_confidence_threshold", 0.001))
    cache_root = Path(ctx["repo_root"]) / config["paths"]["cache_root"]
    threshold = float(config["retrieval"]["rejection"]["initial_threshold"])
    benchmark_count = int(config["evaluation"].get("benchmark_samples", 20))

    detector_rows: list[dict[str, Any]] = []
    ap_rows: list[dict[str, Any]] = []
    per_image_rows: list[dict[str, Any]] = []
    object_results = []

    for detector_index, (detector, configured_path) in enumerate(
        dict(yolo_config["variants"]).items()
    ):
        checkpoint, missing = expand_path(str(configured_path), ctx["repo_root"])
        if missing or checkpoint is None or not checkpoint.is_file():
            raise RuntimeError(f"checkpoint {detector} no disponible: {configured_path}")
        sunrgbd = load_dataset(sun_spec, ctx["repo_root"])
        cache = EmbeddingCache(str(cache_root / f"detector_{detector}"))
        started = time.perf_counter()
        pipeline = SemanticVisionPipeline(
            retrieval_mode="siglip_yolo",
            siglip_model_id=model_id,
            yolo_model_path=str(checkpoint),
            yolo_confidence_threshold=ap_confidence,
            device=ctx["device"],
            processor_fast=bool(siglip_config["processor_fast"]),
            local_files_only=bool(siglip_config.get("local_files_only", False)),
        )
        _sync_cuda()
        load_seconds = time.perf_counter() - started

        detect_objects(
            sunrgbd.nodes, pipeline, cache, str(checkpoint), ap_confidence,
            with_crop_embeddings=False,
        )
        predictions = {
            observation.observation_id: list(observation.objects)
            for node in sunrgbd.nodes for observation in node.observations
        }
        mapping = sunrgbd.metadata["class_mapping"]
        detector_ap, mean_ap = detection_average_precision(
            predictions, sunrgbd.object_ground_truth, mapping
        )
        ap_rows.extend({"detector": detector, **row} for row in detector_ap)

        operational_predictions = {
            observation_id: [
                item for item in items if item.confidence >= operational_confidence
            ]
            for observation_id, items in predictions.items()
        }
        for observation_id, truth in sunrgbd.object_ground_truth.items():
            tp, fp, fn, unmapped = match_detections(
                operational_predictions.get(observation_id, []), truth, mapping
            )
            per_image_rows.append({
                "detector": detector,
                "observation_id": observation_id,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "unmapped_classes": unmapped,
                **precision_recall_f1(tp, fp, fn),
            })
        detector_per_image = [row for row in per_image_rows if row["detector"] == detector]
        totals = pd.DataFrame(detector_per_image)[["tp", "fp", "fn"]].sum()
        sample_paths = [
            observation.image_path for node in sunrgbd.nodes
            for observation in node.observations if observation.image_path
        ][:benchmark_count]
        detector_rows.append({
            "detector": detector,
            "checkpoint": str(checkpoint),
            "checkpoint_size_mb": checkpoint.stat().st_size / 1024**2,
            "model_load_s": load_seconds,
            "ap_min_confidence": ap_confidence,
            "operational_confidence": operational_confidence,
            "mean_average_precision": mean_ap,
            "tp": int(totals["tp"]),
            "fp": int(totals["fp"]),
            "fn": int(totals["fn"]),
            **precision_recall_f1(int(totals["tp"]), int(totals["fp"]), int(totals["fn"])),
            **_detector_latency(pipeline, sample_paths),
        })

        # Re-run at the operational threshold and embed only retained crops.
        pipeline._yolo_conf = operational_confidence
        detect_objects(
            sunrgbd.nodes, pipeline, cache, str(checkpoint), operational_confidence,
            with_crop_embeddings=True, siglip_model_id=cache_model_id,
        )
        encode_observations(sunrgbd.nodes, pipeline, cache, cache_model_id)
        prepared = _prepared_queries(ctx, sun_spec, sunrgbd, require_objects=True)
        embeddings = embed_texts(
            [item.query.text for item in prepared], pipeline, cache, cache_model_id
        )
        for item, embedding in zip(prepared, embeddings):
            item.embedding = embedding
        if detector_index == 0:
            object_results.extend(run_method(
                f"{selected_variant}_single_view",
                prepared,
                sunrgbd,
                RetrievalConfig(method=METHOD_SINGLE_VIEW_SIGLIP),
                threshold,
            ))
        object_results.extend(run_method(
            f"{selected_variant}_{detector}_objects",
            prepared,
            sunrgbd,
            RetrievalConfig(
                method=METHOD_SIGLIP_WITH_OBJECTS,
                weights=_hybrid_weights(config),
            ),
            threshold,
        ))
        _release_pipeline(pipeline)

    relation_summary, relation_examples = _stream_relation_metrics(visual_genome)
    object_cases = pd.DataFrame(results_to_rows(object_results))
    object_summary = pd.DataFrame(summarize(
        object_results, ("dataset_id", "method", "query_type", "language")
    ))
    return {
        "detector_summary": pd.DataFrame(detector_rows),
        "object_average_precision": pd.DataFrame(ap_rows),
        "object_detection_cases": pd.DataFrame(per_image_rows),
        "object_retrieval_cases": object_cases,
        "object_retrieval_summary": object_summary,
        "relation_metrics": relation_summary,
        "relation_examples": relation_examples,
    }


def _save_figure(figure, directory: Path, name: str) -> list[str]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("png", "pdf"):
        target = directory / f"{name}.{suffix}"
        figure.savefig(target, dpi=200, bbox_inches="tight")
        paths.append(str(target))
    return paths


def create_vlm_figures(
    cases: pd.DataFrame, model_costs: pd.DataFrame, output: str | Path
) -> list[str]:
    import matplotlib.pyplot as plt

    apply_plot_style()
    output = Path(output)
    paths: list[str] = []
    positive = cases.loc[
        (cases["dataset_id"] == "sunrgbd")
        & (~cases["is_negative"])
        & cases["method"].isin(VLM_LABELS)
    ].copy()

    types = [item for item in QUERY_LABELS if item != "negative" and item in positive["query_type"].unique()]
    x = np.arange(len(types))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    for index, method in enumerate(VLM_LABELS):
        means, lows, highs = [], [], []
        for query_type in types:
            values = positive.loc[
                (positive["method"] == method) & (positive["query_type"] == query_type),
                "recall_at_1",
            ].dropna().tolist()
            mean, low, high = bootstrap_interval(values)
            means.append(mean); lows.append(mean - low); highs.append(high - mean)
        ax.bar(x + (index - 0.5) * width, means, width, label=VLM_LABELS[method],
               yerr=np.vstack([lows, highs]), capsize=3)
    ax.set_xticks(x, [QUERY_LABELS[item] for item in types])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Recall@1")
    ax.set_title("Recuperación sobre SUN RGB-D por tipo de consulta")
    ax.legend()
    paths.extend(_save_figure(fig, output, "vlm_recall_por_tipo")); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for method in VLM_LABELS:
        group = positive.loc[positive["method"] == method]
        values = [group[f"recall_at_{k}"].mean() for k in (1, 3, 5)]
        ax.plot((1, 3, 5), values, marker="o", linewidth=2, label=VLM_LABELS[method])
    ax.set_xticks((1, 3, 5))
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("K")
    ax.set_ylabel("Recall@K")
    ax.set_title("Evolución de la recuperación con el número de candidatos")
    ax.legend()
    paths.extend(_save_figure(fig, output, "vlm_recall_at_k")); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    language = positive.groupby(["method", "language"])["recall_at_1"].mean().unstack()
    positions = np.arange(len(language))
    ax.plot(positions, language.get("es"), marker="o", label="Español")
    ax.plot(positions, language.get("en"), marker="o", label="Inglés")
    ax.set_xticks(positions, [VLM_LABELS[item] for item in language.index])
    ax.set_ylim(0, 1.03)
    ax.set_ylabel("Recall@1")
    ax.set_title("Comportamiento por idioma sobre SUN RGB-D")
    ax.legend()
    paths.extend(_save_figure(fig, output, "vlm_comparacion_idiomas")); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    labels = [VLM_LABELS[item] for item in model_costs["method"]]
    axes[0].bar(labels, model_costs["mean_image_encoding_ms"], color=PALETTE[:len(labels)])
    axes[0].set_ylabel("Tiempo medio (ms)")
    axes[0].set_title("Codificación de imagen")
    axes[1].bar(labels, model_costs["mean_text_encoding_ms"], color=PALETTE[:len(labels)])
    axes[1].set_ylabel("Tiempo medio (ms)")
    axes[1].set_title("Codificación de texto")
    fig.suptitle("Coste de inferencia de los encoders")
    paths.extend(_save_figure(fig, output, "vlm_coste_inferencia")); plt.close(fig)
    return paths


def create_detector_figures(results: dict[str, pd.DataFrame], output: str | Path) -> list[str]:
    import matplotlib.pyplot as plt

    apply_plot_style()
    output = Path(output)
    paths: list[str] = []
    ap = results["object_average_precision"]
    classes = sorted(ap["class"].unique())
    y = np.arange(len(classes))
    height = 0.36
    fig, ax = plt.subplots(figsize=(8.4, max(4.8, 0.48 * len(classes))))
    for index, detector in enumerate(DETECTOR_LABELS):
        group = ap.loc[ap["detector"] == detector].set_index("class")
        values = [group.loc[item, "average_precision"] if item in group.index else 0 for item in classes]
        ax.barh(y + (index - 0.5) * height, values, height, label=DETECTOR_LABELS[detector])
    ax.set_yticks(y, classes)
    ax.set_xlim(0, max(0.5, float(ap["average_precision"].max()) * 1.1))
    ax.set_xlabel("AP@0,5 IoU")
    ax.set_title("Precisión media por clase sobre SUN RGB-D")
    ax.legend()
    paths.extend(_save_figure(fig, output, "detectores_ap_por_clase")); plt.close(fig)

    summary = results["detector_summary"].set_index("detector")
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    metrics = ["precision", "recall", "f1", "mean_average_precision"]
    x = np.arange(len(metrics))
    width = 0.36
    for index, detector in enumerate(DETECTOR_LABELS):
        ax.bar(x + (index - 0.5) * width,
               [summary.loc[detector, metric] for metric in metrics], width,
               label=DETECTOR_LABELS[detector])
    ax.set_xticks(x, ["Precisión", "Exhaustividad", "F1", "mAP@0,5"])
    ax.set_ylim(0, 1.0)
    ax.set_title("Comparación global de detectores")
    ax.legend()
    paths.extend(_save_figure(fig, output, "detectores_metricas_globales")); plt.close(fig)

    object_cases = results["object_retrieval_cases"]
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    methods = list(object_cases["method"].unique())
    values = [object_cases.loc[object_cases["method"] == method, "recall_at_1"].mean()
              for method in methods]
    display_labels = [
        "SigLIP2" if method.endswith("single_view")
        else ("SigLIP2 + YOLOv8n" if "yolov8n" in method else "SigLIP2 + YOLO26n")
        for method in methods
    ]
    ax.bar(display_labels, values, color=PALETTE[:len(methods)])
    ax.set_ylim(0, 1.03)
    ax.set_ylabel("Recall@1")
    ax.set_title("Efecto de la evidencia de objetos en consultas compatibles")
    paths.extend(_save_figure(fig, output, "objetos_efecto_en_recuperacion")); plt.close(fig)

    relations = results["relation_metrics"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))
    axes[0].barh(relations["predicate"], relations["recall"])
    axes[0].set_xlim(0, 1.03)
    axes[0].set_xlabel("Exhaustividad")
    axes[0].set_title("Cobertura de relaciones anotadas")
    axes[1].barh(relations["predicate"], relations["precision"])
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Precisión (escala logarítmica)")
    axes[1].set_title("Concordancia con anotaciones no exhaustivas")
    paths.extend(_save_figure(fig, output, "relaciones_precision_exhaustividad")); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for detector in DETECTOR_LABELS:
        row = summary.loc[detector]
        ax.scatter(row["mean_detection_ms"], row["mean_average_precision"], s=90,
                   label=DETECTOR_LABELS[detector])
        ax.annotate(DETECTOR_LABELS[detector],
                    (row["mean_detection_ms"], row["mean_average_precision"]),
                    xytext=(6, 5), textcoords="offset points")
    ax.set_xlabel("Latencia media por imagen (ms)")
    ax.set_ylabel("mAP@0,5")
    ax.set_title("Compromiso entre calidad y coste del detector")
    paths.extend(_save_figure(fig, output, "detectores_calidad_coste")); plt.close(fig)
    return paths


def write_results_summary(
    output: str | Path,
    vlm: dict[str, pd.DataFrame],
    detectors: dict[str, pd.DataFrame],
) -> Path:
    """Write a compact, machine-regenerated Markdown record of key values."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    vlm_cases = vlm["cases"]
    sun = vlm_cases.loc[
        (vlm_cases["dataset_id"] == "sunrgbd")
        & (~vlm_cases["is_negative"])
        & vlm_cases["method"].isin(VLM_LABELS)
    ]
    lines = [
        "# Resultados de la experimentación offline",
        "",
        "> Archivo generado por los notebooks. No editar valores manualmente.",
        "",
        "## Recuperación con modelos visión-lenguaje",
        "",
        "| Modelo | Consultas positivas | R@1 | R@3 | R@5 | MRR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method, label in VLM_LABELS.items():
        group = sun.loc[sun["method"] == method]
        lines.append(
            f"| {label} | {len(group)} | {group['recall_at_1'].mean():.3f} | "
            f"{group['recall_at_3'].mean():.3f} | {group['recall_at_5'].mean():.3f} | "
            f"{group['reciprocal_rank'].mean():.3f} |"
        )
    lines.extend(["", "## Detección de objetos", "",
                  "| Detector | mAP@0,5 | Precisión | Exhaustividad | F1 | Latencia (ms) |",
                  "|---|---:|---:|---:|---:|---:|"])
    for _, row in detectors["detector_summary"].iterrows():
        lines.append(
            f"| {DETECTOR_LABELS.get(row['detector'], row['detector'])} | "
            f"{row['mean_average_precision']:.3f} | {row['precision']:.3f} | "
            f"{row['recall']:.3f} | {row['f1']:.3f} | {row['mean_detection_ms']:.1f} |"
        )
    lines.extend(["", "## Recuperación informada por objetos", "",
                  "| Método | Consultas | R@1 | R@3 | R@5 | MRR |",
                  "|---|---:|---:|---:|---:|---:|"])
    object_cases = detectors["object_retrieval_cases"]
    for method, group in object_cases.groupby("method", sort=False):
        lines.append(
            f"| {method} | {len(group)} | {group['recall_at_1'].mean():.3f} | "
            f"{group['recall_at_3'].mean():.3f} | {group['recall_at_5'].mean():.3f} | "
            f"{group['reciprocal_rank'].mean():.3f} |"
        )
    relation = detectors["relation_metrics"]
    lines.extend([
        "",
        "## Relaciones espaciales",
        "",
        f"Se evaluaron {int((relation['tp'] + relation['fn']).sum()):,} relaciones anotadas. "
        "La precisión debe interpretarse con cautela porque Visual Genome no anota exhaustivamente "
        "todas las relaciones geométricamente verdaderas.",
        "",
        "## Alcance",
        "",
        "La evaluación offline no estima la aportación multivista, la contaminación entre "
        "habitaciones, la política espacial ni el éxito de navegación. Esos factores se reservan "
        "a las campañas de simulación.",
        "",
    ])
    output.write_text("\n".join(lines), encoding="utf-8")
    return output
