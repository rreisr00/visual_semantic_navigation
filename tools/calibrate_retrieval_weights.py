#!/usr/bin/env python3
"""Calibra pesos y umbral de rechazo sobre el conjunto de VALIDACION.

Metodologia (manual, seccion 8.3): la calibracion nunca toca la suite de test.
Este script:
  1. cachea los componentes de score por (consulta de validacion, nodo candidato)
     con la misma tuberia que el orquestador (SigLIP + strict_filter + multivista);
  2. barre el simplex de pesos activos (global, objetos, crops, relaciones)
     y elige por orden lexicografico: recall@1 -> F1 de aceptacion -> margen;
  3. elige el umbral en el punto medio del mejor corte positivas/negativas;
  4. escribe frozen_retrieval_config_v2.yaml (mismo esquema que v1) e imprime
     su hash canonico.

Uso:
  PYTHONPATH=semantic_navigation_ws/src/semantic_navigation_core:\
semantic_navigation_ws/src/semantic_vision_core \
  python3 tools/calibrate_retrieval_weights.py [--step 0.05] [--scene aws_small_house]
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
import time

import numpy as np
import yaml

from semantic_navigation_core import retrieval as R
from semantic_navigation_core.configuration import load_frozen_config
from semantic_navigation_core.graph_store import load_rooms, load_semantic_nodes
from semantic_navigation_core.multiview import MultiviewConfig, score_node_views
from semantic_navigation_core.query_semantics import extract_query_semantics
from semantic_navigation_core.retrieval import SemanticQuery
from semantic_vision_core.vision_pipeline import SemanticVisionPipeline

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENES = os.path.join(
    REPO, "semantic_navigation_ws/src/semantic_bringup/config/scenes"
)
FROZEN_V1 = os.path.join(
    REPO, "semantic_navigation_ws/src/semantic_navigation_ros/config/"
    "frozen_retrieval_config.yaml"
)
FROZEN_V2 = os.path.join(
    REPO, "semantic_navigation_ws/src/semantic_navigation_ros/config/"
    "frozen_retrieval_config_v2.yaml"
)


def cache_components(scene: str, base: dict) -> list[tuple[dict, list]]:
    """Una pasada cara (SigLIP) -> lista de (caso, [(node_id, componentes)])."""
    db = os.path.expanduser(f"~/.ros/semantic_maps/{scene}/graph.db")
    images = os.path.expanduser(f"~/.ros/semantic_maps/{scene}/images")
    nodes = load_semantic_nodes(db, images_dir=images, read_only=True)
    rooms = [room.room_id for room in load_rooms(db)]
    vocab = sorted({
        det.label
        for node in nodes for obs in node.observations for det in obs.objects
    })
    mv_cfg = base["multiview_aggregation"]
    multiview = MultiviewConfig(
        method=mv_cfg["method"], top_k=int(mv_cfg["top_k"]),
        max_weight=float(mv_cfg["max_weight"]),
        topk_weight=float(mv_cfg["topk_weight"]),
    )
    pipeline = SemanticVisionPipeline(
        retrieval_mode="siglip_pure",
        siglip_model_id=base["siglip_checkpoint"],
        local_files_only=bool(base["preprocessing"].get("local_files_only", True)),
    )
    suite = yaml.safe_load(
        open(os.path.join(SCENES, f"{scene}_validation_queries.yaml"),
             encoding="utf-8")
    )
    cached = []
    for case in suite["cases"]:
        semantics = extract_query_semantics(case["query_text"], vocab, rooms)
        query = SemanticQuery(
            text=case["query_text"],
            embedding=pipeline.embed_text(case["query_text"]),
            objects=semantics.objects,
            relations=semantics.relations,
            room=semantics.room,
        )
        candidates = nodes
        if base["room_policy"] == "strict_filter" and query.room:
            candidates = [
                node for node in nodes
                if (node.room_id or "").lower() == query.room.lower()
            ]
        rows = [
            (node.node_id, {
                "g": score_node_views(query.embedding, node, multiview),
                "o": R.object_score(query.objects, node),
                "c": R.crop_score(query.embedding, node),
                "r": R.relation_score(query.relations, node),
            })
            for node in candidates
        ]
        cached.append((case, rows))
    return cached


def evaluate(cached, wg, wo, wc, wr):
    """recall@1, F1 de aceptacion con el mejor umbral, margen y ese umbral."""
    hits, top_pos, top_neg = 0, [], []
    for case, rows in cached:
        if not rows:
            continue
        score, node_id = max(
            (wg * d["g"] + wo * d["o"] + wc * d["c"] + wr * d["r"], nid)
            for nid, d in rows
        )
        if case["is_negative"]:
            top_neg.append(score)
        else:
            top_pos.append(score)
            valid = set(case["exact_valid_nodes"]) | set(case["nearby_valid_nodes"])
            hits += node_id in valid
    recall = hits / len(top_pos)
    # mejor corte: probar el punto medio entre cada par de scores consecutivos
    merged = sorted(
        [(s, 1) for s in top_pos] + [(s, 0) for s in top_neg]
    )
    best_f1, best_thr = -1.0, 0.0
    thresholds = [merged[0][0] - 1e-4] + [
        (merged[i][0] + merged[i + 1][0]) / 2 for i in range(len(merged) - 1)
    ]
    for thr in thresholds:
        tp = sum(1 for s in top_pos if s >= thr)
        fp = sum(1 for s in top_neg if s >= thr)
        fn = len(top_pos) - tp
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    margin = min(top_pos) - max(top_neg)
    return recall, best_f1, margin, best_thr


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="aws_small_house")
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument(
        "--out",
        default=None,
        help="Fichero de salida; por defecto frozen_retrieval_config_v2.yaml. "
             "Usar uno propio al calibrar una escena distinta para no pisar "
             "la configuracion ya congelada de otra.",
    )
    args = parser.parse_args()
    out_path = args.out or FROZEN_V2

    base, hash_v1 = load_frozen_config(FROZEN_V1)
    print(f"config base v1: {hash_v1[:16]}…  (umbral {base['rejection_threshold']})")
    started = time.time()
    cached = cache_components(args.scene, base)
    n_pos = sum(1 for case, _ in cached if not case["is_negative"])
    n_neg = len(cached) - n_pos
    print(f"componentes cacheados: {n_pos} positivas + {n_neg} negativas "
          f"en {time.time() - started:.1f}s")

    grid = np.round(np.arange(0.0, 1.0 + 1e-9, args.step), 4)
    best_key, best = None, None
    evaluations = 0
    for wg, wo, wc in itertools.product(grid, grid, grid):
        wr = round(1.0 - wg - wo - wc, 4)
        if wr < 0.0 or wr > 1.0:
            continue
        recall, f1, margin, thr = evaluate(cached, wg, wo, wc, wr)
        evaluations += 1
        key = (recall, f1, margin)
        if best_key is None or key > best_key:
            best_key, best = key, (wg, wo, wc, wr, thr)
    wg, wo, wc, wr, thr = best
    recall, f1, margin = best_key
    print(f"\nbarrido: {evaluations} configuraciones")
    print(f"pesos v1 (renormalizados sin room): "
          f"recall@1={evaluate(cached, 0.50/0.85, 0.15/0.85, 0.10/0.85, 0.10/0.85)[0]:.0%}")
    print(f"pesos v2: global={wg} objetos={wo} crops={wc} relaciones={wr}")
    print(f"  validacion: recall@1={recall:.0%}  F1 aceptacion={f1:.2f}  "
          f"margen={margin:+.4f}  umbral={thr:.4f}")

    config_v2 = dict(base)
    config_v2["retrieval_weights"] = {
        "global_similarity": float(wg),
        "object_match": float(wo),
        "crop_similarity": float(wc),
        "relation_match": float(wr),
        # strict_filter: la sala filtra candidatos, no suma al score.
        "room_match": 0.0,
    }
    config_v2["rejection_threshold"] = round(float(thr), 4)
    config_v2["calibration"] = {
        "suite": f"{args.scene}_validation_queries_v1",
        "method": "simplex_grid_lexicographic_recall_f1_margin",
        "grid_step": args.step,
        "base_config_hash": hash_v1,
    }
    with open(out_path, "w", encoding="utf-8") as stream:
        yaml.safe_dump(config_v2, stream, allow_unicode=True, sort_keys=False)
    _, hash_v2 = load_frozen_config(out_path)
    print(f"\nescrito {os.path.relpath(out_path, REPO)}")
    print(f"hash v2: {hash_v2}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
