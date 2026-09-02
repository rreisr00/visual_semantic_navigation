#!/usr/bin/env python3
"""Recalcula la contaminación semántica sobre un grafo ya capturado.

La clasificación que hace el sistema en línea necesita conocer las habitaciones
en el momento de la captura. Si el grafo se capturó antes de definirlas, las
observaciones quedan como ``unknown`` y las métricas salen vacías.

No hace falta recapturar: el grafo conserva la pose de cada waypoint, la
posición 3D en el frame del mapa de cada detección y, una vez definidas, los
rectángulos de las habitaciones. Con eso se reconstruye offline:

    camera_room       habitación que contiene el waypoint desde el que se observó
    observation_room  habitación que contiene la posición 3D del objeto
    purity            fracción de detecciones de la observación cuya habitación
                      coincide con la de la cámara

y de ahí las dos tasas de la memoria:

    CRDR  cross-room detection rate: detecciones cuya habitación no es la de la
          cámara, sobre las detecciones localizables
    COR   contaminated observation rate: observaciones con pureza por debajo del
          umbral, sobre las observaciones clasificables

Uso:

    python3 tools/recompute_contamination.py --scene aws_office
    python3 tools/recompute_contamination.py --scene aws_small_house --json salida.json
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sqlite3
import sys

from semantic_navigation_core.graph_store import load_rooms
from semantic_navigation_core.rooms import room_of_point

UMBRAL_PUREZA = 0.8


def _propiedades(valor):
    if isinstance(valor, str):
        try:
            return json.loads(valor)
        except json.JSONDecodeError:
            return {}
    return valor or {}


def recomputar(db: str, umbral: float = UMBRAL_PUREZA) -> dict:
    rooms = load_rooms(db)
    if not rooms:
        raise SystemExit(f"{db}: no hay habitaciones definidas; defínelas antes")
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)

    # habitación de cada waypoint, por la arista CONTAINS room -> waypoint
    sala_de_waypoint: dict[str, str] = {}
    ids_sala = {r.room_id for r in rooms}
    for origen, destino in con.execute(
        "select source_node, target_node from edges where type='CONTAINS'"
    ):
        if origen in ids_sala:
            sala_de_waypoint[destino] = origen

    # detecciones: posición 3D en el frame del mapa -> habitación
    objetos: dict[str, dict] = {}
    for nombre, props in con.execute(
        "select name, properties from nodes where type='object'"
    ):
        d = _propiedades(props)
        pos = d.get("position_3d_map")
        if isinstance(pos, str):
            pos = _propiedades(pos)
        if not pos or len(pos) < 2:
            continue
        objetos[nombre] = {
            "label": d.get("label"),
            "sala": room_of_point(float(pos[0]), float(pos[1]), rooms),
            "waypoint": d.get("source_waypoint") or nombre.split("_")[0],
        }

    # agrupar detecciones por waypoint a través de las aristas CONTAINS
    por_waypoint: dict[str, list[str]] = collections.defaultdict(list)
    for origen, destino in con.execute(
        "select source_node, target_node from edges where type='CONTAINS'"
    ):
        if destino in objetos and origen in sala_de_waypoint:
            por_waypoint[origen].append(destino)
    for nombre, info in objetos.items():
        wp = info["waypoint"]
        if wp in sala_de_waypoint and nombre not in por_waypoint[wp]:
            por_waypoint[wp].append(nombre)

    localizables = cruzadas = 0
    purezas: list[float] = []
    por_sala: dict[str, dict[str, int]] = collections.defaultdict(
        lambda: {"detecciones": 0, "cruzadas": 0}
    )
    for wp, nombres in por_waypoint.items():
        sala_camara = sala_de_waypoint.get(wp)
        if sala_camara is None:
            continue
        con_sala = [n for n in nombres if objetos[n]["sala"] is not None]
        if not con_sala:
            continue
        coincidentes = sum(1 for n in con_sala if objetos[n]["sala"] == sala_camara)
        localizables += len(con_sala)
        cruzadas += len(con_sala) - coincidentes
        purezas.append(coincidentes / len(con_sala))
        por_sala[sala_camara]["detecciones"] += len(con_sala)
        por_sala[sala_camara]["cruzadas"] += len(con_sala) - coincidentes

    contaminadas = sum(1 for p in purezas if p < umbral)
    return {
        "detecciones_localizadas": localizables,
        "detecciones_cruzadas": cruzadas,
        "CRDR": (cruzadas / localizables) if localizables else None,
        "waypoints_clasificados": len(purezas),
        "waypoints_contaminados": contaminadas,
        "COR": (contaminadas / len(purezas)) if purezas else None,
        "umbral_pureza": umbral,
        "purezas": purezas,
        "por_sala": {k: dict(v) for k, v in por_sala.items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--umbral", type=float, default=UMBRAL_PUREZA)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    db = os.path.expanduser(f"~/.ros/semantic_maps/{a.scene}/graph.db")
    r = recomputar(db, a.umbral)
    print(f"escena: {a.scene}   (umbral de pureza {r['umbral_pureza']})")
    print(f"  detecciones localizadas : {r['detecciones_localizadas']}")
    print(f"  detecciones cruzadas    : {r['detecciones_cruzadas']}")
    print(f"  CRDR                    : {r['CRDR']:.3f}" if r["CRDR"] is not None else "  CRDR: n/d")
    print(f"  waypoints clasificados  : {r['waypoints_clasificados']}")
    print(f"  waypoints contaminados  : {r['waypoints_contaminados']}")
    print(f"  COR                     : {r['COR']:.3f}" if r["COR"] is not None else "  COR: n/d")
    print("  por sala:")
    for sala, v in sorted(r["por_sala"].items()):
        tasa = v["cruzadas"] / v["detecciones"] if v["detecciones"] else 0.0
        print(f"    {sala:16s} {v['cruzadas']:3d}/{v['detecciones']:3d} cruzadas ({tasa:.2f})")
    if a.json:
        json.dump(r, open(a.json, "w"), indent=1)
        print(f"\nguardado en {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
