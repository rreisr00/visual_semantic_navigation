# `semantic_evaluation` (UF-7) — Integration Report

Arquitectura: núcleo puro `semantic_evaluation.core` (sin `rclpy`) + 3 nodos
wrapper delgados. Parametrización total vía `config/evaluation_params.yaml`.
Sin llamadas bloqueantes dentro de callbacks.

---

## (a) Puntos de integración reales detectados / modificados

Todos los nombres se confirmaron leyendo el código (PASO 0), no se asumieron.

| Concepto | Hallazgo real | Acción tomada |
|---|---|---|
| **Acción consulta→navegación** | `semantic_interfaces/action/NavigateToSemanticGoal` (servida por `semantic_orchestrator_node` como `navigate_to_semantic_goal`). Result tenía `success, matched_node_id, score, message` pero **no** los splits de latencia. | Añadidos al **Result** `float64 visual_extraction_s/retrieval_s/navigation_s`, y al **Goal** `bool decision_only`. El orquestador ahora mide cada fase (embedding / retrieval / nav) y los devuelve; en `decision_only` no llama a Nav2 y `navigation_s = NaN`. |
| **Snapshot del grafo** | **No existía.** Solo `GetWaypoints.srv` (waypoints, sin aristas ni conteos), servido por `knowledge_graph_bridge_node` (lifecycle). La lib pura `knowledge_graph.graph` ofrece `get_nodes()/get_edges()`. | Creado **`GetGraphSnapshot.srv`** + **`GraphEdge.msg`**. Servido desde el bridge (`get_graph_snapshot`) en `on_activate`/`on_deactivate`, usando la lib pura (sin tocarla): devuelve `total_nodes`, `total_edges`, `waypoints[]` y `edges[]`. |
| **Tópico de cámara** | `/camera/image_raw` (único sub del repo, en `kg_manager`, QoS por defecto). | Teleop se suscribe con `qos_profile_sensor_data` (BEST_EFFORT/KEEP_LAST). Tópico parametrizado (`camera_topic`, default `/camera/image_raw`). |
| **Mecanismo de captura** | Acción **`CaptureWaypoint`** (`capture_waypoint`, en `kg_manager`). Goal: `string label`. **`label` se ignoraba**: el id era `waypoint_<time_ns>`. | Teleop dispara la acción (cliente, desacoplado). `kg_manager` **modificado**: si `label != ""`, `node_id = label` (p.ej. `cocina_01`); si no, fallback a `waypoint_<ns>`. |
| **Convención de nombres** | Ids reales eran timestamp; la precisión de sala asumía `cocina_01`→`cocina`. | Con `label`-como-id, `room_key(strip_last)` deriva la sala: `cocina_01`→`cocina`, `sala_estar_02`→`sala_estar` (multi-token correcto). Parametrizable (`room_separator`, `room_strategy`). |

**Archivos modificados fuera de `semantic_evaluation`:**
- `semantic_interfaces/`: `action/NavigateToSemanticGoal.action`, `msg/GraphEdge.msg` (nuevo), `srv/GetGraphSnapshot.srv` (nuevo), `CMakeLists.txt`.
- `semantic_navigation_ros/.../semantic_orchestrator_node.py`: medición de 3 fases + `decision_only`.
- `semantic_navigation_ros/.../kg_manager_node.py`: `label` como `node_id`.
- `knowledge_graph_ros/.../knowledge_graph_bridge_node.py`: servidor `get_graph_snapshot`.

---

## (b) Suposiciones

1. **`decision_only` por-goal** (no parámetro global) para mezclar casos navegables
   y de solo-decisión en una misma campaña; `navigation_s = NaN` en ese modo.
2. **`GetGraphSnapshot` con Request vacío** (snapshot total). `total_nodes/edges`
   cuentan **todos** los tipos (waypoints + objetos + …); `waypoints[]` solo nodos
   tipo `waypoint` (los que tienen pose). Las aristas sin pose en un extremo
   (waypoint→object) se omiten del `LINE_LIST` pero **sí** cuentan en `total_edges`.
3. **Carga de imágenes** (`image_path`) con `cv2.imread` → `bgr8` (el encoder
   convierte a `rgb8`). `cv_bridge`/OpenCV con import guardado.
4. **`room_strategy=strip_last`** por defecto (preserva salas multi-token).
   Alternativa `first_token` disponible.
5. **`output_dir` por defecto `~/ros2_evaluation_results`**; el CSV se nombra
   `<prefix>_<YYYYmmdd_HHMMSS>.csv`.
6. Esquema CSV fijo (`csv_export.CSV_COLUMNS`): NaN → celda vacía; en
   `__AGGREGATE_MEAN__`, precisión = tasas 0..1 y resto = medias NaN-aware.

---

## (c) Preguntas abiertas que podrían bloquear el cableado final

1. **Coexistencia con el `evaluation_node` pasivo** ya presente en
   `semantic_navigation_ros` (suscriptor de `/retrieval_latency`,
   `/retrieval_result`, `/ground_truth_node`). El nuevo `evaluation_collector`
   es el arnés **activo** y los reemplaza funcionalmente. ¿Se deprecia el pasivo
   o se mantienen ambos? (No se ha tocado el pasivo.)
2. **Frame de captura/pose**: el orquestador y el bridge asumen frame `map` y
   TF `map→base_link`. La visualización publica en `frame_id=map` (param). Si el
   despliegue real usa otro frame fijo, ajustar el parámetro.
3. **Aristas a visualizar**: hoy el grafo solo tiene `CONTAINS`
   (waypoint→object). Como los objetos no tienen pose, el `LINE_LIST` queda vacío
   hasta que existan aristas waypoint↔waypoint. ¿Se desea además dibujar los
   nodos `object` (con pose heredada del waypoint) o basta con los waypoints?
4. **`cmd_vel` del teleop**: en simulación Nav2 remapea `cmd_vel→cmd_vel_nav`.
   El teleop publica en `cmd_vel` (param). Confirmar si debe ir a un mux de
   velocidad para no competir con Nav2.

---

## Verificación (criterios de aceptación)

- `grep -r "import rclpy" semantic_evaluation/core/` → **0** resultados.
- `pytest` del núcleo (17 tests) **verde sin entorno ROS**.
- `colcon build --packages-select semantic_interfaces semantic_evaluation` **OK**.
- `ros2 run semantic_evaluation {evaluation_collector,graph_visualizer,teleop_capture}`
  arrancan sin errores de import.
- CSV con el esquema exacto y fila final `__AGGREGATE_MEAN__` (validado E2E
  contra un orquestador/servicio simulados, incluyendo `decision_only`/NaN).
- Estática: sin `spin_until_future_complete` en callbacks; sin nombres/rutas
  hardcodeados en los nodos (todo en `config/evaluation_params.yaml`).

## Uso rápido

```bash
colcon build --packages-select semantic_interfaces semantic_evaluation
source install/setup.bash

# Visualizador + RViz (campaña condicional)
ros2 launch semantic_evaluation evaluation.launch.py
ros2 launch semantic_evaluation evaluation.launch.py \
    run_collector:=true test_suite_path:=/abs/test_suite.yaml decision_only:=true

# Teleop + captura
ros2 run semantic_evaluation teleop_capture --ros-args \
    --params-file install/semantic_evaluation/share/semantic_evaluation/config/evaluation_params.yaml
```
