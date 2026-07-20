# visual_semantic_navigation

A ROS 2 (Jazzy) semantic navigation system for mobile robots that uses
[SigLIP 2](https://huggingface.co/google/siglip-base-patch16-224) vision-language
embeddings (optionally fused with YOLOv8 object detections) to build a spatial
Knowledge Graph of waypoints and navigate to semantically described targets.

The stack is organised as **pure-core + ROS-wrapper pairs**: all business logic
(vision pipeline, graph, ranking, capture state machine) lives in `*_core`
packages with no `rclpy` dependency and is unit-tested outside ROS; the `*_ros`
packages are thin coordinators that wrap that logic in lifecycle nodes, services
and actions.

---

## Architecture

```
semantic_navigation_ws/
└── src/
    ├── semantic_interfaces/          # msg / srv / action definitions
    ├── semantic_vision_core/         # PURE: SigLIP + YOLO pipeline (embed_image/embed_text)
    ├── semantic_vision_ros/          # visual_encoder lifecycle node
    ├── knowledge_graph/              # PURE: third-party knowledge graph lib (+ C++ targets)
    ├── knowledge_graph_ros/          # knowledge_graph_bridge lifecycle node (+ SQLite)
    ├── semantic_navigation_core/     # PURE: ranking (cosine/Jaccard) + capture state machine
    ├── semantic_navigation_ros/      # kg_manager, semantic_orchestrator, lifecycle_manager
    ├── semantic_evaluation/          # UF-7 metrics collector, teleop capture, RViz graph visualizer
    ├── semantic_voice/               # voice control: faster-whisper + intent parser (PURE core/)
    ├── semantic_bringup/             # simulation.launch.py, nav2 params, rviz, worlds, vendored robot model
    └── aws-robomaker-small-house-world/   # simulation world + map
```

### Data flow

```
                       teaching (CaptureWaypoint action)
 /capture_waypoint ──► kg_manager ──► get_visual_features ──► visual_encoder (SigLIP/YOLO)
                            │                                        │
                            └────────► store_waypoint ──► knowledge_graph_bridge ──► SQLite

                       runtime (NavigateToSemanticGoal action)
 /navigate_to_   ──► semantic_orchestrator ──► get_embedding / get_visual_features ──► visual_encoder
   semantic_goal           │
                           ├──► get_waypoints ──► knowledge_graph_bridge
                           ├──► rank_waypoints (semantic_navigation_core, pure)
                           └──► navigate_to_pose (Nav2 action)
```

---

## Prerequisites

| Dependency | Install | Used by |
|---|---|---|
| ROS 2 **Jazzy** | `ros-jazzy-desktop` | everything |
| Nav2 | `ros-jazzy-navigation2 ros-jazzy-nav2-bringup` | navigation + `nav2_msgs` action |
| Gazebo (Harmonic) bridge | `ros-jazzy-ros-gz-sim` | simulation |
| TurtleBot3 | `ros-jazzy-turtlebot3*` | robot model / spawn |
| cv_bridge | `ros-jazzy-cv-bridge` | image conversion |
| diagnostic_updater | `ros-jazzy-diagnostic-updater` | camera watchdog / GPU diagnostics |
| Python ≥ 3.12 | system | union-type hints (`X \| Y`) |

Install the ROS-side dependencies with `rosdep` from the workspace root:

```bash
cd semantic_navigation_ws
rosdep install --from-paths src --ignore-src -r -y
```

> **NumPy ABI note:** the system `cv_bridge` (Jazzy) is compiled against NumPy 1.x.
> The Python deps pin `numpy<2` and `opencv-python<4.11` accordingly — do not
> upgrade past those without rebuilding `cv_bridge`. See
> [`requirements.txt`](requirements.txt).

---

## Installation

```bash
cd semantic_navigation_ws

# 1. Python dependencies (torch, transformers, ultralytics, …)
pip install -r ../requirements.txt

# 2. Build the workspace (colcon resolves the build order from package deps,
#    including the knowledge_graph C++ targets)
colcon build --symlink-install
source install/setup.bash
```

You usually don't need to build by hand: `./run_simulation.sh` builds the
workspace (`colcon build --symlink-install`), sources the overlay and launches
the simulation in one step — see [Usage](#usage).

---

## Usage

### Build + launch the full simulation

The scripts live inside the workspace directory:

```bash
cd semantic_navigation_ws
./run_simulation.sh                       # full colcon build + launch
./run_simulation.sh --no-build            # skip the build (source + launch)
./run_simulation.sh --pkg semantic_bringup  # build a single package
./run_simulation.sh -- use_sim_time:=false map:=/path/to/map.yaml   # forward launch args
./run_simulation.sh -- camera_mast_z:=0.6   # camera height [m] (default 0.45)
```

`run_simulation.sh` sources ROS, builds (`--symlink-install`), kills any stale
Gazebo process, then runs `semantic_bringup simulation.launch.py` (Gazebo + Nav2
+ the semantic layer).

> **Robot camera:** the simulation spawns a vendored TurtleBot3 waffle
> (`semantic_bringup/models/turtlebot3_waffle_semantic`) whose RGB camera is
> raised on a mast to `camera_mast_z` meters (stock height is 0.107 m — too low
> for furniture-level SigLIP captures) at 640×480. TF from
> `robot_state_publisher` still uses the stock URDF camera height; that only
> matters if something starts consuming camera-frame TF (nothing does today).

### Health check

```bash
./check_system.sh        # every 10 s
./check_system.sh 0      # single shot
```

Reports the Nav2 and semantic nodes, lifecycle states of `visual_encoder` /
`knowledge_graph_bridge`, key topics, action servers (`/capture_waypoint`,
`/navigate_to_semantic_goal`, Nav2) and services (`/get_visual_features`,
`/get_embedding`, `/store_waypoint`, `/get_waypoints`).

### Teaching — capture a waypoint (action)

Drive the robot to the desired spot, then send a `CaptureWaypoint` goal:

```bash
ros2 action send_goal --feedback /capture_waypoint \
    semantic_interfaces/action/CaptureWaypoint "{label: ''}"
```

Feedback reports the stage (`got_image` → `got_pose` → `encoded` → `stored`);
the result carries the generated `node_id`.

### Runtime — navigate to a semantic target (action)

Text query:

```bash
ros2 action send_goal --feedback /navigate_to_semantic_goal \
    semantic_interfaces/action/NavigateToSemanticGoal \
    "{query_text: 'Find the sofa', use_image: false}"
```

Image query (`use_image: true` with a `sensor_msgs/Image` in `query_image`).
Feedback reports `embedding` → `ranking` → `navigating` plus
`distance_remaining`; the result carries the matched `node_id` and score.

### Voice control (`semantic_voice`)

Speak English commands; faster-whisper transcribes and a rules/regex parser
dispatches to the actions above:

| You say | Intent | Action |
|---|---|---|
| "move to 2.5 3.0" | direct move (map coords) | Nav2 `/navigate_to_pose` |
| "save waypoint kitchen" | capture with label | `/capture_waypoint` |
| "go to the sofa" (anything else) | semantic query | `/navigate_to_semantic_goal` |

One-time setup: `sudo apt install libportaudio2` (microphone) — the Python deps
(`faster-whisper`, `sounddevice`) come from `requirements.txt`.

```bash
# hands-free (VAD) — venv PYTHONPATH injected by the launch file
ros2 launch semantic_voice voice.launch.py activation_mode:=vad

# push-to-talk / text modes need a real terminal (stdin), so use ros2 run:
PYTHONPATH=../.venv-1/lib/python3.12/site-packages:$PYTHONPATH \
    ros2 run semantic_voice voice_command --ros-args \
    --params-file src/semantic_voice/config/voice_params.yaml \
    -p activation_mode:=push_to_talk        # or text
```

Modes (`activation_mode` param): `push_to_talk` (press `r` to start/stop
recording), `vad` (continuous listening, RMS segmentation), `text` (type
commands — no mic/GPU, ideal for testing the dispatch path). Whisper `small`
runs on the GPU in `int8_float16` (~0.5 GB VRAM next to SigLIP/YOLO); set
`device: cpu` in `config/voice_params.yaml` under VRAM pressure.

---

## Nodes

| Node | Package / executable | Kind | Exposes |
|---|---|---|---|
| `visual_encoder` | `semantic_vision_ros` / `visual_encoder` | Lifecycle | `get_visual_features`, `get_embedding` |
| `knowledge_graph_bridge` | `knowledge_graph_ros` / `knowledge_graph_bridge` | Lifecycle | `store_waypoint`, `get_waypoints` (+ SQLite) |
| `kg_manager` | `semantic_navigation_ros` / `kg_manager` | Action server | `/capture_waypoint` |
| `semantic_orchestrator` | `semantic_navigation_ros` / `semantic_orchestrator` | Action server | `/navigate_to_semantic_goal` |
| `lifecycle_manager` | `semantic_navigation_ros` / `lifecycle_manager` | Supervisor | drives the lifecycle nodes to `active` |
| `evaluation_collector` | `semantic_evaluation` / `evaluation_collector` | Metrics | active campaign harness + CSV export (see `semantic_evaluation`) |
| `voice_command` | `semantic_voice` / `voice_command` | Client | voice → `/navigate_to_pose` · `/capture_waypoint` · `/navigate_to_semantic_goal` |

All coordinator nodes run on a `MultiThreadedExecutor` with the action server
and the service/action clients in separate callback groups (no
`spin_until_future_complete` inside callbacks → no deadlock).

### Key parameters

| Node | Parameter | Default |
|---|---|---|
| `visual_encoder` | `retrieval_mode` | `siglip_pure` \| `siglip_yolo` |
| | `siglip_model_id` | `google/siglip-base-patch16-224` |
| | `yolo_model_path` / `yolo_confidence_threshold` | `~/.ros/semantic_models/yolov8n.pt` / `0.4` |
| | `force_cpu` | `false` (CPU fallback on persistent CUDA OOM) |
| `semantic_orchestrator` | `retrieval_mode` | `siglip_yolo` |
| | `hybrid_embedding_weight` / `hybrid_object_weight` | `0.7` / `0.3` |
| `knowledge_graph_bridge` | `db_file_path` | `~/.ros/semantic_maps/knowledge_graph.db` |
| `lifecycle_manager` | `managed_nodes` | `[visual_encoder, knowledge_graph_bridge]` |
| | `reconcile_period_sec` | `2.0` |

---

## Interfaces (`semantic_interfaces`)

**Services**

| Service | Type | Server |
|---|---|---|
| `get_visual_features` | `GetVisualFeatures` (image → embedding + objects) | `visual_encoder` |
| `get_embedding` | `GetEmbedding` (text **or** image → embedding) | `visual_encoder` |
| `store_waypoint` | `StoreWaypoint` | `knowledge_graph_bridge` |
| `get_waypoints` | `GetWaypoints` → `WaypointInfo[]` (typed embedding + objects) | `knowledge_graph_bridge` |

**Actions**

| Action | Type | Server |
|---|---|---|
| `/capture_waypoint` | `CaptureWaypoint` | `kg_manager` |
| `/navigate_to_semantic_goal` | `NavigateToSemanticGoal` | `semantic_orchestrator` |

---

## Knowledge Graph storage

`knowledge_graph_bridge` owns the `KnowledgeGraph` singleton and mirrors every
mutation to a SQLite database (WAL mode). Per stored waypoint:

- `pose_x/y/z`, `orient_x/y/z/w` — pose in the `map` frame
- `visual_embedding` — SigLIP embedding stored as a **JSON string property**
  (typed, not CSV). Vector properties are deliberately avoided: the vendored
  `knowledge_graph` cannot reconstruct ROS msg vector fields (`array.array`),
  which would crash every `graph_update` receiver (rqt viewer, terminal) and
  the bridge itself on DB reload; legacy float-list rows are migrated to JSON
  on load
- detected objects — stored as separate `object` nodes linked by `CONTAINS`
  edges; `get_waypoints` reconstructs the object label list by walking those edges

The retrieval ranking (`semantic_navigation_core.rank_waypoints`) is a linear
cosine (+ optional Jaccard) scan in memory — fine for tens/hundreds of
waypoints. An ANN index (FAISS/hnswlib) is intentionally **not** included yet.

---

## Fault tolerance

- **Lifecycle manager:** a custom supervisor reconciles `visual_encoder` and
  `knowledge_graph_bridge` to `active` every ~2 s. On startup they may briefly
  show `unconfigured`/`inactive` before reaching `active` — expected, not a fault.
- **CUDA OOM recovery:** `visual_encoder` empties the cache and retries once; on
  persistent OOM it self-recovers (deactivate → cleanup, reload on CPU via
  `force_cpu`) so the supervisor re-activates it. `on_error` releases the model.
- **Camera watchdog:** `kg_manager` publishes `diagnostic_msgs` and warns when
  `/camera/image_raw` frames go stale.

---

## Testing

```bash
# Pure unit tests (no ROS graph required)
colcon test --packages-select semantic_navigation_core
colcon test-result --verbose

# semantic_voice core (intent parser + speech segmenter), plain pytest
python3 -m pytest src/semantic_voice/test -v
```

> **Full experiment guide:** [docs/TESTING_GUIDE.md](docs/TESTING_GUIDE.md)
> (in Spanish) covers end-to-end testing, defining room zones in RViz,
> evaluation campaigns/metrics, and running the additional scenarios
> (turtlebot3_house, AWS bookstore/warehouse).

---

## Python Dependencies

See [`requirements.txt`](requirements.txt) at the repository root. ROS 2 packages
(`rclpy`, `sensor_msgs`, `cv_bridge`, `tf2_ros`, `nav2_msgs`,
`diagnostic_updater`, …) are installed via `rosdep`/`apt`, not pip.
