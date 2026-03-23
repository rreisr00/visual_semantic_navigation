# visual_semantic_navigation

A ROS 2 Semantic Navigation system for mobile robots that uses
[SigLIP 2](https://huggingface.co/google/siglip-base-patch16-224) vision-language
embeddings to build a spatial Knowledge Graph of waypoints and navigate to
semantically described targets.

---

## Architecture

```
semantic_navigation_ws/
└── src/
    ├── semantic_map_manager_interfaces/   # Custom GetEmbedding service definition
    ├── semantic_map_manager/              # Core logic nodes
    │   ├── siglip_inference.py            # Task 1 – AI Inference Node
    │   ├── waypoint_capture.py            # Task 2 – Manual Waypoint Capturer
    │   └── semantic_navigator.py          # Task 3 – Semantic Navigator
    ├── semantic_simulation/               # Task 4 – Simulation setup
    │   ├── launch/simulation.launch.py
    │   ├── config/nav2_params.yaml
    │   ├── maps/
    │   └── worlds/small_house.world
    └── third_party/
        └── knowledge_graph/               # KnowledgeGraphClient stub / shim
```

---

## Prerequisites

| Dependency | Notes |
|---|---|
| ROS 2 Humble or Jazzy | `ros-humble-desktop` / `ros-jazzy-desktop` |
| Nav2 | `ros-${ROS_DISTRO}-navigation2 ros-${ROS_DISTRO}-nav2-bringup` |
| TurtleBot3 packages | `ros-${ROS_DISTRO}-turtlebot3*` |
| cv_bridge | `ros-${ROS_DISTRO}-cv-bridge` |
| Python ≥ 3.10 | Required for union-type hints (`X \| Y`) |

---

## Installation

```bash
# 1. Clone (already done if you are reading this in the workspace)
cd semantic_navigation_ws

# 2. Install Python dependencies
pip install -r ../requirements.txt

# 3. Build the workspace
colcon build --symlink-install

# 4. Source the workspace
source install/setup.bash
```

### Build order (knowledge_graph first)

`knowledge_graph` includes C++ targets. In this workspace layout (`src/knowledge_graph` side-by-side with your packages), compile it first and then build the rest:

```bash
cd semantic_navigation_ws
./build_kg_first.sh
source install/setup.bash
```

---

## Usage

### Launch the full simulation

```bash
# Set the TurtleBot3 model (also handled automatically by the launch file)
export TURTLEBOT3_MODEL=waffle

ros2 launch semantic_simulation simulation.launch.py
```

### Launch AWS house world with Gazebo Harmonic (`gz sim`)

```bash
source semantic_navigation_ws/install/setup.bash
ros2 launch aws_robomaker_small_house_world small_house.launch.py gui:=true
```

Optional arguments:

```bash
ros2 launch semantic_simulation simulation.launch.py \
    map:=/path/to/my_map.yaml \
    world:=/path/to/my_world.world \
    params_file:=/path/to/nav2_params.yaml
```

### Capture a waypoint

Publish a trigger message while the robot is at the desired location:

```bash
ros2 topic pub --once /trigger_capture std_msgs/msg/Empty {}
```

### Navigate to a semantic target

```bash
ros2 topic pub --once /navigate_to_semantic_target std_msgs/msg/String \
    "data: 'Find the sofa'"
```

---

## Nodes

| Node | Executable | Description |
|---|---|---|
| `siglip_inference` | `siglip_inference` | Loads SigLIP 2 and serves `get_embedding` service |
| `waypoint_capturer` | `waypoint_capture` | Captures waypoints on `/trigger_capture` |
| `semantic_navigator` | `semantic_navigator` | Navigates to best semantic match |

### `get_embedding` service

**Service type:** `semantic_map_manager_interfaces/srv/GetEmbedding`

| Field | Direction | Type | Description |
|---|---|---|---|
| `image` | Request | `sensor_msgs/Image` | Input image (when `use_image=true`) |
| `text` | Request | `string` | Input text (when `use_image=false`) |
| `use_image` | Request | `bool` | Selects the input modality |
| `embedding` | Response | `float32[]` | L2-normalised SigLIP 2 embedding |
| `success` | Response | `bool` | Whether the call succeeded |
| `message` | Response | `string` | Status or error message |

---

## Knowledge Graph

The `knowledge_graph` package (under `third_party/`) provides a
`KnowledgeGraphClient` that stores waypoint nodes. Each node contains:

- `id` – UUID
- `pose_x`, `pose_y`, `pose_z` – robot position in the `map` frame
- `orient_x`, `orient_y`, `orient_z`, `orient_w` – quaternion orientation
- `embedding` – comma-separated SigLIP 2 image embedding

When the upstream [mgonzs13/knowledge_graph](https://github.com/mgonzs13/knowledge_graph)
package is installed the client uses its ROS 2 services; otherwise an in-memory
store is used automatically.

---

## Map Generation

See `semantic_navigation_ws/src/semantic_simulation/maps/README.md` for
instructions on generating a map with SLAM.

---

## Python Dependencies

See [`requirements.txt`](requirements.txt) at the repository root.