# Ejecución del sistema semántico

Todos los comandos parten de la raíz del repositorio.

## Compilación y tests

```bash
cd semantic_navigation_ws
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
ROS_LOG_DIR=/tmp/semantic_navigation_tests colcon test
colcon test-result --verbose
```

## Escenas

AWS Small House, con mapa y AMCL:

```bash
ros2 launch semantic_bringup simulation.launch.py \
  scene_id:=aws_small_house start_semantic:=true start_auto_mapping:=false
```

TurtleBot3 House, primero en modo SLAM. El mundo está vendorizado y no usa
recursos Fuel durante la ejecución:

```bash
ros2 launch semantic_bringup simulation.launch.py \
  scene_id:=turtlebot3_house \
  world:="$(ros2 pkg prefix semantic_bringup)/share/semantic_bringup/worlds/turtlebot3_house.world" \
  localization_mode:=slam spawn_x:=-2.0 spawn_y:=-0.5 \
  start_semantic:=true start_auto_mapping:=true
```

Tras explorar la casa, guardar el mapa y relanzar con AMCL para campañas:

```bash
mkdir -p "$HOME/.ros/semantic_maps/turtlebot3_house"
ros2 run nav2_map_server map_saver_cli \
  -f "$HOME/.ros/semantic_maps/turtlebot3_house/nav_map"

ros2 launch semantic_bringup simulation.launch.py \
  scene_id:=turtlebot3_house \
  world:="$(ros2 pkg prefix semantic_bringup)/share/semantic_bringup/worlds/turtlebot3_house.world" \
  map:="$HOME/.ros/semantic_maps/turtlebot3_house/nav_map.yaml" \
  localization_mode:=localization spawn_x:=-2.0 spawn_y:=-0.5 \
  start_semantic:=true
```

Semantic Office Lab, con modelos geométricos locales y mapa reproducible:

```bash
ros2 launch semantic_bringup simulation.launch.py \
  scene_id:=semantic_office_lab \
  world:="$(ros2 pkg prefix semantic_bringup)/share/semantic_bringup/worlds/semantic_office_lab.world" \
  map:="$(ros2 pkg prefix semantic_bringup)/share/semantic_bringup/maps/semantic_office_lab.yaml" \
  spawn_x:=-4.0 spawn_y:=0.0 start_semantic:=true
```

## Construcción del grafo

Captura multivista manual en el mismo punto físico:

```bash
ros2 action send_goal /capture_waypoint \
  semantic_interfaces/action/CaptureWaypoint \
  "{label: workstation_north_01, scene_id: semantic_office_lab,
    relative_view_yaws_deg: [0.0, 90.0, 180.0, 270.0], rotate_robot: true}"
```

Para creación automática, lanzar la escena con `start_auto_mapping:=true`. La
política de distancia, giro, tiempo, duplicados y vistas se encuentra en
`semantic_navigation_ros/config/mapping_config.yaml`. Las imágenes y el grafo
se guardan por escena bajo `~/.ros/semantic_maps/<scene_id>/`.

## Consulta y navegación

Top-K sin navegación:

```bash
ros2 action send_goal /navigate_to_semantic_goal \
  semantic_interfaces/action/NavigateToSemanticGoal \
  "{query_text: 'la taza a la izquierda del monitor', language: es,
    scene_id: semantic_office_lab, top_k: 5, decision_only: true,
    navigate: false}"
```

Consulta con validación de ocupación, comprobación de ruta y `NavigateToPose`:

```bash
ros2 action send_goal /navigate_to_semantic_goal \
  semantic_interfaces/action/NavigateToSemanticGoal \
  "{query_text: 'the meeting table near the plant', language: en,
    scene_id: semantic_office_lab, top_k: 5, decision_only: false,
    navigate: true}"
```

## Campañas

Ejemplo para Office Lab, con restauración de pose entre casos:

```bash
share="$(ros2 pkg prefix semantic_bringup)/share/semantic_bringup"
ros2 launch semantic_evaluation evaluation.launch.py \
  run_collector:=true use_rviz:=false scene_id:=semantic_office_lab \
  method:=hybrid_semantic_retrieval start_pose_id:=lab_entrance \
  test_suite_path:="$share/config/scenes/semantic_office_lab_queries.yaml" \
  start_poses_path:="$share/config/scenes/semantic_office_lab_start_poses.yaml"
```

Para AWS o TurtleBot3 House se sustituyen `scene_id`, `test_suite_path` y
`start_poses_path` por los ficheros homónimos de `config/scenes/`. No se debe
ejecutar una campaña navegada de TurtleBot3 House hasta guardar el mapa SLAM y
relanzar en modo `localization`.

Los resultados se escriben en
`experiments/simulation/campaigns/<scene>/<run>/evaluation.csv`. El análisis se
realiza con:

```bash
jupyter lab experiments/simulation/notebooks/00_ros2_campaign_analysis.ipynb
```

## Smoke tests y regeneración

```bash
cd semantic_navigation_ws
ROS_LOG_DIR=/tmp/aws_smoke src/semantic_bringup/scripts/smoke_test.sh install/setup.bash

ROS_LOG_DIR=/tmp/office_smoke SCENE_ID=semantic_office_lab \
WORLD_FILE="$PWD/install/semantic_bringup/share/semantic_bringup/worlds/semantic_office_lab.world" \
MAP_FILE="$PWD/install/semantic_bringup/share/semantic_bringup/maps/semantic_office_lab.yaml" \
SPAWN_X=-4.0 src/semantic_bringup/scripts/smoke_test.sh install/setup.bash
```

Regenerar el mapa de Office Lab desde su fuente vectorial y reinstalar assets:

```bash
ffmpeg -y -loglevel error \
  -i src/semantic_bringup/maps/semantic_office_lab.svg -frames:v 1 -pix_fmt gray \
  src/semantic_bringup/maps/semantic_office_lab.pgm
colcon build --symlink-install --packages-select semantic_bringup
```

Para apartar datos de una escena de forma recuperable:

```bash
gio trash "$HOME/.ros/semantic_maps/semantic_office_lab"
```
