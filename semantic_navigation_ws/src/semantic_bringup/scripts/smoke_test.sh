#!/usr/bin/env bash
set -eo pipefail

workspace_setup="${1:-install/setup.bash}"
scene_id="${SCENE_ID:-aws_small_house}"
startup_timeout="${STARTUP_TIMEOUT_S:-90}"
smoke_root="${ROS_LOG_DIR:-/tmp/visual-semantic-navigation-smoke}"
world_file="${WORLD_FILE:-}"
map_file="${MAP_FILE:-}"
localization_mode="${LOCALIZATION_MODE:-localization}"
spawn_x="${SPAWN_X:-0.0}"
spawn_y="${SPAWN_Y:-0.0}"

if [[ ! -f "${workspace_setup}" ]]; then
  echo "Workspace setup not found: ${workspace_setup}" >&2
  exit 2
fi

source /opt/ros/jazzy/setup.bash
source "${workspace_setup}"
set -u
export ROS_LOG_DIR="${smoke_root}/ros"
mkdir -p "${ROS_LOG_DIR}"

launch_args=(
  scene_id:="${scene_id}"
  headless:=true
  start_rviz:=false
  start_semantic:=false
  localization_mode:="${localization_mode}"
  spawn_x:="${spawn_x}"
  spawn_y:="${spawn_y}"
)
[[ -n "${world_file}" ]] && launch_args+=(world:="${world_file}")
[[ -n "${map_file}" ]] && launch_args+=(map:="${map_file}")
ros2 launch semantic_bringup simulation.launch.py "${launch_args[@]}" &
launch_pid=$!

cleanup() {
  kill -TERM "${launch_pid}" 2>/dev/null || true
  for _ in {1..15}; do
    kill -0 "${launch_pid}" 2>/dev/null || break
    sleep 1
  done
  kill -KILL "${launch_pid}" 2>/dev/null || true
  wait "${launch_pid}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

required_topics=(/clock /tf /tf_static /odom /scan /map /camera/image_raw /camera/camera_info /camera/depth/image_raw /camera/depth/camera_info)
if [[ "${localization_mode}" == "slam" ]]; then
  required_nodes=(/slam_toolbox /planner_server /controller_server /bt_navigator)
else
  required_nodes=(/map_server /amcl /planner_server /controller_server /bt_navigator)
fi
required_services=(/world/default/set_pose)
deadline=$((SECONDS + startup_timeout))

while (( SECONDS < deadline )); do
  topics="$(ros2 topic list 2>/dev/null || true)"
  nodes="$(ros2 node list 2>/dev/null || true)"
  services="$(ros2 service list 2>/dev/null || true)"
  missing=0
  for topic in "${required_topics[@]}"; do
    [[ "${topics}" == *"${topic}"* ]] || missing=1
  done
  for node in "${required_nodes[@]}"; do
    [[ "${nodes}" == *"${node}"* ]] || missing=1
  done
  for service in "${required_services[@]}"; do
    [[ "${services}" == *"${service}"* ]] || missing=1
  done
  [[ "${missing}" -eq 0 ]] && break
  sleep 1
done

for topic in "${required_topics[@]}"; do
  ros2 topic list | rg -x "${topic}" >/dev/null
done
for node in "${required_nodes[@]}"; do
  ros2 node list | rg -x "${node}" >/dev/null
done
for service in "${required_services[@]}"; do
  ros2 service list | rg -x "${service}" >/dev/null
done

[[ "$(ros2 topic type /clock)" == "rosgraph_msgs/msg/Clock" ]]
[[ "$(ros2 topic type /scan)" == "sensor_msgs/msg/LaserScan" ]]
[[ "$(ros2 topic type /camera/image_raw)" == "sensor_msgs/msg/Image" ]]
[[ "$(ros2 topic type /camera/depth/image_raw)" == "sensor_msgs/msg/Image" ]]
[[ "$(ros2 topic type /camera/depth/camera_info)" == "sensor_msgs/msg/CameraInfo" ]]
ros2 action list | rg -x "/navigate_to_pose" >/dev/null
ros2 action list | rg -x "/compute_path_to_pose" >/dev/null

timeout 10 ros2 topic echo /clock --once >/dev/null
timeout 15 ros2 topic echo /scan --once >/dev/null
timeout 15 ros2 topic echo /map --once >/dev/null
timeout 15 ros2 topic echo /camera/image_raw --once >/dev/null
timeout 15 ros2 topic echo /camera/camera_info --once >/dev/null
timeout 15 ros2 topic echo /camera/depth/image_raw --once >/dev/null
timeout 15 ros2 topic echo /camera/depth/camera_info --once >/dev/null
tf_output="${smoke_root}/tf_map_base_link.txt"
timeout 10 ros2 run tf2_ros tf2_echo map base_link >"${tf_output}" 2>&1 || true
rg "Translation:" "${tf_output}" >/dev/null

echo "Smoke test passed for ${scene_id}"
