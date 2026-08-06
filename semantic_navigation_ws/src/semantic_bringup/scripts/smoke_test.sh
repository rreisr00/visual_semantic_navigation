#!/usr/bin/env bash
set -eo pipefail

workspace_setup="${1:-install/setup.bash}"
scene_config="${SCENE_CONFIG:-}"
scene_id="${SCENE_ID:-}"
startup_timeout="${STARTUP_TIMEOUT_S:-90}"
smoke_root="${ROS_LOG_DIR:-/tmp/visual-semantic-navigation-smoke}"
world_file="${WORLD_FILE:-}"
map_file="${MAP_FILE:-}"
localization_mode="${LOCALIZATION_MODE:-localization}"
spawn_x="${SPAWN_X:-}"
spawn_y="${SPAWN_Y:-}"
smoke_run_id="${SMOKE_RUN_ID:-$$}"

# Keep smoke runs isolated from interactive simulations and from orphaned ROS
# or Gazebo processes.  Two /clock publishers make simulated time jump
# backwards, which clears TF buffers and prevents Nav2 from activating.
if [[ -z "${ROS_DOMAIN_ID:-}" ]]; then
  export ROS_DOMAIN_ID="$((smoke_run_id % 180 + 30))"
fi
export GZ_PARTITION="${GZ_PARTITION:-semantic_navigation_smoke_${smoke_run_id}}"

if [[ ! -f "${workspace_setup}" ]]; then
  echo "Workspace setup not found: ${workspace_setup}" >&2
  exit 2
fi

source /opt/ros/jazzy/setup.bash
source "${workspace_setup}"
set -u
export ROS_LOG_DIR="${smoke_root}/ros"
mkdir -p "${ROS_LOG_DIR}"
echo "Smoke isolation: ROS_DOMAIN_ID=${ROS_DOMAIN_ID}, GZ_PARTITION=${GZ_PARTITION}"

launch_args=(
  headless:=true
  start_rviz:=false
  start_semantic:=false
  localization_mode:="${localization_mode}"
)
if [[ -n "${scene_config}" ]]; then
  launch_args+=(scene_config:="${scene_config}")
  [[ -n "${scene_id}" ]] && launch_args+=(scene_id:="${scene_id}")
else
  launch_args+=(scene_id:="${scene_id:-aws_small_house}")
fi
[[ -n "${spawn_x}" ]] && launch_args+=(spawn_x:="${spawn_x}")
[[ -n "${spawn_y}" ]] && launch_args+=(spawn_y:="${spawn_y}")
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

contains_line() {
  local output="$1"
  local expected="$2"
  local line
  while IFS= read -r line; do
    [[ "${line}" == "${expected}" ]] && return 0
  done <<< "${output}"
  return 1
}

file_contains() {
  local file="$1"
  local expected="$2"
  local line
  while IFS= read -r line; do
    [[ "${line}" == *"${expected}"* ]] && return 0
  done < "${file}"
  return 1
}

while (( SECONDS < deadline )); do
  topics="$(ros2 topic list 2>/dev/null || true)"
  nodes="$(ros2 node list 2>/dev/null || true)"
  services="$(ros2 service list 2>/dev/null || true)"
  missing=0
  for topic in "${required_topics[@]}"; do
    contains_line "${topics}" "${topic}" || missing=1
  done
  for node in "${required_nodes[@]}"; do
    contains_line "${nodes}" "${node}" || missing=1
  done
  for service in "${required_services[@]}"; do
    contains_line "${services}" "${service}" || missing=1
  done
  [[ "${missing}" -eq 0 ]] && break
  sleep 1
done

topics="$(ros2 topic list)"
nodes="$(ros2 node list)"
services="$(ros2 service list)"
for topic in "${required_topics[@]}"; do
  contains_line "${topics}" "${topic}"
done
for node in "${required_nodes[@]}"; do
  contains_line "${nodes}" "${node}"
done
for service in "${required_services[@]}"; do
  contains_line "${services}" "${service}"
done

[[ "$(ros2 topic type /clock)" == "rosgraph_msgs/msg/Clock" ]]
[[ "$(ros2 topic type /scan)" == "sensor_msgs/msg/LaserScan" ]]
[[ "$(ros2 topic type /camera/image_raw)" == "sensor_msgs/msg/Image" ]]
[[ "$(ros2 topic type /camera/depth/image_raw)" == "sensor_msgs/msg/Image" ]]
[[ "$(ros2 topic type /camera/depth/camera_info)" == "sensor_msgs/msg/CameraInfo" ]]
actions="$(ros2 action list)"
contains_line "${actions}" "/navigate_to_pose"
contains_line "${actions}" "/compute_path_to_pose"

timeout 10 ros2 topic echo /clock --once >/dev/null
timeout 15 ros2 topic echo /scan --once >/dev/null
timeout 15 ros2 topic echo /map --once >/dev/null
timeout 15 ros2 topic echo /camera/image_raw --once >/dev/null
timeout 15 ros2 topic echo /camera/camera_info --once >/dev/null
timeout 15 ros2 topic echo /camera/depth/image_raw --once >/dev/null
timeout 15 ros2 topic echo /camera/depth/camera_info --once >/dev/null
tf_output="${smoke_root}/tf_map_base_link.txt"
timeout 10 ros2 run tf2_ros tf2_echo map base_link >"${tf_output}" 2>&1 || true
file_contains "${tf_output}" "Translation:"

echo "Smoke test passed for ${scene_config:-${scene_id:-aws_small_house}}"
