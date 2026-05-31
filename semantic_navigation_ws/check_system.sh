#!/usr/bin/env bash
# Periodic health check for the visual-semantic navigation stack.
#
# Uso:
#   ./check_system.sh              # check every 10 s (default)
#   ./check_system.sh 5            # check every 5 s
#   ./check_system.sh 0            # single shot, then exit

set -uo pipefail

INTERVAL="${1:-10}"

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

# ── Node lists ────────────────────────────────────────────────────────────────
NAV_NODES=(
    /map_server
    /amcl
    /controller_server
    /smoother_server
    /planner_server
    /route_server
    /behavior_server
    /bt_navigator
    /waypoint_follower
    /velocity_smoother
    /collision_monitor
    /docking_server
    /lifecycle_manager_localization
    /lifecycle_manager_navigation
)

SEMANTIC_NODES=(
    /visual_encoder
    /kg_manager
    /semantic_orchestrator
    /evaluation_node
    /knowledge_graph_db
)

LIFECYCLE_NODES=(
    map_server
    amcl
    controller_server
    smoother_server
    planner_server
    route_server
    behavior_server
    bt_navigator
    waypoint_follower
    velocity_smoother
    collision_monitor
    docking_server
)

KEY_TOPICS=(
    /scan
    /odom
    /map
    /tf
    /amcl_pose
    /cmd_vel
    /local_costmap/costmap
    /plan
)

KEY_ACTIONS=(
    /navigate_to_pose
    /navigate_through_poses
)

# ── Helpers ───────────────────────────────────────────────────────────────────
status_icon() {
    # $1 = condition ("ok" | anything else → fail)
    if [[ "$1" == ok ]]; then
        echo -e "${GREEN}✓${RESET}"
    else
        echo -e "${RED}✗${RESET}"
    fi
}

# ── Single check pass ─────────────────────────────────────────────────────────
check_once() {
    local ts
    ts=$(date '+%H:%M:%S')

    # Gather runtime data (one ROS call each)
    local running_nodes all_topics all_actions
    running_nodes=$(ros2 node list 2>/dev/null || true)
    all_topics=$(ros2 topic list 2>/dev/null || true)
    all_actions=$(ros2 action list 2>/dev/null || true)

    local nav_ok=0 nav_fail=0 sem_ok=0 sem_fail=0

    # ── Nav2 nodes ─────────────────────────────────────────────────────────── #
    echo ""
    echo -e "${BOLD}━━━  Nav2 Nodes  $(printf '%.0s━' {1..40}) ${ts}  ━━━${RESET}"
    for node in "${NAV_NODES[@]}"; do
        if echo "$running_nodes" | grep -qx "$node"; then
            echo -e "  $(status_icon ok) $node"
            ((nav_ok++))
        else
            echo -e "  $(status_icon fail) $node"
            ((nav_fail++))
        fi
    done

    # ── Semantic nodes ─────────────────────────────────────────────────────── #
    echo ""
    echo -e "${BOLD}━━━  Semantic Nodes  $(printf '%.0s━' {1..37})  ━━━${RESET}"
    for node in "${SEMANTIC_NODES[@]}"; do
        if echo "$running_nodes" | grep -qx "$node"; then
            echo -e "  $(status_icon ok) $node"
            ((sem_ok++))
        else
            echo -e "  $(status_icon fail) $node"
            ((sem_fail++))
        fi
    done

    # ── Lifecycle states ────────────────────────────────────────────────────── #
    echo ""
    echo -e "${BOLD}━━━  Lifecycle States  $(printf '%.0s━' {1..35})  ━━━${RESET}"
    for node in "${LIFECYCLE_NODES[@]}"; do
        local raw state
        raw=$(ros2 lifecycle get "/${node}" 2>/dev/null || true)
        # output format: "- /<node> [<id>]: <state>"
        # Jazzy format: "active [3]"  (label first, id in brackets after)
        state=$(echo "$raw" | grep -oP '^\w+' | head -1)
        state="${state:-unreachable}"
        case "$state" in
            active)
                echo -e "  ${GREEN}active     ${RESET} /${node}" ;;
            configured|inactive)
                echo -e "  ${YELLOW}${state}  ${RESET} /${node}" ;;
            *)
                echo -e "  ${RED}${state}  ${RESET} /${node}" ;;
        esac
    done

    # ── Key topics ─────────────────────────────────────────────────────────── #
    echo ""
    echo -e "${BOLD}━━━  Key Topics  $(printf '%.0s━' {1..41})  ━━━${RESET}"
    for topic in "${KEY_TOPICS[@]}"; do
        if echo "$all_topics" | grep -qx "$topic"; then
            echo -e "  $(status_icon ok) $topic"
        else
            echo -e "  $(status_icon fail) $topic"
        fi
    done

    # ── Action servers ─────────────────────────────────────────────────────── #
    echo ""
    echo -e "${BOLD}━━━  Action Servers  $(printf '%.0s━' {1..37})  ━━━${RESET}"
    for action in "${KEY_ACTIONS[@]}"; do
        if echo "$all_actions" | grep -qx "$action"; then
            echo -e "  $(status_icon ok) $action"
        else
            echo -e "  $(status_icon fail) $action"
        fi
    done

    # ── TF tree ────────────────────────────────────────────────────────────── #
    echo ""
    echo -e "${BOLD}━━━  TF Frames  $(printf '%.0s━' {1..42})  ━━━${RESET}"
    local key_frames=(map odom base_footprint base_link base_scan)
    # Collect the TF frame list once (tf2_monitor --spin-time 1)
    local tf_frames
    tf_frames=$(timeout 2s ros2 run tf2_ros tf2_monitor --spin-time 1 2>/dev/null \
                | grep -oP 'Frame \K\S+(?= exists)' || true)
    for frame in "${key_frames[@]}"; do
        if echo "$tf_frames" | grep -qx "$frame"; then
            echo -e "  $(status_icon ok) $frame"
        else
            echo -e "  $(status_icon fail) $frame"
        fi
    done

    # ── Summary ────────────────────────────────────────────────────────────── #
    local total_ok total_fail total_nodes
    total_ok=$((nav_ok + sem_ok))
    total_fail=$((nav_fail + sem_fail))
    total_nodes=$((${#NAV_NODES[@]} + ${#SEMANTIC_NODES[@]}))
    echo ""
    echo -e "${BOLD}Summary: ${GREEN}${total_ok}/${total_nodes} nodes up${RESET}" \
            "  nav2=${GREEN}${nav_ok}/${#NAV_NODES[@]}${RESET}" \
            "  semantic=${GREEN}${sem_ok}/${#SEMANTIC_NODES[@]}${RESET}" \
            "${total_fail:+  ${RED}${total_fail} missing${RESET}}"
}

# ── Environment setup ─────────────────────────────────────────────────────────
WS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROS_DISTRO="${ROS_DISTRO:-jazzy}"

set +u
# shellcheck disable=SC1090,SC1091
source "/opt/ros/${ROS_DISTRO}/setup.bash" 2>/dev/null || true
[[ -f "${WS_DIR}/install/setup.bash" ]] && source "${WS_DIR}/install/setup.bash"
set -u

# ── Run ───────────────────────────────────────────────────────────────────────
if [[ "$INTERVAL" == "0" ]]; then
    check_once
else
    echo -e "${CYAN}Visual-Semantic Navigation — system health check every ${INTERVAL}s" \
            "(Ctrl+C to stop)${RESET}"
    while true; do
        check_once
        sleep "$INTERVAL"
    done
fi
