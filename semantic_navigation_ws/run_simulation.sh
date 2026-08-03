#!/usr/bin/env bash
# Compila el workspace, hace source del install y lanza la simulación semántica.
#
# Uso:
#   ./run_simulation.sh                          # build completo
#   ./run_simulation.sh --no-build               # salta el build (solo source + launch)
#   ./run_simulation.sh --pkg semantic_bringup   # build solo de un paquete
#   ./run_simulation.sh --config aws_small_house # carga config/scenes/aws_small_house.yaml
#
# Argumentos extra tras '--' se pasan al launch:
#   ./run_simulation.sh --config aws_small_house -- headless:=true

set -euo pipefail

# ── Colores ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'
BOLD='\033[1m'; NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
die()   { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

# ── Paths ─────────────────────────────────────────────────────────────────────
WS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROS_DISTRO="${ROS_DISTRO:-jazzy}"
ROS_SETUP="/opt/ros/${ROS_DISTRO}/setup.bash"
WS_SETUP="${WS_DIR}/install/setup.bash"

# ── Argparse ──────────────────────────────────────────────────────────────────
DO_BUILD=true
PKG_FILTER=""
SCENE_CONFIG=""
LAUNCH_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-build)   DO_BUILD=false; shift ;;
        --pkg)        PKG_FILTER="$2"; shift 2 ;;
        --config)     SCENE_CONFIG="$2"; shift 2 ;;
        --)           shift; LAUNCH_ARGS=("$@"); break ;;
        *)            die "Opción desconocida: $1  (usa -- para pasar args al launch)" ;;
    esac
done

if [[ -n "$SCENE_CONFIG" ]]; then
    LAUNCH_ARGS=("scene_config:=${SCENE_CONFIG}" "${LAUNCH_ARGS[@]}")
fi

# ── ROS base setup ────────────────────────────────────────────────────────────
[[ -f "$ROS_SETUP" ]] || die "No se encontró ROS $ROS_DISTRO en $ROS_SETUP"
# Los setup.bash de ROS/ament referencian variables no inicializadas (AMENT_TRACE_SETUP_FILES, etc.)
# por lo que es necesario desactivar nounset (-u) durante el source.
set +u
# shellcheck disable=SC1090
source "$ROS_SETUP"
set -u
info "ROS $ROS_DISTRO cargado."

# ── Build ─────────────────────────────────────────────────────────────────────
cd "$WS_DIR"

if [[ "$DO_BUILD" == true ]]; then
    echo -e "\n${BOLD}━━━  COLCON BUILD  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    BUILD_CMD=(colcon build --symlink-install)
    if [[ -n "$PKG_FILTER" ]]; then
        BUILD_CMD+=(--packages-select "$PKG_FILTER")
        info "Build selectivo: ${PKG_FILTER}"
    else
        info "Build completo del workspace…"
    fi

    "${BUILD_CMD[@]}" || die "colcon build falló. Revisa los errores anteriores."
    ok "Build completado."
fi

# ── Source install ────────────────────────────────────────────────────────────
echo -e "\n${BOLD}━━━  SOURCE INSTALL  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
[[ -f "$WS_SETUP" ]] || die "No existe $WS_SETUP — compila el workspace primero."
set +u
# shellcheck disable=SC1090
source "$WS_SETUP"
set -u
ok "Overlay del workspace cargado."

# ── Kill lingering Gazebo processes ───────────────────────────────────────────
# Ctrl+C on the launch often leaves the Gazebo GUI/server alive.
# A stale Gazebo leaves the robot at its last position, causing an AMCL mismatch.
echo -e "\n${BOLD}━━━  GAZEBO CLEANUP  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
if pgrep -fa "gz sim|gzserver|ruby.*gz" | grep -v grep > /dev/null 2>&1; then
    warn "Gazebo process detected — killing for a clean start."
    pkill -SIGTERM -f "gz sim" 2>/dev/null || true
    pkill -SIGTERM -f "gzserver"  2>/dev/null || true
    sleep 1
    pkill -SIGKILL -f "gz sim" 2>/dev/null || true
    pkill -SIGKILL -f "gzserver"  2>/dev/null || true
    ok "Gazebo cleared."
else
    ok "No stale Gazebo process found."
fi

# ── Launch ────────────────────────────────────────────────────────────────────
echo -e "\n${BOLD}━━━  ROS 2 LAUNCH  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
info "Lanzando semantic_bringup simulation.launch.py ${LAUNCH_ARGS[*]+"con args: ${LAUNCH_ARGS[*]}"}"
echo ""

exec ros2 launch semantic_bringup simulation.launch.py "${LAUNCH_ARGS[@]}"
