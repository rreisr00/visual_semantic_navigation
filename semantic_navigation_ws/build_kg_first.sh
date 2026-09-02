#!/usr/bin/env bash
set -euo pipefail

WS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$WS_DIR"

colcon build --symlink-install --packages-select \
  knowledge_graph_msgs \
  knowledge_graph \
  knowledge_graph_db \
  knowledge_graph_terminal \
  knowledge_graph_demos \
  knowledge_graph_viewer

source "$WS_DIR/install/setup.bash"

colcon build --symlink-install --packages-skip \
  knowledge_graph_msgs \
  knowledge_graph \
  knowledge_graph_db \
  knowledge_graph_terminal \
  knowledge_graph_demos \
  knowledge_graph_viewer