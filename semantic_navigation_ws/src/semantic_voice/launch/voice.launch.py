#!/usr/bin/env python3
"""Launch the voice command node.

Injects the project ML venv (.venv-1) into PYTHONPATH the same way
simulation.launch.py does for visual_encoder/orchestrator, so
faster-whisper + sounddevice resolve.

NOTE: `ros2 launch` does not forward stdin, so the interactive modes
(`push_to_talk`, `text`) should be run directly instead:

    PYTHONPATH=<repo>/.venv-1/lib/python3.12/site-packages:$PYTHONPATH \
        ros2 run semantic_voice voice_command --ros-args \
        --params-file $(ros2 pkg prefix semantic_voice)/share/semantic_voice/config/voice_params.yaml

This launch file is intended for the hands-free mode:

    ros2 launch semantic_voice voice.launch.py activation_mode:=vad

Launch arguments
----------------
  activation_mode  (default: vad)   text | push_to_talk | vad
  params_file      (default: semantic_voice/config/voice_params.yaml)
"""

import glob
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def find_venv_site_packages(start_dir: str) -> str:
    """Locate the ML venv's site-packages (same convention as
    simulation.launch.py): SEMANTIC_NAV_VENV_SITE env var override, else
    walk up from the share dir globbing the python minor version."""
    override = os.environ.get("SEMANTIC_NAV_VENV_SITE")
    if override:
        return override
    d = start_dir
    for _ in range(10):
        d = os.path.dirname(d)
        hits = glob.glob(os.path.join(d, ".venv-1", "lib", "python3.*", "site-packages"))
        if hits:
            return sorted(hits)[-1]
    raise RuntimeError(
        f"Could not find .venv-1 above {start_dir}; set SEMANTIC_NAV_VENV_SITE."
    )


def generate_launch_description() -> LaunchDescription:
    _voice_share = get_package_share_directory("semantic_voice")

    _venv_site = find_venv_site_packages(_voice_share)
    _extra_env = {
        "PYTHONPATH": _venv_site + ":" + os.environ.get("PYTHONPATH", "")
    }

    params_file_arg = DeclareLaunchArgument(
        "params_file",
        default_value=os.path.join(_voice_share, "config", "voice_params.yaml"),
        description="Full path to the voice_command parameter file.",
    )
    activation_mode_arg = DeclareLaunchArgument(
        "activation_mode", default_value="vad",
        description="text | push_to_talk | vad (interactive modes need "
                    "'ros2 run' for stdin; see file docstring).",
    )

    voice_node = Node(
        package="semantic_voice",
        executable="voice_command",
        name="voice_command",
        output="screen",
        parameters=[
            LaunchConfiguration("params_file"),
            {"activation_mode": LaunchConfiguration("activation_mode")},
        ],
        additional_env=_extra_env,
    )

    return LaunchDescription([
        params_file_arg,
        activation_mode_arg,
        voice_node,
    ])
