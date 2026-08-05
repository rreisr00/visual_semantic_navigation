"""Launch the graphical console used for mapping and voice navigation."""

import glob
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def find_venv_site_packages(start_dir: str) -> str:
    """Return the optional project ML environment used by Whisper."""
    override = os.environ.get('SEMANTIC_NAV_VENV_SITE')
    if override:
        return override
    directory = start_dir
    for _ in range(10):
        hits = glob.glob(
            os.path.join(
                directory,
                '.venv-1',
                'lib',
                'python3.*',
                'site-packages',
            )
        )
        if hits:
            return sorted(hits)[-1]
        directory = os.path.dirname(directory)
    return ''


def generate_launch_description() -> LaunchDescription:
    share = get_package_share_directory('semantic_evaluation')
    default_params = os.path.join(share, 'config', 'operator_gui_params.yaml')
    venv_site = find_venv_site_packages(share)
    extra_env = {}
    if venv_site:
        extra_env['PYTHONPATH'] = (
            venv_site + ':' + os.environ.get('PYTHONPATH', '')
        )
    return LaunchDescription([
        DeclareLaunchArgument('scene_id', default_value='aws_small_house'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('camera_topic', default_value='/camera/image_raw'),
        DeclareLaunchArgument('rooms_file', default_value=''),
        DeclareLaunchArgument('params_file', default_value=default_params),
        Node(
            package='semantic_evaluation',
            executable='semantic_operator_gui',
            name='semantic_operator_gui',
            output='screen',
            additional_env=extra_env,
            parameters=[
                LaunchConfiguration('params_file'),
                {
                    'scene_id': LaunchConfiguration('scene_id'),
                    'use_sim_time': LaunchConfiguration('use_sim_time'),
                    'camera_topic': LaunchConfiguration('camera_topic'),
                    'rooms_file': LaunchConfiguration('rooms_file'),
                },
            ],
        ),
    ])
