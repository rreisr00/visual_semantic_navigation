"""Launch the graphical console used to create semantic maps."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    share = get_package_share_directory('semantic_evaluation')
    default_params = os.path.join(share, 'config', 'operator_gui_params.yaml')
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
