#!/usr/bin/env python3
"""Launch file for the visual-semantic navigation simulation.

Starts the following subsystems:
- Gazebo with the AWS Small House world
- TurtleBot3 Waffle robot spawn
- Nav2 (navigation stack) with a pre-existing map
- The semantic_map_manager nodes (siglip_inference, waypoint_capture, semantic_navigator)

Usage
-----
    ros2 launch semantic_simulation simulation.launch.py

Optional arguments
------------------
    use_sim_time:=true/false  (default: true)
    map:=<path/to/map.yaml>   (default: maps/small_house.yaml bundled with this package)
    world:=<path/to/world>    (default: small_house.world bundled with this package)
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, Command
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    # ------------------------------------------------------------------ #
    # Package share directories
    # ------------------------------------------------------------------ #
    sim_share = get_package_share_directory("semantic_simulation")
    nav2_bringup_share = get_package_share_directory("nav2_bringup")
    tb3_gazebo_share = get_package_share_directory("turtlebot3_gazebo")
    aws_share = get_package_share_directory("aws_robomaker_small_house_world")
    tb3_desc_share = get_package_share_directory("turtlebot3_description")

    # ------------------------------------------------------------------ #
    # Launch arguments
    # ------------------------------------------------------------------ #
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation (Gazebo) clock.",
    )
    map_arg = DeclareLaunchArgument(
        "map",
        default_value=os.path.join(aws_share, "maps", "turtlebot3_waffle_pi", "map.yaml"),
        description="Full path to the Nav2 map yaml file.",
    )
    world_arg = DeclareLaunchArgument(
        "world",
        default_value=os.path.join(aws_share, "worlds", "small_house.world"),
        description="Full path to the Gazebo world file.",
    )
    params_file_arg = DeclareLaunchArgument(
        "params_file",
        default_value=os.path.join(sim_share, "config", "nav2_params.yaml"),
        description="Full path to the Nav2 parameters file.",
    )
    rviz_config_arg = DeclareLaunchArgument(
        "rviz_config",
        default_value=os.path.join(nav2_bringup_share, "rviz", "nav2_default_view.rviz"),
        description="Full path to the Rviz config file.",
    )

    use_sim_time = LaunchConfiguration("use_sim_time")
    map_yaml = LaunchConfiguration("map")
    world_file = LaunchConfiguration("world")
    params_file = LaunchConfiguration("params_file")
    rviz_config = LaunchConfiguration("rviz_config")

    # ------------------------------------------------------------------ #
    # Environment – TurtleBot3 model & Gazebo Models
    # ------------------------------------------------------------------ #
    set_tb3_model = SetEnvironmentVariable("TURTLEBOT3_MODEL", "waffle")
    
    set_gz_model_path = SetEnvironmentVariable(
        'GZ_SIM_RESOURCE_PATH',
        os.path.join(aws_share, 'models')
    )

    # ------------------------------------------------------------------ #
    # Gazebo
    # ------------------------------------------------------------------ #
    ros_gz_sim_share = get_package_share_directory("ros_gz_sim")
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim_share, "launch", "gz_sim.launch.py")
        ),
        launch_arguments={"gz_args": ["-r ", world_file]}.items(),
    )

    # ------------------------------------------------------------------ #
    # Robot State Publisher & Bridges
    # ------------------------------------------------------------------ #
    robot_state_publisher = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(tb3_gazebo_share, "launch", "robot_state_publisher.launch.py")
        ),
        launch_arguments={"use_sim_time": use_sim_time}.items(),
    )

    spawn_turtlebot3 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(tb3_gazebo_share, "launch", "spawn_turtlebot3.launch.py")
        ),
        launch_arguments={"x_pose": "0.0", "y_pose": "0.0"}.items(),
    )

    # ------------------------------------------------------------------ #
    # Nav2 & Rviz
    # ------------------------------------------------------------------ #
    nav2_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_bringup_share, "launch", "bringup_launch.py")
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "map": map_yaml,
            "params_file": params_file,
        }.items(),
    )
    
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config, '--ros-args', '--log-level', 'debug'],
        parameters=[{'use_sim_time': use_sim_time}],
    )

    # ------------------------------------------------------------------ #
    # Semantic map manager nodes
    # ------------------------------------------------------------------ #
    siglip_node = Node(
        package="semantic_map_manager",
        executable="siglip_inference",
        name="siglip_inference",
        parameters=[{"use_sim_time": use_sim_time}],
        output="screen",
    )

    waypoint_node = Node(
        package="semantic_map_manager",
        executable="waypoint_capture",
        name="waypoint_capturer",
        parameters=[{"use_sim_time": use_sim_time}],
        output="screen",
    )

    navigator_node = Node(
        package="semantic_map_manager",
        executable="semantic_navigator",
        name="semantic_navigator",
        parameters=[{"use_sim_time": use_sim_time}],
        output="screen",
    )

    # ------------------------------------------------------------------ #
    # Launch description
    # ------------------------------------------------------------------ #
    return LaunchDescription(
        [
            # Arguments
            use_sim_time_arg,
            map_arg,
            world_arg,
            params_file_arg,
            rviz_config_arg,
            # Environment
            set_tb3_model,
            set_gz_model_path,
            # Subsystems
            gazebo,
            spawn_turtlebot3,
            robot_state_publisher,
            nav2_bringup,
            rviz,
            siglip_node,
            waypoint_node,
            navigator_node,
        ]
    )
