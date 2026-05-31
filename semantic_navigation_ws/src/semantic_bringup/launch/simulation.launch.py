#!/usr/bin/env python3
"""
Bringup launch for the visual-semantic navigation project.

Starts (in order):
  1. Gazebo          – semantic_test.world via ros_gz_sim
  2. TurtleBot3      – robot_state_publisher + spawn via turtlebot3_gazebo
  3. Localization    – map_server + amcl + lifecycle_manager_localization
  4. Navigation      – controller, smoother, planner, route_server, behavior,
                       bt_navigator, waypoint_follower, velocity_smoother,
                       collision_monitor, docking_server +
                       lifecycle_manager_navigation
  5. RViz2           – project config from this package
  6. Semantic layer  – visual_encoder, kg_manager, semantic_orchestrator,
                       evaluation_node, knowledge_graph_db

Launch arguments
----------------
  use_sim_time  (default: true)
  map           (default: aws_robomaker_small_house_world bundled map)
  world         (default: semantic_bringup/worlds/semantic_test.world)
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterFile
from launch_ros.substitutions import FindPackageShare
from nav2_common.launch import RewrittenYaml


def generate_launch_description() -> LaunchDescription:

    # ------------------------------------------------------------------ #
    # Static paths (resolved at parse time)
    # ------------------------------------------------------------------ #
    _bringup_share = get_package_share_directory("semantic_bringup")
    _aws_share     = get_package_share_directory("aws_robomaker_small_house_world")
    _smm_share     = get_package_share_directory("semantic_map_manager")

    _params_file = os.path.join(_bringup_share, "config", "nav2_params.yaml")
    _rviz_config = os.path.join(_bringup_share, "config", "rviz_config.rviz")
    _bt_to_pose  = os.path.join(
        _bringup_share, "config", "behavior_trees",
        "navigate_to_pose_w_replanning_and_recovery.xml",
    )
    _bt_through  = os.path.join(
        _bringup_share, "config", "behavior_trees",
        "navigate_through_poses_w_replanning_and_recovery.xml",
    )

    # ------------------------------------------------------------------ #
    # Launch arguments
    # ------------------------------------------------------------------ #
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time", default_value="true",
        description="Use Gazebo simulation clock.",
    )
    map_arg = DeclareLaunchArgument(
        "map",
        default_value=os.path.join(
            _aws_share, "maps", "turtlebot3_waffle_pi", "map.yaml"
        ),
        description="Full path to the Nav2 map YAML file.",
    )
    world_arg = DeclareLaunchArgument(
        "world",
        default_value=PathJoinSubstitution(
            [FindPackageShare("semantic_bringup"), "worlds", "semantic_test.world"]
        ),
        description="Full path to the Gazebo world file.",
    )

    use_sim_time = LaunchConfiguration("use_sim_time")
    map_yaml     = LaunchConfiguration("map")
    world_file   = LaunchConfiguration("world")

    # ------------------------------------------------------------------ #
    # Nav2 parameter file
    # RewrittenYaml injects use_sim_time and BT paths at launch time so
    # every node receives correct values without hardcoding in the YAML.
    # ------------------------------------------------------------------ #
    remappings = [("/tf", "tf"), ("/tf_static", "tf_static")]

    configured_params = ParameterFile(
        RewrittenYaml(
            source_file=_params_file,
            root_key="",
            param_rewrites={
                "use_sim_time":                    use_sim_time,
                "default_nav_to_pose_bt_xml":      _bt_to_pose,
                "default_nav_through_poses_bt_xml": _bt_through,
            },
            convert_types=True,
        ),
        allow_substs=True,
    )

    # ------------------------------------------------------------------ #
    # Environment variables
    # ------------------------------------------------------------------ #
    set_tb3_model = SetEnvironmentVariable("TURTLEBOT3_MODEL", "waffle")
    set_gz_resource_path = SetEnvironmentVariable(
        "GZ_SIM_RESOURCE_PATH",
        os.path.join(_aws_share, "models"),
    )

    # ------------------------------------------------------------------ #
    # 1. Gazebo
    # ------------------------------------------------------------------ #
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("ros_gz_sim"), "launch", "gz_sim.launch.py"]
            )
        ),
        launch_arguments={"gz_args": ["-r ", world_file]}.items(),
    )

    # ------------------------------------------------------------------ #
    # 2. TurtleBot3
    # ------------------------------------------------------------------ #
    robot_state_publisher = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("turtlebot3_gazebo"), "launch",
                 "robot_state_publisher.launch.py"]
            )
        ),
        launch_arguments={"use_sim_time": use_sim_time}.items(),
    )

    spawn_turtlebot3 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("turtlebot3_gazebo"), "launch",
                 "spawn_turtlebot3.launch.py"]
            )
        ),
        launch_arguments={"x_pose": "0.0", "y_pose": "0.0"}.items(),
    )

    # ------------------------------------------------------------------ #
    # 3. Localization – map_server + amcl
    # ------------------------------------------------------------------ #
    map_server = Node(
        package="nav2_map_server",
        executable="map_server",
        name="map_server",
        output="screen",
        parameters=[configured_params, {"yaml_filename": map_yaml}],
        remappings=remappings,
    )

    amcl = Node(
        package="nav2_amcl",
        executable="amcl",
        name="amcl",
        output="screen",
        parameters=[configured_params],
        remappings=remappings,
    )

    lifecycle_manager_localization = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_localization",
        output="screen",
        parameters=[{
            "use_sim_time": use_sim_time,
            "autostart":    True,
            "node_names":   ["map_server", "amcl"],
        }],
    )

    # ------------------------------------------------------------------ #
    # 4. Navigation servers
    # ------------------------------------------------------------------ #
    controller_server = Node(
        package="nav2_controller",
        executable="controller_server",
        name="controller_server",
        output="screen",
        parameters=[configured_params],
        remappings=remappings + [("cmd_vel", "cmd_vel_nav")],
    )

    smoother_server = Node(
        package="nav2_smoother",
        executable="smoother_server",
        name="smoother_server",
        output="screen",
        parameters=[configured_params],
        remappings=remappings,
    )

    planner_server = Node(
        package="nav2_planner",
        executable="planner_server",
        name="planner_server",
        output="screen",
        parameters=[configured_params],
        remappings=remappings,
    )

    route_server = Node(
        package="nav2_route",
        executable="route_server",
        name="route_server",
        output="screen",
        parameters=[configured_params],
        remappings=remappings,
    )

    behavior_server = Node(
        package="nav2_behaviors",
        executable="behavior_server",
        name="behavior_server",
        output="screen",
        parameters=[configured_params],
        remappings=remappings + [("cmd_vel", "cmd_vel_nav")],
    )

    bt_navigator = Node(
        package="nav2_bt_navigator",
        executable="bt_navigator",
        name="bt_navigator",
        output="screen",
        parameters=[configured_params],
        remappings=remappings,
    )

    waypoint_follower = Node(
        package="nav2_waypoint_follower",
        executable="waypoint_follower",
        name="waypoint_follower",
        output="screen",
        parameters=[configured_params],
        remappings=remappings,
    )

    velocity_smoother = Node(
        package="nav2_velocity_smoother",
        executable="velocity_smoother",
        name="velocity_smoother",
        output="screen",
        parameters=[configured_params],
        remappings=remappings + [("cmd_vel", "cmd_vel_nav")],
    )

    collision_monitor = Node(
        package="nav2_collision_monitor",
        executable="collision_monitor",
        name="collision_monitor",
        output="screen",
        parameters=[configured_params],
        remappings=remappings,
    )

    docking_server = Node(
        package="opennav_docking",
        executable="opennav_docking",
        name="docking_server",
        output="screen",
        parameters=[configured_params],
        remappings=remappings,
    )

    lifecycle_manager_navigation = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_navigation",
        output="screen",
        parameters=[{
            "use_sim_time": use_sim_time,
            "autostart":    True,
            "node_names":   [
                "controller_server",
                "smoother_server",
                "planner_server",
                "route_server",
                "behavior_server",
                "velocity_smoother",
                "collision_monitor",
                "bt_navigator",
                "waypoint_follower",
                "docking_server",
            ],
        }],
    )

    # ------------------------------------------------------------------ #
    # 5. RViz2
    # ------------------------------------------------------------------ #
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", _rviz_config],
        parameters=[{"use_sim_time": use_sim_time}],
    )

    # ------------------------------------------------------------------ #
    # 6. Semantic map manager – ML venv handling
    #    share path: .../install/<pkg>/share/<pkg>
    #    5 × dirname  →  project root (.../visual_semantic_navigation/)
    # ------------------------------------------------------------------ #
    _project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_smm_share))))
    )
    _venv_site = os.path.join(
        _project_root, ".venv-1", "lib", "python3.12", "site-packages"
    )
    _extra_env = {
        "PYTHONPATH": _venv_site + ":" + os.environ.get("PYTHONPATH", "")
    }

    retrieval_config = os.path.join(_smm_share, "config", "retrieval_config.yaml")

    visual_encoder_node = Node(
        package="semantic_map_manager",
        executable="visual_encoder",
        name="visual_encoder",
        parameters=[retrieval_config, {"use_sim_time": use_sim_time}],
        additional_env=_extra_env,
        output="screen",
    )

    kg_manager_node = Node(
        package="semantic_map_manager",
        executable="kg_manager",
        name="kg_manager",
        parameters=[retrieval_config, {"use_sim_time": use_sim_time}],
        additional_env=_extra_env,
        output="screen",
    )

    orchestrator_node = Node(
        package="semantic_map_manager",
        executable="semantic_orchestrator",
        name="semantic_orchestrator",
        parameters=[retrieval_config, {"use_sim_time": use_sim_time}],
        additional_env=_extra_env,
        output="screen",
    )

    eval_node = Node(
        package="semantic_map_manager",
        executable="evaluation_node",
        name="evaluation_node",
        parameters=[{"use_sim_time": use_sim_time}],
        output="screen",
    )

    knowledge_graph_db_node = Node(
        package="knowledge_graph_db",
        executable="knowledge_graph_db_node",
        name="knowledge_graph_db",
        output="screen",
    )

    # ------------------------------------------------------------------ #
    # Launch description
    # ------------------------------------------------------------------ #
    return LaunchDescription([
        # Arguments
        use_sim_time_arg,
        map_arg,
        world_arg,
        # Environment
        set_tb3_model,
        set_gz_resource_path,
        # 1. Simulation
        gazebo,
        robot_state_publisher,
        spawn_turtlebot3,
        # 2. Localization
        map_server,
        amcl,
        lifecycle_manager_localization,
        # 3. Navigation
        controller_server,
        smoother_server,
        planner_server,
        route_server,
        behavior_server,
        bt_navigator,
        waypoint_follower,
        velocity_smoother,
        collision_monitor,
        docking_server,
        lifecycle_manager_navigation,
        # 4. Visualization
        rviz,
        # 5. Semantic layer
        knowledge_graph_db_node,
        visual_encoder_node,
        kg_manager_node,
        orchestrator_node,
        eval_node,
    ])
