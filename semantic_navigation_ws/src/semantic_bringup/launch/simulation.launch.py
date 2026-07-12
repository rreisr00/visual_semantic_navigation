#!/usr/bin/env python3
"""
Bringup launch for the visual-semantic navigation project.

Starts:
  1. Gazebo          – semantic_test.world via ros_gz_sim
  2. TurtleBot3      – robot_state_publisher + vendored waffle spawn (mast
                       camera) + gz<->ROS bridges; the post-spawn pose reset
                       is event-chained to the spawn process exit
  3. Localization    – map_server + amcl + lifecycle_manager_localization
  4. Navigation      – controller, smoother, planner, route_server, behavior,
                       bt_navigator, waypoint_follower, velocity_smoother,
                       collision_monitor, docking_server +
                       lifecycle_manager_navigation
  5. RViz2           – project config from this package
  6. Semantic layer  – visual_encoder, kg_manager, semantic_orchestrator,
                       knowledge_graph_bridge
                       (metrics: use semantic_evaluation's evaluation_collector)

Launch arguments
----------------
  use_sim_time   (default: true)
  map            (default: aws_robomaker_small_house_world bundled map)
  world          (default: semantic_bringup/worlds/semantic_test.world)
  camera_mast_z  (default: 0.45) camera height [m]; the robot is spawned from
                 the vendored turtlebot3_waffle_semantic model, not the stock
                 turtlebot3_gazebo one
"""

import glob
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
    SetEnvironmentVariable,
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterFile
from launch_ros.substitutions import FindPackageShare
from nav2_common.launch import RewrittenYaml


def find_venv_site_packages(start_dir: str) -> str:
    """Locate the ML venv's site-packages for PYTHONPATH injection.

    Honours the SEMANTIC_NAV_VENV_SITE env var; otherwise walks up from
    ``start_dir`` (an installed share path) until a ``.venv-1`` appears,
    globbing the python minor version so a 3.13 rebuild keeps working.
    """
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

    # ------------------------------------------------------------------ #
    # Static paths (resolved at parse time)
    # ------------------------------------------------------------------ #
    _bringup_share = get_package_share_directory("semantic_bringup")
    _aws_share     = get_package_share_directory("aws_robomaker_small_house_world")
    _snr_share     = get_package_share_directory("semantic_navigation_ros")
    _svr_share     = get_package_share_directory("semantic_vision_ros")
    _kgr_share     = get_package_share_directory("knowledge_graph_ros")
    _tb3_share     = get_package_share_directory("turtlebot3_gazebo")

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
    camera_mast_arg = DeclareLaunchArgument(
        "camera_mast_z", default_value="0.45",
        description="Camera height above base_footprint [m] (stock waffle: 0.107).",
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

    # The stock spawn_turtlebot3.launch.py hardcodes the system waffle SDF, so
    # we replicate its three nodes (create + parameter_bridge + image_bridge)
    # and spawn our vendored model with the camera raised to `camera_mast_z`.
    # The vendored SDF differs from the stock one only in camera height
    # (placeholder-substituted below), 640x480 resolution and a mast visual.
    # TF caveat: robot_state_publisher keeps the stock URDF camera height
    # (~0.12 m); fine while nothing consumes camera-frame TF.
    def _spawn_robot(context):
        camera_z = float(LaunchConfiguration("camera_mast_z").perform(context))
        mast_bottom = 0.107          # stock camera height ≈ robot top plate
        mast_length = max(camera_z - mast_bottom, 0.001)
        sdf_path = os.path.join(
            _bringup_share, "models", "turtlebot3_waffle_semantic", "model.sdf"
        )
        with open(sdf_path, encoding="utf-8") as f:
            sdf = f.read()
        sdf = (
            sdf.replace("@CAMERA_Z@", f"{camera_z:.4f}")
               .replace("@MAST_LENGTH@", f"{mast_length:.4f}")
               .replace("@MAST_CENTER_Z@", f"{-mast_length / 2.0:.4f}")
        )
        create_robot = Node(
            package="ros_gz_sim",
            executable="create",
            arguments=[
                "-name", "waffle",          # reset_robot_pose targets this name
                "-string", sdf,
                "-x", "0.0", "-y", "0.0", "-z", "0.01",
            ],
            output="screen",
        )
        # Reset the robot pose to (0, 0) once `create` has exited (i.e. the
        # entity exists). Guards against a stale Gazebo session where the
        # model sits at its previous position while AMCL re-initialises at
        # the origin. Event-chained instead of a fixed timer so it can never
        # fire before the spawn.
        reset_robot_pose = ExecuteProcess(
            cmd=[
                "gz", "service",
                "-s", "/world/semantic_test/set_pose",
                "--reqtype", "gz.msgs.Pose",
                "--reptype", "gz.msgs.Boolean",
                "--timeout", "2000",
                "--req",
                "name: 'waffle' position: {x: 0.0 y: 0.0 z: 0.01}"
                " orientation: {w: 1.0}",
            ],
            output="screen",
        )
        return [
            create_robot,
            RegisterEventHandler(
                OnProcessExit(
                    target_action=create_robot,
                    on_exit=[reset_robot_pose],
                )
            ),
        ]

    spawn_turtlebot3 = OpaqueFunction(function=_spawn_robot)

    # gz<->ROS bridges normally started by spawn_turtlebot3.launch.py
    # (clock, odom, tf, cmd_vel, imu, scan, joint_states, camera_info).
    tb3_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        parameters=[{
            "config_file": os.path.join(
                _tb3_share, "params", "turtlebot3_waffle_bridge.yaml"
            ),
        }],
        output="screen",
    )

    camera_image_bridge = Node(
        package="ros_gz_image",
        executable="image_bridge",
        arguments=["/camera/image_raw"],
        output="screen",
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
    # 6. Semantic layer – ML venv handling
    # ------------------------------------------------------------------ #
    _venv_site = find_venv_site_packages(_snr_share)
    _extra_env = {
        "PYTHONPATH": _venv_site + ":" + os.environ.get("PYTHONPATH", "")
    }

    retrieval_config = os.path.join(_snr_share, "config", "retrieval_config.yaml")
    vision_config    = os.path.join(_svr_share, "config", "vision_config.yaml")
    kg_config        = os.path.join(_kgr_share, "config", "kg_config.yaml")

    # visual_encoder and knowledge_graph_bridge are lifecycle nodes driven to
    # 'active' by our own semantic_navigation_ros lifecycle_manager, which also
    # recovers them if they self-deactivate (e.g. visual_encoder after CUDA OOM).
    visual_encoder_node = Node(
        package="semantic_vision_ros",
        executable="visual_encoder",
        name="visual_encoder",
        parameters=[vision_config, {"use_sim_time": use_sim_time}],
        additional_env=_extra_env,
        output="screen",
    )

    # knowledge_graph_bridge owns both the /store_waypoint + /get_waypoints
    # services and the SQLite persistence layer.
    kg_bridge_node = Node(
        package="knowledge_graph_ros",
        executable="knowledge_graph_bridge",
        name="knowledge_graph_bridge",
        parameters=[kg_config, {"use_sim_time": use_sim_time}],
        output="screen",
    )

    semantic_lifecycle_manager = Node(
        package="semantic_navigation_ros",
        executable="lifecycle_manager",
        name="lifecycle_manager",
        output="screen",
        parameters=[{
            "use_sim_time":  use_sim_time,
            "managed_nodes": ["visual_encoder", "knowledge_graph_bridge"],
        }],
    )

    kg_manager_node = Node(
        package="semantic_navigation_ros",
        executable="kg_manager",
        name="kg_manager",
        parameters=[{"use_sim_time": use_sim_time}],
        output="screen",
    )

    orchestrator_node = Node(
        package="semantic_navigation_ros",
        executable="semantic_orchestrator",
        name="semantic_orchestrator",
        parameters=[retrieval_config, {"use_sim_time": use_sim_time}],
        additional_env=_extra_env,
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
        camera_mast_arg,
        # Environment
        set_tb3_model,
        set_gz_resource_path,
        # 1. Simulation
        gazebo,
        robot_state_publisher,
        spawn_turtlebot3,          # also chains the post-spawn pose reset
        tb3_bridge,
        camera_image_bridge,
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
        visual_encoder_node,
        kg_bridge_node,
        semantic_lifecycle_manager,
        kg_manager_node,
        orchestrator_node,
    ])
