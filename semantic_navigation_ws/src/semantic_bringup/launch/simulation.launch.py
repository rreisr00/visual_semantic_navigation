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
  scene_config  (default: empty) bundled scene name or path to a scene YAML;
                explicit CLI launch arguments override YAML values
  start_operator_gui (default: false) start the semantic mapping console
  use_sim_time   (default: true)
  map            (default: aws_robomaker_small_house_world bundled map)
  world          (default: semantic_bringup/worlds/semantic_test.world)
  camera_mast_z  (default: 0.45) camera height [m]; the robot is spawned from
                 the vendored turtlebot3_waffle_semantic model, not the stock
                 turtlebot3_gazebo one
  world_name     (default: "default") the <world name=...> inside the .world
                 file — every stock world here uses "default"; needed for the
                 gz set_pose service path
"""

import glob
import math
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
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.logging import get_logger
from launch.substitutions import (
    IfElseSubstitution,
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterFile
from launch_ros.substitutions import FindPackageShare
from nav2_common.launch import RewrittenYaml
from semantic_bringup.scene_config import apply_yaml_defaults, load_scene_launch_config
from semantic_navigation_core.configuration import load_frozen_config


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

    _default_params_file = os.path.join(
        _bringup_share, "config", "nav2_params.yaml"
    )
    _default_rviz_config = os.path.join(
        _bringup_share, "config", "rviz_config.rviz"
    )
    _default_retrieval_config = os.path.join(
        _snr_share, "config", "retrieval_config.yaml"
    )
    _default_mapping_config = os.path.join(
        _snr_share, "config", "mapping_config.yaml"
    )
    _default_vision_config = os.path.join(
        _svr_share, "config", "vision_config.yaml"
    )
    _default_kg_config = os.path.join(_kgr_share, "config", "kg_config.yaml")
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
    scene_config_arg = DeclareLaunchArgument(
        "scene_config",
        default_value="",
        description=(
            "Bundled scene name (for example aws_small_house) or path to a "
            "scene YAML. Explicit launch arguments override YAML values."
        ),
    )
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
    spawn_x_arg = DeclareLaunchArgument("spawn_x", default_value="0.0")
    spawn_y_arg = DeclareLaunchArgument("spawn_y", default_value="0.0")
    spawn_yaw_arg = DeclareLaunchArgument("spawn_yaw", default_value="0.0")
    world_name_arg = DeclareLaunchArgument(
        "world_name", default_value="default",
        description="The <world name=...> declared inside the .world file — "
                    "used for gz service paths like /world/<name>/set_pose.",
    )
    scene_id_arg = DeclareLaunchArgument(
        "scene_id", default_value="aws_small_house",
        description="Stable scene identifier used to isolate graph persistence.",
    )
    graph_database_arg = DeclareLaunchArgument(
        "graph_database", default_value="~/.ros/semantic_maps/{scene_id}/graph.db",
        description="SQLite graph path; {scene_id} is expanded by the bridge.",
    )
    start_semantic_arg = DeclareLaunchArgument(
        "start_semantic", default_value="true",
        description="Start ML, graph, capture and semantic navigation nodes.",
    )
    start_rviz_arg = DeclareLaunchArgument(
        "start_rviz", default_value="true",
        description="Start RViz2.",
    )
    start_operator_gui_arg = DeclareLaunchArgument(
        "start_operator_gui", default_value="false",
        description="Start the camera, teleoperation and semantic mapping GUI.",
    )
    start_auto_mapping_arg = DeclareLaunchArgument(
        "start_auto_mapping", default_value="false",
        description="Create semantic nodes automatically from odometry motion.",
    )
    headless_arg = DeclareLaunchArgument(
        "headless", default_value="false",
        description="Run only the Gazebo server (no graphical client).",
    )
    localization_mode_arg = DeclareLaunchArgument(
        "localization_mode", default_value="localization",
        description="'localization' uses map_server+AMCL; 'slam' uses slam_toolbox.",
    )
    nav2_params_file_arg = DeclareLaunchArgument(
        "nav2_params_file",
        default_value=_default_params_file,
        description="Nav2 ROS parameter YAML.",
    )
    rviz_config_file_arg = DeclareLaunchArgument(
        "rviz_config_file",
        default_value=_default_rviz_config,
        description="RViz configuration file.",
    )
    vision_params_file_arg = DeclareLaunchArgument(
        "vision_params_file",
        default_value=_default_vision_config,
        description="Visual encoder ROS parameter YAML.",
    )
    knowledge_graph_params_file_arg = DeclareLaunchArgument(
        "knowledge_graph_params_file",
        default_value=_default_kg_config,
        description="Knowledge graph bridge ROS parameter YAML.",
    )
    retrieval_params_file_arg = DeclareLaunchArgument(
        "retrieval_params_file",
        default_value=_default_retrieval_config,
        description="Semantic retrieval ROS parameter YAML.",
    )
    mapping_params_file_arg = DeclareLaunchArgument(
        "mapping_params_file",
        default_value=_default_mapping_config,
        description="Automatic topology mapping ROS parameter YAML.",
    )

    def _apply_scene_config(context):
        reference = LaunchConfiguration("scene_config").perform(context).strip()
        if not reference:
            return []
        config_path, yaml_values = load_scene_launch_config(
            reference, _bringup_share
        )
        applied = apply_yaml_defaults(context.launch_configurations, yaml_values)
        overridden = sorted(set(yaml_values) - set(applied))
        logger = get_logger("semantic_bringup.scene_config")
        logger.info(
            f"Loaded scene configuration '{config_path}'"
            + (f" (applied: {', '.join(sorted(applied))})" if applied else "")
        )
        if overridden:
            logger.info(
                "Explicit launch values override YAML fields: "
                + ", ".join(overridden)
            )
        return []

    load_scene_config = OpaqueFunction(function=_apply_scene_config)

    use_sim_time = LaunchConfiguration("use_sim_time")
    map_yaml     = LaunchConfiguration("map")
    world_file   = LaunchConfiguration("world")
    scene_id     = LaunchConfiguration("scene_id")
    graph_database = LaunchConfiguration("graph_database")
    start_semantic = LaunchConfiguration("start_semantic")
    headless = LaunchConfiguration("headless")
    localization_mode = LaunchConfiguration("localization_mode")
    params_file = LaunchConfiguration("nav2_params_file")
    rviz_config = LaunchConfiguration("rviz_config_file")
    vision_config = LaunchConfiguration("vision_params_file")
    kg_config = LaunchConfiguration("knowledge_graph_params_file")
    retrieval_config = LaunchConfiguration("retrieval_params_file")
    mapping_config = LaunchConfiguration("mapping_params_file")
    localization_condition = IfCondition(PythonExpression([
        "'", localization_mode, "' == 'localization'"
    ]))
    slam_condition = IfCondition(PythonExpression([
        "'", localization_mode, "' == 'slam'"
    ]))

    # ------------------------------------------------------------------ #
    # Nav2 parameter file
    # RewrittenYaml injects use_sim_time and BT paths at launch time so
    # every node receives correct values without hardcoding in the YAML.
    # ------------------------------------------------------------------ #
    remappings = [("/tf", "tf"), ("/tf_static", "tf_static")]

    configured_params = ParameterFile(
        RewrittenYaml(
            source_file=params_file,
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
    # Append (never overwrite): a pre-set GZ_SIM_RESOURCE_PATH lets users add
    # model dirs for extra worlds (e.g. cloned AWS bookstore/warehouse), the
    # package-local models are used by semantic_office_lab, and the
    # turtlebot3_gazebo models are needed by the turtlebot3_* worlds.
    set_gz_resource_path = SetEnvironmentVariable(
        "GZ_SIM_RESOURCE_PATH",
        os.pathsep.join(p for p in [
            os.environ.get("GZ_SIM_RESOURCE_PATH", ""),
            os.path.join(_bringup_share, "models"),
            os.path.join(_aws_share, "models"),
            os.path.join(_tb3_share, "models"),
        ] if p),
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
        launch_arguments={
            "gz_args": [IfElseSubstitution(headless, "-s -r ", "-r "), world_file]
        }.items(),
    )

    # ------------------------------------------------------------------ #
    # 2. TurtleBot3
    # ------------------------------------------------------------------ #
    def _robot_state_publisher(context):
        camera_z = float(LaunchConfiguration("camera_mast_z").perform(context))
        urdf_path = os.path.join(_tb3_share, "urdf", "turtlebot3_waffle.urdf")
        with open(urdf_path, encoding="utf-8") as stream:
            robot_description = stream.read()
        stock = '<origin xyz="0.064 -0.065 0.094" rpy="0 0 0"/>'
        raised = f'<origin xyz="0.064 -0.065 {camera_z - 0.013:.4f}" rpy="0 0 0"/>'
        if stock not in robot_description:
            raise RuntimeError("TurtleBot3 camera_joint origin not found in URDF")
        robot_description = robot_description.replace(stock, raised, 1)
        return [Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            output="screen",
            parameters=[{
                "use_sim_time": use_sim_time,
                "robot_description": robot_description,
            }],
        )]

    robot_state_publisher = OpaqueFunction(function=_robot_state_publisher)

    depth_optical_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="depth_optical_tf",
        arguments=[
            "--x", "0", "--y", "0", "--z", "0",
            "--qx", "0", "--qy", "0", "--qz", "0", "--qw", "1",
            "--frame-id", "camera_rgb_optical_frame",
            "--child-frame-id", "camera_depth_optical_frame",
        ],
        parameters=[{"use_sim_time": use_sim_time}],
    )

    # The stock spawn_turtlebot3.launch.py hardcodes the system waffle SDF, so
    # we replicate its three nodes (create + parameter_bridge + image_bridge)
    # and spawn our vendored model with the camera raised to `camera_mast_z`.
    # The vendored SDF differs from the stock one only in camera height
    # (placeholder-substituted below), 640x480 resolution and a mast visual.
    def _spawn_robot(context):
        world_name = LaunchConfiguration("world_name").perform(context)
        camera_z = float(LaunchConfiguration("camera_mast_z").perform(context))
        spawn_x = float(LaunchConfiguration("spawn_x").perform(context))
        spawn_y = float(LaunchConfiguration("spawn_y").perform(context))
        spawn_yaw = float(LaunchConfiguration("spawn_yaw").perform(context))
        mast_bottom = 0.107          # stock camera height ≈ robot top plate
        mast_length = max(camera_z - mast_bottom, 0.001)
        sdf_path = os.path.join(
            _bringup_share, "models", "turtlebot3_waffle_semantic", "model.sdf"
        )
        with open(sdf_path, encoding="utf-8") as f:
            sdf = f.read()
        sdf = (
            sdf.replace("@CAMERA_Z@", f"{camera_z:.4f}")
               .replace("@CAMERA_LINK_Z@", f"{camera_z - 0.013:.4f}")
               .replace("@MAST_LENGTH@", f"{mast_length:.4f}")
               .replace("@MAST_CENTER_Z@", f"{-mast_length / 2.0:.4f}")
        )
        create_robot = Node(
            package="ros_gz_sim",
            executable="create",
            arguments=[
                "-name", "waffle",          # reset_robot_pose targets this name
                "-string", sdf,
                "-x", str(spawn_x), "-y", str(spawn_y), "-z", "0.01",
                "-Y", str(spawn_yaw),
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
                "-s", f"/world/{world_name}/set_pose",
                "--reqtype", "gz.msgs.Pose",
                "--reptype", "gz.msgs.Boolean",
                "--timeout", "2000",
                "--req",
                f"name: 'waffle' position: {{x: {spawn_x} y: {spawn_y} z: 0.01}}"
                f" orientation: {{z: {math.sin(spawn_yaw / 2.0)} "
                f"w: {math.cos(spawn_yaw / 2.0)}}}",
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
        arguments=["/camera/image_raw", "/camera/depth/image_raw"],
        output="screen",
    )

    depth_camera_info_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/camera/depth/camera_info"
            "@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo"
        ],
        output="screen",
    )

    def _set_pose_service_bridge(context):
        world_name = LaunchConfiguration("world_name").perform(context)
        return [Node(
            package="ros_gz_bridge",
            executable="parameter_bridge",
            arguments=[
                f"/world/{world_name}/set_pose"
                "@ros_gz_interfaces/srv/SetEntityPose"
            ],
            output="screen",
        )]

    set_pose_service_bridge = OpaqueFunction(function=_set_pose_service_bridge)

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
        condition=localization_condition,
    )

    amcl = Node(
        package="nav2_amcl",
        executable="amcl",
        name="amcl",
        output="screen",
        parameters=[configured_params],
        remappings=remappings,
        condition=localization_condition,
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
        condition=localization_condition,
    )

    slam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("slam_toolbox"), "launch", "online_async_launch.py"]
            )
        ),
        launch_arguments={"use_sim_time": use_sim_time}.items(),
        condition=slam_condition,
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
        arguments=["-d", rviz_config],
        parameters=[{"use_sim_time": use_sim_time}],    
        condition=IfCondition(LaunchConfiguration("start_rviz")),
    )

    # ------------------------------------------------------------------ #
    # 6. Semantic layer – ML venv handling
    # ------------------------------------------------------------------ #
    _venv_site = find_venv_site_packages(_snr_share)
    _extra_env = {
        "PYTHONPATH": _venv_site + ":" + os.environ.get("PYTHONPATH", "")
    }

    frozen_config_path = os.path.join(
        _snr_share, "config", "frozen_retrieval_config.yaml"
    )
    frozen_config, frozen_hash = load_frozen_config(frozen_config_path)
    repo_root = _bringup_share
    for _ in range(10):
        candidate = os.path.join(repo_root, "experiments", "yolov8n.pt")
        if os.path.isfile(candidate):
            break
        repo_root = os.path.dirname(repo_root)
    yolo_checkpoint = candidate
    aggregation = frozen_config["multiview_aggregation"]
    weights = frozen_config["retrieval_weights"]

    # visual_encoder and knowledge_graph_bridge are lifecycle nodes driven to
    # 'active' by our own semantic_navigation_ros lifecycle_manager, which also
    # recovers them if they self-deactivate (e.g. visual_encoder after CUDA OOM).
    visual_encoder_node = Node(
        package="semantic_vision_ros",
        executable="visual_encoder",
        name="visual_encoder",
        parameters=[vision_config, {
            "use_sim_time": use_sim_time,
            "siglip_model_id": frozen_config["siglip_checkpoint"],
            "yolo_model_path": yolo_checkpoint,
            "local_files_only": bool(
                frozen_config["preprocessing"].get("local_files_only", True)
            ),
        }],
        additional_env=_extra_env,
        output="screen",
        condition=IfCondition(start_semantic),
    )

    # knowledge_graph_bridge owns both the /store_waypoint + /get_waypoints
    # services and the SQLite persistence layer.
    kg_bridge_node = Node(
        package="knowledge_graph_ros",
        executable="knowledge_graph_bridge",
        name="knowledge_graph_bridge",
        parameters=[kg_config, {
            "use_sim_time": use_sim_time,
            "scene_id": scene_id,
            "db_file_path": graph_database,
            "configuration_hash": frozen_hash,
        }],
        output="screen",
        condition=IfCondition(start_semantic),
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
        condition=IfCondition(start_semantic),
    )

    kg_manager_node = Node(
        package="semantic_navigation_ros",
        executable="kg_manager",
        name="kg_manager",
        parameters=[{
            "use_sim_time": use_sim_time,
            "scene_id": scene_id,
            "configuration_hash": frozen_hash,
        }],
        output="screen",
        condition=IfCondition(start_semantic),
    )

    orchestrator_node = Node(
        package="semantic_navigation_ros",
        executable="semantic_orchestrator",
        name="semantic_orchestrator",
        parameters=[retrieval_config, {
            "use_sim_time": use_sim_time,
            "scene_id": scene_id,
            "retrieval_method": frozen_config["retrieval_method"],
            "multiview_aggregation": aggregation["method"],
            "multiview_top_k": int(aggregation["top_k"]),
            "global_similarity_weight": float(weights["global_similarity"]),
            "object_match_weight": float(weights["object_match"]),
            "crop_similarity_weight": float(weights["crop_similarity"]),
            "relation_match_weight": float(weights["relation_match"]),
            "room_match_weight": float(weights["room_match"]),
            "rejection_threshold": float(frozen_config["rejection_threshold"]),
        }],
        additional_env=_extra_env,
        output="screen",
        condition=IfCondition(start_semantic),
    )
    graph_visualizer_node = Node(
        package="semantic_evaluation",
        executable="graph_visualizer",
        name="graph_visualizer",
        parameters=[{"use_sim_time": use_sim_time}],
        output="screen",
        condition=IfCondition(start_semantic),
    )
    topology_mapper_node = Node(
        package="semantic_navigation_ros",
        executable="topology_mapper",
        name="topology_mapper",
        parameters=[mapping_config, {
            "use_sim_time": use_sim_time,
            "scene_id": scene_id,
        }],
        output="screen",
        condition=IfCondition(LaunchConfiguration("start_auto_mapping")),
    )
    operator_gui_node = Node(
        package="semantic_evaluation",
        executable="semantic_operator_gui",
        name="semantic_operator_gui",
        parameters=[{
            "use_sim_time": use_sim_time,
            "scene_id": scene_id,
        }],
        output="screen",
        condition=IfCondition(LaunchConfiguration("start_operator_gui")),
    )

    # ------------------------------------------------------------------ #
    # Launch description
    # ------------------------------------------------------------------ #
    return LaunchDescription([
        # Arguments
        # Load the YAML before declaring other arguments. Include/CLI values
        # are already present in the context, so they keep highest precedence;
        # declarations below only fill values missing from both CLI and YAML.
        scene_config_arg,
        load_scene_config,
        use_sim_time_arg,
        map_arg,
        world_arg,
        camera_mast_arg,
        spawn_x_arg,
        spawn_y_arg,
        spawn_yaw_arg,
        world_name_arg,
        scene_id_arg,
        graph_database_arg,
        start_semantic_arg,
        start_rviz_arg,
        start_operator_gui_arg,
        start_auto_mapping_arg,
        headless_arg,
        localization_mode_arg,
        nav2_params_file_arg,
        rviz_config_file_arg,
        vision_params_file_arg,
        knowledge_graph_params_file_arg,
        retrieval_params_file_arg,
        mapping_params_file_arg,
        # Environment
        set_tb3_model,
        set_gz_resource_path,
        # 1. Simulation
        gazebo,
        robot_state_publisher,
        depth_optical_tf,
        spawn_turtlebot3,          # also chains the post-spawn pose reset
        tb3_bridge,
        camera_image_bridge,
        depth_camera_info_bridge,
        set_pose_service_bridge,
        # 2. Localization
        map_server,
        amcl,
        lifecycle_manager_localization,
        slam,
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
        graph_visualizer_node,
        topology_mapper_node,
        operator_gui_node,
    ])
