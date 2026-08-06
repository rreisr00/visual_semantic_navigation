#!/usr/bin/env python3
"""
Bringup launch for the visual-semantic navigation project.

Starts:
  1. Gazebo          – semantic_test.world via ros_gz_sim
  2. Mobile robot    – robot_state_publisher + scene-selected RGB-D model +
                       gz<->ROS bridges; the post-spawn pose reset
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
  world_package  (default: aws_robomaker_small_house_world) package whose
                 models are added to GZ_SIM_RESOURCE_PATH
  robot_package/model select the Gazebo model declared by the scene
  camera_mast_z  (default: 1.05) RGB-D optical height above base_footprint [m]
  camera_pitch_rad (default: 0.0) camera pitch in radians
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
    TimerAction,
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
from launch_ros.parameter_descriptions import ParameterValue
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
    world_package_arg = DeclareLaunchArgument(
        "world_package",
        default_value="aws_robomaker_small_house_world",
        description="Resource package whose models Gazebo must discover.",
    )
    robot_package_arg = DeclareLaunchArgument(
        "robot_package",
        default_value="semantic_robot_description",
        description="Package containing models/<robot_model>/model.sdf.",
    )
    robot_model_arg = DeclareLaunchArgument(
        "robot_model",
        default_value="semantic_tall_rgbd",
        description="Gazebo model directory below the selected robot package.",
    )
    robot_entity_name_arg = DeclareLaunchArgument(
        "robot_entity_name",
        default_value="semantic_robot",
        description="Gazebo entity name used by spawn and reset operations.",
    )
    camera_mast_arg = DeclareLaunchArgument(
        "camera_mast_z", default_value="1.05",
        description="RGB-D camera height above base_footprint [m].",
    )
    camera_pitch_arg = DeclareLaunchArgument(
        "camera_pitch_rad", default_value="0.0",
        description="RGB-D camera pitch in radians (negative looks down).",
    )
    robot_gz_rgb_topic_arg = DeclareLaunchArgument(
        "robot_gz_rgb_topic", default_value="/camera/image",
        description="Gazebo RGB image topic produced by the selected model.",
    )
    robot_gz_depth_topic_arg = DeclareLaunchArgument(
        "robot_gz_depth_topic", default_value="/camera/depth_image",
        description="Gazebo registered depth image topic.",
    )
    robot_gz_camera_info_topic_arg = DeclareLaunchArgument(
        "robot_gz_camera_info_topic", default_value="/camera/camera_info",
        description="Gazebo camera calibration topic shared by RGB and depth.",
    )
    robot_bridge_config_arg = DeclareLaunchArgument(
        "robot_bridge_config",
        default_value=PathJoinSubstitution([
            FindPackageShare("semantic_robot_description"),
            "config", "base_bridge.yaml",
        ]),
        description="ros_gz_bridge YAML for the mobile base topics.",
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
    # Resolve the selected scene package after its YAML defaults have been
    # applied. This keeps model:// URIs scene-local while preserving paths
    # supplied by the user.
    def _set_gz_resource_path(context):
        package_name = LaunchConfiguration("world_package").perform(context).strip()
        robot_package_name = LaunchConfiguration("robot_package").perform(
            context
        ).strip()
        if not package_name:
            raise RuntimeError("world_package must not be empty")
        if not robot_package_name:
            raise RuntimeError("robot_package must not be empty")
        package_share = get_package_share_directory(package_name)
        robot_package_share = get_package_share_directory(robot_package_name)
        candidates = [
            *os.environ.get("GZ_SIM_RESOURCE_PATH", "").split(os.pathsep),
            os.path.join(_bringup_share, "models"),
            os.path.join(_aws_share, "models"),
            package_share,
            os.path.join(package_share, "models"),
            robot_package_share,
            os.path.join(robot_package_share, "models"),
            os.path.join(_tb3_share, "models"),
        ]
        paths = list(dict.fromkeys(path for path in candidates if path))
        return [SetEnvironmentVariable(
            "GZ_SIM_RESOURCE_PATH",
            os.pathsep.join(paths),
        )]

    set_gz_resource_path = OpaqueFunction(function=_set_gz_resource_path)

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
    # 2. Scene-selected mobile robot
    # ------------------------------------------------------------------ #
    def _robot_state_publisher(context):
        camera_z = float(LaunchConfiguration("camera_mast_z").perform(context))
        camera_pitch = float(
            LaunchConfiguration("camera_pitch_rad").perform(context)
        )
        urdf_path = os.path.join(_tb3_share, "urdf", "turtlebot3_waffle.urdf")
        with open(urdf_path, encoding="utf-8") as stream:
            robot_description = stream.read()
        stock = '<origin xyz="0.064 -0.065 0.094" rpy="0 0 0"/>'
        raised = (
            f'<origin xyz="0.064 -0.065 {camera_z - 0.013:.4f}" '
            f'rpy="0 {camera_pitch:.6f} 0"/>'
        )
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

    # Spawn the package/model selected by the scene after substituting the
    # sensor-tower geometry. The base remains TurtleBot3-compatible so the
    # stock URDF meshes and Nav2 tuning remain valid.
    def _spawn_robot(context):
        world_name = LaunchConfiguration("world_name").perform(context)
        robot_package = LaunchConfiguration("robot_package").perform(context).strip()
        robot_model = LaunchConfiguration("robot_model").perform(context).strip()
        entity_name = LaunchConfiguration("robot_entity_name").perform(context).strip()
        camera_z = float(LaunchConfiguration("camera_mast_z").perform(context))
        camera_pitch = float(
            LaunchConfiguration("camera_pitch_rad").perform(context)
        )
        spawn_x = float(LaunchConfiguration("spawn_x").perform(context))
        spawn_y = float(LaunchConfiguration("spawn_y").perform(context))
        spawn_yaw = float(LaunchConfiguration("spawn_yaw").perform(context))
        mast_bottom = 0.107          # stock camera height ≈ robot top plate
        mast_length = max(camera_z - mast_bottom, 0.001)
        mast_z = mast_bottom + mast_length / 2.0
        mast_mass = 0.15
        mast_radius = 0.012
        mast_ixx = mast_mass * (3.0 * mast_radius ** 2 + mast_length ** 2) / 12.0
        if not robot_package or not robot_model or not entity_name:
            raise RuntimeError(
                "robot_package, robot_model and robot_entity_name must not be empty"
            )
        robot_share = get_package_share_directory(robot_package)
        sdf_path = os.path.join(
            robot_share, "models", robot_model, "model.sdf"
        )
        if not os.path.isfile(sdf_path):
            raise RuntimeError(f"robot model SDF was not found: {sdf_path}")
        with open(sdf_path, encoding="utf-8") as f:
            sdf = f.read()
        sdf = (
            sdf.replace("@CAMERA_Z@", f"{camera_z:.4f}")
               .replace("@CAMERA_LINK_Z@", f"{camera_z - 0.013:.4f}")
               .replace("@CAMERA_PITCH@", f"{camera_pitch:.6f}")
               .replace("@MAST_LENGTH@", f"{mast_length:.4f}")
               .replace("@MAST_Z@", f"{mast_z:.4f}")
               .replace("@MAST_IXX@", f"{mast_ixx:.8f}")
        )
        if "@" in sdf:
            raise RuntimeError(
                f"unresolved placeholder remains in robot model '{sdf_path}'"
            )
        create_robot = Node(
            package="ros_gz_sim",
            executable="create",
            arguments=[
                "-name", entity_name,
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
                f"name: '{entity_name}' position: "
                f"{{x: {spawn_x} y: {spawn_y} z: 0.01}}"
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
                    # Gazebo loads the office meshes and RGB-D renderer while
                    # the entity is being created. Starting every lifecycle
                    # transition at the same time can make DDS service replies
                    # miss their timeout and leave only part of Nav2 active.
                    # Start navigation once odom / scan / TF publishers exist,
                    # then load the semantic ML stack after Nav2 has settled.
                    on_exit=[
                        reset_robot_pose,
                        nav2_after_spawn,
                        semantic_after_spawn,
                    ],
                )
            ),
        ]

    spawn_robot = OpaqueFunction(function=_spawn_robot)

    # Base bridges are robot-configurable; RGB-D topics are bridged separately
    # so every scene exposes one stable ROS camera API.
    robot_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        parameters=[{
            "config_file": LaunchConfiguration("robot_bridge_config"),
        }],
        output="screen",
    )

    def _camera_bridges(context):
        rgb_topic = LaunchConfiguration("robot_gz_rgb_topic").perform(context)
        depth_topic = LaunchConfiguration("robot_gz_depth_topic").perform(context)
        info_topic = LaunchConfiguration(
            "robot_gz_camera_info_topic"
        ).perform(context)
        if not rgb_topic or not depth_topic or not info_topic:
            raise RuntimeError("robot Gazebo camera topics must not be empty")
        image_bridge = Node(
            package="ros_gz_bridge",
            executable="parameter_bridge",
            arguments=[
                f"{rgb_topic}@sensor_msgs/msg/Image[gz.msgs.Image",
                f"{depth_topic}@sensor_msgs/msg/Image[gz.msgs.Image",
            ],
            remappings=[
                (rgb_topic, "/camera/image_raw"),
                (depth_topic, "/camera/depth/image_raw"),
            ],
            output="screen",
        )
        camera_info_bridge = Node(
            package="ros_gz_bridge",
            executable="parameter_bridge",
            arguments=[
                f"{info_topic}@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo"
            ],
            remappings=[(info_topic, "/camera/camera_info")],
            output="screen",
        )
        depth_camera_info_bridge = Node(
            package="ros_gz_bridge",
            executable="parameter_bridge",
            arguments=[
                f"{info_topic}@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo"
            ],
            remappings=[(info_topic, "/camera/depth/camera_info")],
            output="screen",
        )
        return [image_bridge, camera_info_bridge, depth_camera_info_bridge]

    camera_bridges = OpaqueFunction(function=_camera_bridges)

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
        parameters=[configured_params, {
            # Keep localization aligned with the actual Gazebo spawn pose.
            # Scene YAML values are strings at launch time, so preserve the
            # parameter type explicitly instead of relying on inference.
            "initial_pose.x": ParameterValue(
                LaunchConfiguration("spawn_x"), value_type=float
            ),
            "initial_pose.y": ParameterValue(
                LaunchConfiguration("spawn_y"), value_type=float
            ),
            "initial_pose.yaw": ParameterValue(
                LaunchConfiguration("spawn_yaw"), value_type=float
            ),
        }],
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
            # Capture rotation uses the final Gazebo command topic while the
            # collision monitor is temporarily suspended below.  This avoids
            # both its approach limiter and competing zero-velocity outputs.
            "cmd_vel_topic": "/cmd_vel",
            # The approach polygon can reject an in-place mapping rotation
            # close to furniture.  kg_manager suspends it only while rotating
            # and restores it before acquiring each observation.
            "suspend_collision_monitor_during_rotation": True,
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
            "cmd_vel_topic": "/cmd_vel_nav",
        }],
        additional_env=_extra_env,
        output="screen",
        condition=IfCondition(LaunchConfiguration("start_operator_gui")),
    )

    # Lifecycle service calls are sensitive to the CPU / GPU spike produced
    # while Gazebo loads a world and creates the RGB-D sensor. Event-chain both
    # stacks to the successful robot spawn instead of racing that initial load.
    nav2_after_spawn = TimerAction(
        period=2.0,
        actions=[
            map_server,
            amcl,
            lifecycle_manager_localization,
            slam,
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
        ],
    )
    semantic_after_spawn = TimerAction(
        period=10.0,
        actions=[
            visual_encoder_node,
            kg_bridge_node,
            semantic_lifecycle_manager,
            kg_manager_node,
            orchestrator_node,
            graph_visualizer_node,
            topology_mapper_node,
        ],
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
        world_package_arg,
        robot_package_arg,
        robot_model_arg,
        robot_entity_name_arg,
        camera_mast_arg,
        camera_pitch_arg,
        robot_gz_rgb_topic_arg,
        robot_gz_depth_topic_arg,
        robot_gz_camera_info_topic_arg,
        robot_bridge_config_arg,
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
        spawn_robot,               # also chains the post-spawn pose reset
        robot_bridge,
        camera_bridges,
        set_pose_service_bridge,
        # 2-3. Localization and navigation are event-chained by spawn_robot.
        # 4. Visualization
        rviz,
        # 5. Semantic layer is also event-chained after the robot spawn.
        operator_gui_node,
    ])
