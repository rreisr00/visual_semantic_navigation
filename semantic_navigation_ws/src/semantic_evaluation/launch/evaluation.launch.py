"""Launch the knowledge-graph visualizer + RViz, and optionally the campaign.

Always starts:
  * graph_visualizer  (publishes the latched MarkerArray)
  * rviz2             (pre-configured to show that MarkerArray)

Conditionally starts (``run_collector:=true``):
  * evaluation_collector  (runs the test suite and writes the CSV)

Examples
--------
  ros2 launch semantic_evaluation evaluation.launch.py
  ros2 launch semantic_evaluation evaluation.launch.py run_collector:=true \\
      test_suite_path:=/abs/path/test_suite.yaml decision_only:=true
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory("semantic_evaluation")
    default_params = os.path.join(pkg_share, "config", "evaluation_params.yaml")
    default_rviz = os.path.join(pkg_share, "rviz", "semantic_graph.rviz")

    params_file = LaunchConfiguration("params_file")
    rviz_config = LaunchConfiguration("rviz_config")
    run_collector = LaunchConfiguration("run_collector")
    test_suite_path = LaunchConfiguration("test_suite_path")
    decision_only = LaunchConfiguration("decision_only")
    use_rviz = LaunchConfiguration("use_rviz")
    campaign_id = LaunchConfiguration("campaign_id")
    scene_id = LaunchConfiguration("scene_id")
    run_id = LaunchConfiguration("run_id")
    seed = LaunchConfiguration("seed")
    method = LaunchConfiguration("method")
    start_pose_id = LaunchConfiguration("start_pose_id")
    query_suite_id = LaunchConfiguration("query_suite_id")
    frozen_config_hash = LaunchConfiguration("frozen_config_hash")
    success_semantics = LaunchConfiguration("success_semantics")
    start_poses_path = LaunchConfiguration("start_poses_path")
    reset_pose_service = LaunchConfiguration("reset_pose_service")

    args = [
        DeclareLaunchArgument("params_file", default_value=default_params),
        DeclareLaunchArgument("rviz_config", default_value=default_rviz),
        DeclareLaunchArgument("use_rviz", default_value="true"),
        DeclareLaunchArgument("run_collector", default_value="false"),
        DeclareLaunchArgument("test_suite_path", default_value=""),
        DeclareLaunchArgument("decision_only", default_value="false"),
        DeclareLaunchArgument("campaign_id", default_value=""),
        DeclareLaunchArgument("scene_id", default_value="scene_unset"),
        DeclareLaunchArgument("run_id", default_value=""),
        DeclareLaunchArgument("seed", default_value="42"),
        DeclareLaunchArgument("method", default_value="single_view_siglip"),
        DeclareLaunchArgument("start_pose_id", default_value="start_pose_unset"),
        DeclareLaunchArgument("query_suite_id", default_value=""),
        DeclareLaunchArgument("frozen_config_hash", default_value=""),
        DeclareLaunchArgument("success_semantics", default_value=""),
        DeclareLaunchArgument("start_poses_path", default_value=""),
        DeclareLaunchArgument(
            "reset_pose_service", default_value="/world/default/set_pose"
        ),
    ]

    visualizer = Node(
        package="semantic_evaluation",
        executable="graph_visualizer",
        name="graph_visualizer",
        parameters=[params_file],
        output="screen",
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", rviz_config],
        condition=IfCondition(use_rviz),
        output="screen",
    )

    collector = Node(
        package="semantic_evaluation",
        executable="evaluation_collector",
        name="evaluation_collector",
        parameters=[
            params_file,
            {
                "test_suite_path": test_suite_path,
                "decision_only": decision_only,
                "campaign_id": campaign_id,
                "scene_id": scene_id,
                "run_id": run_id,
                "seed": seed,
                "method": method,
                "start_pose_id": start_pose_id,
                "query_suite_id": query_suite_id,
                "frozen_config_hash": frozen_config_hash,
                "success_semantics": success_semantics,
                "start_poses_path": start_poses_path,
                "reset_pose_service": reset_pose_service,
            },
        ],
        condition=IfCondition(run_collector),
        output="screen",
    )

    return LaunchDescription([*args, visualizer, rviz, collector])
