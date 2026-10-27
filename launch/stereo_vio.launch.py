# =============================================================================
# File: launch/stereo_vio.launch.py
#
# Description:
#   Main launch file for the Stereo VIO system.
#   Launches stereo_vio_node with parameters loaded from config/params.yaml.
#
#   Usage:
#     ros2 launch stereo_vio stereo_vio.launch.py
#
#   Optional overrides:
#     ros2 launch stereo_vio stereo_vio.launch.py \
#         bag:=/path/to/euroc.bag \
#         rviz:=true
#
# References:
#   [1] ROS 2 Humble launch documentation
#       https://docs.ros.org/en/humble/Tutorials/Intermediate/Launch/Launch-Main.html
#   [2] EuRoC MAV dataset
#       https://rpg.ifi.uzh.ch/docs/IJRR17_Burri.html
# =============================================================================

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    # ------------------------------------------------------------------
    # Package directories
    # ------------------------------------------------------------------
    pkg_share = get_package_share_directory("stereo_vio")
    default_params = os.path.join(pkg_share, "config", "params.yaml")
    default_rviz   = os.path.join(pkg_share, "config", "stereo_vio.rviz")

    # ------------------------------------------------------------------
    # Launch arguments
    # ------------------------------------------------------------------
    arg_params = DeclareLaunchArgument(
        "params",
        default_value=default_params,
        description="Full path to the params.yaml file"
    )

    arg_rviz = DeclareLaunchArgument(
        "rviz",
        default_value="false",
        description="Launch RViz2 for visualisation (true/false)"
    )

    arg_bag = DeclareLaunchArgument(
        "bag",
        default_value="",
        description="Full path to a ROS 2 bag file (leave empty to skip)"
    )

    arg_bag_rate = DeclareLaunchArgument(
        "bag_rate",
        default_value="1.0",
        description="Playback rate for bag file (1.0 = real-time)"
    )

    arg_log_level = DeclareLaunchArgument(
        "log_level",
        default_value="info",
        description="ROS 2 logger level: debug | info | warn | error"
    )

    # ------------------------------------------------------------------
    # stereo_vio_node
    # ------------------------------------------------------------------
    stereo_vio_node = Node(
        package    = "stereo_vio",
        executable = "stereo_vio_node",
        name       = "stereo_vio_node",
        output     = "screen",
        parameters = [LaunchConfiguration("params")],
        arguments  = [
            "--ros-args",
            "--log-level",
            LaunchConfiguration("log_level"),
        ],
        remappings = [
            # EuRoC topic remappings — adjust if your bag uses different names
            ("/cam0/image_raw",  "/cam0/image_raw"),
            ("/cam1/image_raw",  "/cam1/image_raw"),
            ("/imu0",            "/imu0"),
        ],
    )

    # ------------------------------------------------------------------
    # RViz2 (conditional)
    # ------------------------------------------------------------------
    rviz_node = Node(
        package    = "rviz2",
        executable = "rviz2",
        name       = "rviz2",
        output     = "screen",
        arguments  = ["-d", default_rviz],
        condition  = IfCondition(LaunchConfiguration("rviz")),
    )

    # ------------------------------------------------------------------
    # Optional bag playback
    #   Only launched when bag:= argument is non-empty.
    #   Uses OpaqueFunction to inspect the argument at launch time.
    # ------------------------------------------------------------------
    def maybe_play_bag(context, *args, **kwargs):
        bag_path = LaunchConfiguration("bag").perform(context)
        bag_rate = LaunchConfiguration("bag_rate").perform(context)
        if not bag_path:
            return []
        return [
            ExecuteProcess(
                cmd=[
                    "ros2", "bag", "play",
                    bag_path,
                    "--rate", bag_rate,
                    "--clock",
                ],
                output="screen",
            )
        ]

    bag_player = OpaqueFunction(function=maybe_play_bag)

    # ------------------------------------------------------------------
    # Log info at startup
    # ------------------------------------------------------------------
    log_start = LogInfo(
        msg=["[stereo_vio] Launching with params: ",
             LaunchConfiguration("params")]
    )

    # ------------------------------------------------------------------
    # Assemble LaunchDescription
    # ------------------------------------------------------------------
    return LaunchDescription([
        # Arguments
        arg_params,
        arg_rviz,
        arg_bag,
        arg_bag_rate,
        arg_log_level,

        # Info
        log_start,

        # Nodes
        stereo_vio_node,
        rviz_node,

        # Optional bag player
        bag_player,
    ])