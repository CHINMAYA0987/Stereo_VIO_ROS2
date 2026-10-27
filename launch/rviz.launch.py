# =============================================================================
# File: launch/rviz.launch.py
#
# Description:
#   Standalone RViz2 launch file for the Stereo VIO system.
#   Can be used independently or sourced from stereo_vio.launch.py.
#
#   Configures RViz2 to display:
#     - Odometry arrow          (/stereo_vio/odom)
#     - Trajectory path         (/stereo_vio/path)
#     - Feature debug image     (/vio/debug_image)
#     - TF frames               (world -> body -> cam0)
#     - Grid                    (world frame XY plane)
#
#   Usage (standalone):
#     ros2 launch stereo_vio rviz.launch.py
#
#   Usage (with custom config):
#     ros2 launch stereo_vio rviz.launch.py \
#         rviz_config:=/path/to/my_config.rviz
#
# References:
#   [1] RViz2 display types
#       https://github.com/ros2/rviz/tree/humble/rviz_default_plugins
# =============================================================================

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration


def generate_launch_description():

    # ------------------------------------------------------------------
    # Default RViz config path
    # ------------------------------------------------------------------
    pkg_share   = get_package_share_directory("stereo_vio")
    default_cfg = os.path.join(pkg_share, "config", "stereo_vio.rviz")

    # ------------------------------------------------------------------
    # Launch arguments
    # ------------------------------------------------------------------
    arg_rviz_config = DeclareLaunchArgument(
        "rviz_config",
        default_value=default_cfg,
        description="Full path to an .rviz configuration file"
    )

    arg_rviz_frame = DeclareLaunchArgument(
        "fixed_frame",
        default_value="world",
        description="RViz fixed frame (must match world_frame_id in params.yaml)"
    )

    # ------------------------------------------------------------------
    # Static transform: world -> odom (identity, for RViz compatibility)
    # Published once so RViz TF tree is fully connected even before
    # the VIO node starts publishing the world->body transform.
    # ------------------------------------------------------------------
    static_tf_world_odom = Node(
        package    = "tf2_ros",
        executable = "static_transform_publisher",
        name       = "static_tf_world_odom",
        output     = "screen",
        arguments  = [
            "--x",     "0", "--y", "0", "--z", "0",
            "--roll",  "0", "--pitch", "0", "--yaw", "0",
            "--frame-id",       "world",
            "--child-frame-id", "odom",
        ],
    )

    # ------------------------------------------------------------------
    # Static transform: body -> cam0
    #   T_cam0_imu from params.yaml (approximate identity for display)
    #   For accurate visualisation replace with exact T_cam0_imu values.
    # ------------------------------------------------------------------
    static_tf_body_cam0 = Node(
        package    = "tf2_ros",
        executable = "static_transform_publisher",
        name       = "static_tf_body_cam0",
        output     = "screen",
        arguments  = [
            "--x",     "-0.00692820",
            "--y",      "0.01445325",
            "--z",     "-0.00403378",
            "--roll",   "0",
            "--pitch",  "0",
            "--yaw",    "0",
            "--frame-id",       "body",
            "--child-frame-id", "cam0",
        ],
    )

    # ------------------------------------------------------------------
    # RViz2 node
    # ------------------------------------------------------------------
    rviz_node = Node(
        package    = "rviz2",
        executable = "rviz2",
        name       = "rviz2",
        output     = "screen",
        arguments  = [
            "-d", LaunchConfiguration("rviz_config"),
            "--fixed-frame", LaunchConfiguration("fixed_frame"),
        ],
    )

    # ------------------------------------------------------------------
    # Assemble LaunchDescription
    # ------------------------------------------------------------------
    return LaunchDescription([
        arg_rviz_config,
        arg_rviz_frame,
        static_tf_world_odom,
        static_tf_body_cam0,
        rviz_node,
    ])