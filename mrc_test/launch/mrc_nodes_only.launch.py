"""
MRC Nodes Launch File

Launches MRC master, slave nodes, and evaluator.
Assumes the world and robots are already running.

Usage:
  1. Launch world: ros2 launch mrc_test spawn_world_only.launch.py
  2. Spawn robots: ros2 launch mrc_test spawn_robots_only.launch.py
  3. Start MRC: ros2 launch mrc_test mrc_nodes_only.launch.py
"""

import os
import math
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    TimerAction,
    SetEnvironmentVariable,
)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, PushRosNamespace


def generate_launch_description():
    # Triangle formation parameters (must match spawn_robots_only.launch.py)
    side_length = 1.5
    height = side_length * math.sqrt(3) / 2

    ec_swift_1_pos = (-side_length / 2, -height / 3, 0.15, 0.0)
    ec_swift_2_pos = (side_length / 2, -height / 3, 0.15, 2.094)
    athena_pos = (0.0, 2 * height / 3, 0.15, -1.571)

    return LaunchDescription([
        # Add /usr/local/lib for GTSAM libraries
        SetEnvironmentVariable(
            name='LD_LIBRARY_PATH',
            value='/usr/local/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
        ),

        DeclareLaunchArgument(
            'algorithm',
            default_value='kiss_matcher',
            description='Registration algorithm: kiss_matcher or fricp'
        ),
        DeclareLaunchArgument(
            'use_pose_graph',
            default_value='true',
            description='Use GTSAM pose graph optimization'
        ),

        # MRC Slave for EC Swift 1 (single front lidar)
        # Note: TF frames inside robot namespace don't have namespace prefix
        GroupAction([
            PushRosNamespace('ec_swift_1'),
            Node(
                package='mrc',
                executable='mrc_slave',
                name='mrc_slave',
                parameters=[{
                    'robot_id': 'ec_swift_1',
                    'robot_type': 'ec_swift',
                    'pointcloud_topics': ['/ec_swift_1/front_lidar/points_raw'],
                    'base_frame': 'base_link',  # No namespace prefix in TF
                    'sensor_frame': 'front_lidar_laser_frame',
                    'merge_frame': 'base_link',
                    'accumulation_time_sec': 2.0,
                    'kiss_voxel_size': 0.3,
                }],
                remappings=[
                    ('/tf', 'tf'),  # Use namespaced TF topics
                    ('/tf_static', 'tf_static'),
                ],
                output='screen',
            ),
        ]),

        # MRC Slave for EC Swift 2 (front + back lidar merged)
        GroupAction([
            PushRosNamespace('ec_swift_2'),
            Node(
                package='mrc',
                executable='mrc_slave',
                name='mrc_slave',
                parameters=[{
                    'robot_id': 'ec_swift_2',
                    'robot_type': 'ec_swift',
                    'pointcloud_topics': [
                        '/ec_swift_2/front_lidar/points_raw',
                        '/ec_swift_2/back_lidar/points_raw',
                    ],
                    'base_frame': 'base_link',
                    'merge_frame': 'base_link',  # Merge all clouds to base_link
                    'accumulation_time_sec': 2.0,
                    'kiss_voxel_size': 0.3,
                }],
                remappings=[
                    ('/tf', 'tf'),
                    ('/tf_static', 'tf_static'),
                ],
                output='screen',
            ),
        ]),

        # MRC Slave for Athena (front + back lidar merged)
        GroupAction([
            PushRosNamespace('athena'),
            Node(
                package='mrc',
                executable='mrc_slave',
                name='mrc_slave',
                parameters=[{
                    'robot_id': 'athena',
                    'robot_type': 'athena',
                    'pointcloud_topics': [
                        '/athena/front_lidar/points_raw',
                        '/athena/back_lidar/points_raw',
                    ],
                    'base_frame': 'base_link',
                    'merge_frame': 'base_link',  # Merge all clouds to base_link
                    'accumulation_time_sec': 2.0,
                    'kiss_voxel_size': 0.3,
                }],
                remappings=[
                    ('/tf', 'tf'),
                    ('/tf_static', 'tf_static'),
                ],
                output='screen',
            ),
        ]),

        # MRC Master node (after slaves are ready)
        # Spawned in ec_swift_1 namespace to publish TF to the robot's TF tree
        TimerAction(
            period=3.0,
            actions=[
                GroupAction([
                    PushRosNamespace('ec_swift_1'),
                    Node(
                        package='mrc',
                        executable='mrc_master',
                        name='mrc_master',
                        parameters=[{
                            'robot_id': 'ec_swift_1',  # Master uses ec_swift_1 as reference
                            'master_map_frame': 'map',  # Use map frame (exists in robot's TF tree: map -> odom -> base_link)
                            'default_slave_base_frame': 'base_link',  # No prefix
                            'algorithm': LaunchConfiguration('algorithm'),
                            'use_pose_graph': LaunchConfiguration('use_pose_graph'),
                            'announce_topic': '/robot_announcement',
                            'request_timeout_sec': 10.0,
                            # Point to ec_swift_1's slave as the local/reference data source
                            'local_slave_data_service': '/ec_swift_1/mrc_slave/get_registration_data',
                            # Static robot configuration (Zenoh transient_local has issues)
                            'static_robots': ['ec_swift_2:/ec_swift_2', 'athena:/athena'],
                        }],
                        remappings=[
                            ('/tf', 'tf'),  # Use namespaced TF topics
                            ('/tf_static', 'tf_static'),
                        ],
                        output='screen',
                    ),
                ]),
            ]
        ),

        # MRC Evaluator - also in ec_swift_1 namespace to use same TF tree
        TimerAction(
            period=5.0,
            actions=[
                GroupAction([
                    PushRosNamespace('ec_swift_1'),
                    Node(
                        package='mrc_test',
                        executable='mrc_evaluator',
                        name='mrc_evaluator',
                        parameters=[{
                            # Reference frame for TF lookups (ec_swift_1's map frame)
                            'reference_frame': 'map',
                            # Robot names - ec_swift_1 uses 'base_link', others use prefixed names
                            'robot_names': ['ec_swift_1', 'ec_swift_2', 'athena'],
                            # Frame mapping: ec_swift_1 -> base_link, others -> <name>/base_link
                            'frame_ec_swift_1': 'base_link',
                            'frame_ec_swift_2': 'ec_swift_2/base_link',
                            'frame_athena': 'athena/base_link',
                            'ground_truth_ec_swift_1_x': ec_swift_1_pos[0],
                            'ground_truth_ec_swift_1_y': ec_swift_1_pos[1],
                            'ground_truth_ec_swift_1_z': ec_swift_1_pos[2],
                            'ground_truth_ec_swift_1_yaw': ec_swift_1_pos[3],
                            'ground_truth_ec_swift_2_x': ec_swift_2_pos[0],
                            'ground_truth_ec_swift_2_y': ec_swift_2_pos[1],
                            'ground_truth_ec_swift_2_z': ec_swift_2_pos[2],
                            'ground_truth_ec_swift_2_yaw': ec_swift_2_pos[3],
                            'ground_truth_athena_x': athena_pos[0],
                            'ground_truth_athena_y': athena_pos[1],
                            'ground_truth_athena_z': athena_pos[2],
                            'ground_truth_athena_yaw': athena_pos[3],
                            'evaluation_rate_hz': 1.0,
                        }],
                        remappings=[
                            ('/tf', 'tf'),  # Use namespaced TF topics
                            ('/tf_static', 'tf_static'),
                        ],
                        output='screen',
                    ),
                ]),
            ]
        ),
    ])
