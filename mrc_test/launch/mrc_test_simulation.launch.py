"""
MRC Test Simulation Launch File

Spawns 3 robots (2 EC Swift, 1 Athena) in a triangle formation with 1.5m side length.
Launches MRC master and slave nodes for each robot.
Launches the evaluation node to compare ground truth with MRC output.
"""

import os
import math
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    ExecuteProcess,
    TimerAction,
    GroupAction,
    SetEnvironmentVariable,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, PushRosNamespace
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Package paths
    mrc_test_share = get_package_share_directory('mrc_test')
    mrc_share = get_package_share_directory('mrc')

    # Triangle formation: equilateral triangle with side length 1.5m
    # Centered roughly at origin
    side_length = 1.5
    height = side_length * math.sqrt(3) / 2  # ~1.299m

    # Robot positions (x, y, z, yaw)
    # ec_swift_1: bottom-left vertex
    ec_swift_1_pos = (-side_length / 2, -height / 3, 0.15, 0.0)
    # ec_swift_2: bottom-right vertex
    ec_swift_2_pos = (side_length / 2, -height / 3, 0.15, 2.094)  # 120 degrees
    # athena: top vertex
    athena_pos = (0.0, 2 * height / 3, 0.15, -1.571)  # -90 degrees (facing down)

    # World file
    world_file = os.path.join(mrc_test_share, 'worlds', 'mrc_test_room.sdf')

    # RGL Plugin environment variables
    rgl_plugin_path = os.path.expanduser('~/workspace/RGLGazeboPlugin/install/RGLServerPlugin')
    rgl_patterns_dir = os.path.expanduser('~/workspace/RGLGazeboPlugin/lidar_patterns')

    return LaunchDescription([
        # Environment variables for RGL Plugin
        SetEnvironmentVariable(
            name='GZ_SIM_SYSTEM_PLUGIN_PATH',
            value=rgl_plugin_path + ':' + os.environ.get('GZ_SIM_SYSTEM_PLUGIN_PATH', '')
        ),
        SetEnvironmentVariable(
            name='RGL_PATTERNS_DIR',
            value=rgl_patterns_dir
        ),
        # Add GTSAM library path
        SetEnvironmentVariable(
            name='LD_LIBRARY_PATH',
            value=os.path.expanduser('~/workspace/gtsam/build/gtsam') + ':' + os.environ.get('LD_LIBRARY_PATH', '')
        ),

        # Declare launch arguments
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
        DeclareLaunchArgument(
            'comparison_trials',
            default_value='3',
            description='Number of trials per mode for pose graph comparison test'
        ),
        DeclareLaunchArgument(
            'pose_graph_topology',
            default_value='star',
            description='Pose graph topology: star (master<->slaves) or mesh (all pairwise)'
        ),

        # Launch Gazebo with the test world
        ExecuteProcess(
            cmd=['gz', 'sim', '-r', world_file],
            output='screen',
            additional_env={
                'GZ_SIM_SYSTEM_PLUGIN_PATH': rgl_plugin_path,
                'RGL_PATTERNS_DIR': rgl_patterns_dir,
            }
        ),

        # RGL bridge config path
        # Note: Bridge is started per robot namespace to handle namespaced topics

        # Wait for Gazebo to start, then spawn robots
        TimerAction(
            period=5.0,
            actions=[
                # Spawn EC Swift 1
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource([
                        '/opt/hector/jazzy/share/gazebo_robot_sim_ec_swift/launch/spawn_robot.launch.py'
                    ]),
                    launch_arguments={
                        'robot_name': 'ec_swift_1',
                        'position': f'{ec_swift_1_pos[0]},{ec_swift_1_pos[1]},{ec_swift_1_pos[2]}',
                        'yaw': str(ec_swift_1_pos[3]),
                    }.items(),
                ),
                # RGL bridge for EC Swift 1
                GroupAction([
                    PushRosNamespace('ec_swift_1'),
                    Node(
                        package='ros_gz_bridge',
                        executable='parameter_bridge',
                        name='rgl_bridge',
                        parameters=[
                            {'config_file': os.path.join(mrc_test_share, 'config', 'rgl_bridge_params.yaml')},
                            {'expand_gz_topic_names': True},
                        ],
                        output='screen',
                    ),
                ]),
            ]
        ),

        TimerAction(
            period=8.0,
            actions=[
                # Spawn EC Swift 2
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource([
                        '/opt/hector/jazzy/share/gazebo_robot_sim_ec_swift/launch/spawn_robot.launch.py'
                    ]),
                    launch_arguments={
                        'robot_name': 'ec_swift_2',
                        'position': f'{ec_swift_2_pos[0]},{ec_swift_2_pos[1]},{ec_swift_2_pos[2]}',
                        'yaw': str(ec_swift_2_pos[3]),
                    }.items(),
                ),
                # RGL bridge for EC Swift 2
                GroupAction([
                    PushRosNamespace('ec_swift_2'),
                    Node(
                        package='ros_gz_bridge',
                        executable='parameter_bridge',
                        name='rgl_bridge',
                        parameters=[
                            {'config_file': os.path.join(mrc_test_share, 'config', 'rgl_bridge_params.yaml')},
                            {'expand_gz_topic_names': True},
                        ],
                        output='screen',
                    ),
                ]),
            ]
        ),

        TimerAction(
            period=11.0,
            actions=[
                # Spawn Athena
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource([
                        '/opt/hector/jazzy/share/gazebo_robot_sim_athena/launch/spawn_robot.launch.py'
                    ]),
                    launch_arguments={
                        'robot_name': 'athena',
                        'position': f'{athena_pos[0]},{athena_pos[1]},{athena_pos[2]}',
                        'yaw': str(athena_pos[3]),
                    }.items(),
                ),
                # RGL bridge for Athena
                GroupAction([
                    PushRosNamespace('athena'),
                    Node(
                        package='ros_gz_bridge',
                        executable='parameter_bridge',
                        name='rgl_bridge',
                        parameters=[
                            {'config_file': os.path.join(mrc_test_share, 'config', 'rgl_bridge_params.yaml')},
                            {'expand_gz_topic_names': True},
                        ],
                        output='screen',
                    ),
                ]),
            ]
        ),

        # Launch MRC Slave nodes for each robot (after robots are spawned)
        TimerAction(
            period=15.0,
            actions=[
                # MRC Slave for EC Swift 1 (single front lidar)
                # Note: TF frames inside robot namespace don't have namespace prefix
                # Topic: RGL bridge publishes to /<ns>/front_lidar/points_raw
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
                # Topics: RGL bridge publishes to /<ns>/front_lidar/points_raw and back_lidar/points_raw
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
                # Topics: RGL bridge publishes to /<ns>/front_lidar/points_raw and back_lidar/points_raw
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
            ]
        ),

        # Launch MRC Master node (after slaves are ready)
        # Spawned in ec_swift_1 namespace to publish TF to the robot's TF tree
        TimerAction(
            period=18.0,
            actions=[
                GroupAction([
                    PushRosNamespace('ec_swift_1'),
                    Node(
                        package='mrc',
                        executable='mrc_master',
                        name='mrc_master',
                        parameters=[{
                            'robot_id': 'ec_swift_1',  # Master uses ec_swift_1 as reference
                            'master_map_frame': 'base_link',  # Publish relative to master's base_link
                            'default_slave_base_frame': 'base_link',  # No prefix
                            'algorithm': LaunchConfiguration('algorithm'),
                            'use_pose_graph': LaunchConfiguration('use_pose_graph'),
                            'pose_graph_topology': LaunchConfiguration('pose_graph_topology'),
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

        # Launch MRC Evaluator node - also in ec_swift_1 namespace to use same TF tree
        TimerAction(
            period=20.0,
            actions=[
                GroupAction([
                    PushRosNamespace('ec_swift_1'),
                    Node(
                        package='mrc_test',
                        executable='mrc_evaluator',
                        name='mrc_evaluator',
                        parameters=[{
                            # Reference frame for TF lookups (ec_swift_1's base_link)
                            'reference_frame': 'base_link',
                            # Robot names - ec_swift_1 uses 'base_link', others use prefixed names
                            'robot_names': ['ec_swift_1', 'ec_swift_2', 'athena'],
                            # Frame mapping: ec_swift_1 -> base_link, others -> <name>/base_link
                            'frame_ec_swift_1': 'base_link',
                            'frame_ec_swift_2': 'ec_swift_2/base_link',
                            'frame_athena': 'athena/base_link',
                            # Ground truth relative to ec_swift_1's base_link (master is at origin)
                            'ground_truth_ec_swift_1_x': 0.0,
                            'ground_truth_ec_swift_1_y': 0.0,
                            'ground_truth_ec_swift_1_z': 0.0,
                            'ground_truth_ec_swift_1_yaw': 0.0,
                            # ec_swift_2 relative to ec_swift_1
                            'ground_truth_ec_swift_2_x': ec_swift_2_pos[0] - ec_swift_1_pos[0],
                            'ground_truth_ec_swift_2_y': ec_swift_2_pos[1] - ec_swift_1_pos[1],
                            'ground_truth_ec_swift_2_z': ec_swift_2_pos[2] - ec_swift_1_pos[2],
                            'ground_truth_ec_swift_2_yaw': ec_swift_2_pos[3] - ec_swift_1_pos[3],
                            # athena relative to ec_swift_1
                            'ground_truth_athena_x': athena_pos[0] - ec_swift_1_pos[0],
                            'ground_truth_athena_y': athena_pos[1] - ec_swift_1_pos[1],
                            'ground_truth_athena_z': athena_pos[2] - ec_swift_1_pos[2],
                            'ground_truth_athena_yaw': athena_pos[3] - ec_swift_1_pos[3],
                            'evaluation_rate_hz': 1.0,
                            'comparison_trials': LaunchConfiguration('comparison_trials'),
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
