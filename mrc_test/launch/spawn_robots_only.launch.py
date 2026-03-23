"""
MRC Test Robot Spawn Launch File

Spawns 3 robots (2 EC Swift, 1 Athena) in a triangle formation.
Assumes the world is already running.

Usage:
  1. First launch the world: ros2 launch mrc_test spawn_world_only.launch.py
  2. Then spawn robots: ros2 launch mrc_test spawn_robots_only.launch.py
"""

import os
import math
from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    # Triangle formation: equilateral triangle with 1.5m side length
    side_length = 1.5
    height = side_length * math.sqrt(3) / 2  # ~1.299m

    # Robot positions (x, y, z, yaw)
    ec_swift_1_pos = (-side_length / 2, -height / 3, 0.15, 0.0)
    ec_swift_2_pos = (side_length / 2, -height / 3, 0.15, 2.094)
    athena_pos = (0.0, 2 * height / 3, 0.15, -1.571)

    return LaunchDescription([
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

        # Spawn EC Swift 2 with delay
        TimerAction(
            period=3.0,
            actions=[
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
            ]
        ),

        # Spawn Athena with delay
        TimerAction(
            period=6.0,
            actions=[
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
            ]
        ),
    ])
