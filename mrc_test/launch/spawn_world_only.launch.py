"""
MRC Test World Launch File

Launches only the Gazebo world with the room and objects (no robots).
Useful for testing the world setup before spawning robots.
"""

import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, SetEnvironmentVariable
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    mrc_test_share = get_package_share_directory('mrc_test')
    world_file = os.path.join(mrc_test_share, 'worlds', 'mrc_test_room.sdf')

    # RGL Plugin paths
    rgl_plugin_path = os.path.expanduser('~/workspace/RGLGazeboPlugin/install/RGLServerPlugin')
    rgl_patterns_dir = os.path.expanduser('~/workspace/RGLGazeboPlugin/lidar_patterns')

    return LaunchDescription([
        SetEnvironmentVariable(
            name='GZ_SIM_SYSTEM_PLUGIN_PATH',
            value=rgl_plugin_path + ':' + os.environ.get('GZ_SIM_SYSTEM_PLUGIN_PATH', '')
        ),
        SetEnvironmentVariable(
            name='RGL_PATTERNS_DIR',
            value=rgl_patterns_dir
        ),

        ExecuteProcess(
            cmd=['gz', 'sim', '-r', world_file],
            output='screen',
            additional_env={
                'GZ_SIM_SYSTEM_PLUGIN_PATH': rgl_plugin_path,
                'RGL_PATTERNS_DIR': rgl_patterns_dir,
            }
        ),
    ])
