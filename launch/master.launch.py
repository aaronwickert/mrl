from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
        return LaunchDescription([
            DeclareLaunchArgument('is_master', default_value='true'),
            Node(
                package='mrl',
                executable='mrl_master',
                namespace='master',
                name='mrl_master',
                    parameters=[{
                    'is_master' : LaunchConfiguration('is_master'),
                }]
            ),
        ])