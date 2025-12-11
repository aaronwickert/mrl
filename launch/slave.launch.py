from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
        return LaunchDescription([
            DeclareLaunchArgument('is_master', default_value='false'),
            Node(
                package='mrl',
                executable='mrl_slave',
                namespace='slave',
                name='mrl_slave',
                    parameters=[{
                    'is_master' : LaunchConfiguration('is_master'),
                }]
            ),
        ])