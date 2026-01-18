from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Declare launch arguments
    namespace_arg = DeclareLaunchArgument(
        'namespace',
        default_value='',
        description='ROS namespace for the master node'
    )

    # Config file
    mrl_config = PathJoinSubstitution([
        FindPackageShare('mrl'),
        'config',
        'mrl_config.yaml'
    ])

    # MRL Master node
    master_node = Node(
        package='mrl',
        executable='mrl_master',
        namespace=LaunchConfiguration('namespace'),
        name='mrl_master',
        parameters=[mrl_config],
        output='screen'
    )

    return LaunchDescription([
        namespace_arg,
        master_node,
    ])
