from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Declare launch arguments
    robot_type_arg = DeclareLaunchArgument(
        'robot_type',
        default_value='generic',
        description='Robot type (e.g., ec_swift, spot, generic)'
    )

    namespace_arg = DeclareLaunchArgument(
        'namespace',
        default_value='',
        description='ROS namespace for the slave node'
    )

    # Build path to config file based on robot_type
    config_file = PathJoinSubstitution([
        FindPackageShare('mrl'),
        'config',
        [LaunchConfiguration('robot_type'), '.yaml']
    ])

    # MRL Slave node
    slave_node = Node(
        package='mrl',
        executable='mrl_slave',
        namespace=LaunchConfiguration('namespace'),
        name='mrl_slave',
        parameters=[config_file],
        output='screen'
    )

    return LaunchDescription([
        robot_type_arg,
        namespace_arg,
        slave_node,
    ])
