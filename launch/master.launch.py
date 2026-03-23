from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    master = Node(
        package="mrc",
        executable="mrc_master",
        name="mrc_master",
        output="screen",
        parameters=[{
            "robot_id": "master",
            "master_map_frame": "map",

            # hector_multi_robot global announcement topic
            "announce_topic": "/robot_announcement",

            # where each robot's slave service is expected (relative to robot namespace)
            "slave_service_relative": "mrc_slave/get_registration_data",

            "algorithm": "kiss_matcher",  # or "fricp"
            "local_slave_data_service": "/mrc_slave/get_registration_data",
            "default_slave_base_frame": "base_link",
            "request_timeout_sec": 2.0,
        }],
    )
    return LaunchDescription([master])