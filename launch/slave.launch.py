from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    slave = Node(
        package="mrc",
        executable="mrc_slave",
        name="mrc_slave",
        output="screen",
        parameters=[{
            "robot_id": "robot_01",
            "robot_type": "unknown",

            "pointcloud_topic": "/points",
            "base_frame": "base_link",
            "sensor_frame": "lidar_frame",

            # Service name (relative to this node)
            "data_service": "~/get_registration_data",

            # Accumulate PointCloud2 messages for non-repetitive LiDARs.
            # 0.0 => use one message (latest)
            "accumulation_time_sec": 0.0,

            # KISS feature extraction params
            "kiss_voxel_size": 0.3,
            "kiss_thr_linearity": 1.0,
            "kiss_normal_radius_gain": 3.0,
            "kiss_fpfh_radius_gain": 5.0,
        }],
    )
    return LaunchDescription([slave])