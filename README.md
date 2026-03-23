# MRC - Multi-Robot Calibration

ROS 2 package for multi-robot extrinsic calibration using point cloud registration.

## Overview

MRC calibrates the relative poses between multiple robots by registering their LiDAR point clouds. It supports two registration backends (KISS-Matcher, FRICP) and optional pose graph optimization with GTSAM.

## Nodes

### mrc_master
Coordinates calibration across all robots. Collects point clouds from slaves, performs registration, and publishes static TF transforms.

**Parameters:**
- `algorithm`: Registration backend (`kiss_matcher` or `fricp`)
- `use_pose_graph`: Enable GTSAM pose graph optimization
- `pose_graph_topology`: `star` (master↔slaves) or `mesh` (all pairwise)
- `static_robots`: List of robot IDs and namespaces

**Services:**
- `~/calibrate` - Trigger calibration (legacy)
- `~/calibrate_action` - Action-based calibration with feedback

### mrc_slave
Runs on each robot. Accumulates and preprocesses point clouds for registration.

**Parameters:**
- `robot_id`: Unique robot identifier
- `pointcloud_topics`: List of LiDAR topics to subscribe
- `accumulation_time_sec`: Point cloud accumulation window

**Services:**
- `~/get_registration_data` - Returns accumulated point cloud

## Usage

```bash
# Launch master (on reference robot)
ros2 run mrc mrc_master --ros-args -p algorithm:=kiss_matcher -p use_pose_graph:=true

# Launch slave (on each robot)
ros2 run mrc mrc_slave --ros-args -p robot_id:=robot1 -p pointcloud_topics:='["/robot1/lidar"]'

# Trigger calibration
ros2 service call /mrc_master/calibrate std_srvs/srv/Trigger
```

## Dependencies

- PCL, KISS-ICP (for KISS-Matcher)
- GTSAM (for pose graph optimization)
