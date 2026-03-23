# MRC Test

Simulation and evaluation package for MRC (Multi-Robot Calibration).

## Overview

Provides a Gazebo simulation environment with multiple robots and an evaluator node to benchmark MRC accuracy against ground truth.

## Components

### mrc_evaluator
Compares MRC calibration results against known ground truth poses.

**Services:**
- `~/trigger_evaluation` - Run single calibration + evaluation
- `~/trigger_pose_graph_comparison` - Compare pose graph topologies (none/star/mesh)
- `~/trigger_full_comparison` - Comprehensive test of all backends and configurations

### Simulation World
- 8x8m room with furniture and obstacles
- 3 robots: 2x EC Swift, 1x Athena
- Livox Mid360 LiDAR (RGL plugin with realistic scan pattern)

## Usage

```bash
# Launch full simulation
ros2 launch mrc_test mrc_test_simulation.launch.py

# Trigger evaluation
ros2 service call /ec_swift_1/mrc_evaluator/trigger_evaluation std_srvs/srv/Trigger

# Run full backend comparison (kiss_matcher/fricp × none/star/mesh)
ros2 service call /ec_swift_1/mrc_evaluator/trigger_full_comparison std_srvs/srv/Trigger
```

## Launch Arguments

- `algorithm`: `kiss_matcher` (default) or `fricp`
- `use_pose_graph`: Enable pose graph optimization (default: true)
- `pose_graph_topology`: `star` or `mesh`
- `comparison_trials`: Trials per configuration (default: 3)

## Dependencies

- mrc, ros_gz_sim, ros_gz_bridge
- RGLGazeboPlugin (for Livox Mid360 simulation)
- GTSAM (for pose graph optimization)
