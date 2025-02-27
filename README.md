# SMART-TRACK V2

SMART-TRACK V2 is an advanced drone tracking system built on ROS2 that combines LiDAR, camera data, and multi-target Kalman filtering to provide robust aerial target tracking capabilities.

## Overview

This package implements two primary tracking modes:
1. **L2D (LiDAR-to-Depth Map)**: Converts LiDAR point clouds to depth maps for object detection using YOLO
2. **L2I (LiDAR-to-Image)**: Fuses LiDAR point clouds with RGB camera data for improved detection accuracy

Both modes feed detected object positions to a multi-target Kalman filter for stable tracking even with occasional occlusions or missed detections.

## System Architecture

![System Architecture](docs/system_architecture.png)

The SMART-TRACK V2 system consists of several key components:

- **Depth Map Detection Module**: Converts 3D LiDAR point clouds into 2D depth maps
- **YOLOv11 Object Detection**: Performs object detection on depth maps or RGB images
- **LiDAR-Camera Fusion**: Associates 3D points with 2D detections
- **Multi-Target Kalman Filter**: Tracks multiple objects across time
- **Pose Estimation**: Provides 3D position estimates of detected objects
- **Visualization**: Tools for monitoring and debugging

## Features

- Real-time drone detection and tracking
- Multiple detection pathways (depth-map based and image based)
- Robust tracking using Kalman filtering
- Built-in simulation environment for testing
- Visualization for monitoring system performance
- TF2 integration for transformations between coordinate frames

## Prerequisites

- ROS2 Humble
- Gazebo Ignition/Gazebo Garden
- CUDA-capable GPU (for YOLOv11)
- PX4 Autopilot (for simulation)
- OpenCV 4.x
- PCL (Point Cloud Library)

## Installation

1. Clone the SMART-TRACK V2 Repository:
   
```bash
git clone https://github.com/khaledgabr77/smart_track_v2.git
```

2. Install dependencies:

```bash
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
```

3. Build the ROS2 Workspace:

```bash
cd ~/ros2_ws
colcon build --symlink-install
source install/setup.bash
```

## Usage

### Launch the Simulation Environment with Observer Drone

```bash
ros2 launch smart_track_v2 l2d.launch.py  # For LiDAR-to-Depth Map tracking
# or
ros2 launch smart_track_v2 l2i.launch.py  # For LiDAR-to-Image tracking
```

### Launch a Target Drone

```bash
ros2 launch smart_track_v2 target.launch.py
```

### Launch Multiple Targets

```bash
ros2 launch smart_track_v2 multi_target.launch.py
```

### Visualize in RViz

The launch files already include RViz configuration, but you can also launch RViz separately:

```bash
ros2 run rviz2 rviz2 -d /path/to/ros2_ws/src/smart_track_v2/rviz/l2d.rviz
```

## Configuration

The system behavior can be configured through parameters in the launch files:

- `std_scaler`: Controls the size of the bounding box for point cloud filtering
- `min_range`/`max_range`: Sets the range limits for LiDAR data
- `yolo_measurement_only`: When true, uses only YOLO detections without Kalman filter refinement
- `kf_feedback`: Enables KF-based predictions when no detection is available

## Nodes

### Main Nodes:

- `depth_map_detection_localization`: Converts LiDAR data to depth maps
- `smart_track_node`: Core tracking logic and data fusion
- `lidar_camera_fusion_with_detection`: Fuses LiDAR points with camera detections
- `results`: Calculates and publishes tracking error metrics

### Supporting Nodes:

- `gimbal_stabilizer`: Controls the gimbal for camera stabilization
- `offboard_control_node`: Handles drone trajectory planning and control
- `gt_target_tf`: Provides ground truth target positions for evaluation

## Topics

### Input Topics:
- `/observer/lidar_points` - LiDAR point cloud data
- `/observer/rgb_image` - Camera images
- `/tracking` - YOLO detection results

### Output Topics:
- `/depth_map` - Converted depth map for visualization
- `/detected_object_pose` - Detected object poses
- `/final_fused_pose` - Final tracked poses with Kalman filtering
- `/kf_bounding_boxes` - Visualization markers for tracking

## Development

For detailed development documentation, see the [Developer Guide](docs/developer_guide.md).

## License

BSD 3-Clause License. See LICENSE file for details.

## Contact

Khaled Gabr - khaledgabr77@gmail.com

## Acknowledgements

- PX4 Team for the PX4-Autopilot framework
- Ultralytics for YOLOv11