# SmartTrack v2

## Installation

1. Clone the SMART-TRACK V2 Repository:
   
```bash
git clone https://github.com/khaledgabr77/smart_track_v2.git
```

2. Build the ROS2 Workspace

```bash
cd ~/ros2_ws
colcon build
source install/setup.bash
```

## Getting Started

1. Launch the Pose Estimator

```bash
ros2 launch smart_track_v2 smart_track_v2.launch.py
```

2. Launch the Target

```bash
ros2 launch smart_track_v2 target.launch.py
```