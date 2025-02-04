#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():

    # Launch Description
    ld = LaunchDescription()


    # Namespace and topic setup
    ns = 'observer'

    # World and Model Configuration
    world = {'gz_world': 'default'}
    model_name = {'gz_model_name': 'x500_lidar_camera'}
    autostart_id = {'px4_autostart_id': '4022'}
    instance_id = {'instance_id': '1'}
    xpos = {'xpos': '0.0'}
    ypos = {'ypos': '0.0'}
    zpos = {'zpos': '0.1'}
    headless = {'headless': '0'}

    # PX4 SITL + Gazebo Simulation Launch
    gz_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('smart_track_v2'),
                'launch',
                'gz_sim.launch.py'
            ])
        ]),
        launch_arguments={
            'gz_ns': ns,
            'headless': headless['headless'],
            'gz_world': world['gz_world'],
            'gz_model_name': model_name['gz_model_name'],
            'px4_autostart_id': autostart_id['px4_autostart_id'],
            'instance_id': instance_id['instance_id'],
            'xpos': xpos['xpos'],
            'ypos': ypos['ypos'],
            'zpos': zpos['zpos']
        }.items()
    )

    # MAVROS Launch
    package_share_directory = get_package_share_directory('smart_track_v2')
    plugins_file_path = os.path.join(package_share_directory, 'config', 'mavros', 'observer_px4_pluginlists.yaml')
    config_file_path = os.path.join(package_share_directory, 'config', 'mavros', 'observer_px4_config.yaml')
    mavros_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('smart_track_v2'),
                'launch',
                'mavros.launch.py'
            ])
        ]),
        launch_arguments={
            'mavros_namespace': ns + '/mavros',
            'tgt_system': '2',
            'fcu_url': 'udp://:14541@127.0.0.1:14558',
            'pluginlists_yaml': plugins_file_path,
            'config_yaml': config_file_path,
            'base_link_frame': 'observer/base_link',
            'odom_frame': 'observer/odom',
            'map_frame': 'map'
        }.items()
    )

    # Static Transform Publishers
    map2pose_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=[
            xpos['xpos'], ypos['ypos'], zpos['zpos'], '0.0', '0', '0',
            'map', f"{ns}/odom"
        ]
    )

    cam_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=[
            '0.1', '0', '0.13', '1.5708', '0', '1.5708',
            f"{ns}/base_link", f"{ns}/camera_link"
        ]
    )

    lidar_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=[
            '0', '0', '0.295', '0', '0', '0',
            f"{ns}/base_link", f"x500_lidar_camera_1/lidar_link/gpu_lidar"
        ]
    )

    # ROS-GZ Bridge Node
    ros_gz_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock',
            '/scan@sensor_msgs/msg/LaserScan[ignition.msgs.LaserScan',
            '/scan/points@sensor_msgs/msg/PointCloud2[ignition.msgs.PointCloudPacked',
            '/rgb_image@sensor_msgs/msg/Image[ignition.msgs.Image',
            '/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo',
            '/gimbal/cmd_yaw@std_msgs/msg/Float64[ignition.msgs.Double',
            '/gimbal/cmd_pitch@std_msgs/msg/Float64[ignition.msgs.Double',
            '--ros-args',
            '-r', '/scan/points:=' + ns + '/lidar_points',
            '-r', '/rgb_image:=' + ns + '/rgb_image',
            '-r', '/camera_info:=' + ns + '/camera_info',
        ],
    )

    # RViz2 Node
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', '/home/user/shared_volume/ros2_ws/src/smart_track_v2/rviz/l2i.rviz']
    )

    # Depth Map Detection and Localization Node
    depth_map_detection_localization_node = Node(
        package='smart_track_v2',
        executable='depth_map_detection_localization',
        name='point_cloud_to_depth_map',
        parameters=[
            {
                'width': 650,
                'height': 650,
                'ScaleVector': 4.0,
                'MinDepth': 0.2,
                'MaxDepth': 30.0,
            }
        ],
        remappings=[
            ('/observer/lidar_points', '/observer/lidar_points'),
            ('/tracking', '/tracking'),
        ]
    )

    # L2D Pose Node
    l2d_pose_node = Node(
        package='smart_track_v2',
        executable='l2d_pose_node',
        name='l2d_pose_node',
        parameters=[
            {
                'std_range': 5.0,
                'lidar_frame': 'x500_lidar_camera_1/lidar_link/gpu_lidar',
                'reference_frame': 'observer/odom',
            }
        ],
        remappings=[
            ('/observer/lidar_points', '/observer/lidar_points'),
            # ('/tracking', '/detection'),
            ('/depth_map', '/depth_map'),
            # ('/final_fused_pose', '/yolo_detections_poses'),
        ]
    )

    # Lidar-Camera Fusion Node
    lidar_camera_fusion_node = Node(
        package='smart_track_v2',
        executable='lidar_camera_fusion_with_detection',
        name='lidar_camera_fusion_node',
        parameters=[
            {'min_range': 0.2, 'max_range': 10.0,
            'lidar_frame': 'x500_lidar_camera_1/lidar_link/gpu_lidar',
            'camera_frame': 'observer/camera_link'}
        ],
        remappings=[
            ('/observer/lidar_points', '/observer/lidar_points'),
            ('/observer/camera_info', '/observer/camera_info'),
            ('/observer/rgb_image', '/observer/rgb_image'),
            ('/tracking', '/tracking')
        ]
    )

    # L2D Pose Node
    l2i_pose_node = Node(
        package='smart_track_v2',
        executable='l2i_pose_node',
        name='l2i_pose_node',
        parameters=[
            {
                'std_range': 5.0,
                'lidar_frame': 'x500_lidar_camera_1/lidar_link/gpu_lidar',
                'reference_frame': 'observer/odom',
            }
        ],
    )
    # YOLO Launch for Depth Map Detection
    yolo_launch_depth_map = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('yolo_bringup'),
                'launch/yolov11.launch.py'
            ])
        ]),
        launch_arguments={
            'model': '/home/user/shared_volume/ros2_ws/src/smart_track_v2/config/rgb.pt',
            # 'model': '/home/user/shared_volume/ros2_ws/src/smart_track_v2/config/depth.pt',
            'threshold': '0.5',
            # 'input_image_topic': 'depth_map',
            'input_image_topic': '/observer/rgb_image',
            'namespace': '',
            'device': 'cuda:0'
        }.items()
    )

    # Kalman Filter Launch
    kf_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('multi_target_kf'),
                'launch/kf_const_vel.launch.py'
            ])
        ]),
        launch_arguments={
            'detections_topic': 'final_fused_pose',
            'kf_ns': '',
            'kf_yaml': os.path.join(package_share_directory, 'kf_param.yaml'),
        }.items()
    )

    # Add all nodes and launches to the launch description
    ld.add_action(gz_launch)
    ld.add_action(map2pose_tf_node)
    ld.add_action(cam_tf_node)
    ld.add_action(lidar_tf_node)
    ld.add_action(ros_gz_bridge)
    ld.add_action(mavros_launch)
    ld.add_action(rviz_node)
    # ld.add_action(depth_map_detection_localization_node)
    ld.add_action(yolo_launch_depth_map)
    # ld.add_action(l2d_pose_node)
    ld.add_action(kf_launch)
    ld.add_action(lidar_camera_fusion_node)
    ld.add_action(l2i_pose_node)
    return ld
