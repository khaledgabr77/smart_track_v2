#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch_ros.actions import Node
from ament_index_python import get_package_share_directory
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution

def generate_launch_description():
    ld = LaunchDescription()

    # Node for Drone 2
    model_name = {'gz_model_name': 'x3_uav'}
    autostart_id = {'px4_autostart_id': '4021'}
    # For ihunter_world
    # xpos = {'xpos': '200.0'}
    # ypos = {'ypos': '100.0'}
    # zpos = {'zpos': '7.0'}
    headless= {'headless' : '1'}

    #############################################################
    #              Target 1
    #############################################################

    instance_id_1 = {'instance_id': '2'}
    # For default world
    xpos_1 = {'xpos': '4.0'}
    ypos_1 = {'ypos': '0.0'}
    zpos_1 = {'zpos': '0.05'}

    # Namespace
    ns_1='target'

    # PX4 SITL + Spawn x3
    gz_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('smart_track_v2'),
                'launch',
                'gz_sim.launch.py'
            ])
        ]),
        launch_arguments={
            'gz_ns': ns_1,
            'headless': headless['headless'],
            'gz_model_name': model_name['gz_model_name'],
            'px4_autostart_id': autostart_id['px4_autostart_id'],
            'instance_id': instance_id_1['instance_id'],
            'xpos': xpos_1['xpos'],
            'ypos': ypos_1['ypos'],
            'zpos': zpos_1['zpos']
        }.items()
    )

    # MAVROS
    file_name = 'target_px4_pluginlists.yaml'
    package_share_directory = get_package_share_directory('smart_track_v2')
    plugins_file_path = os.path.join(package_share_directory, 'config', 'mavros', file_name)
    file_name = 'target_px4_config.yaml'
    config_file_path = os.path.join(package_share_directory, 'config', 'mavros', file_name)
    mavros_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('smart_track_v2'),
                'launch',
                'mavros.launch.py'
            ])
        ]),
        launch_arguments={
            'mavros_namespace' :ns_1+'/mavros',
            'tgt_system': '3',
            'fcu_url': 'udp://:14542@127.0.0.1:14559',
            'pluginlists_yaml': plugins_file_path,
            'config_yaml': config_file_path,
            'base_link_frame': 'target/base_link',
            'odom_frame': 'target/odom',
            'map_frame': 'map'
        }.items()
    )    

    # Static TF map(or world) -> local_pose_ENU
    map_frame = 'map'
    odom_frame= 'odom'
    map2pose_tf_node = Node(
        package='tf2_ros',
        name='map2px4_'+ns_1+'_tf_node',
        executable='static_transform_publisher',
        arguments=[str(xpos_1['xpos']), str(ypos_1['ypos']), str(zpos_1['zpos']), '0', '0', '0', map_frame, ns_1+'/'+odom_frame],
    )

    
    offboard_control_node = Node(
        package='smart_track_v2',
        executable='offboard_control_node.py',
        name='offboard_control_node',
        output='screen',
        namespace=ns_1,
        parameters=[ {'trajectory_type': 'circle'},
                    {'system_id': 3},
                    {'radius': 5.0},
                    {'omega': 0.5},
                    {'normal_vector': [0.0, 0.0, 1.0]},
                    {'center': [-4.0, 0.0, 10.0]},
        ],
        remappings=[
            ('mavros/state', 'mavros/state'),
            ('mavros/local_position/odom', 'mavros/local_position/odom'),
            ('mavros/setpoint_raw/local', 'mavros/setpoint_raw/local')
        ]
    )

    #############################################################
    #              END Target 1
    #############################################################


    #############################################################
    #              Target 2
    #############################################################
    instance_id_2 = {'instance_id': '3'}
    # For default world
    xpos_2 = {'xpos': '4.0'}
    ypos_2 = {'ypos': '4.0'}
    zpos_2 = {'zpos': '0.05'}
    ns_2='target2'

    # PX4 SITL + Spawn x3
    gz_launch2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('smart_track_v2'),
                'launch',
                'gz_sim.launch.py'
            ])
        ]),
        launch_arguments={
            'gz_ns': ns_2,
            'headless': headless['headless'],
            'gz_model_name': model_name['gz_model_name'],
            'px4_autostart_id': autostart_id['px4_autostart_id'],
            'instance_id': instance_id_2['instance_id'],
            'xpos': xpos_2['xpos'],
            'ypos': ypos_2['ypos'],
            'zpos': zpos_2['zpos']
        }.items()
    )

    # MAVROS
    # TODO need custom yaml file for target 2
    file_name = 'target2_px4_pluginlists.yaml'
    package_share_directory = get_package_share_directory('smart_track_v2')
    plugins_file_path = os.path.join(package_share_directory, 'config', 'mavros', file_name)
    file_name = 'target2_px4_config.yaml'
    config_file_path = os.path.join(package_share_directory, 'config', 'mavros', file_name)
    mavros_launch2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('smart_track_v2'),
                'launch',
                'mavros.launch.py'
            ])
        ]),
        launch_arguments={
            'mavros_namespace' :ns_2+'/mavros',
            'tgt_system': '4',
            'fcu_url': 'udp://:14543@127.0.0.1:14560',
            'pluginlists_yaml': plugins_file_path,
            'config_yaml': config_file_path,
            'base_link_frame': 'target2/base_link', # TODO need to change for target 2 ( in the target_px4_config.yaml)
            'odom_frame': 'target2/odom', # TODO need to change for target 2
            'map_frame': 'map'
        }.items()
    )    

    # Static TF map(or world) -> local_pose_ENU
    map_frame = 'map'
    odom_frame= 'odom'
    map2pose_tf_node2 = Node(
        package='tf2_ros',
        name='map2px4_'+ns_2+'_tf_node',
        executable='static_transform_publisher',
        arguments=[str(xpos_2['xpos']), str(ypos_2['ypos']), str(zpos_2['zpos']), '0', '0', '0', map_frame, ns_2+'/'+odom_frame],
    )

    
    offboard_control_node2 = Node(
        package='smart_track_v2',
        executable='offboard_control_node.py',
        name='offboard_control_node',
        output='screen',
        namespace=ns_2,
        parameters=[ {'trajectory_type': 'circle'},
                    {'system_id': 3},
                    {'radius': 5.0},
                    {'omega': 0.2},
                    {'normal_vector': [0.0, 0.0, 1.0]},
                    {'center': [-4.0, -4.0, 11.0]},
        ],
        remappings=[
            ('mavros/state', 'mavros/state'),
            ('mavros/local_position/odom', 'mavros/local_position/odom'),
            ('mavros/setpoint_raw/local', 'mavros/setpoint_raw/local')
        ]
    )
    #############################################################
    #              END Target 2
    #############################################################
    
    # Drone marker in RViz
    quadcopter_marker_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('smart_track_v2'),
                'quadcopter_marker.launch.py'
            ])
        ]),
        launch_arguments={
            'node_ns':ns_1,
            'propeller_size': '0.15',                # Set propeller_size directly
            'arm_length': '0.3',                    # Set arm_length directly
            'body_color': '[1.0, 0.0, 0.0, 1.0]',   # Set body_color directly
            'propeller_color': '[1.0, 1.0, 1.0, 1.0]',  # Set propeller_color directly
            'odom_topic': '/target/mavros/local_position/odom',     # Set odom_topic directly
        }.items(),
    )

    ld.add_action(gz_launch)
    # ld.add_action(px4_ros_node)
    ld.add_action(map2pose_tf_node)
    ld.add_action(offboard_control_node)
    ld.add_action(mavros_launch)
    # ld.add_action(quadcopter_marker_launch)

    ld.add_action(gz_launch2)
    ld.add_action(map2pose_tf_node2)
    ld.add_action(offboard_control_node2)
    ld.add_action(mavros_launch2)

    return ld