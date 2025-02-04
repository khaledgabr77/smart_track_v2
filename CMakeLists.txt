cmake_minimum_required(VERSION 3.8)
project(smart_track_v2)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(yolo_msgs REQUIRED)
find_package(multi_target_kf REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(message_filters REQUIRED)
find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(image_geometry REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
# Include directories
include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}

)

# Add executable for the main depth map node
add_executable(depth_map_detection_localization src/depth_map_detection_localization.cpp)
add_executable(l2d_pose_node src/l2d_pose_node.cpp)
add_executable(lidar_camera_fusion_with_detection src/lidar_camera_fusion_with_detection.cpp)
add_executable(l2i_pose_node src/l2i_pose_node.cpp)

# Link libraries
target_link_libraries(depth_map_detection_localization 
  ${PCL_LIBRARIES} 
  ${OpenCV_LIBS} 
)


target_link_libraries(l2d_pose_node 
  ${PCL_LIBRARIES} 
  ${OpenCV_LIBS} 
)
target_link_libraries(l2i_pose_node 
  ${PCL_LIBRARIES} 
  ${OpenCV_LIBS} 
)
# Link additional libraries
target_link_libraries(lidar_camera_fusion_with_detection
  ${OpenCV_LIBRARIES}
  ${Eigen3_LIBRARIES}
  ${PCL_LIBRARIES}
)

ament_target_dependencies(
  depth_map_detection_localization
  rclcpp
  sensor_msgs
  geometry_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  cv_bridge
  OpenCV
  pcl_conversions
  pcl_ros
  message_filters
  yolo_msgs
  multi_target_kf
)
ament_target_dependencies(
  l2d_pose_node
  rclcpp
  sensor_msgs
  geometry_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  cv_bridge
  OpenCV
  pcl_conversions
  pcl_ros
  message_filters
  yolo_msgs
  multi_target_kf
)
ament_target_dependencies(
  l2i_pose_node
  rclcpp
  sensor_msgs
  geometry_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  cv_bridge
  OpenCV
  pcl_conversions
  pcl_ros
  message_filters
  yolo_msgs
  multi_target_kf
)
# Specify libraries to link a library or executable target against
ament_target_dependencies(
  lidar_camera_fusion_with_detection
  rclcpp
  sensor_msgs
  vision_msgs
  geometry_msgs
  tf2_ros
  tf2_sensor_msgs
  tf2_geometry_msgs
  tf2_eigen
  pcl_conversions
  cv_bridge
  image_geometry
  OpenCV
  yolo_msgs
  message_filters
)
# Install the targets
install(TARGETS 
  depth_map_detection_localization 
  DESTINATION lib/${PROJECT_NAME})

# Install launch files
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

# Install the targets
install(TARGETS 
  l2d_pose_node
  DESTINATION lib/${PROJECT_NAME})

install(TARGETS 
  l2i_pose_node
  DESTINATION lib/${PROJECT_NAME})

# Install launch files
install(DIRECTORY 
          launch/ 
  DESTINATION share/${PROJECT_NAME}/launch
)
install(
  DIRECTORY
    config/
  DESTINATION share/${PROJECT_NAME}/config
)
# Install project RViz files
install(
  DIRECTORY
    rviz/
  DESTINATION share/${PROJECT_NAME}/rviz
)
# Install Python nodes
install(PROGRAMS
  scripts/gimbal_stabilizer.py
  DESTINATION lib/${PROJECT_NAME}
)
install(PROGRAMS
  scripts/offboard_control_node.py
  DESTINATION lib/${PROJECT_NAME}
)
install(PROGRAMS
  scripts/gt_target_tf.py
  DESTINATION lib/${PROJECT_NAME}
)
install(PROGRAMS
  scripts/trajectories.py
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS
  lidar_camera_fusion_with_detection
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
