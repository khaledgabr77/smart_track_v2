/*
BSD 3-Clause License

Copyright (c) 2025, Khaled Hammad Gabr
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <optional>
#include <array>

// ROS2 core
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/logger.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/qos.hpp>
#include <rclcpp/time.hpp>
#include <rclcpp/duration.hpp>
#include <rclcpp/clock.hpp>
#include <rclcpp/timer.hpp>

// TF2
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

// Message filters
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

// Messages
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/pose_array.hpp>

#include "yolo_msgs/msg/detection_array.hpp"
#include "multi_target_kf/msg/kf_tracks.hpp"

#include <sensor_msgs/point_cloud2_iterator.hpp>

// tf2 / linear algebra
#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Matrix3x3.h>

// PCL
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.hpp>  // Or <pcl_ros/point_cloud.h> if needed
#include <pcl/filters/crop_box.h>
#include <pcl/common/centroid.h>

#include <visualization_msgs/msg/marker_array.hpp>
#include <visualization_msgs/msg/marker.hpp>

using namespace std::placeholders;

/**
 * @brief Convert a builtin_interfaces::msg::Time to a double (seconds).
 */
static inline double timeToSec(const builtin_interfaces::msg::Time & t)
{
  return static_cast<double>(t.sec) + 1.0e-9 * static_cast<double>(t.nanosec);
}

/**
 * @brief Transform a PoseWithCovarianceStamped from frame A to frame B,
 *        applying a block-diagonal rotation to the 6×6 covariance (ignoring cross-terms).
 *
 * The covariance order is assumed to be (x, y, z, roll, pitch, yaw).
 * Cross terms between position/orientation are not transformed, effectively zeroed.
 * This is a common approximation in many ROS robotics pipelines.
 *
 * @param pose_in    Input pose + covariance in frame A
 * @param tfAtoB     Transform from A->B
 * @param pose_out   Output in frame B (pose + rotated covariance)
 * @return true if successful, false if exception
 */
bool transformPoseWithCovariance(
  const geometry_msgs::msg::PoseWithCovarianceStamped & pose_in,
  const geometry_msgs::msg::TransformStamped & tfAtoB,
  geometry_msgs::msg::PoseWithCovarianceStamped & pose_out)
{
  try
  {
    // 1) Transform the pose (position + orientation) using tf2::doTransform
    geometry_msgs::msg::PoseStamped tmp_in, tmp_out;
    tmp_in.header = pose_in.header;
    tmp_in.pose   = pose_in.pose.pose;  // ignoring the input covariance for the transform

    tf2::doTransform(tmp_in, tmp_out, tfAtoB);

    // Fill the output message
    pose_out.header       = tmp_out.header;          // new frame + stamp
    pose_out.pose.pose    = tmp_out.pose;            // position + orientation in new frame

    // 2) Extract the rotation (assuming no scale/shear), for the covariance rotation
    tf2::Quaternion q(
      tfAtoB.transform.rotation.x,
      tfAtoB.transform.rotation.y,
      tfAtoB.transform.rotation.z,
      tfAtoB.transform.rotation.w);
    tf2::Matrix3x3 R_pos(q);

    // In many cases, we use the same rotation for orientation uncertainties
    tf2::Matrix3x3 R_ori(q);

    // 3) Copy input covariance into a local 6×6 array
    // Covariance is row-major: (x, y, z, roll, pitch, yaw)
    double cov_in[36];
    for (size_t i = 0; i < 36; i++) {
      cov_in[i] = pose_in.pose.covariance[i];
    }

    // 4) We'll build cov_out = J * cov_in * J^T, ignoring cross-terms
    double cov_out[36];
    for (size_t i = 0; i < 36; i++) {
      cov_out[i] = 0.0;
    }

    // Index helper for row-major 6×6
    auto idx = [](int r, int c) {
      return r * 6 + c;
    };

    // A) Position block (top-left 3×3)
    //    cov_out_pos = R_pos * cov_in_pos * R_pos^T
    for (int r = 0; r < 3; r++) {
      for (int c = 0; c < 3; c++) {
        double sumRC = 0.0;
        for (int k = 0; k < 3; k++) {
          for (int m = 0; m < 3; m++) {
            sumRC += R_pos[r][k] * cov_in[idx(k, m)] * R_pos[c][m];
          }
        }
        cov_out[idx(r, c)] = sumRC;
      }
    }

    // B) Orientation block (bottom-right 3×3, rows 3..5, cols 3..5)
    //    cov_out_ori = R_ori * cov_in_ori * R_ori^T
    for (int r = 0; r < 3; r++) {
      for (int c = 0; c < 3; c++) {
        double sumRC = 0.0;
        for (int k = 0; k < 3; k++) {
          for (int m = 0; m < 3; m++) {
            sumRC += R_ori[r][k] * cov_in[idx(k+3, m+3)] * R_ori[c][m];
          }
        }
        cov_out[idx(r+3, c+3)] = sumRC;
      }
    }

    // 5) Write back into the output
    for (size_t i = 0; i < 36; i++) {
      pose_out.pose.covariance[i] = cov_out[i];
    }

    return true;
  }
  catch (const std::exception & e)
  {
    std::cerr << "[transformPoseWithCovariance] Exception: " << e.what() << std::endl;
    return false;
  }
}

/**
 * @class IntegratedLidarKFNode
 * @brief Example node that merges LiDAR data + Kalman Filter tracks + detection data
 *
 * - Subscribes to YOLO detections (publishes if new detection arrives).
 * - Subscribes to PoseArray from depth map node (and transforms it).
 * - Subscribes to multi-target KF tracks + LiDAR points, transforms them,
 *   and stores them for final multi-track updates.
 * - Publishes final PoseArray with either new detection poses or new KF poses.
 */
class IntegratedLidarKFNode : public rclcpp::Node
{
public:
  IntegratedLidarKFNode() : Node("integrated_lidar_kf_node")
  {
    // Declare/get parameters
    this->declare_parameter<bool>("debug", true);
    debug_ = this->get_parameter("debug").as_bool();

    this->declare_parameter<std::string>("lidar_frame", "lidar_link");
    lidar_frame_ = this->get_parameter("lidar_frame").as_string();

    // A parameter controlling how big the depth range is around the KF's z
    this->declare_parameter<double>("std_scaler", 5.0);
    std_scaler_ = this->get_parameter("std_scaler").as_double();

    this->declare_parameter<std::string>("reference_frame", "map");
    reference_frame_ = this->get_parameter("reference_frame").as_string();

    this->declare_parameter<bool>("yolo_measurement_only", true);
    this->declare_parameter<bool>("kf_feedback", true);

    // TF buffer + listener
    tf_buffer_   = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    // Initialize timestamps
    last_detection_t_ = 0.0;
    last_kf_measurements_t_ = 0.0;

    // Subscriptions
    detection_sub_ = this->create_subscription<yolo_msgs::msg::DetectionArray>(
      "/tracking", 10,
      std::bind(&IntegratedLidarKFNode::detectionCallback, this, _1));

    depth_map_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseArray>(
      "/detected_object_depthmap_pose", 10,
      std::bind(&IntegratedLidarKFNode::depthMapPoseCallback, this, _1));

    // Message filters for KF + Lidar
    kf_tracks_filter_.subscribe(this, "/kf/good_tracks");
    lidar_sub_.subscribe(this, "/observer/lidar_points");

    using KfDepthSyncPolicy =
      message_filters::sync_policies::ApproximateTime<
        multi_target_kf::msg::KFTracks,
        sensor_msgs::msg::PointCloud2>;

    kftracks_lidar_sync_ = std::make_shared<
      message_filters::Synchronizer<KfDepthSyncPolicy>>(KfDepthSyncPolicy(10),
                                                        kf_tracks_filter_,
                                                        lidar_sub_);

    kftracks_lidar_sync_->registerCallback(
      std::bind(&IntegratedLidarKFNode::kftracksLidarCallback, this, _1, _2));

    // Publisher: PoseArray
    fused_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseArray>(
      "/final_fused_pose", 10);
    // Create a publisher for the bounding boxes visualization markers
    marker_array_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
      "/kf_bounding_boxes", 10);
    // Timer checks for new data and publishes if needed
    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(100),
      std::bind(&IntegratedLidarKFNode::timerCallback, this));

    RCLCPP_INFO(this->get_logger(), "Integrated Lidar+KF node started.");
  }

private:

  /**
   * @brief For each track in KFTracks, transform it (pose+covariance) into the LiDAR frame,
   *        store the resulting (x,y,z,cov_x,cov_y,cov_z) for later LiDAR-based refinement.
   */
  void processAndStoreTrackData(const multi_target_kf::msg::KFTracks::ConstSharedPtr &kftracks_msg)
  {
    latest_positions_.clear();
    // latest_covariances_3d_.clear();
    latest_std_.clear();

    if (kftracks_msg->tracks.empty()) {
      RCLCPP_INFO(this->get_logger(), "No KF tracks found in message.");
      return;
    }
    // RCLCPP_INFO(this->get_logger(), "KF tracks found in message: %zu", kftracks_msg->tracks.size());

    // 1) Lookup transform from the KF track frame -> LiDAR frame
    geometry_msgs::msg::TransformStamped transform;
    try {
      transform = tf_buffer_->lookupTransform(
        lidar_frame_,                     // target
        kftracks_msg->header.frame_id,    // source
        tf2::TimePointZero,
        tf2::durationFromSec(1.0));
    }
    catch (const tf2::TransformException &ex) {
      RCLCPP_ERROR(
        this->get_logger(),
        "[processAndStoreTrackData] TF transform error %s -> %s: %s",
        kftracks_msg->header.frame_id.c_str(), lidar_frame_.c_str(), ex.what());
      return;
    }

    // 2) Transform each track from kftracks_msg->tracks
    for (const auto &track : kftracks_msg->tracks)
    {
      // Build PoseWithCovarianceStamped from track
      geometry_msgs::msg::PoseWithCovarianceStamped track_pose_in, track_pose_out;
      track_pose_in.header = kftracks_msg->header;  // same time stamp, frame
      track_pose_in.pose   = track.pose;            // includes covariance

      // Transform pose+cov into LiDAR frame
      if (!transformPoseWithCovariance(track_pose_in, transform, track_pose_out))
      {
        RCLCPP_WARN(this->get_logger(),
          "[processAndStoreTrackData] Failed to transform track pose+cov!");
        continue;
      }

      // Now track_pose_out is in the LiDAR frame
      double x = track_pose_out.pose.pose.position.x;
      double y = track_pose_out.pose.pose.position.y;
      double z = track_pose_out.pose.pose.position.z;

      // const auto &cov = track_pose_out.pose.covariance;
      Eigen::Matrix3d sub_cov = Eigen::Map<const Eigen::Matrix<double, 6, 6, Eigen::RowMajor>>(track_pose_out.pose.covariance.data()).topLeftCorner<3, 3>();
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(sub_cov);
      if (solver.info() != Eigen::Success) {
        // Handle numerical failure...
        RCLCPP_WARN(this->get_logger(),
          "[processAndStoreTrackData] Eigen decomposition failed! -> skip track.");
        continue ;
      }

      // The eigenvalues are returned in increasing order for SelfAdjointEigenSolver.
      Eigen::Vector3d eigen_values = solver.eigenvalues();
      double max_eigen_value = eigen_values.maxCoeff();
      if (max_eigen_value < 0)
      {
        RCLCPP_WARN(this->get_logger(),
          "[processAndStoreTrackData] Negative variance -> skip track.");
          continue ;
      }

      // DEBUG
      // std::cout << "Eigenvalues: " << eigen_values.transpose() << std::endl;
      // std::cout << "Max eigenvalue: " << max_eigen_value << std::endl;


      // The maximum standard deviation of all axes
      double max_std = std::sqrt(max_eigen_value);

      // double roi_min = std::max(0.0, z - std_scaler_ * sigma_z);
      // double roi_max = z + std_scaler_ * sigma_z;

      // Store them for later LiDAR-based averaging in kfProcessPoses()
      latest_positions_.push_back({x, y, z});
      // latest_covariances_3d_.push_back({cov_x, cov_y, cov_z});
      latest_std_.push_back(max_std);
    }
  }

  /**
   * @brief After we have updated (x,y,z) & (z_min,z_max) for each track in LiDAR frame,
   *        gather LiDAR points in that z-range, do an average, then transform each final
   *        position into the reference frame for output. Publish multiple poses if multi-track.
   */
  std::optional<geometry_msgs::msg::PoseArray> kfProcessPoses(
    const multi_target_kf::msg::KFTracks::ConstSharedPtr &kftracks_msg,
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr &lidar_msg)
  {
    if (latest_positions_.empty()) {
      return std::nullopt;
    }

    // Clear the MarkerArray from any previous iteration
    marker_array_.markers.clear();
    
    // Basic check
    if (latest_positions_.size() != latest_std_.size()) {
      RCLCPP_WARN(this->get_logger(),
        "[kfProcessPoses] Mismatch in positions vs. depth ranges.");
      return std::nullopt;
    }

    // We'll transform from LiDAR frame -> reference_frame_
    geometry_msgs::msg::TransformStamped transform_lidar_to_ref;
    try {
      transform_lidar_to_ref = tf_buffer_->lookupTransform(
        reference_frame_,  // target
        lidar_frame_,      // source
        tf2::TimePointZero,
        tf2::durationFromSec(1.0));
    }
    catch (const tf2::TransformException &ex) {
      RCLCPP_ERROR(this->get_logger(),
        "[kfProcessPoses] TF transform error %s -> %s: %s",
        lidar_frame_.c_str(), reference_frame_.c_str(), ex.what());
      return std::nullopt;
    }

    // Build the final PoseArray in the reference frame
    geometry_msgs::msg::PoseArray fused_pose_array;
    fused_pose_array.header.stamp = kftracks_msg->header.stamp;
    fused_pose_array.header.frame_id = reference_frame_;

    // 1. Convert ROS msg to PCL
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::fromROSMsg(*lidar_msg, *pcl_cloud);

    // Loop over each track
    for (size_t i = 0; i < latest_positions_.size(); ++i) {
      // The current track's center position
      double cx = latest_positions_[i][0];
      double cy = latest_positions_[i][1];
      double cz = latest_positions_[i][2];

      // Single standard deviation for this track
      double std_dev = latest_std_[i];  // e.g. 1σ

      // Define scaled-width of our box
      double scaled_w = std_scaler_ * std_dev;  // scaled std
      // DEBUG
      RCLCPP_INFO(this->get_logger(), "[kfProcessPoses] scaled_w: %f", scaled_w);

      // Create CropBox filter
      pcl::CropBox<pcl::PointXYZ> crop_filter;
      crop_filter.setInputCloud(pcl_cloud);

      // Set bounding box
      Eigen::Vector4f min_pt(cx - scaled_w, cy - scaled_w, cz - scaled_w, 1.0f);
      Eigen::Vector4f max_pt(cx + scaled_w, cy + scaled_w, cz + scaled_w, 1.0f);
      crop_filter.setMin(min_pt);
      crop_filter.setMax(max_pt);

      // Filter the cloud
      pcl::PointCloud<pcl::PointXYZ>::Ptr cropped_cloud(new pcl::PointCloud<pcl::PointXYZ>());
      crop_filter.filter(*cropped_cloud);

      // Compute the average if we have any points
      Eigen::Vector4f centroid;
      bool success = pcl::compute3DCentroid(*cropped_cloud, centroid);
      double avg_x = 0.0, avg_y = 0.0, avg_z = 0.0;
      if (success) {
        // centroid[0], centroid[1], centroid[2] => x, y, z
        avg_x = static_cast<double>(centroid[0]);
        avg_y = static_cast<double>(centroid[1]);
        avg_z = static_cast<double>(centroid[2]);

        RCLCPP_INFO(this->get_logger(), "[kfProcessPoses] Centroid: [%f, %f, %f]", avg_x, avg_y, avg_z);
      } else {
        RCLCPP_WARN(this->get_logger(), "[kfProcessPoses] Could not compute centroid (empty or invalid cloud?) for this track");
        continue;
      }


      // Build PoseStamped in the LiDAR frame
      geometry_msgs::msg::PoseStamped fused_pose_lidar;
      fused_pose_lidar.header.stamp = kftracks_msg->header.stamp;
      fused_pose_lidar.header.frame_id = lidar_msg->header.frame_id;  // The LiDAR frame
      fused_pose_lidar.pose.position.x = avg_x;
      fused_pose_lidar.pose.position.y = avg_y;
      fused_pose_lidar.pose.position.z = avg_z;
      fused_pose_lidar.pose.orientation.x = 0.0;
      fused_pose_lidar.pose.orientation.y = 0.0;
      fused_pose_lidar.pose.orientation.z = 0.0;
      fused_pose_lidar.pose.orientation.w = 1.0;

      // Transform to reference frame
      geometry_msgs::msg::PoseStamped fused_pose_ref;
      try {
        tf2::doTransform(fused_pose_lidar, fused_pose_ref, transform_lidar_to_ref);
      } catch (const std::exception & e) {
        RCLCPP_ERROR(this->get_logger(),
                     "[kfProcessPoses] Exception while transforming pose: %s", e.what());
        continue;
      }

      // Add to the PoseArray
      fused_pose_array.poses.push_back(fused_pose_ref.pose);

      // --------------------------
      //  CREATE BOUNDING-BOX MARKER
      // --------------------------
      // We want to transform the corner points min_pt and max_pt from LiDAR -> reference frame
      // to get an axis-aligned bounding box in reference frame.
      // In practice, because the frames might be rotated, the bounding box in the reference frame
      // might no longer be axis-aligned. If you prefer to visualize it axis-aligned in the LiDAR frame,
      // you can simply set `marker.header.frame_id = lidar_frame_`. 
      // But here we do it in the reference frame for consistency.
      
      // 1) Transform min and max corners to reference frame
      geometry_msgs::msg::Point pt_min_ref, pt_max_ref;

      // Helper to transform an Eigen::Vector4f into geometry_msgs::msg::Point in ref frame
      auto transformPoint = [&](const Eigen::Vector4f &pt_lidar) {
        geometry_msgs::msg::PointStamped in, out;
        in.header.frame_id = lidar_frame_;
        in.point.x = pt_lidar.x();
        in.point.y = pt_lidar.y();
        in.point.z = pt_lidar.z();
        // Transform to reference
        tf2::doTransform(in, out, transform_lidar_to_ref);
        return out.point;
      };

      pt_min_ref = transformPoint(min_pt);
      pt_max_ref = transformPoint(max_pt);

      // 2) Calculate the center (in ref frame) of that bounding box
      geometry_msgs::msg::Point box_center;
      box_center.x = 0.5 * (pt_min_ref.x + pt_max_ref.x);
      box_center.y = 0.5 * (pt_min_ref.y + pt_max_ref.y);
      box_center.z = 0.5 * (pt_min_ref.z + pt_max_ref.z);

      // 3) Box scale = difference in x, y, z
      double dx = std::fabs(pt_max_ref.x - pt_min_ref.x);
      double dy = std::fabs(pt_max_ref.y - pt_min_ref.y);
      double dz = std::fabs(pt_max_ref.z - pt_min_ref.z);

      // 4) Create the Marker
      visualization_msgs::msg::Marker marker;
      marker.header.stamp = this->now();
      marker.header.frame_id = reference_frame_;
      marker.ns = "kf_bounding_box";
      marker.id = static_cast<int>(i);  // unique ID for each track

      marker.type = visualization_msgs::msg::Marker::CUBE;
      marker.action = visualization_msgs::msg::Marker::ADD;

      // Marker position (center of the bounding box)
      marker.pose.position.x = box_center.x;
      marker.pose.position.y = box_center.y;
      marker.pose.position.z = box_center.z;

      // Orientation: for an axis-aligned bounding box in the reference frame,
      // we keep the orientation as identity (0,0,0,1).
      marker.pose.orientation.w = 1.0;

      // Size
      marker.scale.x = dx;
      marker.scale.y = dy;
      marker.scale.z = dz;

      // Color & transparency (RGBA)
      marker.color.r = 1.0f;   // Red
      marker.color.g = 0.0f;
      marker.color.b = 0.0f;
      marker.color.a = 0.3f;   // 30% opacity -> semi-transparent

      // Lifetime: 0 = forever
      marker.lifetime = rclcpp::Duration(0, 5e8);

      marker_array_.markers.push_back(marker);
    } // end for each track


    if (fused_pose_array.poses.empty()) {
      return std::nullopt;
    }
    return fused_pose_array;
  }

  // YOLO detection callback (just store the message for a "new detection" check)
  void detectionCallback(const yolo_msgs::msg::DetectionArray::SharedPtr msg)
  {
    // RCLCPP_INFO(this->get_logger(), "[detectionCallback] Received detection message with %zu detections.", msg->detections.size());

    latest_detections_msg_ = msg;
  }

  // Depth map pose callback (PoseArray in some frame, transform to reference_frame_ and store)
  void depthMapPoseCallback(const geometry_msgs::msg::PoseArray::SharedPtr msg)
  {
    if (msg->poses.empty()) {
      return; // no poses
    }

    geometry_msgs::msg::TransformStamped transform_lidar_to_ref;
    try {
      transform_lidar_to_ref = tf_buffer_->lookupTransform(
        reference_frame_,
        msg->header.frame_id,
        tf2::TimePointZero,
        tf2::durationFromSec(1.0));
    }
    catch (const tf2::TransformException &ex) {
      RCLCPP_ERROR(this->get_logger(),
        "[depthMapPoseCallback] TF error %s -> %s: %s",
        msg->header.frame_id.c_str(), reference_frame_.c_str(), ex.what());
      return;
    }

    geometry_msgs::msg::PoseArray transformed_pose_array;
    transformed_pose_array.header.stamp = msg->header.stamp;
    transformed_pose_array.header.frame_id = reference_frame_;

    for (size_t i = 0; i < msg->poses.size(); ++i) {
      geometry_msgs::msg::PoseStamped ps_in, ps_out;
      ps_in.header = msg->header;
      ps_in.pose   = msg->poses[i];

      try {
        tf2::doTransform(ps_in, ps_out, transform_lidar_to_ref);
        transformed_pose_array.poses.push_back(ps_out.pose);
      }
      catch (const std::exception & e) {
        RCLCPP_ERROR(this->get_logger(),
          "[depthMapPoseCallback] transform exception: %s", e.what());
      }
    }

    objects_pose_msg_ = std::make_shared<geometry_msgs::msg::PoseArray>(transformed_pose_array);
  }

  // Synchronized callback (multi_target_kf::KFTracks + Lidar points)
  void kftracksLidarCallback(
    const multi_target_kf::msg::KFTracks::ConstSharedPtr &kftracks_msg,
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr &lidar_msg)
  {
    latest_kftracks_msg_ = kftracks_msg;
    last_lidar_msg_      = lidar_msg;

    // Transform each track's pose+cov to LiDAR frame and store
    processAndStoreTrackData(kftracks_msg);
  }

  // Timer callback
  void timerCallback()
  {
    bool use_yolo = this->get_parameter("yolo_measurement_only").as_bool();
    bool use_kf   = this->get_parameter("kf_feedback").as_bool();
    // Check whether new detection or KF data arrived
    bool detectionUpdated = isNewDetections();
    bool kfUpdated        = isNewKfTracks();

    if(use_yolo){
      // 1) If new detections arrived, publish them (priority)
      if (detectionUpdated && objects_pose_msg_) {
        fused_pose_pub_->publish(*objects_pose_msg_);
        //DEBUG
        RCLCPP_INFO(this->get_logger(),
        "[timerCallback] Got detection. Publishing directly to KF.");
        // if (debug_) {
        //   RCLCPP_INFO(this->get_logger(),
        //     "[timerCallback] Published new depth-based PoseArray (detection).");
        // }
        return;
      }
      else if (detectionUpdated && !objects_pose_msg_) {
        RCLCPP_WARN(this->get_logger(),
          "[timerCallback] We got a new detection, but objects_pose_msg_ is null!");
        return;
      }
  }
    if(use_kf){
      // 2) If new KF tracks arrived, publish them
      if (kfUpdated) {
        // DEBUG
        RCLCPP_INFO(this->get_logger(),
        "[timerCallback] Trying KF feedback.");

        if (!latest_kftracks_msg_ || !last_lidar_msg_) {
          RCLCPP_WARN(this->get_logger(),
            "[timerCallback] Missing KF or lidar data, cannot publish fused pose.");
          return;
        }

        auto kf_pose_array_opt = kfProcessPoses(latest_kftracks_msg_, last_lidar_msg_);
        if (kf_pose_array_opt) {
          fused_pose_pub_->publish(kf_pose_array_opt.value());
          marker_array_pub_->publish(marker_array_);
          // RCLCPP_INFO(this->get_logger(),
          // "[timerCallback] fused_pose_pub_ KF PUBLISHING.");
          if (debug_) {
            RCLCPP_INFO(this->get_logger(),
              "[timerCallback] Published new KF-based PoseArray with %zu poses.",
              kf_pose_array_opt->poses.size());
          }
        } else {
          RCLCPP_WARN(this->get_logger(),
            "[timerCallback] Got new KF tracks but no valid fused pose from them!");
        }
      }
      else
      {
        RCLCPP_WARN(this->get_logger(),
            "[timerCallback] No new KF feedback!");
      }
    }
  }
  // Convert ROS2 stamp to double seconds
  inline double timeToSec(const builtin_interfaces::msg::Time & stamp)
  {
    return static_cast<double>(stamp.sec) +
          static_cast<double>(stamp.nanosec) / 1e9;
  }

  bool isNewDetections()
  {
    if (!latest_detections_msg_) {
      return false;
    }
    if (!latest_detections_msg_->detections.empty()) {
      double t = timeToSec(latest_detections_msg_->header.stamp);
      if (t > last_detection_t_) {
        last_detection_t_ = t;
        return true;
      }
    }
    return false;
  }

  bool isNewKfTracks()
  {
    if (!latest_kftracks_msg_) {
      return false;
    }
    if (!latest_kftracks_msg_->tracks.empty()) {
      double t = timeToSec(latest_kftracks_msg_->header.stamp);
      // IMPORTANT: Add the extra check that matches Python:
      //   if (t > last_kf_measurements_t_ AND t > last_detection_t_)
      // so we only consider KF if it’s newer than the last detection.
      if ((t > last_kf_measurements_t_) && (t > last_detection_t_)) {
        last_kf_measurements_t_ = t;
        return true;
      }
    }
    return false;
  }


private:
  // Parameters
  bool        debug_{false};
  std::string lidar_frame_;
  double      std_scaler_{5.0};
  std::string reference_frame_;

  // Timestamps for detection and KF
  double last_detection_t_{0.0};
  double last_kf_measurements_t_{0.0};

  // Stored messages
  yolo_msgs::msg::DetectionArray::SharedPtr latest_detections_msg_;
  geometry_msgs::msg::PoseArray::SharedPtr objects_pose_msg_;
  multi_target_kf::msg::KFTracks::ConstSharedPtr latest_kftracks_msg_;
  sensor_msgs::msg::PointCloud2::ConstSharedPtr last_lidar_msg_;

  // Subscriptions
  rclcpp::Subscription<yolo_msgs::msg::DetectionArray>::SharedPtr detection_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr depth_map_pose_sub_;

  message_filters::Subscriber<multi_target_kf::msg::KFTracks> kf_tracks_filter_;
  message_filters::Subscriber<sensor_msgs::msg::PointCloud2>  lidar_sub_;

  std::shared_ptr<message_filters::Synchronizer<
    message_filters::sync_policies::ApproximateTime<
      multi_target_kf::msg::KFTracks,
      sensor_msgs::msg::PointCloud2
    >>> kftracks_lidar_sync_;

  // TF
  std::shared_ptr<tf2_ros::Buffer>           tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener>tf_listener_;

  // Publisher
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr fused_pose_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_array_pub_;

  // Timer
  rclcpp::TimerBase::SharedPtr timer_;

  // Data from KF tracks (in LiDAR frame)
  std::vector<std::array<double,3>> latest_positions_;      // x,y,z
  // std::vector<std::array<double,3>> latest_covariances_3d_; // diag cov_x,cov_y,cov_z
  std::vector<double> latest_std_;   // roi_min, roi_max
  // A marker array to hold your bounding boxes
  visualization_msgs::msg::MarkerArray marker_array_;
};

// main
int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<IntegratedLidarKFNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}