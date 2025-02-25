#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include "multi_target_kf/msg/kf_tracks.hpp"
#include <vector>
#include <cmath>

#include <rclcpp/time.hpp>
#include <rclcpp/duration.hpp>
#include <rclcpp/clock.hpp>

class KFErrorCalculator : public rclcpp::Node
{
public:
    KFErrorCalculator()
        : Node("kf_error_calculator"), msg_count_(0), msg_limit_(200)
    {
        kf_tracks_sub_ = this->create_subscription<multi_target_kf::msg::KFTracks>(
            "/kf/good_tracks", 10, std::bind(&KFErrorCalculator::collectData, this, std::placeholders::_1));

        mse_publisher_ = this->create_publisher<std_msgs::msg::Float64>("mse", 10);
        rmse_publisher_ = this->create_publisher<std_msgs::msg::Float64>("rmse", 10);
        abs_publisher_ = this->create_publisher<std_msgs::msg::Float64>("abs_error", 10);
        pose_publisher_ = this->create_publisher<geometry_msgs::msg::Pose>("actual_pose", 10);

        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
    }

private:
    void collectData(const multi_target_kf::msg::KFTracks::SharedPtr msg)
    {
        auto actual_pose_values = getActualPose(msg->header.stamp);
        for (const auto &track : msg->tracks)
        {
            double x = track.pose.pose.position.x;
            double y = track.pose.pose.position.y;
            double z = track.pose.pose.position.z;

            track_data_.emplace_back(x, y, z);
            ground_truth_pose_.emplace_back(actual_pose_values);
            
            msg_count_++;

            if (msg_count_ >= msg_limit_)
            {
                calculateErrors();
                resetData();
            }
        }
    }

    void calculateErrors()
    {
        if (ground_truth_pose_.size() < msg_limit_ || track_data_.size() < msg_limit_)
        {
            RCLCPP_WARN(this->get_logger(), "Insufficient data for calculations");
            return;
        }

        double squared_error = 0.0;
        double abs_error = 0.0;

        for (size_t i = 0; i < msg_limit_; i++)
        {
            auto [pose_x, pose_y, pose_z] = ground_truth_pose_[i];
            auto [x, y, z] = track_data_[i];

            squared_error += std::pow(pose_x - x, 2) + std::pow(pose_y - y, 2) + std::pow(pose_z - z, 2);
            abs_error += std::abs(pose_x - x) + std::abs(pose_y - y) + std::abs(pose_z - z);
        }

        double mse = squared_error / msg_limit_;
        double rmse = std::sqrt(mse);
        double avg_abs_error = abs_error;

        publishError(mse_publisher_, mse);
        publishError(rmse_publisher_, rmse);
        publishError(abs_publisher_, avg_abs_error);
    }

    void resetData()
    {
        track_data_.clear();
        ground_truth_pose_.clear();
        msg_count_ = 0;
    }

    void publishError(rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr publisher, double value)
    {
        auto msg = std_msgs::msg::Float64();
        msg.data = value;
        publisher->publish(msg);
    }

    std::tuple<double, double, double> getActualPose(const builtin_interfaces::msg::Time &kf_time)
    {
    
        geometry_msgs::msg::TransformStamped transform;
        try {
        tf2::TimePoint tf_time = tf2_ros::fromMsg(kf_time);
        transform = tf_buffer_->lookupTransform(
            "observer/odom",        // target
            "target/base_link",    // source
            tf_time,
            tf2::durationFromSec(1.0));
        }

        catch (const tf2::TransformException &ex) {
            RCLCPP_ERROR(
                this->get_logger(),
                "[getActualPose] TF transform error %s -> %s: %s",
                "target/base_link", "observer/odom", ex.what());
            return {0.0, 0.0, 0.0}; 
        }

        return {transform.transform.translation.x, 
                transform.transform.translation.y, 
                transform.transform.translation.z};

    }

    rclcpp::Subscription<multi_target_kf::msg::KFTracks>::SharedPtr kf_tracks_sub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr mse_publisher_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr rmse_publisher_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr abs_publisher_;
    rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr pose_publisher_;

    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    std::vector<std::tuple<double, double, double>> track_data_;
    std::vector<std::tuple<double, double, double>> ground_truth_pose_;

    size_t msg_count_;
    size_t msg_limit_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<KFErrorCalculator>());
    rclcpp::shutdown();
    return 0;
}
