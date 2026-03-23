#include <chrono>
#include <deque>
#include <mutex>
#include <string>
#include <vector>
#include <unordered_map>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>

#include <Eigen/Core>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <kiss_matcher/FasterPFH.hpp>
#include <kiss_matcher/points/downsampling.hpp>

#include "mrc/msg/kiss_features.hpp"
#include "mrc/srv/get_registration_data.hpp"

using namespace std::chrono_literals;

class MRCSlaveNode : public rclcpp::Node {
public:
  explicit MRCSlaveNode(const rclcpp::NodeOptions& options)
      : rclcpp::Node("mrc_slave", options) {
    robot_id_ = declare_parameter<std::string>("robot_id", "robot_01");
    robot_type_ = declare_parameter<std::string>("robot_type", "unknown");

    // Support both single topic (backward compatible) and multiple topics
    declare_parameter<std::string>("pointcloud_topic", "");
    declare_parameter<std::vector<std::string>>("pointcloud_topics", std::vector<std::string>{});

    base_frame_ = declare_parameter<std::string>("base_frame", "base_link");
    sensor_frame_ = declare_parameter<std::string>("sensor_frame", "lidar_frame");
    merge_frame_ = declare_parameter<std::string>("merge_frame", "");  // Frame to merge clouds into

    // TF prefix to prepend to frame names from point clouds
    // Use this when TF frames are namespaced but point cloud frame_ids are not
    tf_prefix_ = declare_parameter<std::string>("tf_prefix", "");

    // Accumulation type controls whether point clouds are accumulated over time:
    // - "none" or "latest": Use only the latest cloud(s) - for spinning/repetitive lidars (default)
    // - "time_window": Accumulate clouds over accumulation_time_sec - for non-repetitive/solid-state lidars
    accumulation_type_ = declare_parameter<std::string>("accumulation_type", "none");
    accumulation_time_sec_ = declare_parameter<double>("accumulation_time_sec", 1.0);
    max_buffer_size_ = declare_parameter<int>("max_buffer_size", 100);  // Prevent OOM

    // Validate accumulation_type
    if (accumulation_type_ != "none" && accumulation_type_ != "latest" &&
        accumulation_type_ != "time_window") {
      RCLCPP_WARN(get_logger(),
                  "Unknown accumulation_type '%s', defaulting to 'none'. "
                  "Valid options: 'none', 'latest', 'time_window'",
                  accumulation_type_.c_str());
      accumulation_type_ = "none";
    }

    // Enable accumulation only for time_window type
    use_accumulation_ = (accumulation_type_ == "time_window");

    // KISS feature extraction params
    voxel_size_ = declare_parameter<double>("kiss_voxel_size", 0.3);
    thr_linearity_ = declare_parameter<double>("kiss_thr_linearity", 1.0);
    normal_radius_gain_ = declare_parameter<double>("kiss_normal_radius_gain", 3.0);
    fpfh_radius_gain_ = declare_parameter<double>("kiss_fpfh_radius_gain", 5.0);

    data_service_name_ = declare_parameter<std::string>("data_service", "~/get_registration_data");

    // Get topic configuration
    auto single_topic = get_parameter("pointcloud_topic").as_string();
    auto multi_topics = get_parameter("pointcloud_topics").as_string_array();

    // Use multi-topic if provided, otherwise fall back to single topic
    if (!multi_topics.empty()) {
      pointcloud_topics_ = multi_topics;
    } else if (!single_topic.empty()) {
      pointcloud_topics_.push_back(single_topic);
    } else {
      pointcloud_topics_.push_back("/points");  // Default
    }

    // If merge_frame not set, use base_frame for multi-topic or sensor_frame for single
    if (merge_frame_.empty()) {
      merge_frame_ = (pointcloud_topics_.size() > 1) ? base_frame_ : sensor_frame_;
    }

    // TF2 buffer and listener for transforming point clouds
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    // Create subscriptions for each topic
    for (size_t i = 0; i < pointcloud_topics_.size(); ++i) {
      const auto& topic = pointcloud_topics_[i];
      auto sub = create_subscription<sensor_msgs::msg::PointCloud2>(
          topic, rclcpp::SensorDataQoS(),
          [this, i](const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
            onCloud(msg, i);
          });
      cloud_subs_.push_back(sub);
      RCLCPP_INFO(get_logger(), "Subscribed to pointcloud topic [%zu]: %s", i, topic.c_str());
    }

    RCLCPP_INFO(get_logger(),
                "MRC Slave initialized: robot_id='%s', %zu topic(s), merge_frame='%s', "
                "accumulation_type='%s'%s",
                robot_id_.c_str(), pointcloud_topics_.size(), merge_frame_.c_str(),
                accumulation_type_.c_str(),
                use_accumulation_ ?
                    (", time_window=" + std::to_string(accumulation_time_sec_) + "s").c_str() : "");

    data_srv_ = create_service<mrc::srv::GetRegistrationData>(
        data_service_name_,
        std::bind(&MRCSlaveNode::onGetRegistrationData, this,
                  std::placeholders::_1, std::placeholders::_2));
  }

private:
  void onCloud(const sensor_msgs::msg::PointCloud2::SharedPtr msg, size_t topic_idx) {
    std::lock_guard<std::mutex> lk(mutex_);

    if (!ready_) {
      ready_ = true;
      RCLCPP_INFO(get_logger(), "First point cloud received on topic [%zu], slave is now READY", topic_idx);
    }

    // Get frame from cloud or fallback to sensor_frame
    std::string cloud_frame = msg->header.frame_id.empty() ? sensor_frame_ : msg->header.frame_id;

    // Apply TF prefix if configured (for namespaced TF trees)
    const std::string source_frame = tf_prefix_.empty() ? cloud_frame : (tf_prefix_ + "/" + cloud_frame);

    sensor_msgs::msg::PointCloud2 transformed_cloud;

    // Transform point cloud to merge_frame if needed
    if (source_frame != merge_frame_) {
      try {
        auto transform = tf_buffer_->lookupTransform(
            merge_frame_, source_frame, tf2::TimePointZero, tf2::durationFromSec(0.1));
        tf2::doTransform(*msg, transformed_cloud, transform);
        transformed_cloud.header.frame_id = merge_frame_;
      } catch (const tf2::TransformException& ex) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                             "Could not transform cloud from '%s' to '%s': %s",
                             source_frame.c_str(), merge_frame_.c_str(), ex.what());
        // Fall back to using the cloud as-is
        transformed_cloud = *msg;
      }
    } else {
      transformed_cloud = *msg;
    }

    // Store in per-topic buffer for merging
    last_clouds_[topic_idx] = transformed_cloud;
    last_stamp_ = msg->header.stamp;
    last_frame_id_ = source_frame;
    has_cloud_ = true;

    // Add to accumulation buffer only if using time_window accumulation
    if (use_accumulation_) {
      cloud_buffer_.push_back(transformed_cloud);
      pruneBufferLocked();
    }

    cached_features_valid_ = false;
  }

  void pruneBufferLocked() {
    if (!use_accumulation_ || accumulation_time_sec_ <= 0.0) {
      cloud_buffer_.clear();
      return;
    }
    if (!has_cloud_) return;

    // Enforce maximum buffer size to prevent OOM
    while (cloud_buffer_.size() > static_cast<size_t>(max_buffer_size_)) {
      cloud_buffer_.pop_front();
    }

    // Prune by time window
    const rclcpp::Time newest(last_stamp_);
    const rclcpp::Duration window = rclcpp::Duration::from_seconds(accumulation_time_sec_);
    const rclcpp::Time oldest_allowed = newest - window;

    while (!cloud_buffer_.empty()) {
      const auto& front = cloud_buffer_.front();
      const rclcpp::Time t(front.header.stamp);
      if (t < oldest_allowed) {
        cloud_buffer_.pop_front();
      } else {
        break;
      }
    }
  }

  static void appendCloudXYZ(pcl::PointCloud<pcl::PointXYZ>& dst,
                             const sensor_msgs::msg::PointCloud2& msg) {
    pcl::PointCloud<pcl::PointXYZ> pcl_cloud;
    pcl::fromROSMsg(msg, pcl_cloud);

    dst.reserve(dst.size() + pcl_cloud.size());
    for (const auto& p : pcl_cloud.points) {
      if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) continue;
      dst.push_back(p);
    }
  }

  sensor_msgs::msg::PointCloud2 buildAccumulatedCloudLocked(std::string& out_error) {
    if (!has_cloud_) {
      out_error = "No point cloud received yet.";
      return sensor_msgs::msg::PointCloud2();
    }

    if (!use_accumulation_) {
      // No accumulation (none/latest): merge latest clouds from all topics
      pcl::PointCloud<pcl::PointXYZ> merged;
      for (const auto& [idx, cloud] : last_clouds_) {
        appendCloudXYZ(merged, cloud);
      }
      if (merged.size() < 10) {
        out_error = "Merged cloud too small.";
        return sensor_msgs::msg::PointCloud2();
      }
      sensor_msgs::msg::PointCloud2 out;
      pcl::toROSMsg(merged, out);
      out.header.stamp = last_stamp_;
      out.header.frame_id = merge_frame_;
      return out;
    }

    // time_window accumulation
    pruneBufferLocked();
    if (cloud_buffer_.empty()) {
      out_error = "Accumulation enabled but buffer is empty (no clouds in window).";
      return sensor_msgs::msg::PointCloud2();
    }

    pcl::PointCloud<pcl::PointXYZ> merged;
    for (const auto& msg : cloud_buffer_) {
      appendCloudXYZ(merged, msg);
    }

    if (merged.size() < 10) {
      out_error = "Accumulated cloud too small after filtering invalid points.";
      return sensor_msgs::msg::PointCloud2();
    }

    sensor_msgs::msg::PointCloud2 out;
    pcl::toROSMsg(merged, out);
    out.header.stamp = last_stamp_;
    out.header.frame_id = merge_frame_;
    return out;
  }

  static std::vector<Eigen::Vector3f> cloudMsgToEigenPoints(const sensor_msgs::msg::PointCloud2& msg) {
    pcl::PointCloud<pcl::PointXYZ> pcl_cloud;
    pcl::fromROSMsg(msg, pcl_cloud);

    std::vector<Eigen::Vector3f> pts;
    pts.reserve(pcl_cloud.size());
    for (const auto& p : pcl_cloud.points) {
      if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) continue;
      pts.emplace_back(p.x, p.y, p.z);
    }
    return pts;
  }


  bool computeKissFeaturesLocked(mrc::msg::KissFeatures& out_features,
                                 std::string& out_error) {
    if (!has_cloud_) {
      out_error = "No point cloud received yet.";
      return false;
    }

    // Cache is safe if the "effective input" is unchanged.
    const bool accum_unchanged = use_accumulation_ ?
        (cached_features_buffer_size_ == cloud_buffer_.size()) : true;
    if (cached_features_valid_ &&
        cached_features_stamp_ == last_stamp_ &&
        accum_unchanged &&
        cached_features_use_accumulation_ == use_accumulation_) {
      out_features = cached_features_;
      return true;
    }

    std::string acc_err;
    const auto cloud_for_features = buildAccumulatedCloudLocked(acc_err);
    if (!acc_err.empty()) {
      out_error = acc_err;
      return false;
    }

    const auto input_pts = cloudMsgToEigenPoints(cloud_for_features);
    if (input_pts.size() < 10) {
      out_error = "Too few points to extract features.";
      return false;
    }

    const float voxel = static_cast<float>(voxel_size_);
    const auto down = kiss_matcher::VoxelgridSampling(input_pts, voxel);

    const float normal_radius = static_cast<float>(normal_radius_gain_ * voxel_size_);
    const float fpfh_radius = static_cast<float>(fpfh_radius_gain_ * voxel_size_);
    const float thr_lin = static_cast<float>(thr_linearity_);

    kiss_matcher::FasterPFH fpfh(normal_radius, fpfh_radius, thr_lin);
    fpfh.setInputCloud(down);

    std::vector<Eigen::Vector3f> keypoints;
    std::vector<Eigen::VectorXf> descriptors;
    fpfh.ComputeFeature(keypoints, descriptors);

    if (keypoints.size() < 3 || descriptors.size() < 3) {
      out_error = "Feature extraction produced too few keypoints/descriptors.";
      return false;
    }
    if (keypoints.size() != descriptors.size()) {
      out_error = "Feature extraction mismatch: keypoints.size != descriptors.size.";
      return false;
    }

    std_msgs::msg::Header h;
    h.stamp = last_stamp_;
    h.frame_id = last_frame_id_;

    out_features.header = h;
    // Flatten keypoints to float32
    std::vector<float> kp_flat;
    kp_flat.reserve(keypoints.size() * 3);
    for (const auto& p : keypoints) {
      kp_flat.push_back(p.x());
      kp_flat.push_back(p.y());
      kp_flat.push_back(p.z());
    }
    out_features.keypoint_xyz = std::move(kp_flat);

    // Flatten descriptors to float32
    const uint32_t dim = descriptors.empty() ? 0u : static_cast<uint32_t>(descriptors.front().size());
    std::vector<float> desc_flat;
    desc_flat.reserve(descriptors.size() * dim);
    for (const auto& d : descriptors) {
      for (int k = 0; k < d.size(); ++k) {
        desc_flat.push_back(d(k));
      }
    }
    out_features.descriptor_dim = dim;
    out_features.descriptors = std::move(desc_flat);

    cached_features_ = out_features;
    cached_features_stamp_ = last_stamp_;
    cached_features_buffer_size_ = cloud_buffer_.size();
    cached_features_use_accumulation_ = use_accumulation_;
    cached_features_valid_ = true;

    return true;
  }

  void onGetRegistrationData(
      const std::shared_ptr<mrc::srv::GetRegistrationData::Request> req,
      std::shared_ptr<mrc::srv::GetRegistrationData::Response> res) {
    std::lock_guard<std::mutex> lk(mutex_);

    res->robot_id = robot_id_;
    // Return merge_frame since that's where point clouds are transformed to
    // This ensures registration is done in base_link frame for all robots
    res->frame_id = merge_frame_;
    res->stamp = last_stamp_;

    if (!ready_) {
      res->ok = false;
      res->message = "MRC slave not ready yet, try again later";
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                           "Registration data requested but slave is not ready (no point cloud received)");
      return;
    }

    if (req->algorithm == "fricp") {
      std::string err;
      const auto cloud = buildAccumulatedCloudLocked(err);
      if (!err.empty()) {
        res->ok = false;
        res->message = err;
        return;
      }
      res->cloud = cloud;
      res->ok = true;
      res->message = use_accumulation_ ? "OK (cloud, time_window)" : "OK (cloud, latest)";
      return;
    }

    if (req->algorithm == "kiss_matcher") {
      mrc::msg::KissFeatures feat;
      std::string err;
      if (!computeKissFeaturesLocked(feat, err)) {
        res->ok = false;
        res->message = err;
        return;
      }

      const size_t kp_count = feat.keypoint_xyz.size() / 3;
      const size_t expected = kp_count * static_cast<size_t>(feat.descriptor_dim);
      if (expected == 0 || feat.descriptors.size() != expected) {
        res->ok = false;
        res->message = "Invalid features: descriptors.size != (keypoint_xyz.size/3)*descriptor_dim.";
        return;
      }

      // Also include raw cloud for potential ICP refinement on master
      std::string cloud_err;
      res->cloud = buildAccumulatedCloudLocked(cloud_err);

      // Message size comparison
      const size_t feat_bytes =
          feat.keypoint_xyz.size() * sizeof(float) +
          feat.descriptors.size() * sizeof(float) +
          sizeof(uint32_t);  // descriptor_dim
      const size_t cloud_bytes = res->cloud.data.size();
      const size_t cloud_points = cloud_bytes / (res->cloud.point_step > 0 ? res->cloud.point_step : 1);
      RCLCPP_INFO(get_logger(),
          "[SIZE] KissFeatures: %zu keypoints, %.1f KB  |  PointCloud2: %zu pts, %.1f KB  |  %.1fx smaller",
          kp_count, feat_bytes / 1024.0,
          cloud_points, cloud_bytes / 1024.0,
          cloud_bytes > 0 ? static_cast<double>(cloud_bytes) / static_cast<double>(feat_bytes) : 0.0);

      res->features = std::move(feat);

      res->ok = true;
      res->message = use_accumulation_ ? "OK (features, time_window)" : "OK (features, latest)";
      return;
    }

    res->ok = false;
    res->message = "Unknown algorithm '" + req->algorithm + "'";
  }

private:
  std::mutex mutex_;

  bool ready_{false};
  bool has_cloud_{false};
  std::unordered_map<size_t, sensor_msgs::msg::PointCloud2> last_clouds_;  // Per-topic last cloud
  builtin_interfaces::msg::Time last_stamp_;
  std::string last_frame_id_;

  std::deque<sensor_msgs::msg::PointCloud2> cloud_buffer_;
  std::string accumulation_type_{"none"};
  double accumulation_time_sec_{1.0};
  int max_buffer_size_{100};
  bool use_accumulation_{false};

  bool cached_features_valid_{false};
  builtin_interfaces::msg::Time cached_features_stamp_;
  size_t cached_features_buffer_size_{0};
  bool cached_features_use_accumulation_{false};
  mrc::msg::KissFeatures cached_features_;

  std::string robot_id_;
  std::string robot_type_;
  std::vector<std::string> pointcloud_topics_;  // Multiple topics support
  std::string base_frame_;
  std::string sensor_frame_;
  std::string merge_frame_;  // Frame to merge all clouds into
  std::string tf_prefix_;    // Prefix for TF frame lookups
  std::string data_service_name_;

  double voxel_size_{0.3};
  double thr_linearity_{1.0};
  double normal_radius_gain_{3.0};
  double fpfh_radius_gain_{5.0};

  // TF2 for transforming point clouds
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  // Multiple subscriptions for multi-topic support
  std::vector<rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr> cloud_subs_;
  rclcpp::Service<mrc::srv::GetRegistrationData>::SharedPtr data_srv_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MRCSlaveNode>(rclcpp::NodeOptions{}));
  rclcpp::shutdown();
  return 0;
}