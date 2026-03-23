#pragma once

#include <optional>
#include <string>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include "mrc/msg/kiss_features.hpp"

namespace mrc {

  struct RegistrationInput {
    std::optional<sensor_msgs::msg::PointCloud2> cloud;
    std::optional<mrc::msg::KissFeatures> features;
    std::string frame_id;
  };

  struct RegistrationResult {
    bool ok{false};
    std::string message;
    geometry_msgs::msg::TransformStamped T_target_from_source;

    // Generic score (keep for backwards compatibility / quick sorting)
    double score{0.0};

    // Pose-graph relevant quality metrics (optional if backend can't provide)
    std::optional<double> rmse_t_m;        // translation RMSE [m]
    std::optional<double> rmse_r_rad;      // rotation RMSE [rad]
    std::optional<int> num_inliers;
    std::optional<int> num_correspondences;
    std::optional<double> confidence;      // 0..1 (master may compute this too)
  };

}  // namespace mrc