#ifndef MRL_UTILS_HPP
#define MRL_UTILS_HPP

#include <string>
#include <vector>
#include <memory>
#include <Eigen/Core>
#include <geometry_msgs/msg/point.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <mrl/srv/get_features.hpp>
#include <rclcpp/rclcpp.hpp>

namespace mrl {
namespace utils {
inline std::string extract_namespace(const std::string& full_node_name) {
    size_t last_slash = full_node_name.find_last_of('/');
    if (last_slash == 0) return "/";
    if (last_slash != std::string::npos) {
        return full_node_name.substr(0, last_slash);
    }
    return "/";
}

inline std::pair<std::vector<Eigen::Vector3f>, std::vector<Eigen::VectorXf>>
convert_feature_response(
    std::shared_ptr<mrl::srv::GetFeatures::Response> response,
    rclcpp::Logger logger) {
    
    std::vector<Eigen::Vector3f> points_3d;
    std::vector<Eigen::VectorXf> features;
    
    
    // Convert geometry_msgs/Point[] to std::vector<Eigen::Vector3f>
    points_3d.reserve(response->points_3f.size());
    for (const auto& point : response->points_3f) {
        points_3d.emplace_back(
            static_cast<float>(point.x),
            static_cast<float>(point.y),
            static_cast<float>(point.z)
        );
    }
    
    // Convert std_msgs/Float64MultiArray[] to std::vector<Eigen::VectorXf>
    features.reserve(response->points_xf.size());
    for (const auto& feature_array : response->points_xf) {
        // Get the size of the feature vector
        size_t feature_size = feature_array.data.size();
        
        // Create Eigen vector and fill it
        Eigen::VectorXf feature(feature_size);
        for (size_t i = 0; i < feature_size; i++) {
            feature(i) = static_cast<float>(feature_array.data[i]);
        }
        
        features.push_back(feature);
    }
    return {points_3d, features};
}
}