/**
 * @file mrl_utils.hpp
 * @brief Utility functions for the MRL (Multi-Robot Localization) system
 * @author MRL Team
 *
 * This file contains helper functions used throughout the MRL system,
 * including namespace extraction and ROS message conversion utilities.
 */

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

/**
 * @namespace mrl
 * @brief Main namespace for the Multi-Robot Localization system
 */
namespace mrl {

/**
 * @namespace mrl::utils
 * @brief Utility functions for MRL operations
 */
namespace utils {

/**
 * @brief Extracts the namespace from a fully qualified ROS2 node name
 *
 * Parses a full node name (e.g., "/robot1/mrl_slave") and returns
 * the namespace portion (e.g., "/robot1").
 *
 * @param full_node_name The complete node name including namespace
 * @return The namespace portion of the node name
 *
 * @par Examples
 * - "/robot1/mrl_slave" -> "/robot1"
 * - "/mrl_slave" -> "/"
 * - "mrl_slave" -> "/"
 *
 * @note Returns "/" if no namespace is found or for root namespace nodes
 */
inline std::string extract_namespace(const std::string& full_node_name) {
    size_t last_slash = full_node_name.find_last_of('/');
    if (last_slash == 0) return "/";
    if (last_slash != std::string::npos) {
        return full_node_name.substr(0, last_slash);
    }
    return "/";
}

/**
 * @brief Converts a GetFeatures service response to Eigen vector format
 *
 * Transforms the ROS2 message types used in the GetFeatures service
 * response into Eigen vector types suitable for the MRL registration pipeline.
 *
 * @param response Shared pointer to the GetFeatures service response
 * @param logger ROS2 logger for diagnostic output (currently unused)
 * @return std::pair containing:
 *         - first: Vector of 3D keypoint positions (Eigen::Vector3f)
 *         - second: Vector of FPFH descriptors (Eigen::VectorXf)
 *
 * @par Conversion Details
 * - geometry_msgs/Point[] -> std::vector<Eigen::Vector3f>
 *   - Converts double precision to float
 * - std_msgs/Float64MultiArray[] -> std::vector<Eigen::VectorXf>
 *   - Each array becomes one Eigen vector
 *   - Converts double precision to float
 *
 * @note The returned vectors are value types, not references, to ensure
 *       data ownership is transferred to the caller.
 *
 * @see mrl::srv::GetFeatures
 */
inline std::pair<std::vector<Eigen::Vector3f>, std::vector<Eigen::VectorXf>>
convert_feature_response(
    std::shared_ptr<mrl::srv::GetFeatures::Response> response,
    rclcpp::Logger /*logger*/) {

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

}  // namespace utils
}  // namespace mrl

#endif  // MRL_UTILS_HPP
