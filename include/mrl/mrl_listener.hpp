/**
 * @file mrl_listener.hpp
 * @brief Listener for discovering MRL slaves via robot announcements
 * @author MRL Team
 *
 * Subscribes to robot announcement messages to discover slaves.
 * Designed to be expandable for different robot announcement types.
 */

#pragma once

#include <rclcpp/rclcpp.hpp>
#include <hector_multi_robot_msgs/msg/robot_announcement.hpp>

#include <string>
#include <map>
#include <mutex>
#include <functional>

namespace mrl {

/**
 * @brief Information about a discovered robot/slave
 */
struct SlaveInfo {
    std::string slave_id;           ///< Unique identifier (from announcement)
    std::string robot_name;         ///< Human-readable name
    std::string ros_namespace;      ///< ROS namespace for services
    std::string robot_type;         ///< Robot type (derived from namespace/id)
    rclcpp::Time last_seen;         ///< Timestamp of last announcement

    /**
     * @brief Get the full service path for GetFeatures
     */
    std::string get_features_service_path() const {
        if (ros_namespace.empty() || ros_namespace == "/") {
            return "/get_features";
        }
        return ros_namespace + "/get_features";
    }

    /**
     * @brief Get the full service path for GetStatus
     */
    std::string get_status_service_path() const {
        if (ros_namespace.empty() || ros_namespace == "/") {
            return "/get_status";
        }
        return ros_namespace + "/get_status";
    }
};

using SlaveDiscoveredCallback = std::function<void(const SlaveInfo&)>;

/**
 * @brief Listener for discovering MRL slaves via robot announcements
 *
 * Subscribes to hector_multi_robot_msgs/RobotAnnouncement messages
 * published by robot announcers (e.g., ec_swift_announcer).
 */
class MRLListener {
public:
    explicit MRLListener(rclcpp::Node* node) : node_(node) {
        setup_subscribers();
    }

    ~MRLListener() = default;

    void set_slave_discovered_callback(SlaveDiscoveredCallback callback) {
        discovered_callback_ = std::move(callback);
    }

    std::map<std::string, SlaveInfo> get_discovered_slaves() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return slaves_;
    }

    bool has_slaves() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return !slaves_.empty();
    }

    size_t slave_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return slaves_.size();
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        slaves_.clear();
    }

private:
    void setup_subscribers() {
        const rclcpp::QoS qos = rclcpp::QoS(10).reliable().transient_local();

        // Subscribe to global robot announcements
        global_sub_ = node_->create_subscription<hector_multi_robot_msgs::msg::RobotAnnouncement>(
            "/robot_announcement", qos,
            [this](const hector_multi_robot_msgs::msg::RobotAnnouncement::SharedPtr msg) {
                handle_announcement(msg);
            });

        // Subscribe to local announcements
        local_sub_ = node_->create_subscription<hector_multi_robot_msgs::msg::RobotAnnouncement>(
            "robot_announcement", qos,
            [this](const hector_multi_robot_msgs::msg::RobotAnnouncement::SharedPtr msg) {
                handle_announcement(msg);
            });

        RCLCPP_INFO(node_->get_logger(), "MRL Listener: Subscribed to robot announcements");
    }

    void handle_announcement(const hector_multi_robot_msgs::msg::RobotAnnouncement::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mutex_);

        bool is_new = (slaves_.find(msg->id) == slaves_.end());

        SlaveInfo info;
        info.slave_id = msg->id;
        info.robot_name = msg->name;
        info.ros_namespace = msg->ros_namespace;
        info.robot_type = derive_robot_type(msg->id, msg->ros_namespace);
        info.last_seen = node_->now();

        slaves_[msg->id] = info;

        if (is_new) {
            RCLCPP_INFO(node_->get_logger(), "Discovered robot: id='%s', name='%s', ns='%s', type='%s'",
                info.slave_id.c_str(), info.robot_name.c_str(),
                info.ros_namespace.c_str(), info.robot_type.c_str());

            if (discovered_callback_) {
                discovered_callback_(info);
            }
        }
    }

    /**
     * @brief Derive robot type from ID or namespace
     * Expandable: add more robot type detection logic here
     */
    std::string derive_robot_type(const std::string& id, const std::string& ns) {
        // Check for known robot types in ID or namespace
        if (id.find("ec_swift") != std::string::npos || ns.find("ec_swift") != std::string::npos) {
            return "ec_swift";
        }
        return "generic";
    }

    rclcpp::Node* node_;
    mutable std::mutex mutex_;
    std::map<std::string, SlaveInfo> slaves_;

    rclcpp::Subscription<hector_multi_robot_msgs::msg::RobotAnnouncement>::SharedPtr global_sub_;
    rclcpp::Subscription<hector_multi_robot_msgs::msg::RobotAnnouncement>::SharedPtr local_sub_;

    SlaveDiscoveredCallback discovered_callback_;
};

} // namespace mrl
