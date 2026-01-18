/**
 * @file mrl_listener.hpp
 * @brief Listener class for MRL master to discover slaves
 * @author MRL Team
 *
 * This file provides the MRLListener class that the master node uses to
 * discover and track slave nodes. It subscribes to SlaveAnnouncement messages
 * and maintains a registry of known slaves with their service endpoints.
 *
 * @par Usage
 * @code
 * class MRLMaster : public rclcpp::Node {
 *     MRLMaster() : Node("mrl_master") {
 *         listener_ = std::make_unique<mrl::MRLListener>(this);
 *         listener_->set_slave_discovered_callback(
 *             [this](const mrl::SlaveInfo& info) {
 *                 RCLCPP_INFO(get_logger(), "Discovered: %s", info.slave_id.c_str());
 *             });
 *     }
 *
 *     void execute_mrl() {
 *         auto slaves = listener_->get_discovered_slaves();
 *         for (const auto& [id, info] : slaves) {
 *             // Create service clients using info.ros_namespace
 *         }
 *     }
 * };
 * @endcode
 */

#pragma once

#include <rclcpp/rclcpp.hpp>
#include "mrl/msg/slave_announcement.hpp"

#include <string>
#include <map>
#include <mutex>
#include <functional>
#include <chrono>

namespace mrl {

/**
 * @brief Information about a discovered slave
 */
struct SlaveInfo {
    std::string slave_id;           ///< Unique slave identifier
    std::string robot_name;         ///< Human-readable name
    std::string ros_namespace;      ///< ROS namespace for services
    std::string robot_type;         ///< Robot type (e.g., "ec_swift")
    uint8_t sensor_type;            ///< Sensor type (0=LiDAR, 1=RGB-D, 2=Stereo)
    std::string get_features_service;  ///< GetFeatures service name
    std::string get_status_service;    ///< GetStatus service name
    rclcpp::Time last_seen;         ///< Timestamp of last announcement

    /**
     * @brief Get the full service path for GetFeatures
     * @return Full service path (namespace + service name)
     */
    std::string get_features_service_path() const {
        if (ros_namespace == "/" || ros_namespace.empty()) {
            return "/" + get_features_service;
        }
        return ros_namespace + "/" + get_features_service;
    }

    /**
     * @brief Get the full service path for GetStatus
     * @return Full service path (namespace + service name)
     */
    std::string get_status_service_path() const {
        if (ros_namespace == "/" || ros_namespace.empty()) {
            return "/" + get_status_service;
        }
        return ros_namespace + "/" + get_status_service;
    }
};

/**
 * @brief Callback type for slave discovery events
 */
using SlaveDiscoveredCallback = std::function<void(const SlaveInfo&)>;

/**
 * @brief Callback type for slave removal events (timeout/disconnect)
 */
using SlaveRemovedCallback = std::function<void(const std::string& slave_id)>;

/**
 * @brief Listener for discovering MRL slaves via announcements
 *
 * The MRLListener subscribes to slave announcement topics and maintains
 * a registry of known slaves. It supports filtering by robot type and
 * provides callbacks for discovery events.
 */
class MRLListener {
public:
    /**
     * @brief Construct a new MRLListener
     *
     * @param node The ROS2 node to attach subscribers to
     * @param slave_timeout Duration after which a slave is considered stale
     *                      (set to 0 to disable timeout checking)
     */
    explicit MRLListener(rclcpp::Node* node,
                         std::chrono::seconds slave_timeout = std::chrono::seconds(0))
        : node_(node), slave_timeout_(slave_timeout) {
        setup_subscribers();
    }

    ~MRLListener() = default;

    /**
     * @brief Set callback for when a new slave is discovered
     * @param callback Function to call with SlaveInfo
     */
    void set_slave_discovered_callback(SlaveDiscoveredCallback callback) {
        discovered_callback_ = std::move(callback);
    }

    /**
     * @brief Set callback for when a slave is removed (timeout)
     * @param callback Function to call with slave_id
     */
    void set_slave_removed_callback(SlaveRemovedCallback callback) {
        removed_callback_ = std::move(callback);
    }

    /**
     * @brief Get all discovered slaves
     * @return Map of slave_id to SlaveInfo
     */
    std::map<std::string, SlaveInfo> get_discovered_slaves() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return slaves_;
    }

    /**
     * @brief Get discovered slaves filtered by robot type
     * @param robot_type Robot type to filter by
     * @return Map of slave_id to SlaveInfo for matching slaves
     */
    std::map<std::string, SlaveInfo> get_slaves_by_type(const std::string& robot_type) const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::map<std::string, SlaveInfo> filtered;
        for (const auto& [id, info] : slaves_) {
            if (info.robot_type == robot_type) {
                filtered[id] = info;
            }
        }
        return filtered;
    }

    /**
     * @brief Get a specific slave by ID
     * @param slave_id The slave ID to look up
     * @return Pointer to SlaveInfo if found, nullptr otherwise
     */
    const SlaveInfo* get_slave(const std::string& slave_id) const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = slaves_.find(slave_id);
        if (it != slaves_.end()) {
            return &it->second;
        }
        return nullptr;
    }

    /**
     * @brief Check if any slaves have been discovered
     * @return true if at least one slave is known
     */
    bool has_slaves() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return !slaves_.empty();
    }

    /**
     * @brief Get the number of discovered slaves
     * @return Number of slaves
     */
    size_t slave_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return slaves_.size();
    }

    /**
     * @brief Remove stale slaves that haven't announced recently
     *
     * Only effective if slave_timeout > 0 was set in constructor.
     * Call this periodically if you want to prune disconnected slaves.
     */
    void prune_stale_slaves() {
        if (slave_timeout_.count() == 0) {
            return;
        }

        std::lock_guard<std::mutex> lock(mutex_);
        auto now = node_->now();
        std::vector<std::string> to_remove;

        for (const auto& [id, info] : slaves_) {
            auto age = now - info.last_seen;
            if (age > rclcpp::Duration(slave_timeout_)) {
                to_remove.push_back(id);
            }
        }

        for (const auto& id : to_remove) {
            RCLCPP_WARN(node_->get_logger(), "Removing stale slave: %s", id.c_str());
            slaves_.erase(id);
            if (removed_callback_) {
                removed_callback_(id);
            }
        }
    }

    /**
     * @brief Clear all discovered slaves
     */
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        slaves_.clear();
    }

private:
    void setup_subscribers() {
        // Use transient_local QoS to receive announcements from slaves that
        // published before this listener started
        const rclcpp::QoS announcement_qos = rclcpp::QoS(10).reliable().transient_local();

        // Subscribe to global announcement topic
        global_subscriber_ = node_->create_subscription<mrl::msg::SlaveAnnouncement>(
            "/mrl/slave_announcement", announcement_qos,
            [this](const mrl::msg::SlaveAnnouncement::SharedPtr msg) {
                handle_announcement(msg);
            });

        // Also subscribe to local topic for slaves in same namespace
        local_subscriber_ = node_->create_subscription<mrl::msg::SlaveAnnouncement>(
            "slave_announcement", announcement_qos,
            [this](const mrl::msg::SlaveAnnouncement::SharedPtr msg) {
                handle_announcement(msg);
            });

        RCLCPP_INFO(node_->get_logger(), "MRL Listener initialized, waiting for slave announcements...");
    }

    void handle_announcement(const mrl::msg::SlaveAnnouncement::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mutex_);

        bool is_new = (slaves_.find(msg->slave_id) == slaves_.end());

        SlaveInfo info;
        info.slave_id = msg->slave_id;
        info.robot_name = msg->robot_name;
        info.ros_namespace = msg->ros_namespace;
        info.robot_type = msg->robot_type;
        info.sensor_type = msg->sensor_type;
        info.get_features_service = msg->get_features_service;
        info.get_status_service = msg->get_status_service;
        info.last_seen = node_->now();

        slaves_[msg->slave_id] = info;

        if (is_new) {
            RCLCPP_INFO(node_->get_logger(),
                "Discovered slave: id='%s', name='%s', ns='%s', type='%s'",
                info.slave_id.c_str(), info.robot_name.c_str(),
                info.ros_namespace.c_str(), info.robot_type.c_str());

            if (discovered_callback_) {
                discovered_callback_(info);
            }
        } else {
            RCLCPP_DEBUG(node_->get_logger(), "Updated slave: %s", info.slave_id.c_str());
        }
    }

    rclcpp::Node* node_;
    std::chrono::seconds slave_timeout_;
    mutable std::mutex mutex_;

    std::map<std::string, SlaveInfo> slaves_;

    rclcpp::Subscription<mrl::msg::SlaveAnnouncement>::SharedPtr global_subscriber_;
    rclcpp::Subscription<mrl::msg::SlaveAnnouncement>::SharedPtr local_subscriber_;

    SlaveDiscoveredCallback discovered_callback_;
    SlaveRemovedCallback removed_callback_;
};

} // namespace mrl
