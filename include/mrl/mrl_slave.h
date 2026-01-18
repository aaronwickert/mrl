/**
 * @file mrl_slave.h
 * @brief MRL Slave node for individual robot feature extraction
 * @author MRL Team
 *
 * This file contains the MRLSlave class which runs on each robot participating
 * in the MRL (Multi-Robot Localization) system. The slave node subscribes to
 * point cloud data, extracts FPFH features on demand, and provides them to
 * the master node for multi-robot registration.
 *
 * Robot announcement/registration is handled externally by the robot's own
 * announcer (e.g., ec_swift_announcer). The master discovers slaves via
 * MRLListener which listens for SlaveAnnouncement messages.
 */

#pragma once

#include <mutex>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>

#include "mrl/mrl.h"
#include "mrl/srv/get_ready.hpp"
#include "mrl/srv/get_features.hpp"
#include "mrl/srv/execute_mrl.hpp"

/**
 * @enum MRLSlaveState
 * @brief Represents the operational state of an MRL slave node
 */
enum class MRLSlaveState {
    NOT_READY,   ///< Node is initializing or waiting for master
    READY,       ///< Node is ready to process feature requests
    PROCESSING   ///< Node is currently extracting features
};

/**
 * @class MRLSlave
 * @brief ROS2 node that provides point cloud features for multi-robot localization
 *
 * The MRLSlave node is responsible for:
 * - Subscribing to point cloud data from the robot's LiDAR sensor
 * - Storing the latest point cloud for feature extraction
 * - Extracting FPFH features when requested by the master node
 * - Reporting readiness status to the master
 *
 * @par Communication
 * The slave provides two services:
 * - GetReady: Returns whether the slave is ready for feature extraction
 * - GetFeatures: Extracts and returns FPFH features from the latest point cloud
 *
 * @par Discovery
 * Robot announcement is handled externally by the robot's startup announcer
 * (e.g., ec_swift_announcer). The master discovers slaves via MRLListener.
 *
 * @par Startup Sequence
 * 1. Node initializes and declares parameters
 * 2. Subscribes to point cloud topic
 * 3. Transitions to READY state
 * 4. Responds to service requests
 *
 * @par Usage
 * @code
 * // Launch with default parameters
 * ros2 run mrl mrl_slave
 *
 * // Launch with custom point cloud topic
 * ros2 run mrl mrl_slave --ros-args -p pointcloud_topic:=/lidar/points
 * @endcode
 *
 * @see MRLMaster
 */
class MRLSlave : public rclcpp::Node {
public:
    /**
     * @brief Constructs the MRL Slave node
     *
     * Initializes ROS2 interfaces, parameters, MRL feature extractor,
     * and waits for the master node to come online.
     * The node name is set to "mrl_slave".
     *
     * @note Constructor blocks until master node is detected
     */
    MRLSlave();

    /**
     * @brief Destructor
     */
    ~MRLSlave();

private:
    /// @name ROS2 Node Setup
    /// @{

    /**
     * @brief Declares all ROS2 parameters for the node
     *
     * Parameters declared:
     * - pointcloud_topic (string): Topic to subscribe for point clouds
     *   Default: "/pointcloud"
     */
    void declare_parameters();

    /**
     * @brief Retrieves parameter values from the ROS2 parameter server
     */
    void get_parameters();

    /**
     * @brief Initializes ROS2 publishers
     * @note Currently no publishers are used
     */
    void init_publishers();

    /**
     * @brief Initializes ROS2 subscribers
     *
     * Creates subscription to point cloud topic configured via parameters.
     */
    void init_subscribers();

    /**
     * @brief Initializes ROS2 services
     *
     * Creates the following services:
     * - get_features: Returns FPFH features from latest point cloud
     * - get_status: Returns current readiness status
     */
    void init_services();

    /// @}

    /// @name MRL Operations
    /// @{

    /**
     * @brief Performs initial MRL setup
     *
     * Transitions to READY state immediately (non-blocking).
     */
    void mrl_setup();

    /// @}

    /// @name Callbacks
    /// @{

    /**
     * @brief Callback for incoming point cloud messages
     *
     * Converts the ROS2 PointCloud2 message to PCL format and stores
     * it for later feature extraction. Thread-safe via mutex.
     *
     * @param msg Incoming point cloud message
     */
    void pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

    /// @}

    /// @name Service Handlers
    /// @{

    /**
     * @brief Handles GetFeatures service requests
     *
     * Extracts FPFH features from the latest stored point cloud:
     * 1. Checks if a point cloud is available
     * 2. Uses MRL to extract FPFH keypoints and descriptors
     * 3. Converts results to ROS2 message format
     * 4. Returns features in the service response
     *
     * @param request Service request (contains timestamp, currently unused)
     * @param response Service response containing:
     *                 - success: Whether extraction succeeded
     *                 - points_3f: 3D keypoint positions
     *                 - points_xf: FPFH descriptors as Float64MultiArray
     *                 - sensor_type: Sensor type identifier (0 = LiDAR)
     */
    void handle_get_features(const std::shared_ptr<mrl::srv::GetFeatures::Request> request,
        std::shared_ptr<mrl::srv::GetFeatures::Response> response);

    /**
     * @brief Handles GetReady service requests
     *
     * Returns the current readiness status of the slave node.
     *
     * @param request Service request (empty)
     * @param response Service response containing ready status
     */
    void handle_get_status(const std::shared_ptr<mrl::srv::GetReady::Request> request,
        std::shared_ptr<mrl::srv::GetReady::Response> response);

    /// @}

    /// @name ROS2 Interfaces
    /// @{

    /** @brief Service server for GetFeatures requests */
    rclcpp::Service<mrl::srv::GetFeatures>::SharedPtr srv_get_features;

    /** @brief Service server for GetReady requests */
    rclcpp::Service<mrl::srv::GetReady>::SharedPtr srv_get_status;

    /** @brief Subscription to point cloud topic */
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_pointcloud;

    /// @}

    /// @name MRL Components
    /// @{

    /** @brief MRL instance for FPFH feature extraction */
    std::unique_ptr<MRL> mrl_;

    /// @}

    /// @name State
    /// @{

    /** @brief Current operational state of the slave node */
    MRLSlaveState state_ = MRLSlaveState::NOT_READY;

    /// @}

    /// @name Point Cloud Storage
    /// @{

    /** @brief Latest received point cloud (PCL format) */
    pcl::PointCloud<pcl::PointXYZ>::Ptr latest_cloud_;

    /** @brief Accumulated point cloud for non-repetitive mode */
    pcl::PointCloud<pcl::PointXYZ>::Ptr accumulated_cloud_;

    /** @brief Mutex protecting access to point clouds */
    std::mutex cloud_mutex_;

    /** @brief Flag indicating if a point cloud has been received */
    bool has_cloud_ = false;

    /** @brief Timestamp when accumulation started */
    rclcpp::Time accumulation_start_time_;

    /** @brief Flag indicating if currently accumulating */
    bool is_accumulating_ = false;

    /// @}

    /// @name Parameters
    /// @{

    /** @brief Topic name for point cloud subscription */
    std::string pointcloud_topic_;

    /** @brief Robot type identifier (e.g., "ec_swift", "spot") */
    std::string robot_type_;

    /** @brief Sensor type: 0=LiDAR, 1=RGB-D, 2=Stereo */
    int sensor_type_;

    /// @}
};
