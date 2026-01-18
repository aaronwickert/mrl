/**
 * @file mrl_master.h
 * @brief MRL Master node for coordinating multi-robot localization
 * @author MRL Team
 *
 * This file contains the MRLMaster class which acts as the central coordinator
 * in the MRL (Multi-Robot Localization) system. The master node is a pure
 * coordinator that discovers slave nodes, collects their point cloud features,
 * and computes relative transformations between all robots.
 *
 * @par Deployment Model
 * - Each robot runs exactly one MRLSlave node for feature extraction
 * - One robot additionally runs the MRLMaster node for coordination
 * - The master does NOT extract features itself; it queries all slaves
 * - The robot running the master also runs a slave for its own features
 *
 * @code
 * Robot 1: [MRLMaster] + [MRLSlave] ← coordinates + provides own features
 * Robot 2: [MRLSlave]               ← provides features only
 * Robot 3: [MRLSlave]               ← provides features only
 * @endcode
 */

#pragma once

#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/static_transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>

#include "mrl/mrl.h"
#include "mrl/mrl_utils.hpp"
#include "mrl/mrl_listener.hpp"

#include "mrl/srv/get_ready.hpp"
#include "mrl/srv/get_features.hpp"
#include "mrl/srv/execute_mrl.hpp"

/**
 * @enum MRLMasterState
 * @brief Represents the operational state of the MRL master node
 */
enum class MRLMasterState {
    NOT_READY,   ///< Node is initializing
    READY,       ///< Node is ready to coordinate registration
    PROCESSING   ///< Node is currently processing a registration request
};

/**
 * @class MRLMaster
 * @brief ROS2 node that coordinates the multi-robot localization process
 *
 * The MRLMaster node is a **pure coordinator** responsible for:
 * - Discovering active MRL slave nodes in the ROS2 network
 * - Collecting FPFH features from ALL slave robots (including co-located slave)
 * - Computing pairwise transformations between robots
 * - Providing the ExecuteMRL service to trigger the localization pipeline
 *
 * @par Architecture
 * The master node does NOT perform feature extraction itself. Instead:
 * - All robots run MRLSlave nodes that handle feature extraction
 * - The master queries all slaves for features via GetFeatures service
 * - One robot runs both master and slave nodes
 *
 * @par Transformation Convention
 * Transformations are computed pairwise between all slave nodes.
 * The first discovered slave is used as the reference frame.
 *
 * @par Usage
 * @code
 * // On the coordinating robot, launch both nodes:
 * ros2 run mrl mrl_master &
 * ros2 run mrl mrl_slave --ros-args -p pointcloud_topic:=/robot1/points
 *
 * // On other robots, launch only slave:
 * ros2 run mrl mrl_slave --ros-args -p pointcloud_topic:=/robot2/points
 *
 * // Trigger localization
 * ros2 service call /execute_mrl mrl/srv/ExecuteMRL
 * @endcode
 *
 * @see MRLSlave
 */
class MRLMaster : public rclcpp::Node {
public:
    /**
     * @brief Constructs the MRL Master node
     *
     * Initializes ROS2 interfaces and parameters.
     * The node name is set to "mrl_master".
     *
     * @note The master does NOT create an MRL instance; it only coordinates.
     */
    MRLMaster();

    /**
     * @brief Destructor
     */
    ~MRLMaster();

private:
    /// @name ROS2 Node Setup
    /// @{

    /**
     * @brief Declares all ROS2 parameters for the node
     */
    void declare_parameters();

    /**
     * @brief Retrieves parameter values from the ROS2 parameter server
     */
    void get_parameters();

    /**
     * @brief Initializes ROS2 publishers
     */
    void init_publishers();

    /**
     * @brief Initializes ROS2 subscribers
     */
    void init_subscribers();

    /**
     * @brief Initializes ROS2 services
     *
     * Creates the ExecuteMRL service for triggering registration.
     */
    void init_services();

    /// @}

    /// @name MRL Coordination
    /// @{

    /**
     * @brief Performs initial setup
     */
    void mrl_setup();

    /**
     * @brief Resets the MRL state for a new registration cycle
     *
     * Clears all stored features, transforms, and service clients.
     */
    void mrl_reset();

    /**
     * @brief Creates service clients for all discovered slaves
     *
     * Uses the MRLListener to get discovered slaves and creates
     * service clients for each one.
     */
    void mrl_create_clients_for_slaves();

    /**
     * @brief Checks if all discovered slaves are ready
     *
     * @return true if all slaves report ready, false otherwise
     */
    bool mrl_check_ready();

    /**
     * @brief Computes pairwise transformations between all robots
     *
     * Uses the first slave as reference frame and computes transforms
     * from reference to all other slaves.
     */
    void compute_pairwise_transforms();

    /**
     * @brief Publishes computed transforms as static TF transforms
     *
     * Broadcasts each pairwise transformation as a static transform from
     * the reference robot's frame to each other robot's frame.
     */
    void publish_transforms();

    /// @}

    /// @name Service Handlers
    /// @{

    /**
     * @brief Handles ExecuteMRL service requests
     *
     * Executes the full multi-robot localization pipeline:
     * 1. Discovers all slave nodes (including co-located slave)
     * 2. Verifies all slaves are ready
     * 3. Collects FPFH features from ALL slaves
     * 4. Computes pairwise transformations
     *
     * @param request Service request (empty)
     * @param response Service response containing success status
     */
    void handle_execute_mrl(const std::shared_ptr<mrl::srv::ExecuteMRL::Request> request,
        std::shared_ptr<mrl::srv::ExecuteMRL::Response> response);

    /// @}

    /// @name ROS2 Interfaces
    /// @{

    /** @brief Service server for ExecuteMRL requests */
    rclcpp::Service<mrl::srv::ExecuteMRL>::SharedPtr srv_execute_mrl;

    /** @brief Static transform broadcaster for publishing robot transforms */
    std::shared_ptr<tf2_ros::StaticTransformBroadcaster> tf_static_broadcaster_;

    /** @brief Listener for discovering slaves via announcements */
    std::unique_ptr<mrl::MRLListener> listener_;

    /** @brief Service clients for checking slave readiness */
    std::vector<rclcpp::Client<mrl::srv::GetReady>::SharedPtr> slave_clients_status;

    /** @brief Service clients for retrieving slave features */
    std::vector<rclcpp::Client<mrl::srv::GetFeatures>::SharedPtr> slave_clients_features;

    /** @brief Info for discovered slaves (from listener) */
    std::vector<mrl::SlaveInfo> slave_infos;

    /// @}

    /// @name State
    /// @{

    /** @brief Current operational state of the master node */
    MRLMasterState state_ = MRLMasterState::NOT_READY;

    /// @}

    /// @name Collected Data
    /// @{

    /**
     * @brief FPFH features collected from all slave robots
     *
     * Vector of pairs, one per slave, each containing:
     * - first: 3D keypoint positions
     * - second: FPFH descriptors
     *
     * Index corresponds to slave_namespaces order.
     */
    std::vector<std::pair<std::vector<Eigen::Vector3f>, std::vector<Eigen::VectorXf>>> slave_features;

    /**
     * @brief Computed pairwise transformations between robots
     *
     * Transforms from reference robot (index 0) to each other robot.
     * Index i contains transform from robot 0 to robot i.
     */
    std::vector<kiss_matcher::RegistrationSolution> pairwise_transforms;

    /**
     * @brief MRL instance for computing transformations
     *
     * Used only for estimate_transformation(), not for feature extraction.
     */
    std::unique_ptr<MRL> mrl_;

    /// @}
};
