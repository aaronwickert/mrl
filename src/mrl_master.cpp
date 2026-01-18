/**
 * @file mrl_master.cpp
 * @brief Implementation of the MRLMaster ROS2 node
 * @author MRL Team
 *
 * This file implements the MRLMaster class which coordinates the
 * multi-robot localization process. The master is a pure coordinator
 * that queries all slave nodes for features and computes pairwise
 * transformations.
 */

#include <../include/mrl/mrl_master.h>

MRLMaster::MRLMaster() : Node("mrl_master") {
    declare_parameters();
    get_parameters();
    init_publishers();
    init_subscribers();
    init_services();

    // Initialize static transform broadcaster
    tf_static_broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);

    // Initialize listener for slave discovery via announcements
    listener_ = std::make_unique<mrl::MRLListener>(this);
    listener_->set_slave_discovered_callback(
        [this](const mrl::SlaveInfo& info) {
            RCLCPP_INFO(this->get_logger(), "Slave discovered: %s (%s)",
                info.slave_id.c_str(), info.robot_type.c_str());
        });

    mrl_setup();

    // Initialize MRL with registration parameters
    float voxel_size = this->get_parameter("kissmatcher.voxel_size").as_double();
    float robin_gain = this->get_parameter("kissmatcher.robin_noise_bound_gain").as_double();
    float solver_gain = this->get_parameter("kissmatcher.solver_noise_bound_gain").as_double();

    kiss_matcher::KISSMatcherConfig config;
    config.voxel_size_ = voxel_size;
    config.robin_noise_bound_gain_ = robin_gain;
    config.robin_noise_bound_ = voxel_size * robin_gain;
    config.solver_noise_bound_gain_ = solver_gain;
    config.solver_noise_bound_ = voxel_size * solver_gain;

    RCLCPP_INFO(this->get_logger(), "Registration config: voxel=%.2f, robin_bound=%.2f, solver_bound=%.2f",
        config.voxel_size_, config.robin_noise_bound_, config.solver_noise_bound_);

    mrl_ = std::make_unique<MRL>(config);

    state_ = MRLMasterState::READY;
    RCLCPP_INFO(this->get_logger(), "MRL Master node initialized and ready.");
}

MRLMaster::~MRLMaster() {
}

void MRLMaster::declare_parameters() {
    RCLCPP_INFO(this->get_logger(), "Declaring ros2 parameters.");

    // KISSMatcher registration parameters
    this->declare_parameter("kissmatcher.voxel_size", 0.3);
    this->declare_parameter("kissmatcher.robin_noise_bound_gain", 1.0);
    this->declare_parameter("kissmatcher.solver_noise_bound_gain", 0.75);

    RCLCPP_INFO(this->get_logger(), "Finished declaring ros2 parameters.");
}

void MRLMaster::get_parameters() {
    RCLCPP_INFO(this->get_logger(), "Getting ros2 parameters.");
    RCLCPP_INFO(this->get_logger(), "Finished getting ros2 parameters.");
}

void MRLMaster::init_publishers() {
    RCLCPP_INFO(this->get_logger(), "Initializing ros2 publishers.");
    RCLCPP_INFO(this->get_logger(), "Finished initializing ros2 publishers.");
}

void MRLMaster::init_subscribers() {
    RCLCPP_INFO(this->get_logger(), "Initializing ros2 subscribers.");
    RCLCPP_INFO(this->get_logger(), "Finished initializing ros2 subscribers.");
}

void MRLMaster::init_services() {
    RCLCPP_INFO(this->get_logger(), "Initializing ros2 services.");

    srv_execute_mrl = this->create_service<mrl::srv::ExecuteMRL>("execute_mrl",
        std::bind(&MRLMaster::handle_execute_mrl, this, std::placeholders::_1, std::placeholders::_2));

    RCLCPP_INFO(this->get_logger(), "Finished initializing ros2 services.");
}

void MRLMaster::mrl_setup() {
    RCLCPP_INFO(this->get_logger(), "Starting MRL setup.");
    RCLCPP_INFO(this->get_logger(), "Finished MRL setup.");
}

void MRLMaster::mrl_reset() {
    slave_features.clear();
    pairwise_transforms.clear();
    slave_clients_status.clear();
    slave_clients_features.clear();
    slave_infos.clear();
}

void MRLMaster::mrl_create_clients_for_slaves() {
    auto discovered = listener_->get_discovered_slaves();

    if (discovered.empty()) {
        RCLCPP_WARN(this->get_logger(), "No slaves have announced. Start slaves and try again.");
        return;
    }

    RCLCPP_INFO(this->get_logger(), "Found %zu announced slave(s):", discovered.size());

    for (const auto& [id, info] : discovered) {
        slave_infos.push_back(info);

        std::string status_service = info.get_status_service_path();
        std::string features_service = info.get_features_service_path();

        slave_clients_status.push_back(this->create_client<mrl::srv::GetReady>(status_service));
        slave_clients_features.push_back(this->create_client<mrl::srv::GetFeatures>(features_service));

        RCLCPP_INFO(this->get_logger(), "  - %s: id='%s', type='%s', ns='%s'",
            info.robot_name.c_str(), info.slave_id.c_str(),
            info.robot_type.c_str(), info.ros_namespace.c_str());
    }
}

bool MRLMaster::mrl_check_ready() {
    if (slave_clients_status.empty()) {
        RCLCPP_ERROR(this->get_logger(), "No slave clients registered.");
        return false;
    }

    RCLCPP_INFO(this->get_logger(), "Checking readiness of %zu slave(s)...", slave_clients_status.size());

    auto request = std::make_shared<mrl::srv::GetReady::Request>();

    for (size_t i = 0; i < slave_clients_status.size(); ++i) {
        auto& client = slave_clients_status[i];
        auto future = client->async_send_request(request);

        // Wait for response with timeout
        auto status = future.wait_for(std::chrono::seconds(5));
        if (status != std::future_status::ready) {
            RCLCPP_ERROR(this->get_logger(), "Timeout waiting for slave %s", slave_infos[i].slave_id.c_str());
            return false;
        }

        auto response = future.get();
        if (!response->ready) {
            RCLCPP_WARN(this->get_logger(), "Slave %s is not ready.", slave_infos[i].slave_id.c_str());
            return false;
        }

        RCLCPP_INFO(this->get_logger(), "Slave %s is ready.", slave_infos[i].slave_id.c_str());
    }

    return true;
}

void MRLMaster::compute_pairwise_transforms() {
    if (slave_features.size() < 2) {
        RCLCPP_WARN(this->get_logger(), "Need at least 2 robots for pairwise registration.");
        return;
    }

    // Use first slave as reference frame
    const auto& ref_points = slave_features[0].first;
    RCLCPP_INFO(this->get_logger(), "Using %s as reference frame (%zu keypoints).",
        slave_infos[0].slave_id.c_str(), ref_points.size());

    // Identity transform for reference robot
    kiss_matcher::RegistrationSolution identity;
    identity.valid = true;
    identity.rotation = Eigen::Matrix3d::Identity();
    identity.translation = Eigen::Vector3d::Zero();
    pairwise_transforms.push_back(identity);

    // Compute transforms from reference to each other robot
    for (size_t i = 1; i < slave_features.size(); ++i) {
        const auto& tgt_points = slave_features[i].first;

        RCLCPP_INFO(this->get_logger(), "Computing transform: %s -> %s (%zu keypoints)...",
            slave_infos[0].slave_id.c_str(), slave_infos[i].slave_id.c_str(), tgt_points.size());

        auto result = mrl_->estimate_transformation(ref_points, tgt_points);
        pairwise_transforms.push_back(result);

        if (result.valid) {
            RCLCPP_INFO(this->get_logger(), "  Transform to %s: VALID", slave_infos[i].slave_id.c_str());
            RCLCPP_INFO(this->get_logger(), "    Translation: [%.3f, %.3f, %.3f]",
                result.translation.x(), result.translation.y(), result.translation.z());
        } else {
            RCLCPP_WARN(this->get_logger(), "  Transform to %s: INVALID", slave_infos[i].slave_id.c_str());
        }
    }
}

void MRLMaster::publish_transforms() {
    if (pairwise_transforms.empty() || slave_infos.empty()) {
        RCLCPP_WARN(this->get_logger(), "No transforms to publish.");
        return;
    }

    std::vector<geometry_msgs::msg::TransformStamped> transforms;
    auto stamp = this->now();

    // Reference frame is the first slave
    std::string reference_frame = slave_infos[0].slave_id + "/base_link";

    for (size_t i = 0; i < pairwise_transforms.size(); ++i) {
        const auto& result = pairwise_transforms[i];

        if (!result.valid) {
            RCLCPP_WARN(this->get_logger(), "Skipping invalid transform for %s",
                slave_infos[i].slave_id.c_str());
            continue;
        }

        geometry_msgs::msg::TransformStamped t;
        t.header.stamp = stamp;
        t.header.frame_id = reference_frame;
        t.child_frame_id = slave_infos[i].slave_id + "/base_link";

        // Set translation
        t.transform.translation.x = result.translation.x();
        t.transform.translation.y = result.translation.y();
        t.transform.translation.z = result.translation.z();

        // Convert rotation matrix to quaternion
        Eigen::Quaterniond q(result.rotation);
        t.transform.rotation.x = q.x();
        t.transform.rotation.y = q.y();
        t.transform.rotation.z = q.z();
        t.transform.rotation.w = q.w();

        transforms.push_back(t);

        RCLCPP_INFO(this->get_logger(), "Publishing transform: %s -> %s",
            t.header.frame_id.c_str(), t.child_frame_id.c_str());
    }

    if (!transforms.empty()) {
        tf_static_broadcaster_->sendTransform(transforms);
        RCLCPP_INFO(this->get_logger(), "Published %zu static transforms.", transforms.size());
    }
}

void MRLMaster::handle_execute_mrl(const std::shared_ptr<mrl::srv::ExecuteMRL::Request> /*request*/,
        std::shared_ptr<mrl::srv::ExecuteMRL::Response> response) {

    RCLCPP_INFO(this->get_logger(), "========================================");
    RCLCPP_INFO(this->get_logger(), "Executing Multi-Robot Localization...");
    RCLCPP_INFO(this->get_logger(), "========================================");

    state_ = MRLMasterState::PROCESSING;

    // Reset state for new registration cycle
    mrl_reset();

    // Step 1: Create clients for all announced slaves
    RCLCPP_INFO(this->get_logger(), "[Step 1/4] Getting announced slaves...");
    mrl_create_clients_for_slaves();

    if (slave_infos.empty()) {
        RCLCPP_ERROR(this->get_logger(), "No slaves announced. Aborting.");
        response->success = false;
        state_ = MRLMasterState::READY;
        return;
    }

    // Step 2: Check all slaves are ready
    RCLCPP_INFO(this->get_logger(), "[Step 2/4] Checking slave readiness...");
    if (!mrl_check_ready()) {
        RCLCPP_ERROR(this->get_logger(), "Not all slaves are ready. Aborting.");
        response->success = false;
        state_ = MRLMasterState::READY;
        return;
    }

    // Step 3: Collect features from ALL slaves
    RCLCPP_INFO(this->get_logger(), "[Step 3/4] Collecting features from all slaves...");
    auto feature_request = std::make_shared<mrl::srv::GetFeatures::Request>();

    for (size_t i = 0; i < slave_clients_features.size(); ++i) {
        auto& client = slave_clients_features[i];

        RCLCPP_INFO(this->get_logger(), "  Requesting features from %s...", slave_infos[i].slave_id.c_str());

        auto future = client->async_send_request(feature_request);
        auto status = future.wait_for(std::chrono::seconds(30));

        if (status != std::future_status::ready) {
            RCLCPP_ERROR(this->get_logger(), "Timeout getting features from %s", slave_infos[i].slave_id.c_str());
            response->success = false;
            state_ = MRLMasterState::READY;
            return;
        }

        auto feature_response = future.get();

        if (!feature_response->success) {
            RCLCPP_ERROR(this->get_logger(), "Failed to get features from %s", slave_infos[i].slave_id.c_str());
            response->success = false;
            state_ = MRLMasterState::READY;
            return;
        }

        auto features = mrl::utils::convert_feature_response(feature_response, this->get_logger());
        slave_features.push_back(features);

        RCLCPP_INFO(this->get_logger(), "  Received %zu keypoints from %s",
            features.first.size(), slave_infos[i].slave_id.c_str());
    }

    // Step 4: Compute pairwise transformations
    RCLCPP_INFO(this->get_logger(), "[Step 4/5] Computing pairwise transformations...");
    compute_pairwise_transforms();

    // Step 5: Publish transforms as static TF
    RCLCPP_INFO(this->get_logger(), "[Step 5/5] Publishing static transforms...");
    publish_transforms();

    RCLCPP_INFO(this->get_logger(), "========================================");
    RCLCPP_INFO(this->get_logger(), "Multi-Robot Localization COMPLETE");
    RCLCPP_INFO(this->get_logger(), "  Robots: %zu", slave_infos.size());
    RCLCPP_INFO(this->get_logger(), "  Transforms computed: %zu", pairwise_transforms.size());
    RCLCPP_INFO(this->get_logger(), "========================================");

    response->success = true;
    state_ = MRLMasterState::READY;
}

/**
 * @brief Main entry point for the MRL Master node
 *
 * Initializes ROS2, creates the MRLMaster node, and spins until shutdown.
 *
 * @param argc Argument count
 * @param argv Argument values
 * @return 0 on successful exit
 */
int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MRLMaster>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
