/**
 * @file mrl_slave.cpp
 * @brief Implementation of the MRLSlave ROS2 node
 * @author MRL Team
 *
 * This file implements the MRLSlave class which handles point cloud
 * subscription, FPFH feature extraction, and service responses for
 * the multi-robot localization system.
 */

#include <../include/mrl/mrl_slave.h>

MRLSlave::MRLSlave() : Node("mrl_slave") {
    latest_cloud_ = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());

    declare_parameters();
    get_parameters();
    init_publishers();
    init_subscribers();
    init_services();

    // Initialize MRL with default configuration
    kiss_matcher::KISSMatcherConfig config;
    mrl_ = std::make_unique<MRL>(config);

    mrl_setup();
}

MRLSlave::~MRLSlave() {
}

void MRLSlave::declare_parameters() {
    RCLCPP_INFO(this->get_logger(), "Declaring ros2 parameters.");

    this->declare_parameter("pointcloud_topic", "/pointcloud");
    this->declare_parameter("robot_type", "generic");
    this->declare_parameter("sensor_type", 0);  // 0=LiDAR, 1=RGB-D, 2=Stereo

    RCLCPP_INFO(this->get_logger(), "Finished declaring ros2 parameters.");
}

void MRLSlave::get_parameters() {
    RCLCPP_INFO(this->get_logger(), "Getting ros2 parameters.");

    pointcloud_topic_ = this->get_parameter("pointcloud_topic").as_string();
    robot_type_ = this->get_parameter("robot_type").as_string();
    sensor_type_ = this->get_parameter("sensor_type").as_int();

    RCLCPP_INFO(this->get_logger(), "Robot type: %s", robot_type_.c_str());
    RCLCPP_INFO(this->get_logger(), "Point cloud topic: %s", pointcloud_topic_.c_str());
    RCLCPP_INFO(this->get_logger(), "Sensor type: %d", sensor_type_);
    RCLCPP_INFO(this->get_logger(), "Finished getting ros2 parameters.");
}

void MRLSlave::init_publishers() {
    RCLCPP_INFO(this->get_logger(), "Initializing ros2 publishers.");
    RCLCPP_INFO(this->get_logger(), "Finished initializing ros2 publishers.");
}

void MRLSlave::init_subscribers() {
    RCLCPP_INFO(this->get_logger(), "Initializing ros2 subscribers.");

    sub_pointcloud = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        pointcloud_topic_, 10,
        std::bind(&MRLSlave::pointcloud_callback, this, std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(), "Subscribed to pointcloud topic: %s", pointcloud_topic_.c_str());
    RCLCPP_INFO(this->get_logger(), "Finished initializing ros2 subscribers.");
}

void MRLSlave::init_services() {
    RCLCPP_INFO(this->get_logger(), "Initializing ros2 services.");

    srv_get_features = this->create_service<mrl::srv::GetFeatures>("get_features",
        std::bind(&MRLSlave::handle_get_features, this, std::placeholders::_1, std::placeholders::_2));

    srv_get_status = this->create_service<mrl::srv::GetReady>("get_status",
        std::bind(&MRLSlave::handle_get_status, this, std::placeholders::_1, std::placeholders::_2));

    RCLCPP_INFO(this->get_logger(), "Finished initializing ros2 services.");
}

void MRLSlave::mrl_setup() {
    RCLCPP_INFO(this->get_logger(), "Starting MRL setup.");

    // Slave is immediately ready - master discovers via external announcements
    state_ = MRLSlaveState::READY;

    RCLCPP_INFO(this->get_logger(), "MRL Slave READY. Waiting for feature requests.");
    RCLCPP_INFO(this->get_logger(), "Finished MRL setup.");
}

void MRLSlave::pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(cloud_mutex_);

    // Convert ROS2 PointCloud2 message to PCL PointCloud format
    pcl::fromROSMsg(*msg, *latest_cloud_);
    has_cloud_ = true;

    RCLCPP_DEBUG(this->get_logger(), "Received pointcloud with %zu points", latest_cloud_->size());
}

void MRLSlave::handle_get_features(const std::shared_ptr<mrl::srv::GetFeatures::Request> /*request*/,
        std::shared_ptr<mrl::srv::GetFeatures::Response> response) {

    RCLCPP_INFO(this->get_logger(), "Received GetFeatures request.");

    std::lock_guard<std::mutex> lock(cloud_mutex_);

    // Check if point cloud data is available
    if (!has_cloud_ || latest_cloud_->empty()) {
        RCLCPP_WARN(this->get_logger(), "No point cloud available for feature extraction.");
        response->success = false;
        return;
    }

    state_ = MRLSlaveState::PROCESSING;

    // Extract FPFH features using MRL
    auto [keypoints, descriptors] = mrl_->get_fpfh(latest_cloud_);

    if (keypoints.empty()) {
        RCLCPP_WARN(this->get_logger(), "Feature extraction returned no keypoints.");
        response->success = false;
        state_ = MRLSlaveState::READY;
        return;
    }

    RCLCPP_INFO(this->get_logger(), "Extracted %zu keypoints with FPFH descriptors.", keypoints.size());

    // Convert keypoints to geometry_msgs/Point[] format
    response->points_3f.reserve(keypoints.size());
    for (const auto& kp : keypoints) {
        geometry_msgs::msg::Point point;
        point.x = kp.x();
        point.y = kp.y();
        point.z = kp.z();
        response->points_3f.push_back(point);
    }

    // Convert descriptors to std_msgs/Float64MultiArray[] format
    response->points_xf.reserve(descriptors.size());
    for (const auto& desc : descriptors) {
        std_msgs::msg::Float64MultiArray feature_array;
        feature_array.data.reserve(desc.size());
        for (int i = 0; i < desc.size(); ++i) {
            feature_array.data.push_back(static_cast<double>(desc(i)));
        }
        response->points_xf.push_back(feature_array);
    }

    response->sensor_type = 0;  // LiDAR sensor type
    response->success = true;
    state_ = MRLSlaveState::READY;

    RCLCPP_INFO(this->get_logger(), "GetFeatures request completed successfully.");
}

void MRLSlave::handle_get_status(const std::shared_ptr<mrl::srv::GetReady::Request> /*request*/,
        std::shared_ptr<mrl::srv::GetReady::Response> response) {
    response->ready = (state_ == MRLSlaveState::READY);
}

/**
 * @brief Main entry point for the MRL Slave node
 *
 * Initializes ROS2, creates the MRLSlave node, and spins until shutdown.
 * The node will block during construction until the master node is detected.
 *
 * @param argc Argument count
 * @param argv Argument values
 * @return 0 on successful exit
 */
int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MRLSlave>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
