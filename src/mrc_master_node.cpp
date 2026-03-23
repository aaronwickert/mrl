#include <atomic>
#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <rclcpp/qos.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <tf2_ros/static_transform_broadcaster.h>
#include <visualization_msgs/msg/marker_array.hpp>

#include <hector_multi_robot_msgs/msg/robot_announcement.hpp>

#include "mrc/srv/get_registration_data.hpp"
#include "mrc/action/calibrate.hpp"

#include "mrc/icp_refiner.hpp"
#include "mrc/registration_factory.hpp"
#include "mrc/registration_types.hpp"

#ifdef MRC_USE_GTSAM
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/inference/Symbol.h>
#include "mrc/pose_graph/pose_graph_optimizer.hpp"
#endif

using CalibrateAction = mrc::action::Calibrate;
using GoalHandleCalibrate = rclcpp_action::ServerGoalHandle<CalibrateAction>;

struct RobotInfo {
  std::string robot_id;
  std::string robot_type;
  std::string ns;
  std::string data_service;
  rclcpp::Time last_seen;
};

class MRCMasterNode : public rclcpp::Node {
public:
  explicit MRCMasterNode(const rclcpp::NodeOptions& options)
      : rclcpp::Node("mrc_master", options),
        tf_static_broadcaster_(std::make_unique<tf2_ros::StaticTransformBroadcaster>(*this)) {
    robot_id_ = declare_parameter<std::string>("robot_id", "master");
    master_map_frame_ = declare_parameter<std::string>("master_map_frame", "base_link");
    announce_topic_ = declare_parameter<std::string>("announce_topic", "/robot_announcement");
    slave_service_relative_ = declare_parameter<std::string>(
        "slave_service_relative", "mrc_slave/get_registration_data");
    algorithm_ = declare_parameter<std::string>("algorithm", "kiss_matcher");
    local_slave_data_service_ = declare_parameter<std::string>(
        "local_slave_data_service", "/mrc_slave/get_registration_data");
    request_timeout_sec_ = declare_parameter<double>("request_timeout_sec", 2.0);
    default_slave_base_frame_ =
        declare_parameter<std::string>("default_slave_base_frame", "base_link");
    use_pose_graph_ = declare_parameter<bool>("use_pose_graph", true);
    pose_graph_topology_ = declare_parameter<std::string>("pose_graph_topology", "star");

    use_icp_refinement_ = declare_parameter<bool>("use_icp_refinement", false);
    icp_max_iterations_ = declare_parameter<int>("icp_max_iterations", 50);
    icp_max_correspondence_distance_ = declare_parameter<double>("icp_max_correspondence_distance", 1.0);
    icp_voxel_leaf_size_ = declare_parameter<double>("icp_voxel_leaf_size", 0.1);
    max_nn_rmse_t_m_ = declare_parameter<double>("max_nn_rmse_t_m", 2.0);
    max_corr_rmse_t_m_ = declare_parameter<double>("max_corr_rmse_t_m", 1.0);

#ifdef MRC_USE_GTSAM
    pg_default_sigma_t_m_ = declare_parameter<double>("pg_default_sigma_t_m", 0.3);
    pg_default_sigma_r_rad_ = declare_parameter<double>("pg_default_sigma_r_rad", 0.2);
    pg_min_sigma_t_m_ = declare_parameter<double>("pg_min_sigma_t_m", 0.10);
    pg_min_sigma_r_rad_ = declare_parameter<double>("pg_min_sigma_r_rad", 0.10);
    pg_max_sigma_t_m_ = declare_parameter<double>("pg_max_sigma_t_m", 5.0);
    pg_max_sigma_r_rad_ = declare_parameter<double>("pg_max_sigma_r_rad", 3.14159);
    pg_use_robust_ = declare_parameter<bool>("pg_use_robust", true);
    pg_huber_k_ = declare_parameter<double>("pg_huber_k", 1.345);
    anchor_sigma_t_m_ = declare_parameter<double>("anchor_sigma_t_m", 1e-3);
    anchor_sigma_r_rad_ = declare_parameter<double>("anchor_sigma_r_rad", 1e-3);
    min_inliers_ = declare_parameter<int>("min_inliers", 10);
#endif

    if (!createBackendForAlgorithm(algorithm_)) {
      auto available = mrc::availableAlgorithms();
      std::string avail_str;
      for (size_t i = 0; i < available.size(); ++i) {
        if (i > 0) avail_str += ", ";
        avail_str += "'" + available[i] + "'";
      }
      RCLCPP_FATAL(get_logger(),
                   "Unknown registration algorithm '%s'. Available: %s",
                   algorithm_.c_str(), avail_str.c_str());
      throw std::runtime_error("Unknown registration algorithm: " + algorithm_);
    }

    param_callback_handle_ = add_on_set_parameters_callback(
        std::bind(&MRCMasterNode::onParameterChange, this, std::placeholders::_1));

    // Static robot configuration
    declare_parameter<std::vector<std::string>>("static_robots", std::vector<std::string>{});
    auto static_robots = get_parameter("static_robots").as_string_array();
    for (const auto& entry : static_robots) {
      auto pos = entry.find(':');
      if (pos != std::string::npos) {
        RobotInfo info;
        info.robot_id = entry.substr(0, pos);
        info.ns = entry.substr(pos + 1);
        info.robot_type = "static";
        info.last_seen = now();
        info.data_service = qualifyService(info.ns, slave_service_relative_);
        robots_[info.robot_id] = info;
        RCLCPP_INFO(get_logger(), "Static robot: id='%s', ns='%s'",
                    info.robot_id.c_str(), info.ns.c_str());
      }
    }

    service_callback_group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);

    const rclcpp::QoS announcement_qos = rclcpp::QoS(1).reliable().transient_local();
    ann_sub_ = create_subscription<hector_multi_robot_msgs::msg::RobotAnnouncement>(
        announce_topic_, announcement_qos,
        std::bind(&MRCMasterNode::onAnnouncement, this, std::placeholders::_1));

    // Legacy service (for backwards compatibility)
    calibrate_srv_ = create_service<std_srvs::srv::Trigger>(
        "~/calibrate",
        std::bind(&MRCMasterNode::onCalibrateTrigger, this,
                  std::placeholders::_1, std::placeholders::_2));

    // Action server for calibration with feedback
    calibrate_action_server_ = rclcpp_action::create_server<CalibrateAction>(
        this,
        "~/calibrate_action",
        std::bind(&MRCMasterNode::handleGoal, this, std::placeholders::_1, std::placeholders::_2),
        std::bind(&MRCMasterNode::handleCancel, this, std::placeholders::_1),
        std::bind(&MRCMasterNode::handleAccepted, this, std::placeholders::_1));

    // Debug visualization publishers
    unoptimized_poses_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
        "~/debug/unoptimized_poses", 10);
    optimized_poses_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
        "~/debug/optimized_poses", 10);

    RCLCPP_INFO(get_logger(), "MRC Master initialized (action server enabled)");
  }

private:
  // ============== Action Server Callbacks ==============

  rclcpp_action::GoalResponse handleGoal(
      const rclcpp_action::GoalUUID& /*uuid*/,
      std::shared_ptr<const CalibrateAction::Goal> goal) {
    RCLCPP_INFO(get_logger(), "Received calibration goal: use_pose_graph=%s, topology=%s",
                goal->use_pose_graph ? "true" : "false",
                goal->pose_graph_topology.c_str());

    // Validate topology
    if (goal->pose_graph_topology != "star" && goal->pose_graph_topology != "mesh") {
      RCLCPP_WARN(get_logger(), "Invalid topology '%s', rejecting goal",
                  goal->pose_graph_topology.c_str());
      return rclcpp_action::GoalResponse::REJECT;
    }

    // Check if we're already calibrating
    if (is_calibrating_.load()) {
      RCLCPP_WARN(get_logger(), "Already calibrating, rejecting new goal");
      return rclcpp_action::GoalResponse::REJECT;
    }

    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
  }

  rclcpp_action::CancelResponse handleCancel(
      const std::shared_ptr<GoalHandleCalibrate> /*goal_handle*/) {
    RCLCPP_INFO(get_logger(), "Received cancel request");
    cancel_requested_.store(true);
    return rclcpp_action::CancelResponse::ACCEPT;
  }

  void handleAccepted(const std::shared_ptr<GoalHandleCalibrate> goal_handle) {
    // Execute in a separate thread to not block the executor
    std::thread{std::bind(&MRCMasterNode::executeCalibration, this, goal_handle)}.detach();
  }

  void executeCalibration(const std::shared_ptr<GoalHandleCalibrate> goal_handle) {
    is_calibrating_.store(true);
    cancel_requested_.store(false);

    const auto goal = goal_handle->get_goal();
    auto feedback = std::make_shared<CalibrateAction::Feedback>();
    auto result = std::make_shared<CalibrateAction::Result>();

    const auto start_time = std::chrono::steady_clock::now();

    // Use goal parameters or node defaults
    const bool use_pg = goal->use_pose_graph;
    const std::string topology = goal->pose_graph_topology;

    // Capture backend atomically
    mrc::MRCRegistrationPtr current_backend;
    std::string current_algorithm;
    {
      std::lock_guard<std::mutex> lock(backend_mutex_);
      current_backend = backend_;
      current_algorithm = algorithm_;
    }

    // Publish initializing feedback
    feedback->phase = "initializing";
    feedback->progress_percent = 0.0f;
    feedback->status_message = "Starting calibration...";
    goal_handle->publish_feedback(feedback);

    // Get robot list
    const auto robots_snapshot = robots_;
    std::vector<RobotInfo> robot_list;
    for (const auto& [id, info] : robots_snapshot) {
      if (id == robot_id_) continue;
      // Filter by goal robot_ids if specified
      if (!goal->robot_ids.empty()) {
        bool found = false;
        for (const auto& rid : goal->robot_ids) {
          if (rid == id) { found = true; break; }
        }
        if (!found) continue;
      }
      robot_list.push_back(info);
    }

    feedback->robots_total = static_cast<int32_t>(robot_list.size());
    feedback->robots_completed = 0;

    if (robot_list.empty()) {
      result->success = false;
      result->message = "No robots to calibrate";
      goal_handle->succeed(result);
      is_calibrating_.store(false);
      return;
    }

    // Fetch master (local) data
    feedback->phase = "fetching_data";
    feedback->current_robot = robot_id_;
    feedback->status_message = "Fetching master point cloud...";
    goal_handle->publish_feedback(feedback);

    mrc::srv::GetRegistrationData::Response target_res;
    if (!getDataFromService(local_slave_data_service_, target_res)) {
      result->success = false;
      result->message = "Local slave data unavailable";
      goal_handle->abort(result);
      is_calibrating_.store(false);
      return;
    }

    mrc::RegistrationInput target_in;
    target_in.frame_id = target_res.frame_id;
    if (current_algorithm == "kiss_matcher") target_in.features = target_res.features;
    if (current_algorithm == "fricp") target_in.cloud = target_res.cloud;
    if (use_icp_refinement_) target_in.cloud = target_res.cloud;

    // Collect transforms and results
    std::vector<geometry_msgs::msg::TransformStamped> calibration_transforms;
    std::map<std::string, geometry_msgs::msg::TransformStamped> unoptimized_poses;
    int ok_count = 0;
    int fail_count = 0;

#ifdef MRC_USE_GTSAM
    std::unordered_map<std::string, size_t> id_to_idx;
    id_to_idx[robot_id_] = 0;
    for (size_t i = 0; i < robot_list.size(); ++i) {
      id_to_idx[robot_list[i].robot_id] = i + 1;
    }

    mrc::PoseGraphParams pgp;
    pgp.default_sigma_t_m = pg_default_sigma_t_m_;
    pgp.default_sigma_r_rad = pg_default_sigma_r_rad_;
    pgp.min_sigma_t_m = pg_min_sigma_t_m_;
    pgp.min_sigma_r_rad = pg_min_sigma_r_rad_;
    pgp.max_sigma_t_m = pg_max_sigma_t_m_;
    pgp.max_sigma_r_rad = pg_max_sigma_r_rad_;
    pgp.use_robust = pg_use_robust_;
    pgp.huber_k = pg_huber_k_;

    RCLCPP_INFO(get_logger(),
        "[PG] params: default_sigma_t=%.4fm default_sigma_r=%.4frad "
        "min_sigma_t=%.4f min_sigma_r=%.4f max_sigma_t=%.4f max_sigma_r=%.4f "
        "robust=%s huber_k=%.3f anchor_sigma_t=%.1e anchor_sigma_r=%.1e "
        "min_inliers=%d topology=%s",
        pgp.default_sigma_t_m, pgp.default_sigma_r_rad,
        pgp.min_sigma_t_m, pgp.min_sigma_r_rad,
        pgp.max_sigma_t_m, pgp.max_sigma_r_rad,
        pgp.use_robust ? "true" : "false", pgp.huber_k,
        anchor_sigma_t_m_, anchor_sigma_r_rad_,
        min_inliers_, topology.c_str());

    mrc::PoseGraphOptimizer pgo(pgp);
    pgo.reset();

    const gtsam::Key k_master = keyForRobot(0);
    pgo.addAnchor(k_master, gtsam::Pose3(), anchor_sigma_t_m_, anchor_sigma_r_rad_);
    pgo.setInitial(k_master, gtsam::Pose3());
    // Don't pre-set initial values for all robots here — only set them
    // when an edge is actually added. GTSAM requires every variable in
    // initial_ to be referenced by at least one factor; if a robot's
    // registration fails, its key would have no factors and cause a
    // "leftover keys after elimination" error.

    // Store registration results for mesh topology
    std::unordered_map<std::string, mrc::RegistrationInput> robot_data;
    robot_data[robot_id_] = target_in;
#endif

    // Star registration: master <-> each robot
    feedback->phase = "registering";
    for (size_t i = 0; i < robot_list.size(); ++i) {
      if (cancel_requested_.load()) {
        result->success = false;
        result->message = "Calibration cancelled";
        goal_handle->canceled(result);
        is_calibrating_.store(false);
        return;
      }

      const auto& info = robot_list[i];
      feedback->current_robot = info.robot_id;
      feedback->robots_completed = static_cast<int32_t>(i);
      feedback->progress_percent = static_cast<float>(i) / robot_list.size() * 50.0f;
      feedback->status_message = "Registering " + info.robot_id + "...";
      goal_handle->publish_feedback(feedback);

      // Fetch data
      mrc::srv::GetRegistrationData::Response source_res;
      if (!getDataFromService(info.data_service, source_res)) {
        RCLCPP_WARN(get_logger(), "No data from '%s'", info.robot_id.c_str());
        result->result_robot_ids.push_back(info.robot_id);
        result->result_success.push_back(false);
        result->result_num_inliers.push_back(0);
        result->result_translation_errors_m.push_back(-1.0);
        result->result_rotation_errors_rad.push_back(-1.0);
        ++fail_count;
        continue;
      }

      mrc::RegistrationInput source_in;
      source_in.frame_id = source_res.frame_id;
      if (current_algorithm == "kiss_matcher") source_in.features = source_res.features;
      if (current_algorithm == "fricp") source_in.cloud = source_res.cloud;
      if (use_icp_refinement_) source_in.cloud = source_res.cloud;

#ifdef MRC_USE_GTSAM
      robot_data[info.robot_id] = source_in;
#endif

      // Register
      auto reg = current_backend->registerSourceToTarget(source_in, target_in);
      if (reg.ok) reg = maybeRefineWithICP(reg, source_in, target_in);

      result->result_robot_ids.push_back(info.robot_id);
      result->result_num_inliers.push_back(reg.num_inliers.value_or(0));

      if (!reg.ok) {
        RCLCPP_WARN(get_logger(), "Registration failed for '%s': %s",
                    info.robot_id.c_str(), reg.message.c_str());
        result->result_success.push_back(false);
        result->result_translation_errors_m.push_back(-1.0);
        result->result_rotation_errors_rad.push_back(-1.0);
        ++fail_count;
        continue;
      }

      // Reject registrations with excessive RMSE
      if (exceedsRmseThreshold(reg, current_algorithm)) {
        RCLCPP_WARN(get_logger(), "Registration for '%s' rejected (rmse=%.4fm > threshold)",
                    info.robot_id.c_str(), *reg.rmse_t_m);
        result->result_success.push_back(false);
        result->result_translation_errors_m.push_back(-1.0);
        result->result_rotation_errors_rad.push_back(-1.0);
        ++fail_count;
        continue;
      }

#ifdef MRC_USE_GTSAM
      if (use_pg) {
        if (reg.num_inliers.has_value() && *reg.num_inliers < min_inliers_) {
          RCLCPP_WARN(get_logger(), "Registration for '%s' rejected (inliers=%d < %d)",
                      info.robot_id.c_str(), *reg.num_inliers, min_inliers_);
          result->result_success.push_back(false);
          result->result_translation_errors_m.push_back(-1.0);
          result->result_rotation_errors_rad.push_back(-1.0);
          ++fail_count;
          continue;
        }

        // Capture raw (unoptimized) pose for debug visualization
        unoptimized_poses[info.robot_id] = reg.T_target_from_source;

        const gtsam::Pose3 T_master_from_robot = tfToPose3(reg.T_target_from_source);
        const gtsam::Key k_robot = keyForRobot(id_to_idx.at(info.robot_id));
        pgo.setInitial(k_robot, gtsam::Pose3());

        {
          double sigma_t = 0.0, sigma_r = 0.0;
          mrc::PoseGraphOptimizer::computeSigmas(pgp, reg, sigma_t, sigma_r);
          RCLCPP_INFO(get_logger(),
              "[PG_EDGE] type=star robot=%s t_x=%.6f t_y=%.6f t_z=%.6f "
              "rmse_t=%.6f rmse_r=%.6f inliers=%d correspondences=%d "
              "sigma_t=%.6f sigma_r=%.6f score=%.6f",
              info.robot_id.c_str(),
              reg.T_target_from_source.transform.translation.x,
              reg.T_target_from_source.transform.translation.y,
              reg.T_target_from_source.transform.translation.z,
              reg.rmse_t_m.value_or(-1.0),
              reg.rmse_r_rad.value_or(-1.0),
              reg.num_inliers.value_or(-1),
              reg.num_correspondences.value_or(-1),
              sigma_t, sigma_r, reg.score);
        }

        // BetweenFactor(a,b,z) expects z = x_a^{-1} * x_b = T_a_from_b
        pgo.addRegistrationEdge(k_master, k_robot, T_master_from_robot, reg);

        result->result_success.push_back(true);
        result->result_translation_errors_m.push_back(0.0);  // Will be filled after optimization
        result->result_rotation_errors_rad.push_back(0.0);
        ++ok_count;
        continue;
      }
#endif

      // Non-pose-graph path — also capture for visualization
      unoptimized_poses[info.robot_id] = reg.T_target_from_source;

      geometry_msgs::msg::TransformStamped tf = reg.T_target_from_source;
      tf.header.stamp = now();
      tf.header.frame_id = master_map_frame_;
      tf.child_frame_id = makeChildFrame(info);
      calibration_transforms.push_back(tf);
      result->result_success.push_back(true);
      result->result_translation_errors_m.push_back(0.0);
      result->result_rotation_errors_rad.push_back(0.0);
      ++ok_count;
    }

#ifdef MRC_USE_GTSAM
    // Mesh topology: add slave-to-slave edges
    if (use_pg && topology == "mesh") {
      feedback->phase = "registering";
      feedback->status_message = "Adding mesh edges (slave-to-slave)...";
      feedback->progress_percent = 60.0f;
      goal_handle->publish_feedback(feedback);

      RCLCPP_INFO(get_logger(), "Adding slave-to-slave edges (mesh topology)...");

      for (size_t i = 0; i < robot_list.size(); ++i) {
        for (size_t j = i + 1; j < robot_list.size(); ++j) {
          if (cancel_requested_.load()) {
            result->success = false;
            result->message = "Calibration cancelled during mesh registration";
            goal_handle->canceled(result);
            is_calibrating_.store(false);
            return;
          }

          const auto& ri = robot_list[i];
          const auto& rj = robot_list[j];

          auto it_i = robot_data.find(ri.robot_id);
          auto it_j = robot_data.find(rj.robot_id);
          if (it_i == robot_data.end() || it_j == robot_data.end()) continue;

          auto reg = current_backend->registerSourceToTarget(it_j->second, it_i->second);
          if (!reg.ok) {
            RCLCPP_WARN(get_logger(), "Mesh: %s<->%s failed", ri.robot_id.c_str(), rj.robot_id.c_str());
            continue;
          }
          if (reg.ok) reg = maybeRefineWithICP(reg, it_j->second, it_i->second);

          if (reg.num_inliers.has_value() && *reg.num_inliers < min_inliers_) continue;
          if (exceedsRmseThreshold(reg, current_algorithm)) {
            RCLCPP_WARN(get_logger(), "Mesh: %s<->%s rejected (rmse=%.4fm > threshold)",
                        ri.robot_id.c_str(), rj.robot_id.c_str(), *reg.rmse_t_m);
            continue;
          }

          const gtsam::Pose3 T_i_from_j = tfToPose3(reg.T_target_from_source);
          const gtsam::Key k_i = keyForRobot(id_to_idx.at(ri.robot_id));
          const gtsam::Key k_j = keyForRobot(id_to_idx.at(rj.robot_id));
          pgo.setInitial(k_i, gtsam::Pose3());
          pgo.setInitial(k_j, gtsam::Pose3());

          {
            double sigma_t = 0.0, sigma_r = 0.0;
            mrc::PoseGraphOptimizer::computeSigmas(pgp, reg, sigma_t, sigma_r);
            RCLCPP_INFO(get_logger(),
                "[PG_EDGE] type=mesh robot=%s peer=%s t_x=%.6f t_y=%.6f t_z=%.6f "
                "rmse_t=%.6f rmse_r=%.6f inliers=%d correspondences=%d "
                "sigma_t=%.6f sigma_r=%.6f score=%.6f",
                ri.robot_id.c_str(), rj.robot_id.c_str(),
                reg.T_target_from_source.transform.translation.x,
                reg.T_target_from_source.transform.translation.y,
                reg.T_target_from_source.transform.translation.z,
                reg.rmse_t_m.value_or(-1.0),
                reg.rmse_r_rad.value_or(-1.0),
                reg.num_inliers.value_or(-1),
                reg.num_correspondences.value_or(-1),
                sigma_t, sigma_r, reg.score);
          }

          // BetweenFactor(a,b,z) expects z = T_a_from_b
          pgo.addRegistrationEdge(k_i, k_j, T_i_from_j, reg);
        }
      }
    }

    // Optimize pose graph
    if (use_pg) {
      feedback->phase = "optimizing";
      feedback->progress_percent = 80.0f;
      feedback->status_message = "Running pose graph optimization...";
      goal_handle->publish_feedback(feedback);

      try {
        const auto values = pgo.optimize();

        for (const auto& ri : robot_list) {
          const auto it = id_to_idx.find(ri.robot_id);
          if (it == id_to_idx.end()) continue;

          const gtsam::Key k = keyForRobot(it->second);
          if (!values.exists(k)) continue;

          const gtsam::Pose3 X_master_from_robot = values.at<gtsam::Pose3>(k);

          geometry_msgs::msg::TransformStamped tf = pose3ToTf(X_master_from_robot);
          tf.header.stamp = now();
          tf.header.frame_id = master_map_frame_;
          tf.child_frame_id = makeChildFrame(ri);
          calibration_transforms.push_back(tf);
        }
      } catch (const std::exception& e) {
        result->success = false;
        result->message = std::string("Pose graph optimization failed: ") + e.what();
        goal_handle->abort(result);
        is_calibrating_.store(false);
        return;
      }
    }
#endif

    // Publish debug visualization markers
    if (!unoptimized_poses.empty()) {
      publishPoseMarkers(unoptimized_poses_pub_, unoptimized_poses,
                         1.0f, 0.0f, 0.0f, "unoptimized", topology);  // RED
    }
    {
      std::map<std::string, geometry_msgs::msg::TransformStamped> optimized_poses;
      for (const auto& tf : calibration_transforms) {
        // Extract robot name from child_frame_id (strip /base_link suffix)
        std::string name = tf.child_frame_id;
        auto slash_pos = name.find('/');
        if (slash_pos != std::string::npos) name = name.substr(0, slash_pos);
        optimized_poses[name] = tf;
      }
      if (!optimized_poses.empty()) {
        publishPoseMarkers(optimized_poses_pub_, optimized_poses,
                           0.0f, 0.3f, 1.0f, "optimized", topology);  // BLUE
      }
    }

    // Publish transforms
    feedback->phase = "publishing";
    feedback->progress_percent = 95.0f;
    feedback->status_message = "Publishing static transforms...";
    goal_handle->publish_feedback(feedback);

    if (!calibration_transforms.empty()) {
      tf_static_broadcaster_->sendTransform(calibration_transforms);
      RCLCPP_INFO(get_logger(), "Published %zu static transforms", calibration_transforms.size());
    }

    // Complete
    const auto end_time = std::chrono::steady_clock::now();
    const double elapsed_sec = std::chrono::duration<double>(end_time - start_time).count();

    result->success = (fail_count == 0);
    result->total_time_sec = elapsed_sec;
    result->robots_calibrated = ok_count;
    result->robots_failed = fail_count;

    std::string pg_status = use_pg ? " (pose_graph=" + topology + ")" : "";
    result->message = "Calibration complete (algorithm=" + current_algorithm + "). ok=" +
                      std::to_string(ok_count) + " fail=" + std::to_string(fail_count) + pg_status;

    feedback->phase = "complete";
    feedback->progress_percent = 100.0f;
    feedback->robots_completed = feedback->robots_total;
    feedback->status_message = result->message;
    goal_handle->publish_feedback(feedback);

    goal_handle->succeed(result);
    is_calibrating_.store(false);

    RCLCPP_INFO(get_logger(), "Calibration completed in %.2fs", elapsed_sec);
  }

  // ============== Legacy Service (backwards compatibility) ==============

  void onCalibrateTrigger(const std::shared_ptr<std_srvs::srv::Trigger::Request> /*req*/,
                          std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
    // Create a goal with current node parameters
    auto goal = std::make_shared<CalibrateAction::Goal>();
    goal->use_pose_graph = use_pose_graph_;
    goal->pose_graph_topology = pose_graph_topology_;

    // Execute synchronously (blocking)
    auto result = executeCalibrateSync(goal);
    res->success = result->success;
    res->message = result->message;
  }

  std::shared_ptr<CalibrateAction::Result> executeCalibrateSync(
      std::shared_ptr<CalibrateAction::Goal> goal) {
    auto result = std::make_shared<CalibrateAction::Result>();

    if (is_calibrating_.load()) {
      result->success = false;
      result->message = "Already calibrating";
      return result;
    }

    is_calibrating_.store(true);
    cancel_requested_.store(false);

    const auto start_time = std::chrono::steady_clock::now();
    const bool use_pg = goal->use_pose_graph;
    const std::string topology = goal->pose_graph_topology;

    mrc::MRCRegistrationPtr current_backend;
    std::string current_algorithm;
    {
      std::lock_guard<std::mutex> lock(backend_mutex_);
      current_backend = backend_;
      current_algorithm = algorithm_;
    }

    const auto robots_snapshot = robots_;
    std::vector<RobotInfo> robot_list;
    for (const auto& [id, info] : robots_snapshot) {
      if (id == robot_id_) continue;
      robot_list.push_back(info);
    }

    if (robot_list.empty()) {
      result->success = false;
      result->message = "No robots to calibrate";
      is_calibrating_.store(false);
      return result;
    }

    mrc::srv::GetRegistrationData::Response target_res;
    if (!getDataFromService(local_slave_data_service_, target_res)) {
      result->success = false;
      result->message = "Local slave data unavailable";
      is_calibrating_.store(false);
      return result;
    }

    mrc::RegistrationInput target_in;
    target_in.frame_id = target_res.frame_id;
    if (current_algorithm == "kiss_matcher") target_in.features = target_res.features;
    if (current_algorithm == "fricp") target_in.cloud = target_res.cloud;
    if (use_icp_refinement_) target_in.cloud = target_res.cloud;

    std::vector<geometry_msgs::msg::TransformStamped> calibration_transforms;
    std::map<std::string, geometry_msgs::msg::TransformStamped> unoptimized_poses;
    int ok_count = 0;
    int fail_count = 0;
    std::string per_robot_inlier_info;

#ifdef MRC_USE_GTSAM
    std::unordered_map<std::string, size_t> id_to_idx;
    id_to_idx[robot_id_] = 0;
    for (size_t i = 0; i < robot_list.size(); ++i) {
      id_to_idx[robot_list[i].robot_id] = i + 1;
    }

    mrc::PoseGraphParams pgp;
    pgp.default_sigma_t_m = pg_default_sigma_t_m_;
    pgp.default_sigma_r_rad = pg_default_sigma_r_rad_;
    pgp.min_sigma_t_m = pg_min_sigma_t_m_;
    pgp.min_sigma_r_rad = pg_min_sigma_r_rad_;
    pgp.max_sigma_t_m = pg_max_sigma_t_m_;
    pgp.max_sigma_r_rad = pg_max_sigma_r_rad_;
    pgp.use_robust = pg_use_robust_;
    pgp.huber_k = pg_huber_k_;

    RCLCPP_INFO(get_logger(),
        "[PG] params: default_sigma_t=%.4fm default_sigma_r=%.4frad "
        "min_sigma_t=%.4f min_sigma_r=%.4f max_sigma_t=%.4f max_sigma_r=%.4f "
        "robust=%s huber_k=%.3f anchor_sigma_t=%.1e anchor_sigma_r=%.1e "
        "min_inliers=%d topology=%s",
        pgp.default_sigma_t_m, pgp.default_sigma_r_rad,
        pgp.min_sigma_t_m, pgp.min_sigma_r_rad,
        pgp.max_sigma_t_m, pgp.max_sigma_r_rad,
        pgp.use_robust ? "true" : "false", pgp.huber_k,
        anchor_sigma_t_m_, anchor_sigma_r_rad_,
        min_inliers_, topology.c_str());

    mrc::PoseGraphOptimizer pgo(pgp);
    pgo.reset();
    pgo.addAnchor(keyForRobot(0), gtsam::Pose3(), anchor_sigma_t_m_, anchor_sigma_r_rad_);
    pgo.setInitial(keyForRobot(0), gtsam::Pose3());
    // Don't pre-set initial values for all robots — only set them when an
    // edge is actually added, to avoid GTSAM "leftover keys" errors when
    // a robot's registration fails.

    std::unordered_map<std::string, mrc::RegistrationInput> robot_data;
    robot_data[robot_id_] = target_in;
#endif

    for (const auto& info : robot_list) {
      mrc::srv::GetRegistrationData::Response source_res;
      if (!getDataFromService(info.data_service, source_res)) {
        ++fail_count;
        continue;
      }

      mrc::RegistrationInput source_in;
      source_in.frame_id = source_res.frame_id;
      if (current_algorithm == "kiss_matcher") source_in.features = source_res.features;
      if (current_algorithm == "fricp") source_in.cloud = source_res.cloud;
      if (use_icp_refinement_) source_in.cloud = source_res.cloud;

#ifdef MRC_USE_GTSAM
      robot_data[info.robot_id] = source_in;
#endif

      auto reg = current_backend->registerSourceToTarget(source_in, target_in);
      if (reg.ok) reg = maybeRefineWithICP(reg, source_in, target_in);

      // Track per-robot inlier ratio
      if (reg.num_inliers.has_value() && reg.num_correspondences.has_value() &&
          *reg.num_correspondences > 0) {
        double ratio = static_cast<double>(*reg.num_inliers) /
                       static_cast<double>(*reg.num_correspondences) * 100.0;
        char buf[128];
        std::snprintf(buf, sizeof(buf), " [%s inliers=%d/%d (%.0f%%)]",
                      info.robot_id.c_str(), *reg.num_inliers,
                      *reg.num_correspondences, ratio);
        per_robot_inlier_info += buf;
      }

      if (!reg.ok) {
        ++fail_count;
        continue;
      }

      // Reject registrations with excessive RMSE
      if (exceedsRmseThreshold(reg, current_algorithm)) {
        RCLCPP_WARN(get_logger(), "Registration for '%s' rejected (rmse=%.4fm > threshold)",
                    info.robot_id.c_str(), *reg.rmse_t_m);
        ++fail_count;
        continue;
      }

#ifdef MRC_USE_GTSAM
      if (use_pg) {
        if (reg.num_inliers.has_value() && *reg.num_inliers < min_inliers_) {
          ++fail_count;
          continue;
        }
        // Capture raw (unoptimized) pose for debug visualization
        unoptimized_poses[info.robot_id] = reg.T_target_from_source;

        const gtsam::Pose3 T_master_from_robot = tfToPose3(reg.T_target_from_source);
        pgo.setInitial(keyForRobot(id_to_idx.at(info.robot_id)), gtsam::Pose3());

        {
          double sigma_t = 0.0, sigma_r = 0.0;
          mrc::PoseGraphOptimizer::computeSigmas(pgp, reg, sigma_t, sigma_r);
          RCLCPP_INFO(get_logger(),
              "[PG_EDGE] type=star robot=%s t_x=%.6f t_y=%.6f t_z=%.6f "
              "rmse_t=%.6f rmse_r=%.6f inliers=%d correspondences=%d "
              "sigma_t=%.6f sigma_r=%.6f score=%.6f",
              info.robot_id.c_str(),
              reg.T_target_from_source.transform.translation.x,
              reg.T_target_from_source.transform.translation.y,
              reg.T_target_from_source.transform.translation.z,
              reg.rmse_t_m.value_or(-1.0),
              reg.rmse_r_rad.value_or(-1.0),
              reg.num_inliers.value_or(-1),
              reg.num_correspondences.value_or(-1),
              sigma_t, sigma_r, reg.score);
        }

        // BetweenFactor(a,b,z) expects z = T_a_from_b
        pgo.addRegistrationEdge(keyForRobot(0), keyForRobot(id_to_idx.at(info.robot_id)),
                                T_master_from_robot, reg);
        ++ok_count;
        continue;
      }
#endif

      // Non-pose-graph path — also capture for visualization
      unoptimized_poses[info.robot_id] = reg.T_target_from_source;

      geometry_msgs::msg::TransformStamped tf = reg.T_target_from_source;
      tf.header.stamp = now();
      tf.header.frame_id = master_map_frame_;
      tf.child_frame_id = makeChildFrame(info);
      calibration_transforms.push_back(tf);
      ++ok_count;
    }

#ifdef MRC_USE_GTSAM
    if (use_pg && topology == "mesh") {
      for (size_t i = 0; i < robot_list.size(); ++i) {
        for (size_t j = i + 1; j < robot_list.size(); ++j) {
          const auto& ri = robot_list[i];
          const auto& rj = robot_list[j];
          auto it_i = robot_data.find(ri.robot_id);
          auto it_j = robot_data.find(rj.robot_id);
          if (it_i == robot_data.end() || it_j == robot_data.end()) continue;

          auto reg = current_backend->registerSourceToTarget(it_j->second, it_i->second);
          if (!reg.ok) continue;
          if (reg.ok) reg = maybeRefineWithICP(reg, it_j->second, it_i->second);
          if (!reg.ok) continue;
          if (reg.num_inliers.has_value() && *reg.num_inliers < min_inliers_) continue;
          if (exceedsRmseThreshold(reg, current_algorithm)) continue;

          const gtsam::Pose3 T_i_from_j = tfToPose3(reg.T_target_from_source);
          pgo.setInitial(keyForRobot(id_to_idx.at(ri.robot_id)), gtsam::Pose3());
          pgo.setInitial(keyForRobot(id_to_idx.at(rj.robot_id)), gtsam::Pose3());

          {
            double sigma_t = 0.0, sigma_r = 0.0;
            mrc::PoseGraphOptimizer::computeSigmas(pgp, reg, sigma_t, sigma_r);
            RCLCPP_INFO(get_logger(),
                "[PG_EDGE] type=mesh robot=%s peer=%s t_x=%.6f t_y=%.6f t_z=%.6f "
                "rmse_t=%.6f rmse_r=%.6f inliers=%d correspondences=%d "
                "sigma_t=%.6f sigma_r=%.6f score=%.6f",
                ri.robot_id.c_str(), rj.robot_id.c_str(),
                reg.T_target_from_source.transform.translation.x,
                reg.T_target_from_source.transform.translation.y,
                reg.T_target_from_source.transform.translation.z,
                reg.rmse_t_m.value_or(-1.0),
                reg.rmse_r_rad.value_or(-1.0),
                reg.num_inliers.value_or(-1),
                reg.num_correspondences.value_or(-1),
                sigma_t, sigma_r, reg.score);
          }

          // BetweenFactor(a,b,z) expects z = T_a_from_b
          pgo.addRegistrationEdge(keyForRobot(id_to_idx.at(ri.robot_id)),
                                  keyForRobot(id_to_idx.at(rj.robot_id)),
                                  T_i_from_j, reg);
        }
      }
    }

    if (use_pg) {
      try {
        const auto values = pgo.optimize();
        for (const auto& ri : robot_list) {
          const auto it = id_to_idx.find(ri.robot_id);
          if (it == id_to_idx.end()) continue;
          const gtsam::Key k = keyForRobot(it->second);
          if (!values.exists(k)) continue;
          geometry_msgs::msg::TransformStamped tf = pose3ToTf(values.at<gtsam::Pose3>(k));
          tf.header.stamp = now();
          tf.header.frame_id = master_map_frame_;
          tf.child_frame_id = makeChildFrame(ri);
          calibration_transforms.push_back(tf);
        }
      } catch (const std::exception& e) {
        result->success = false;
        result->message = std::string("Pose graph optimization failed: ") + e.what();
        is_calibrating_.store(false);
        return result;
      }
    }
#endif

    // Publish debug visualization markers
    if (!unoptimized_poses.empty()) {
      publishPoseMarkers(unoptimized_poses_pub_, unoptimized_poses,
                         1.0f, 0.0f, 0.0f, "unoptimized", topology);  // RED
    }
    {
      std::map<std::string, geometry_msgs::msg::TransformStamped> optimized_poses;
      for (const auto& tf : calibration_transforms) {
        std::string name = tf.child_frame_id;
        auto slash_pos = name.find('/');
        if (slash_pos != std::string::npos) name = name.substr(0, slash_pos);
        optimized_poses[name] = tf;
      }
      if (!optimized_poses.empty()) {
        publishPoseMarkers(optimized_poses_pub_, optimized_poses,
                           0.0f, 0.3f, 1.0f, "optimized", topology);  // BLUE
      }
    }

    if (!calibration_transforms.empty()) {
      tf_static_broadcaster_->sendTransform(calibration_transforms);
    }

    const auto end_time = std::chrono::steady_clock::now();
    result->total_time_sec = std::chrono::duration<double>(end_time - start_time).count();
    result->success = (fail_count == 0);
    result->robots_calibrated = ok_count;
    result->robots_failed = fail_count;

    std::string pg_status = use_pg ? " (pose_graph=" + topology + ")" : "";
    result->message = "Calibration complete (algorithm=" + current_algorithm + "). ok=" +
                      std::to_string(ok_count) + " fail=" + std::to_string(fail_count) +
                      pg_status + per_robot_inlier_info;

    is_calibrating_.store(false);
    return result;
  }

  // ============== Helper Methods ==============

  bool exceedsRmseThreshold(const mrc::RegistrationResult& reg,
                            const std::string& algorithm) const {
    if (!reg.rmse_t_m.has_value()) return false;
    // KISS-Matcher computes RMSE on matched correspondences (sparse keypoints).
    // FRICP computes nearest-neighbor RMSE on dense clouds.
    // ICP refiner also produces NN RMSE. Use the appropriate threshold.
    const double threshold = (algorithm == "kiss_matcher") ? max_corr_rmse_t_m_ : max_nn_rmse_t_m_;
    return *reg.rmse_t_m > threshold;
  }

  mrc::RegistrationResult maybeRefineWithICP(const mrc::RegistrationResult& coarse,
                                             const mrc::RegistrationInput& source_in,
                                             const mrc::RegistrationInput& target_in) {
    if (!use_icp_refinement_) return coarse;
    if (!source_in.cloud.has_value() || !target_in.cloud.has_value()) return coarse;

    mrc::ICPRefinementParams params;
    params.max_iterations = icp_max_iterations_;
    params.max_correspondence_distance = icp_max_correspondence_distance_;
    params.voxel_leaf_size = icp_voxel_leaf_size_;

    mrc::ICPRefiner refiner(params);
    try {
      auto refined = refiner.refine(source_in.cloud.value(), target_in.cloud.value(), coarse);
      if (refined.ok) {
        bool accepted = (refined.message.find("GICP refined") != std::string::npos);
        RCLCPP_INFO(get_logger(),
                    "GICP: coarse_rmse=%.4f, gicp_rmse=%.4f -> %s",
                    coarse.rmse_t_m.value_or(-1.0),
                    accepted ? refined.rmse_t_m.value_or(-1.0) : refined.score,
                    accepted ? "accepted" : "rejected (worse)");
      }
      return refined;
    } catch (const std::exception& e) {
      RCLCPP_WARN(get_logger(), "GICP refinement failed: %s", e.what());
      return coarse;
    }
  }

  bool createBackendForAlgorithm(const std::string& algorithm) {
    auto new_backend = mrc::createBackend(algorithm);
    if (!new_backend) return false;
    std::lock_guard<std::mutex> lock(backend_mutex_);
    backend_ = new_backend;
    RCLCPP_INFO(get_logger(), "Registration backend set to '%s'", algorithm.c_str());
    return true;
  }

  rcl_interfaces::msg::SetParametersResult onParameterChange(
      const std::vector<rclcpp::Parameter>& parameters) {
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true;

    for (const auto& param : parameters) {
      if (param.get_name() == "algorithm") {
        const std::string new_algo = param.as_string();
        if (new_algo != algorithm_ && !createBackendForAlgorithm(new_algo)) {
          result.successful = false;
          result.reason = "Unknown algorithm: " + new_algo;
        } else {
          algorithm_ = new_algo;
        }
      } else if (param.get_name() == "use_icp_refinement") {
        use_icp_refinement_ = param.as_bool();
      } else if (param.get_name() == "icp_max_iterations") {
        icp_max_iterations_ = static_cast<int>(param.as_int());
      } else if (param.get_name() == "icp_max_correspondence_distance") {
        icp_max_correspondence_distance_ = param.as_double();
      } else if (param.get_name() == "icp_voxel_leaf_size") {
        icp_voxel_leaf_size_ = param.as_double();
      } else if (param.get_name() == "max_nn_rmse_t_m") {
        max_nn_rmse_t_m_ = param.as_double();
      } else if (param.get_name() == "max_corr_rmse_t_m") {
        max_corr_rmse_t_m_ = param.as_double();
      } else if (param.get_name() == "use_pose_graph") {
        use_pose_graph_ = param.as_bool();
      } else if (param.get_name() == "pose_graph_topology") {
        const std::string new_val = param.as_string();
        if (new_val != "star" && new_val != "mesh") {
          result.successful = false;
          result.reason = "pose_graph_topology must be 'star' or 'mesh'";
        } else {
          pose_graph_topology_ = new_val;
        }
      }
    }
    return result;
  }

  static std::string qualifyService(const std::string& ns, const std::string& srv_rel_or_abs) {
    if (!srv_rel_or_abs.empty() && srv_rel_or_abs[0] == '/') return srv_rel_or_abs;
    if (ns.empty() || ns == "/") return "/" + srv_rel_or_abs;
    return ns + "/" + srv_rel_or_abs;
  }

  void onAnnouncement(const hector_multi_robot_msgs::msg::RobotAnnouncement& msg) {
    RobotInfo info;
    info.robot_id = msg.id;
    info.robot_type = msg.name;
    info.ns = msg.ros_namespace;
    info.last_seen = now();
    info.data_service = qualifyService(info.ns, slave_service_relative_);
    robots_[info.robot_id] = info;
    RCLCPP_INFO(get_logger(), "Robot announced: id='%s', ns='%s'",
                info.robot_id.c_str(), info.ns.c_str());
  }

  bool getDataFromService(const std::string& service_name,
                          mrc::srv::GetRegistrationData::Response& out) {
    auto client = create_client<mrc::srv::GetRegistrationData>(
        service_name, rclcpp::ServicesQoS(), service_callback_group_);

    auto timeout_duration = std::chrono::duration<double>(request_timeout_sec_);
    if (!client->wait_for_service(timeout_duration)) {
      RCLCPP_WARN(get_logger(), "Service %s not available", service_name.c_str());
      return false;
    }

    auto req = std::make_shared<mrc::srv::GetRegistrationData::Request>();
    req->requester_robot_id = robot_id_;
    req->algorithm = algorithm_;

    auto future = client->async_send_request(req);
    if (future.wait_for(timeout_duration) != std::future_status::ready) {
      RCLCPP_WARN(get_logger(), "Service call to %s timed out", service_name.c_str());
      return false;
    }

    try {
      out = *future.get();
      if (!out.ok) {
        RCLCPP_WARN(get_logger(), "Slave %s returned not-ok: %s",
                    service_name.c_str(), out.message.c_str());
      }
      return out.ok;
    } catch (const std::exception& e) {
      RCLCPP_ERROR(get_logger(), "Exception from %s: %s", service_name.c_str(), e.what());
      return false;
    }
  }

  std::string makeChildFrame(const RobotInfo& robot) const {
    const std::string base = default_slave_base_frame_;
    if (robot.ns.empty() || robot.ns == "/") return base;
    std::string ns = robot.ns;
    if (!ns.empty() && ns[0] == '/') ns = ns.substr(1);
    return ns + "/" + base;
  }

#ifdef MRC_USE_GTSAM
  static gtsam::Pose3 tfToPose3(const geometry_msgs::msg::TransformStamped& tf) {
    const auto& tr = tf.transform.translation;
    const auto& qr = tf.transform.rotation;
    return gtsam::Pose3(gtsam::Rot3::Quaternion(qr.w, qr.x, qr.y, qr.z),
                        gtsam::Point3(tr.x, tr.y, tr.z));
  }

  static geometry_msgs::msg::TransformStamped pose3ToTf(const gtsam::Pose3& p) {
    geometry_msgs::msg::TransformStamped tf;
    const auto t = p.translation();
    tf.transform.translation.x = t.x();
    tf.transform.translation.y = t.y();
    tf.transform.translation.z = t.z();
    const auto q = p.rotation().toQuaternion();
    tf.transform.rotation.w = q.w();
    tf.transform.rotation.x = q.x();
    tf.transform.rotation.y = q.y();
    tf.transform.rotation.z = q.z();
    return tf;
  }

  static gtsam::Key keyForRobot(size_t idx) {
    return gtsam::Symbol('r', static_cast<uint64_t>(idx));
  }
#endif

  // ============== Debug Visualization ==============

  void publishPoseMarkers(
      const rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr& pub,
      const std::map<std::string, geometry_msgs::msg::TransformStamped>& poses,
      float r, float g, float b, const std::string& ns,
      const std::string& topology = "star") {
    visualization_msgs::msg::MarkerArray markers;

    // Delete all previous markers in this namespace
    visualization_msgs::msg::Marker delete_all;
    delete_all.action = visualization_msgs::msg::Marker::DELETEALL;
    delete_all.header.frame_id = master_map_frame_;
    delete_all.header.stamp = now();
    delete_all.ns = ns;
    markers.markers.push_back(delete_all);

    int id = 0;

    // Master at origin
    {
      visualization_msgs::msg::Marker cube;
      cube.header.frame_id = master_map_frame_;
      cube.header.stamp = now();
      cube.ns = ns;
      cube.id = id++;
      cube.type = visualization_msgs::msg::Marker::CUBE;
      cube.action = visualization_msgs::msg::Marker::ADD;
      cube.pose.position.x = 0.0;
      cube.pose.position.y = 0.0;
      cube.pose.position.z = 0.0;
      cube.pose.orientation.w = 1.0;
      cube.scale.x = 0.2;
      cube.scale.y = 0.2;
      cube.scale.z = 0.2;
      cube.color.r = r;
      cube.color.g = g;
      cube.color.b = b;
      cube.color.a = 1.0;
      cube.lifetime = rclcpp::Duration(0, 0);
      markers.markers.push_back(cube);

      visualization_msgs::msg::Marker text;
      text.header = cube.header;
      text.ns = ns + "_text";
      text.id = id++;
      text.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
      text.action = visualization_msgs::msg::Marker::ADD;
      text.pose.position.x = 0.0;
      text.pose.position.y = 0.0;
      text.pose.position.z = 0.3;
      text.pose.orientation.w = 1.0;
      text.scale.z = 0.15;
      text.color.r = r;
      text.color.g = g;
      text.color.b = b;
      text.color.a = 1.0;
      text.text = robot_id_ + " (master)";
      text.lifetime = rclcpp::Duration(0, 0);
      markers.markers.push_back(text);
    }

    // Each robot pose
    for (const auto& [robot_name, tf] : poses) {
      visualization_msgs::msg::Marker sphere;
      sphere.header.frame_id = master_map_frame_;
      sphere.header.stamp = now();
      sphere.ns = ns;
      sphere.id = id++;
      sphere.type = visualization_msgs::msg::Marker::SPHERE;
      sphere.action = visualization_msgs::msg::Marker::ADD;
      sphere.pose.position.x = tf.transform.translation.x;
      sphere.pose.position.y = tf.transform.translation.y;
      sphere.pose.position.z = tf.transform.translation.z;
      sphere.pose.orientation = tf.transform.rotation;
      sphere.scale.x = 0.15;
      sphere.scale.y = 0.15;
      sphere.scale.z = 0.15;
      sphere.color.r = r;
      sphere.color.g = g;
      sphere.color.b = b;
      sphere.color.a = 1.0;
      sphere.lifetime = rclcpp::Duration(0, 0);
      markers.markers.push_back(sphere);

      visualization_msgs::msg::Marker text;
      text.header = sphere.header;
      text.ns = ns + "_text";
      text.id = id++;
      text.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
      text.action = visualization_msgs::msg::Marker::ADD;
      text.pose.position.x = tf.transform.translation.x;
      text.pose.position.y = tf.transform.translation.y;
      text.pose.position.z = tf.transform.translation.z + 0.25;
      text.pose.orientation.w = 1.0;
      text.scale.z = 0.12;
      text.color.r = r;
      text.color.g = g;
      text.color.b = b;
      text.color.a = 1.0;
      text.text = robot_name;
      text.lifetime = rclcpp::Duration(0, 0);
      markers.markers.push_back(text);

      // Line from master to robot (always drawn — star edge)
      visualization_msgs::msg::Marker line;
      line.header = sphere.header;
      line.ns = ns + "_edges";
      line.id = id++;
      line.type = visualization_msgs::msg::Marker::LINE_STRIP;
      line.action = visualization_msgs::msg::Marker::ADD;
      geometry_msgs::msg::Point p0, p1;
      p0.x = 0.0; p0.y = 0.0; p0.z = 0.0;
      p1.x = tf.transform.translation.x;
      p1.y = tf.transform.translation.y;
      p1.z = tf.transform.translation.z;
      line.points.push_back(p0);
      line.points.push_back(p1);
      line.scale.x = 0.02;
      line.color.r = r;
      line.color.g = g;
      line.color.b = b;
      line.color.a = 0.5;
      line.lifetime = rclcpp::Duration(0, 0);
      markers.markers.push_back(line);
    }

    // Mesh topology: add slave-to-slave edges
    if (topology == "mesh") {
      std::vector<std::pair<std::string, geometry_msgs::msg::TransformStamped>> pose_vec(
          poses.begin(), poses.end());
      for (size_t i = 0; i < pose_vec.size(); ++i) {
        for (size_t j = i + 1; j < pose_vec.size(); ++j) {
          visualization_msgs::msg::Marker line;
          line.header.frame_id = master_map_frame_;
          line.header.stamp = now();
          line.ns = ns + "_edges";
          line.id = id++;
          line.type = visualization_msgs::msg::Marker::LINE_STRIP;
          line.action = visualization_msgs::msg::Marker::ADD;
          geometry_msgs::msg::Point pa, pb;
          pa.x = pose_vec[i].second.transform.translation.x;
          pa.y = pose_vec[i].second.transform.translation.y;
          pa.z = pose_vec[i].second.transform.translation.z;
          pb.x = pose_vec[j].second.transform.translation.x;
          pb.y = pose_vec[j].second.transform.translation.y;
          pb.z = pose_vec[j].second.transform.translation.z;
          line.points.push_back(pa);
          line.points.push_back(pb);
          line.scale.x = 0.015;
          line.color.r = r;
          line.color.g = g;
          line.color.b = b;
          line.color.a = 0.35;
          line.lifetime = rclcpp::Duration(0, 0);
          markers.markers.push_back(line);
        }
      }
    }

    pub->publish(markers);
    RCLCPP_DEBUG(get_logger(), "Published %zu markers on ns '%s'",
                 markers.markers.size(), ns.c_str());
  }

  // ============== Member Variables ==============

  std::string robot_id_;
  std::string master_map_frame_;
  std::string announce_topic_;
  std::string slave_service_relative_;
  std::string algorithm_;
  std::string local_slave_data_service_;
  std::string default_slave_base_frame_;
  double request_timeout_sec_{2.0};

  bool use_pose_graph_{true};
  std::string pose_graph_topology_{"star"};

  bool use_icp_refinement_{false};
  int icp_max_iterations_{50};
  double icp_max_correspondence_distance_{1.0};
  double icp_voxel_leaf_size_{0.1};
  double max_nn_rmse_t_m_{2.0};
  double max_corr_rmse_t_m_{1.0};

#ifdef MRC_USE_GTSAM
  double pg_default_sigma_t_m_{0.3};
  double pg_default_sigma_r_rad_{0.2};
  double pg_min_sigma_t_m_{0.10};
  double pg_min_sigma_r_rad_{0.10};
  double pg_max_sigma_t_m_{5.0};
  double pg_max_sigma_r_rad_{3.14159};
  bool pg_use_robust_{true};
  double pg_huber_k_{1.345};
  double anchor_sigma_t_m_{1e-3};
  double anchor_sigma_r_rad_{1e-3};
  int min_inliers_{10};
#endif

  std::unordered_map<std::string, RobotInfo> robots_;

  mrc::MRCRegistrationPtr backend_;
  std::mutex backend_mutex_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_callback_handle_;

  rclcpp::Subscription<hector_multi_robot_msgs::msg::RobotAnnouncement>::SharedPtr ann_sub_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr calibrate_srv_;
  rclcpp_action::Server<CalibrateAction>::SharedPtr calibrate_action_server_;
  std::unique_ptr<tf2_ros::StaticTransformBroadcaster> tf_static_broadcaster_;

  rclcpp::CallbackGroup::SharedPtr service_callback_group_;

  // Debug visualization
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr unoptimized_poses_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr optimized_poses_pub_;

  std::atomic<bool> is_calibrating_{false};
  std::atomic<bool> cancel_requested_{false};
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<MRCMasterNode>(rclcpp::NodeOptions{});

  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();

  rclcpp::shutdown();
  return 0;
}
