/**
 * MRC Evaluator Node
 *
 * Compares ground truth robot poses from simulation with MRC calibration output.
 * Subscribes to Gazebo pose topics for ground truth and TF for MRC output.
 * Publishes evaluation metrics.
 *
 * Also provides pose graph comparison testing to evaluate whether pose graph
 * optimization improves localization accuracy.
 */

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <std_msgs/msg/float64.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <rcl_interfaces/srv/set_parameters.hpp>
#include <rcl_interfaces/msg/parameter.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include "mrc/action/calibrate.hpp"

using CalibrateAction = mrc::action::Calibrate;
using GoalHandleCalibrate = rclcpp_action::ClientGoalHandle<CalibrateAction>;

#include <atomic>
#include <cmath>
#include <future>
#include <iomanip>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace mrc_test {

struct GroundTruth {
  double x;
  double y;
  double z;
  double yaw;
};

struct EvaluationResult {
  std::string robot_name;
  double translation_error_m;
  double rotation_error_rad;
  bool valid;
};

struct ComparisonMetrics {
  double trans_rms_m;
  double rot_rms_rad;
  double trans_avg_m;
  double rot_avg_rad;
  double trans_max_m;
  double rot_max_rad;
  int valid_robots;
  std::vector<EvaluationResult> per_robot;
};

struct PoseGraphComparisonResult {
  ComparisonMetrics no_pg;         // No pose graph (direct pairwise)
  ComparisonMetrics star_pg;       // Pose graph with star topology (master<->slaves)
  ComparisonMetrics mesh_pg;       // Pose graph with mesh topology (all pairwise edges)

  // Improvements: star vs none
  double star_trans_improvement_percent;
  double star_rot_improvement_percent;

  // Improvements: mesh vs star
  double mesh_trans_improvement_percent;
  double mesh_rot_improvement_percent;

  std::string best_mode;
};

// Configuration for a single test case
struct TestConfig {
  std::string algorithm;           // "kiss_matcher" or "fricp"
  bool use_pose_graph;
  std::string topology;            // "star" or "mesh" (only if use_pose_graph)

  std::string name() const {
    if (!use_pose_graph) {
      return algorithm + "_no_pg";
    }
    return algorithm + "_" + topology;
  }
};

// Result for a single configuration
struct ConfigResult {
  TestConfig config;
  ComparisonMetrics metrics;
  double avg_calibration_time_sec;
  int successful_trials;
  int total_trials;
};

class MRCEvaluator : public rclcpp::Node {
public:
  MRCEvaluator() : Node("mrc_evaluator") {
    // Declare parameters
    declare_parameter<std::vector<std::string>>("robot_names", std::vector<std::string>{});
    declare_parameter<double>("evaluation_rate_hz", 1.0);
    declare_parameter<std::string>("reference_frame", "odom");

    // Get robot names and reference frame
    robot_names_ = get_parameter("robot_names").as_string_array();
    reference_frame_ = get_parameter("reference_frame").as_string();

    // Load ground truth and frame mapping for each robot
    for (const auto& name : robot_names_) {
      GroundTruth gt;
      declare_parameter<double>("ground_truth_" + name + "_x", 0.0);
      declare_parameter<double>("ground_truth_" + name + "_y", 0.0);
      declare_parameter<double>("ground_truth_" + name + "_z", 0.0);
      declare_parameter<double>("ground_truth_" + name + "_yaw", 0.0);

      gt.x = get_parameter("ground_truth_" + name + "_x").as_double();
      gt.y = get_parameter("ground_truth_" + name + "_y").as_double();
      gt.z = get_parameter("ground_truth_" + name + "_z").as_double();
      gt.yaw = get_parameter("ground_truth_" + name + "_yaw").as_double();

      ground_truth_[name] = gt;

      // Frame mapping: allow custom frame per robot, default to <name>/base_link
      declare_parameter<std::string>("frame_" + name, name + "/base_link");
      robot_frames_[name] = get_parameter("frame_" + name).as_string();

      RCLCPP_INFO(get_logger(), "Robot %s: frame=%s, GT=(%.3f, %.3f, %.3f, %.3f)",
                  name.c_str(), robot_frames_[name].c_str(),
                  gt.x, gt.y, gt.z, gt.yaw);
    }

    // TF2 buffer and listener
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    // Publishers for evaluation metrics
    translation_error_pub_ = create_publisher<std_msgs::msg::Float64>(
        "~/translation_error_rms", 10);
    rotation_error_pub_ = create_publisher<std_msgs::msg::Float64>(
        "~/rotation_error_rms", 10);

    // Per-robot error publishers
    for (const auto& name : robot_names_) {
      trans_error_pubs_[name] = create_publisher<std_msgs::msg::Float64>(
          "~/" + name + "/translation_error", 10);
      rot_error_pubs_[name] = create_publisher<std_msgs::msg::Float64>(
          "~/" + name + "/rotation_error", 10);
    }

    // Create a reentrant callback group for service client
    // This allows the response to be processed while we're waiting in the service callback
    client_callback_group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);

    // Service to trigger evaluation
    trigger_eval_srv_ = create_service<std_srvs::srv::Trigger>(
        "~/trigger_evaluation",
        std::bind(&MRCEvaluator::onTriggerEvaluation, this,
                  std::placeholders::_1, std::placeholders::_2));

    // Legacy service client (for backwards compatibility)
    calibrate_client_ = create_client<std_srvs::srv::Trigger>(
        "mrc_master/calibrate",
        rclcpp::ServicesQoS(),
        client_callback_group_);

    // Action client for calibration with feedback
    calibrate_action_client_ = rclcpp_action::create_client<CalibrateAction>(
        this, "mrc_master/calibrate_action");

    // Parameter service client for toggling pose graph on/off
    set_params_client_ = create_client<rcl_interfaces::srv::SetParameters>(
        "mrc_master/set_parameters",
        rclcpp::ServicesQoS(),
        client_callback_group_);

    // Service to trigger pose graph comparison test
    comparison_srv_ = create_service<std_srvs::srv::Trigger>(
        "~/trigger_pose_graph_comparison",
        std::bind(&MRCEvaluator::onTriggerComparison, this,
                  std::placeholders::_1, std::placeholders::_2));

    // Service to trigger comprehensive backend comparison
    full_comparison_srv_ = create_service<std_srvs::srv::Trigger>(
        "~/trigger_full_comparison",
        std::bind(&MRCEvaluator::onTriggerFullComparison, this,
                  std::placeholders::_1, std::placeholders::_2));

    // Number of trials for comparison (more trials = more robust statistics)
    declare_parameter<int>("comparison_trials", 3);
    comparison_trials_ = get_parameter("comparison_trials").as_int();

    // Timer for periodic evaluation
    double rate = get_parameter("evaluation_rate_hz").as_double();
    eval_timer_ = create_wall_timer(
        std::chrono::duration<double>(1.0 / rate),
        std::bind(&MRCEvaluator::evaluationCallback, this));

    RCLCPP_INFO(get_logger(), "MRC Evaluator initialized with %zu robots", robot_names_.size());
  }

private:
  void onTriggerEvaluation(
      const std::shared_ptr<std_srvs::srv::Trigger::Request> /*request*/,
      std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    // First trigger MRC calibration
    if (!calibrate_client_->wait_for_service(std::chrono::seconds(2))) {
      response->success = false;
      response->message = "MRC calibrate service not available";
      return;
    }

    auto calibrate_request = std::make_shared<std_srvs::srv::Trigger::Request>();
    auto future = calibrate_client_->async_send_request(calibrate_request);

    if (future.wait_for(std::chrono::seconds(30)) != std::future_status::ready) {
      response->success = false;
      response->message = "Calibration request timed out";
      return;
    }

    auto result = future.get();
    if (!result->success) {
      response->success = false;
      response->message = "Calibration failed: " + result->message;
      return;
    }

    // Wait a bit for TF to propagate
    std::this_thread::sleep_for(std::chrono::milliseconds(5500));

    // Run evaluation
    auto eval_results = runEvaluation();

    // Build response
    std::stringstream ss;
    ss << "Evaluation Results:\n";
    for (const auto& res : eval_results) {
      if (res.valid) {
        ss << "  " << res.robot_name
           << ": trans_err=" << res.translation_error_m << "m"
           << ", rot_err=" << (res.rotation_error_rad * 180.0 / M_PI) << "deg\n";
      } else {
        ss << "  " << res.robot_name << ": INVALID (no TF available)\n";
      }
    }

    response->success = true;
    response->message = ss.str();
  }

  void evaluationCallback() {
    auto results = runEvaluation();

    double trans_sum_sq = 0.0;
    double rot_sum_sq = 0.0;
    int valid_count = 0;

    for (const auto& res : results) {
      if (res.valid) {
        // Publish per-robot errors
        std_msgs::msg::Float64 trans_msg, rot_msg;
        trans_msg.data = res.translation_error_m;
        rot_msg.data = res.rotation_error_rad;

        trans_error_pubs_[res.robot_name]->publish(trans_msg);
        rot_error_pubs_[res.robot_name]->publish(rot_msg);

        trans_sum_sq += res.translation_error_m * res.translation_error_m;
        rot_sum_sq += res.rotation_error_rad * res.rotation_error_rad;
        valid_count++;
      }
    }

    // Publish RMS errors
    if (valid_count > 0) {
      std_msgs::msg::Float64 rms_trans, rms_rot;
      rms_trans.data = std::sqrt(trans_sum_sq / valid_count);
      rms_rot.data = std::sqrt(rot_sum_sq / valid_count);

      translation_error_pub_->publish(rms_trans);
      rotation_error_pub_->publish(rms_rot);

      RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 5000,
                           "RMS Errors: translation=%.4fm, rotation=%.4fdeg",
                           rms_trans.data, rms_rot.data * 180.0 / M_PI);
    }
  }

  std::vector<EvaluationResult> runEvaluation() {
    std::vector<EvaluationResult> results;

    for (const auto& name : robot_names_) {
      EvaluationResult res;
      res.robot_name = name;
      res.valid = false;

      // Try to get MRC transform from reference frame to robot's target frame
      const std::string& target_frame = robot_frames_.at(name);
      try {
        auto transform = tf_buffer_->lookupTransform(
            reference_frame_, target_frame, tf2::TimePointZero, tf2::durationFromSec(0.1));

        // Get ground truth
        const auto& gt = ground_truth_[name];

        // Calculate translation error
        double dx = transform.transform.translation.x - gt.x;
        double dy = transform.transform.translation.y - gt.y;
        double dz = transform.transform.translation.z - gt.z;
        res.translation_error_m = std::sqrt(dx * dx + dy * dy + dz * dz);

        // Calculate rotation error (yaw only for simplicity)
        tf2::Quaternion q(
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w);

        double roll, pitch, yaw;
        tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);

        // Angular difference
        double yaw_error = yaw - gt.yaw;
        // Normalize to [-pi, pi]
        while (yaw_error > M_PI) yaw_error -= 2 * M_PI;
        while (yaw_error < -M_PI) yaw_error += 2 * M_PI;
        res.rotation_error_rad = std::abs(yaw_error);

        res.valid = true;

        RCLCPP_DEBUG(get_logger(), "%s: MRC(%.3f,%.3f,%.3f,%.3f) vs GT(%.3f,%.3f,%.3f,%.3f) "
                     "-> trans_err=%.4fm, rot_err=%.4fdeg",
                     name.c_str(),
                     transform.transform.translation.x,
                     transform.transform.translation.y,
                     transform.transform.translation.z,
                     yaw,
                     gt.x, gt.y, gt.z, gt.yaw,
                     res.translation_error_m,
                     res.rotation_error_rad * 180.0 / M_PI);

      } catch (const tf2::TransformException& ex) {
        RCLCPP_DEBUG_THROTTLE(get_logger(), *get_clock(), 5000,
                              "Could not get transform for %s: %s",
                              name.c_str(), ex.what());
      }

      results.push_back(res);
    }

    return results;
  }

  bool setMasterParameters(bool use_pose_graph, const std::string& topology = "star") {
    if (!set_params_client_->wait_for_service(std::chrono::seconds(2))) {
      RCLCPP_ERROR(get_logger(), "Set parameters service not available");
      return false;
    }

    auto request = std::make_shared<rcl_interfaces::srv::SetParameters::Request>();

    // Set use_pose_graph
    rcl_interfaces::msg::Parameter pg_param;
    pg_param.name = "use_pose_graph";
    pg_param.value.type = rcl_interfaces::msg::ParameterType::PARAMETER_BOOL;
    pg_param.value.bool_value = use_pose_graph;
    request->parameters.push_back(pg_param);

    // Set pose_graph_topology
    rcl_interfaces::msg::Parameter topo_param;
    topo_param.name = "pose_graph_topology";
    topo_param.value.type = rcl_interfaces::msg::ParameterType::PARAMETER_STRING;
    topo_param.value.string_value = topology;
    request->parameters.push_back(topo_param);

    auto future = set_params_client_->async_send_request(request);
    if (future.wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
      RCLCPP_ERROR(get_logger(), "Set parameters request timed out");
      return false;
    }

    auto result = future.get();
    for (size_t i = 0; i < result->results.size(); ++i) {
      if (!result->results[i].successful) {
        RCLCPP_ERROR(get_logger(), "Failed to set parameter %zu: %s",
                     i, result->results[i].reason.c_str());
        return false;
      }
    }

    RCLCPP_INFO(get_logger(), "Set use_pose_graph=%s, topology=%s",
                use_pose_graph ? "true" : "false", topology.c_str());
    return true;
  }

  bool setTestConfig(const TestConfig& config) {
    if (!set_params_client_->wait_for_service(std::chrono::seconds(2))) {
      RCLCPP_ERROR(get_logger(), "Set parameters service not available");
      return false;
    }

    auto request = std::make_shared<rcl_interfaces::srv::SetParameters::Request>();

    // Set algorithm
    rcl_interfaces::msg::Parameter algo_param;
    algo_param.name = "algorithm";
    algo_param.value.type = rcl_interfaces::msg::ParameterType::PARAMETER_STRING;
    algo_param.value.string_value = config.algorithm;
    request->parameters.push_back(algo_param);

    // Set use_pose_graph
    rcl_interfaces::msg::Parameter pg_param;
    pg_param.name = "use_pose_graph";
    pg_param.value.type = rcl_interfaces::msg::ParameterType::PARAMETER_BOOL;
    pg_param.value.bool_value = config.use_pose_graph;
    request->parameters.push_back(pg_param);

    // Set pose_graph_topology
    rcl_interfaces::msg::Parameter topo_param;
    topo_param.name = "pose_graph_topology";
    topo_param.value.type = rcl_interfaces::msg::ParameterType::PARAMETER_STRING;
    topo_param.value.string_value = config.topology;
    request->parameters.push_back(topo_param);

    auto future = set_params_client_->async_send_request(request);
    if (future.wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
      RCLCPP_ERROR(get_logger(), "Set parameters request timed out");
      return false;
    }

    auto result = future.get();
    for (size_t i = 0; i < result->results.size(); ++i) {
      if (!result->results[i].successful) {
        RCLCPP_ERROR(get_logger(), "Failed to set parameter: %s",
                     result->results[i].reason.c_str());
        return false;
      }
    }

    RCLCPP_INFO(get_logger(), "Config: algorithm=%s, pose_graph=%s, topology=%s",
                config.algorithm.c_str(),
                config.use_pose_graph ? "true" : "false",
                config.topology.c_str());
    return true;
  }

  bool triggerCalibrationService() {
    if (!calibrate_client_->wait_for_service(std::chrono::seconds(5))) {
      RCLCPP_ERROR(get_logger(), "Calibrate service not available");
      return false;
    }

    auto request = std::make_shared<std_srvs::srv::Trigger::Request>();
    RCLCPP_DEBUG(get_logger(), "Sending calibration request...");
    auto future = calibrate_client_->async_send_request(request);

    if (future.wait_for(std::chrono::seconds(60)) != std::future_status::ready) {
      RCLCPP_ERROR(get_logger(), "Calibration request timed out");
      return false;
    }

    auto result = future.get();
    if (!result->success) {
      RCLCPP_WARN(get_logger(), "Calibration service returned failure: %s", result->message.c_str());
    }
    return result->success;
  }

  // Action-based calibration (for future use when executor issues are resolved)
  bool triggerCalibrationAction(bool use_pose_graph, const std::string& topology) {
    if (!calibrate_action_client_->wait_for_action_server(std::chrono::seconds(2))) {
      RCLCPP_ERROR(get_logger(), "Calibrate action server not available");
      return false;
    }

    auto goal = CalibrateAction::Goal();
    goal.use_pose_graph = use_pose_graph;
    goal.pose_graph_topology = topology;

    // Use synchronous result callback pattern
    std::promise<std::shared_ptr<CalibrateAction::Result>> result_promise;
    auto result_future = result_promise.get_future();
    bool goal_accepted = false;

    auto send_goal_options = rclcpp_action::Client<CalibrateAction>::SendGoalOptions();

    send_goal_options.goal_response_callback =
        [&goal_accepted, this](const GoalHandleCalibrate::SharedPtr& goal_handle) {
          if (!goal_handle) {
            RCLCPP_ERROR(get_logger(), "Calibration goal was rejected");
            goal_accepted = false;
          } else {
            goal_accepted = true;
            RCLCPP_DEBUG(get_logger(), "Calibration goal accepted");
          }
        };

    send_goal_options.feedback_callback =
        [this](GoalHandleCalibrate::SharedPtr,
               const std::shared_ptr<const CalibrateAction::Feedback> feedback) {
          RCLCPP_DEBUG(get_logger(), "[%s] %.0f%% - %s",
                       feedback->phase.c_str(),
                       feedback->progress_percent,
                       feedback->status_message.c_str());
        };

    send_goal_options.result_callback =
        [&result_promise, this](const GoalHandleCalibrate::WrappedResult& wrapped_result) {
          if (wrapped_result.code == rclcpp_action::ResultCode::SUCCEEDED) {
            RCLCPP_INFO(get_logger(), "Calibration completed: %s",
                        wrapped_result.result->message.c_str());
          } else {
            RCLCPP_WARN(get_logger(), "Calibration failed with code %d",
                        static_cast<int>(wrapped_result.code));
          }
          result_promise.set_value(wrapped_result.result);
        };

    calibrate_action_client_->async_send_goal(goal, send_goal_options);

    // Wait for result with timeout (callbacks are processed by executor)
    auto status = result_future.wait_for(std::chrono::seconds(60));
    if (status != std::future_status::ready) {
      RCLCPP_ERROR(get_logger(), "Calibration action timed out");
      return false;
    }

    auto result = result_future.get();
    return result && result->success;
  }

  ComparisonMetrics computeMetrics(const std::vector<EvaluationResult>& results) {
    ComparisonMetrics metrics{};
    metrics.per_robot = results;

    double trans_sum = 0.0;
    double rot_sum = 0.0;
    double trans_sum_sq = 0.0;
    double rot_sum_sq = 0.0;
    metrics.trans_max_m = 0.0;
    metrics.rot_max_rad = 0.0;
    metrics.valid_robots = 0;

    for (const auto& res : results) {
      if (res.valid) {
        trans_sum += res.translation_error_m;
        rot_sum += res.rotation_error_rad;
        trans_sum_sq += res.translation_error_m * res.translation_error_m;
        rot_sum_sq += res.rotation_error_rad * res.rotation_error_rad;
        metrics.trans_max_m = std::max(metrics.trans_max_m, res.translation_error_m);
        metrics.rot_max_rad = std::max(metrics.rot_max_rad, res.rotation_error_rad);
        metrics.valid_robots++;
      }
    }

    if (metrics.valid_robots > 0) {
      metrics.trans_avg_m = trans_sum / metrics.valid_robots;
      metrics.rot_avg_rad = rot_sum / metrics.valid_robots;
      metrics.trans_rms_m = std::sqrt(trans_sum_sq / metrics.valid_robots);
      metrics.rot_rms_rad = std::sqrt(rot_sum_sq / metrics.valid_robots);
    }

    return metrics;
  }

  ComparisonMetrics runTrialsAndAverage(const std::string& mode_name, int num_trials,
                                         bool use_pose_graph, const std::string& topology) {
    std::vector<ComparisonMetrics> trial_results;

    for (int trial = 0; trial < num_trials; ++trial) {
      RCLCPP_INFO(get_logger(), "  Trial %d/%d (%s)...",
                  trial + 1, num_trials, mode_name.c_str());

      // Set parameters and use legacy service (more reliable than action client)
      bool cal_success = false;
      if (setMasterParameters(use_pose_graph, topology)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        cal_success = triggerCalibrationService();
      }

      if (!cal_success) {
        RCLCPP_WARN(get_logger(), "  Trial %d calibration failed, skipping", trial + 1);
        continue;
      }

      // Wait for TF to propagate
      std::this_thread::sleep_for(std::chrono::milliseconds(5500));

      auto results = runEvaluation();
      auto metrics = computeMetrics(results);

      if (metrics.valid_robots > 0) {
        trial_results.push_back(metrics);
        RCLCPP_INFO(get_logger(), "  Trial %d: trans_rms=%.4fm, rot_rms=%.2fdeg",
                    trial + 1, metrics.trans_rms_m, metrics.rot_rms_rad * 180.0 / M_PI);
      }
    }

    // Average the results across trials
    ComparisonMetrics avg{};
    if (trial_results.empty()) {
      return avg;
    }

    for (const auto& m : trial_results) {
      avg.trans_rms_m += m.trans_rms_m;
      avg.rot_rms_rad += m.rot_rms_rad;
      avg.trans_max_m += m.trans_max_m;
      avg.rot_max_rad += m.rot_max_rad;
      avg.valid_robots += m.valid_robots;
    }

    size_t n = trial_results.size();
    avg.trans_rms_m /= n;
    avg.rot_rms_rad /= n;
    avg.trans_max_m /= n;
    avg.rot_max_rad /= n;
    avg.valid_robots /= static_cast<int>(n);

    // Use per-robot results from last trial for detailed breakdown
    if (!trial_results.empty()) {
      avg.per_robot = trial_results.back().per_robot;
    }

    return avg;
  }

  void onTriggerComparison(
      const std::shared_ptr<std_srvs::srv::Trigger::Request> /*request*/,
      std::shared_ptr<std_srvs::srv::Trigger::Response> response) {

    // Check if already running
    if (comparison_running_.load()) {
      response->success = false;
      response->message = "Comparison test already running";
      return;
    }

    // Run comparison in a separate thread to not block the executor
    // This allows action client callbacks to be processed
    std::promise<std::string> result_promise;
    auto result_future = result_promise.get_future();

    std::thread([this, &result_promise]() {
      auto result = runComparisonTest();
      result_promise.set_value(result);
    }).detach();

    // Wait for result (with timeout)
    if (result_future.wait_for(std::chrono::minutes(10)) != std::future_status::ready) {
      response->success = false;
      response->message = "Comparison test timed out";
      return;
    }

    response->success = true;
    response->message = result_future.get();
  }

  std::string runComparisonTest() {
    comparison_running_.store(true);

    RCLCPP_INFO(get_logger(), "=== Starting Pose Graph Comparison Test ===");
    RCLCPP_INFO(get_logger(), "Running %d trials for each of 3 modes...", comparison_trials_);

    PoseGraphComparisonResult comparison;

    // Mode 1: No pose graph (direct pairwise registration)
    RCLCPP_INFO(get_logger(), "\n--- Mode 1: NO pose graph (direct pairwise) ---");
    comparison.no_pg = runTrialsAndAverage("no_pose_graph", comparison_trials_, false, "star");

    // Mode 2: Pose graph with star topology (master <-> each slave)
    RCLCPP_INFO(get_logger(), "\n--- Mode 2: Pose graph STAR topology (master<->slaves) ---");
    comparison.star_pg = runTrialsAndAverage("star_topology", comparison_trials_, true, "star");

    // Mode 3: Pose graph with mesh topology (all pairwise edges)
    RCLCPP_INFO(get_logger(), "\n--- Mode 3: Pose graph MESH topology (all edges) ---");
    comparison.mesh_pg = runTrialsAndAverage("mesh_topology", comparison_trials_, true, "mesh");

    // Calculate improvements: star vs none
    if (comparison.no_pg.trans_rms_m > 0) {
      comparison.star_trans_improvement_percent =
          (comparison.no_pg.trans_rms_m - comparison.star_pg.trans_rms_m) /
          comparison.no_pg.trans_rms_m * 100.0;
    } else {
      comparison.star_trans_improvement_percent = 0.0;
    }
    if (comparison.no_pg.rot_rms_rad > 0) {
      comparison.star_rot_improvement_percent =
          (comparison.no_pg.rot_rms_rad - comparison.star_pg.rot_rms_rad) /
          comparison.no_pg.rot_rms_rad * 100.0;
    } else {
      comparison.star_rot_improvement_percent = 0.0;
    }

    // Calculate improvements: mesh vs star
    if (comparison.star_pg.trans_rms_m > 0) {
      comparison.mesh_trans_improvement_percent =
          (comparison.star_pg.trans_rms_m - comparison.mesh_pg.trans_rms_m) /
          comparison.star_pg.trans_rms_m * 100.0;
    } else {
      comparison.mesh_trans_improvement_percent = 0.0;
    }
    if (comparison.star_pg.rot_rms_rad > 0) {
      comparison.mesh_rot_improvement_percent =
          (comparison.star_pg.rot_rms_rad - comparison.mesh_pg.rot_rms_rad) /
          comparison.star_pg.rot_rms_rad * 100.0;
    } else {
      comparison.mesh_rot_improvement_percent = 0.0;
    }

    // Determine best mode
    double best_trans = comparison.no_pg.trans_rms_m;
    comparison.best_mode = "none";
    if (comparison.star_pg.trans_rms_m < best_trans) {
      best_trans = comparison.star_pg.trans_rms_m;
      comparison.best_mode = "star";
    }
    if (comparison.mesh_pg.trans_rms_m < best_trans) {
      comparison.best_mode = "mesh";
    }

    // Build response message
    std::stringstream ss;
    ss << "\n";
    ss << "================================================================\n";
    ss << "           POSE GRAPH TOPOLOGY COMPARISON RESULTS\n";
    ss << "================================================================\n\n";

    auto printMetrics = [&ss](const std::string& name, const ComparisonMetrics& m) {
      ss << name << ":\n";
      ss << "  Translation RMS: " << std::fixed << std::setprecision(4) << m.trans_rms_m << " m\n";
      ss << "  Rotation RMS:    " << std::fixed << std::setprecision(2)
         << (m.rot_rms_rad * 180.0 / M_PI) << " deg\n";
      ss << "  Translation Max: " << std::fixed << std::setprecision(4) << m.trans_max_m << " m\n";
      ss << "  Rotation Max:    " << std::fixed << std::setprecision(2)
         << (m.rot_max_rad * 180.0 / M_PI) << " deg\n\n";
    };

    printMetrics("1. NO Pose Graph (direct pairwise)", comparison.no_pg);
    printMetrics("2. STAR Topology (master <-> slaves)", comparison.star_pg);
    printMetrics("3. MESH Topology (all pairwise edges)", comparison.mesh_pg);

    ss << "----------------------------------------------------------------\n";
    ss << "IMPROVEMENTS (positive = better):\n\n";

    ss << "Star vs None:\n";
    ss << "  Translation: " << std::fixed << std::setprecision(1)
       << comparison.star_trans_improvement_percent << "%\n";
    ss << "  Rotation:    " << std::fixed << std::setprecision(1)
       << comparison.star_rot_improvement_percent << "%\n\n";

    ss << "Mesh vs Star:\n";
    ss << "  Translation: " << std::fixed << std::setprecision(1)
       << comparison.mesh_trans_improvement_percent << "%\n";
    ss << "  Rotation:    " << std::fixed << std::setprecision(1)
       << comparison.mesh_rot_improvement_percent << "%\n\n";

    ss << "----------------------------------------------------------------\n";
    ss << "VERDICT: Best topology for translation accuracy: " << comparison.best_mode << "\n";

    if (comparison.best_mode == "mesh") {
      ss << "  -> Full mesh pose graph provides the best results.\n";
    } else if (comparison.best_mode == "star") {
      ss << "  -> Star topology is sufficient; mesh adds no benefit.\n";
    } else {
      ss << "  -> Pose graph optimization does not help in this scenario.\n";
    }
    ss << "================================================================\n";

    RCLCPP_INFO(get_logger(), "%s", ss.str().c_str());

    comparison_running_.store(false);
    return ss.str();
  }

  void onTriggerFullComparison(
      const std::shared_ptr<std_srvs::srv::Trigger::Request> /*request*/,
      std::shared_ptr<std_srvs::srv::Trigger::Response> response) {

    // Check if already running
    if (comparison_running_.load()) {
      response->success = false;
      response->message = "Comparison test already running";
      return;
    }

    // Run comparison in a separate thread to not block the executor
    std::promise<std::string> result_promise;
    auto result_future = result_promise.get_future();

    std::thread([this, &result_promise]() {
      auto result = runFullComparison();
      result_promise.set_value(result);
    }).detach();

    // Wait for result (with extended timeout for 6 configurations)
    if (result_future.wait_for(std::chrono::minutes(20)) != std::future_status::ready) {
      response->success = false;
      response->message = "Full comparison test timed out";
      return;
    }

    response->success = true;
    response->message = result_future.get();
  }

  std::string runFullComparison() {
    comparison_running_.store(true);

    RCLCPP_INFO(get_logger(), "=== Starting FULL MRC Backend Comparison ===");
    RCLCPP_INFO(get_logger(), "Testing 2 algorithms x 3 pose graph modes = 6 configurations");
    RCLCPP_INFO(get_logger(), "Running %d trials per configuration...\n", comparison_trials_);

    // Define all test configurations
    std::vector<TestConfig> configs = {
      {"kiss_matcher", false, "star"},  // KISS-Matcher without pose graph
      {"kiss_matcher", true, "star"},   // KISS-Matcher with star topology
      {"kiss_matcher", true, "mesh"},   // KISS-Matcher with mesh topology
      {"fricp", false, "star"},         // FRICP without pose graph
      {"fricp", true, "star"},          // FRICP with star topology
      {"fricp", true, "mesh"},          // FRICP with mesh topology
    };

    std::vector<ConfigResult> results;

    for (size_t i = 0; i < configs.size(); ++i) {
      const auto& config = configs[i];
      RCLCPP_INFO(get_logger(), "\n--- Configuration %zu/%zu: %s ---",
                  i + 1, configs.size(), config.name().c_str());

      ConfigResult result;
      result.config = config;
      result.successful_trials = 0;
      result.total_trials = comparison_trials_;
      result.avg_calibration_time_sec = 0.0;

      std::vector<ComparisonMetrics> trial_metrics;

      for (int trial = 0; trial < comparison_trials_; ++trial) {
        RCLCPP_INFO(get_logger(), "  Trial %d/%d...", trial + 1, comparison_trials_);

        // Set the configuration
        if (!setTestConfig(config)) {
          RCLCPP_WARN(get_logger(), "  Failed to set config, skipping trial");
          std::this_thread::sleep_for(std::chrono::seconds(2));
          continue;
        }

        // Wait for parameters to take effect
        std::this_thread::sleep_for(std::chrono::seconds(1));

        // Time the calibration
        auto start_time = std::chrono::steady_clock::now();
        bool cal_success = triggerCalibrationService();
        auto end_time = std::chrono::steady_clock::now();

        if (!cal_success) {
          RCLCPP_WARN(get_logger(), "  Calibration failed, skipping trial");
          // Wait before next trial to let system recover
          std::this_thread::sleep_for(std::chrono::seconds(3));
          continue;
        }

        double cal_time = std::chrono::duration<double>(end_time - start_time).count();
        result.avg_calibration_time_sec += cal_time;

        // Wait for TF to propagate
        std::this_thread::sleep_for(std::chrono::seconds(6));

        auto eval_results = runEvaluation();
        auto metrics = computeMetrics(eval_results);

        if (metrics.valid_robots > 0) {
          trial_metrics.push_back(metrics);
          result.successful_trials++;
          RCLCPP_INFO(get_logger(), "  Trial %d: trans_avg=%.4fm, trans_max=%.4fm, rot_avg=%.2fdeg, time=%.2fs",
                      trial + 1, metrics.trans_avg_m, metrics.trans_max_m,
                      metrics.rot_avg_rad * 180.0 / M_PI, cal_time);
        }

        // Wait between trials
        std::this_thread::sleep_for(std::chrono::seconds(2));
      }

      // Average the metrics
      if (!trial_metrics.empty()) {
        result.avg_calibration_time_sec /= result.successful_trials;

        for (const auto& m : trial_metrics) {
          result.metrics.trans_avg_m += m.trans_avg_m;
          result.metrics.rot_avg_rad += m.rot_avg_rad;
          result.metrics.trans_rms_m += m.trans_rms_m;
          result.metrics.rot_rms_rad += m.rot_rms_rad;
          result.metrics.trans_max_m = std::max(result.metrics.trans_max_m, m.trans_max_m);
          result.metrics.rot_max_rad = std::max(result.metrics.rot_max_rad, m.rot_max_rad);
        }
        size_t n = trial_metrics.size();
        result.metrics.trans_avg_m /= n;
        result.metrics.rot_avg_rad /= n;
        result.metrics.trans_rms_m /= n;
        result.metrics.rot_rms_rad /= n;
        // Keep max as actual max across all trials (not averaged)
        result.metrics.valid_robots = trial_metrics.back().valid_robots;
      }

      results.push_back(result);

      // Wait between configurations
      RCLCPP_INFO(get_logger(), "  Waiting before next configuration...");
      std::this_thread::sleep_for(std::chrono::seconds(3));
    }

    // Generate comprehensive report
    std::stringstream ss;
    ss << "\n";
    ss << "====================================================================\n";
    ss << "              COMPREHENSIVE MRC BACKEND COMPARISON\n";
    ss << "====================================================================\n\n";

    ss << "CONFIGURATIONS TESTED:\n";
    ss << "  - Algorithms: kiss_matcher, fricp\n";
    ss << "  - Pose Graph: none, star, mesh\n";
    ss << "  - Trials per config: " << comparison_trials_ << "\n\n";

    ss << "--------------------------------------------------------------------\n";
    ss << "                         RESULTS SUMMARY\n";
    ss << "--------------------------------------------------------------------\n\n";

    ss << std::left << std::setw(20) << "Configuration"
       << std::right << std::setw(11) << "Trans Avg"
       << std::setw(11) << "Trans Max"
       << std::setw(10) << "Rot Avg"
       << std::setw(10) << "Rot Max"
       << std::setw(8) << "Time"
       << std::setw(8) << "OK" << "\n";
    ss << std::string(78, '-') << "\n";

    // Find best configuration
    double best_trans = std::numeric_limits<double>::max();
    std::string best_config_name;

    for (const auto& r : results) {
      ss << std::left << std::setw(20) << r.config.name()
         << std::right << std::fixed << std::setprecision(4)
         << std::setw(9) << r.metrics.trans_avg_m << " m"
         << std::setw(9) << r.metrics.trans_max_m << " m"
         << std::setprecision(2)
         << std::setw(7) << (r.metrics.rot_avg_rad * 180.0 / M_PI) << " deg"
         << std::setw(7) << (r.metrics.rot_max_rad * 180.0 / M_PI) << " deg"
         << std::setw(6) << r.avg_calibration_time_sec << " s"
         << std::setw(4) << r.successful_trials << "/" << r.total_trials << "\n";

      if (r.metrics.trans_avg_m < best_trans && r.successful_trials > 0) {
        best_trans = r.metrics.trans_avg_m;
        best_config_name = r.config.name();
      }
    }

    ss << "\n";
    ss << "--------------------------------------------------------------------\n";
    ss << "                    ALGORITHM COMPARISON\n";
    ss << "--------------------------------------------------------------------\n\n";

    // Group by algorithm
    auto printAlgorithmComparison = [&ss, &results](const std::string& algo) {
      ss << algo << ":\n";
      for (const auto& r : results) {
        if (r.config.algorithm == algo) {
          std::string pg_mode = r.config.use_pose_graph ?
              ("pg_" + r.config.topology) : "no_pg";
          ss << "  " << std::left << std::setw(10) << pg_mode
             << std::fixed << std::setprecision(4)
             << "trans: avg=" << r.metrics.trans_avg_m << "m, max=" << r.metrics.trans_max_m << "m  "
             << std::setprecision(2)
             << "rot: avg=" << (r.metrics.rot_avg_rad * 180.0 / M_PI) << "deg, "
             << "max=" << (r.metrics.rot_max_rad * 180.0 / M_PI) << "deg\n";
        }
      }
      ss << "\n";
    };

    printAlgorithmComparison("kiss_matcher");
    printAlgorithmComparison("fricp");

    ss << "--------------------------------------------------------------------\n";
    ss << "                 POSE GRAPH IMPACT BY ALGORITHM\n";
    ss << "--------------------------------------------------------------------\n\n";

    // Calculate improvements for each algorithm
    auto calcImprovement = [&results](const std::string& algo, const std::string& from_topo,
                                       const std::string& to_topo, bool from_pg, bool to_pg) {
      const ConfigResult* from_result = nullptr;
      const ConfigResult* to_result = nullptr;

      for (const auto& r : results) {
        if (r.config.algorithm == algo) {
          if (r.config.use_pose_graph == from_pg &&
              (!from_pg || r.config.topology == from_topo)) {
            from_result = &r;
          }
          if (r.config.use_pose_graph == to_pg &&
              (!to_pg || r.config.topology == to_topo)) {
            to_result = &r;
          }
        }
      }

      if (from_result && to_result && from_result->metrics.trans_rms_m > 0) {
        return (from_result->metrics.trans_rms_m - to_result->metrics.trans_rms_m) /
               from_result->metrics.trans_rms_m * 100.0;
      }
      return 0.0;
    };

    for (const auto& algo : {"kiss_matcher", "fricp"}) {
      ss << algo << ":\n";
      double star_vs_none = calcImprovement(algo, "", "star", false, true);
      double mesh_vs_star = calcImprovement(algo, "star", "mesh", true, true);
      double mesh_vs_none = calcImprovement(algo, "", "mesh", false, true);

      ss << "  Star vs None:  " << std::showpos << std::fixed << std::setprecision(1)
         << star_vs_none << "%" << std::noshowpos << "\n";
      ss << "  Mesh vs Star:  " << std::showpos << std::fixed << std::setprecision(1)
         << mesh_vs_star << "%" << std::noshowpos << "\n";
      ss << "  Mesh vs None:  " << std::showpos << std::fixed << std::setprecision(1)
         << mesh_vs_none << "%" << std::noshowpos << "\n\n";
    }

    ss << "====================================================================\n";
    ss << "BEST CONFIGURATION: " << best_config_name << "\n";
    ss << "  Translation RMS: " << std::fixed << std::setprecision(4) << best_trans << " m\n";
    ss << "====================================================================\n";

    RCLCPP_INFO(get_logger(), "%s", ss.str().c_str());

    comparison_running_.store(false);
    return ss.str();
  }

  // Robot names
  std::vector<std::string> robot_names_;

  // Reference frame for TF lookups
  std::string reference_frame_;

  // Robot frame mapping (robot_name -> tf_frame)
  std::map<std::string, std::string> robot_frames_;

  // Ground truth poses
  std::map<std::string, GroundTruth> ground_truth_;

  // TF2
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  // Publishers
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr translation_error_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr rotation_error_pub_;
  std::map<std::string, rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr> trans_error_pubs_;
  std::map<std::string, rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr> rot_error_pubs_;

  // Services
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr trigger_eval_srv_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr comparison_srv_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr full_comparison_srv_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr calibrate_client_;
  rclcpp_action::Client<CalibrateAction>::SharedPtr calibrate_action_client_;
  rclcpp::Client<rcl_interfaces::srv::SetParameters>::SharedPtr set_params_client_;

  // Callback group for service client
  rclcpp::CallbackGroup::SharedPtr client_callback_group_;

  // Comparison test settings
  int comparison_trials_{3};
  std::atomic<bool> comparison_running_{false};

  // Timer
  rclcpp::TimerBase::SharedPtr eval_timer_;
};

}  // namespace mrc_test

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<mrc_test::MRCEvaluator>();

  // Use MultiThreadedExecutor to avoid deadlock when calling services
  // from within service callbacks (nested service calls).
  // The onTriggerEvaluation callback calls the calibrate service and waits
  // for the response - with a single-threaded executor this would deadlock.
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();

  rclcpp::shutdown();
  return 0;
}
