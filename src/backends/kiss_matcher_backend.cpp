#include "mrc/backends/kiss_matcher_backend.hpp"

#include <cmath>
#include <cstdio>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>

#include <geometry_msgs/msg/transform_stamped.hpp>

#include <kiss_matcher/GncSolver.hpp>
#include <kiss_matcher/ROBINMatching.hpp>

namespace mrc {

namespace {

static std::vector<Eigen::Vector3f> toEigenVec3fFromFlat(const std::vector<float>& flat) {
  const size_t n = flat.size() / 3;
  std::vector<Eigen::Vector3f> pts;
  pts.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    pts.emplace_back(flat[i * 3], flat[i * 3 + 1], flat[i * 3 + 2]);
  }
  return pts;
}

static std::vector<Eigen::VectorXf> toDescriptorVectors(const std::vector<float>& flat,
                                                        uint32_t dim,
                                                        size_t count) {
  if (dim == 0) throw std::runtime_error("descriptor_dim must be > 0");
  if (flat.size() != count * static_cast<size_t>(dim)) {
    throw std::runtime_error("Descriptor array size mismatch: expected keypoint_count*descriptor_dim elements");
  }

  std::vector<Eigen::VectorXf> out;
  out.reserve(count);

  for (size_t i = 0; i < count; ++i) {
    Eigen::VectorXf d(static_cast<int>(dim));
    const size_t base = i * static_cast<size_t>(dim);
    for (uint32_t k = 0; k < dim; ++k) {
      d(static_cast<int>(k)) = flat[base + static_cast<size_t>(k)];
    }
    out.emplace_back(std::move(d));
  }
  return out;
}

static geometry_msgs::msg::TransformStamped eigenRtToTf(const Eigen::Matrix3d& R,
                                                        const Eigen::Vector3d& t) {
  geometry_msgs::msg::TransformStamped tf;
  tf.transform.translation.x = t.x();
  tf.transform.translation.y = t.y();
  tf.transform.translation.z = t.z();

  Eigen::Quaterniond q(R);
  q.normalize();
  tf.transform.rotation.w = q.w();
  tf.transform.rotation.x = q.x();
  tf.transform.rotation.y = q.y();
  tf.transform.rotation.z = q.z();
  return tf;
}

}  // namespace

void KissMatcherBackend::configureFromYaml(const std::string& yaml_text) {
  // Simple key: value parsing (one per line)
  // Supports: descriptor_dim, noise_bound, num_max_corr, tuple_scale
  std::istringstream stream(yaml_text);
  std::string line;

  while (std::getline(stream, line)) {
    // Skip empty lines and comments
    if (line.empty() || line[0] == '#') continue;

    auto colon_pos = line.find(':');
    if (colon_pos == std::string::npos) continue;

    std::string key = line.substr(0, colon_pos);
    std::string value = line.substr(colon_pos + 1);

    // Trim whitespace
    auto trim = [](std::string& s) {
      s.erase(0, s.find_first_not_of(" \t"));
      s.erase(s.find_last_not_of(" \t") + 1);
    };
    trim(key);
    trim(value);

    try {
      if (key == "descriptor_dim") {
        descriptor_dim_ = static_cast<uint32_t>(std::stoul(value));
      } else if (key == "noise_bound") {
        noise_bound_ = std::stof(value);
      } else if (key == "num_max_corr") {
        num_max_corr_ = std::stoi(value);
      } else if (key == "tuple_scale") {
        tuple_scale_ = std::stof(value);
      } else if (key == "min_inlier_ratio") {
        min_inlier_ratio_ = std::stof(value);
      }
    } catch (const std::exception&) {
      // Ignore parse errors for individual values
    }
  }
}

RegistrationResult KissMatcherBackend::registerSourceToTarget(
    const RegistrationInput& source,
    const RegistrationInput& target) {
  RegistrationResult out;

  if (!source.features || !target.features) {
    out.ok = false;
    out.message = "KissMatcherBackend requires features (source.features and target.features).";
    return out;
  }

  const auto& sf = *source.features;
  const auto& tf = *target.features;

  if (sf.descriptor_dim == 0 || tf.descriptor_dim == 0) {
    out.ok = false;
    out.message = "descriptor_dim must be > 0 on both source and target.";
    return out;
  }
  if (sf.descriptor_dim != tf.descriptor_dim) {
    out.ok = false;
    out.message = "Descriptor dim mismatch between source and target.";
    return out;
  }

  const size_t src_n = sf.keypoint_xyz.size() / 3;
  const size_t tgt_n = tf.keypoint_xyz.size() / 3;

  if (src_n < 3 || tgt_n < 3) {
    out.ok = false;
    out.message = "Too few keypoints for registration (need >= 3).";
    return out;
  }

  const auto src_pts = toEigenVec3fFromFlat(sf.keypoint_xyz);
  const auto tgt_pts = toEigenVec3fFromFlat(tf.keypoint_xyz);

  const auto src_desc = toDescriptorVectors(sf.descriptors, sf.descriptor_dim, src_n);
  const auto tgt_desc = toDescriptorVectors(tf.descriptors, tf.descriptor_dim, tgt_n);

  const float noise_bound = noise_bound_;
  const int num_max_corr = num_max_corr_;
  const float tuple_scale = tuple_scale_;
  const bool use_ratio_test = true;
  const std::string robin_mode = "max_core";

  kiss_matcher::ROBINMatching matcher(noise_bound, num_max_corr, tuple_scale);

  auto src_pts_mut = src_pts;
  auto tgt_pts_mut = tgt_pts;
  auto src_desc_mut = src_desc;
  auto tgt_desc_mut = tgt_desc;

  const auto corres = matcher.establishCorrespondences(
      src_pts_mut, tgt_pts_mut, src_desc_mut, tgt_desc_mut, robin_mode, tuple_scale, use_ratio_test);

  if (corres.size() < 3) {
    out.ok = false;
    out.message = "Too few correspondences after matching.";
    return out;
  }

  const size_t M = corres.size();
  Eigen::Matrix<double, 3, Eigen::Dynamic> src_matched(3, static_cast<int>(M));
  Eigen::Matrix<double, 3, Eigen::Dynamic> tgt_matched(3, static_cast<int>(M));

  for (size_t i = 0; i < M; ++i) {
    const int si = corres[i].first;
    const int ti = corres[i].second;

    if (si < 0 || ti < 0 ||
        static_cast<size_t>(si) >= src_pts.size() ||
        static_cast<size_t>(ti) >= tgt_pts.size()) {
      out.ok = false;
      out.message = "Invalid correspondence indices.";
      return out;
    }

    src_matched.col(static_cast<int>(i)) = src_pts[static_cast<size_t>(si)].cast<double>();
    tgt_matched.col(static_cast<int>(i)) = tgt_pts[static_cast<size_t>(ti)].cast<double>();
  }

  kiss_matcher::RobustRegistrationSolver::Params params;
  params.noise_bound = static_cast<double>(noise_bound);
  params.rotation_estimation_algorithm =
      kiss_matcher::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::GNC_TLS;

  kiss_matcher::RobustRegistrationSolver solver(params);
  const auto sol = solver.solve(src_matched, tgt_matched);

  if (!sol.valid) {
    out.ok = false;
    out.message = "KISS robust solver returned invalid solution.";
    return out;
  }

  out.ok = true;
  out.message = "OK";
  out.T_target_from_source = eigenRtToTf(sol.rotation, sol.translation);

  const auto& inlier_indices = solver.getTranslationInliers();
  const int inliers = static_cast<int>(inlier_indices.size());
  out.score = static_cast<double>(inliers);
  out.num_correspondences = static_cast<int>(corres.size());
  out.num_inliers = inliers;

  // Check inlier ratio — low ratio means the solver is likely fitting noise
  if (out.num_correspondences.has_value() && *out.num_correspondences > 0) {
    const double ratio = static_cast<double>(inliers) /
                         static_cast<double>(*out.num_correspondences);
    if (ratio < static_cast<double>(min_inlier_ratio_)) {
      out.ok = false;
      std::ostringstream oss;
      oss << "Inlier ratio too low: " << inliers << "/" << *out.num_correspondences
          << " = " << std::fixed << std::setprecision(1) << (ratio * 100.0)
          << "% < " << (static_cast<double>(min_inlier_ratio_) * 100.0) << "%";
      out.message = oss.str();
      fprintf(stderr, "[KISS] FAIL | corr=%d inliers=%d ratio=%.1f%% (threshold %.1f%%) | t=(%.3f,%.3f,%.3f)\n",
              *out.num_correspondences, inliers, ratio * 100.0,
              static_cast<double>(min_inlier_ratio_) * 100.0,
              sol.translation.x(), sol.translation.y(), sol.translation.z());
      return out;
    }
  }

  // Compute RMSE on inlier correspondences after applying estimated transform
  if (inliers >= 3) {
    double sum_sq = 0.0;
    double sum_dist_from_centroid = 0.0;

    // Compute centroid of target inlier points (for rotation RMSE approximation)
    Eigen::Vector3d tgt_centroid = Eigen::Vector3d::Zero();
    for (int idx : inlier_indices) {
      tgt_centroid += tgt_matched.col(idx);
    }
    tgt_centroid /= static_cast<double>(inliers);

    for (int idx : inlier_indices) {
      // Apply estimated transform to source point
      Eigen::Vector3d transformed = sol.rotation * src_matched.col(idx) + sol.translation;
      double residual = (transformed - tgt_matched.col(idx)).norm();
      sum_sq += residual * residual;
      sum_dist_from_centroid += (tgt_matched.col(idx) - tgt_centroid).norm();
    }

    double rmse = std::sqrt(sum_sq / static_cast<double>(inliers));
    out.rmse_t_m = rmse;

    // Approximate rotation RMSE: angular error ~ point_rmse / lever_arm
    double mean_lever_arm = sum_dist_from_centroid / static_cast<double>(inliers);
    if (mean_lever_arm > 0.1) {
      out.rmse_r_rad = rmse / mean_lever_arm;
    }
  }

  // Debug print
  const double ratio = (out.num_correspondences.has_value() && *out.num_correspondences > 0)
      ? static_cast<double>(inliers) / static_cast<double>(*out.num_correspondences) * 100.0
      : 0.0;
  fprintf(stderr, "[KISS] %s | corr=%d inliers=%d ratio=%.1f%% | rmse_t=%.4fm rmse_r=%.2fdeg | t=(%.3f,%.3f,%.3f)\n",
          out.ok ? "OK" : "FAIL",
          out.num_correspondences.value_or(0), inliers, ratio,
          out.rmse_t_m.value_or(-1.0),
          out.rmse_r_rad.value_or(-1.0) * 180.0 / M_PI,
          sol.translation.x(), sol.translation.y(), sol.translation.z());

  return out;
}

}  // namespace mrc