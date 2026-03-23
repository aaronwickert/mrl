#include "mrc/backends/fricp_backend.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <ICP.h>
#include <FRICP.h>

namespace mrc {

namespace {

static Eigen::Matrix<double, 3, Eigen::Dynamic> toEigenMat3Xd(const sensor_msgs::msg::PointCloud2& msg) {
  pcl::PointCloud<pcl::PointXYZ> pcl_cloud;
  pcl::fromROSMsg(msg, pcl_cloud);

  std::vector<Eigen::Vector3d> pts;
  pts.reserve(pcl_cloud.size());
  for (const auto& p : pcl_cloud.points) {
    if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) continue;
    pts.emplace_back(static_cast<double>(p.x), static_cast<double>(p.y), static_cast<double>(p.z));
  }

  Eigen::Matrix<double, 3, Eigen::Dynamic> out(3, static_cast<int>(pts.size()));
  for (size_t i = 0; i < pts.size(); ++i) {
    out.col(static_cast<int>(i)) = pts[i];
  }
  return out;
}

static geometry_msgs::msg::TransformStamped mat4ToTf(const Eigen::Matrix4d& T) {
  geometry_msgs::msg::TransformStamped tf;
  tf.transform.translation.x = T(0, 3);
  tf.transform.translation.y = T(1, 3);
  tf.transform.translation.z = T(2, 3);

  Eigen::Matrix3d R = T.block<3, 3>(0, 0);
  Eigen::Quaterniond q(R);
  q.normalize();
  tf.transform.rotation.w = q.w();
  tf.transform.rotation.x = q.x();
  tf.transform.rotation.y = q.y();
  tf.transform.rotation.z = q.z();
  return tf;
}

static Eigen::Vector3d meanOfCols(const Eigen::Matrix<double, 3, Eigen::Dynamic>& X) {
  if (X.cols() == 0) return Eigen::Vector3d::Zero();
  return X.rowwise().mean();
}

}  // namespace

void FRICPBackend::configureFromYaml(const std::string& yaml_text) {
  // Simple key: value parsing (one per line)
  // Supports: voxel_leaf, max_corr_dist, max_iter
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
      if (key == "voxel_leaf") {
        voxel_leaf_ = std::stod(value);
      } else if (key == "max_corr_dist") {
        max_corr_dist_ = std::stod(value);
      } else if (key == "max_iter") {
        max_iter_ = std::stoi(value);
      }
    } catch (const std::exception&) {
      // Ignore parse errors for individual values
    }
  }
}

RegistrationResult FRICPBackend::registerSourceToTarget(
    const RegistrationInput& source,
    const RegistrationInput& target) {
  RegistrationResult out;

  if (!source.cloud || !target.cloud) {
    out.ok = false;
    out.message = "FRICPBackend requires point clouds (source.cloud and target.cloud).";
    return out;
  }

  Eigen::Matrix<double, 3, Eigen::Dynamic> X = toEigenMat3Xd(*source.cloud); // source
  Eigen::Matrix<double, 3, Eigen::Dynamic> Y = toEigenMat3Xd(*target.cloud); // target

  if (X.cols() < 10 || Y.cols() < 10) {
    out.ok = false;
    out.message = "Too few points for FRICP (need >= 10).";
    return out;
  }

  // FRICP code expects demeaned clouds + means passed in.
  Eigen::Vector3d source_mean = meanOfCols(X);
  Eigen::Vector3d target_mean = meanOfCols(Y);

  X.colwise() -= source_mean;
  Y.colwise() -= target_mean;

  ICP::Parameters pars;

  // Use robust ICP variant (RICP) style: welsch + AA is typical in their examples.
  pars.f = ICP::WELSCH;
  pars.use_AA = true;

  // Reasonable defaults; tune later and/or map to your backend members.
  pars.max_icp = max_iter_;
  pars.stop = 1e-6;
  pars.print_energy = false;
  pars.print_output = false;
  pars.use_init = false;

  FRICP<3> fricp;
  fricp.point_to_point(X, Y, source_mean, target_mean, pars);

  // pars.res_trans is the resulting 4x4
  if (pars.res_trans.rows() != 4 || pars.res_trans.cols() != 4) {
    out.ok = false;
    out.message = "FRICP returned invalid transform shape.";
    return out;
  }

  Eigen::Matrix4d T = pars.res_trans;
  out.ok = true;
  out.message = "OK";
  out.T_target_from_source = mat4ToTf(T);
  out.score = 0.0;

  // Compute RMSE: transform source points, find nearest target point, measure residuals.
  // X and Y were demeaned in-place; restore originals for residual computation.
  Eigen::Matrix3d R = T.block<3, 3>(0, 0);
  Eigen::Vector3d t = T.block<3, 1>(0, 3);

  X.colwise() += source_mean;  // restore original source
  Y.colwise() += target_mean;  // restore original target

  const int n_src = static_cast<int>(X.cols());
  const int n_tgt = static_cast<int>(Y.cols());

  // For each transformed source point, find nearest target point distance
  std::vector<double> dists;
  dists.reserve(static_cast<size_t>(n_src));
  for (int i = 0; i < n_src; ++i) {
    Eigen::Vector3d p = R * X.col(i) + t;
    double min_d = std::numeric_limits<double>::max();
    for (int j = 0; j < n_tgt; ++j) {
      double d = (p - Y.col(j)).squaredNorm();
      if (d < min_d) min_d = d;
    }
    dists.push_back(std::sqrt(min_d));
  }

  // Reject outliers > 3x median distance
  std::vector<double> sorted_dists = dists;
  std::sort(sorted_dists.begin(), sorted_dists.end());
  double median_dist = sorted_dists[sorted_dists.size() / 2];
  double outlier_thresh = 3.0 * median_dist;

  double sum_sq = 0.0;
  double sum_lever = 0.0;
  int inlier_count = 0;
  Eigen::Vector3d tgt_centroid = meanOfCols(Y);
  for (int i = 0; i < n_src; ++i) {
    if (dists[static_cast<size_t>(i)] <= outlier_thresh) {
      sum_sq += dists[static_cast<size_t>(i)] * dists[static_cast<size_t>(i)];
      Eigen::Vector3d p = R * X.col(i) + t;
      sum_lever += (p - tgt_centroid).norm();
      ++inlier_count;
    }
  }

  if (inlier_count >= 3) {
    double rmse = std::sqrt(sum_sq / static_cast<double>(inlier_count));
    out.rmse_t_m = rmse;
    out.num_inliers = inlier_count;

    double mean_lever = sum_lever / static_cast<double>(inlier_count);
    if (mean_lever > 0.1) {
      out.rmse_r_rad = rmse / mean_lever;
    }
  }

  return out;
}

}  // namespace mrc