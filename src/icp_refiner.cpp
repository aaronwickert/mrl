#include "mrc/icp_refiner.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <Eigen/Geometry>

#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/gicp.h>
#include <pcl_conversions/pcl_conversions.h>

namespace mrc {

namespace {

/// Compute nearest-neighbor RMSE with 3x-median outlier rejection.
/// This mirrors the metric used by FRICP and KISS-Matcher backends so
/// that the before/after comparison in refine() is apples-to-apples.
static double computeNNRmse(const pcl::PointCloud<pcl::PointXYZ>& source,
                            const pcl::PointCloud<pcl::PointXYZ>& target,
                            const Eigen::Matrix4f& T) {
  const Eigen::Matrix3f R = T.block<3, 3>(0, 0);
  const Eigen::Vector3f t = T.block<3, 1>(0, 3);

  const int n_src = static_cast<int>(source.size());
  const int n_tgt = static_cast<int>(target.size());
  if (n_src == 0 || n_tgt == 0) return std::numeric_limits<double>::max();

  // Compute nearest-neighbor distance for each source point
  std::vector<double> dists;
  dists.reserve(static_cast<size_t>(n_src));
  for (int i = 0; i < n_src; ++i) {
    Eigen::Vector3f p = R * source[i].getVector3fMap() + t;
    float min_sq = std::numeric_limits<float>::max();
    for (int j = 0; j < n_tgt; ++j) {
      float sq = (p - target[j].getVector3fMap()).squaredNorm();
      if (sq < min_sq) min_sq = sq;
    }
    dists.push_back(static_cast<double>(std::sqrt(min_sq)));
  }

  // 3x-median outlier rejection
  std::vector<double> sorted_dists = dists;
  std::sort(sorted_dists.begin(), sorted_dists.end());
  const double median = sorted_dists[sorted_dists.size() / 2];
  const double thresh = 3.0 * median;

  double sum_sq = 0.0;
  int inlier_count = 0;
  for (int i = 0; i < n_src; ++i) {
    if (dists[static_cast<size_t>(i)] <= thresh) {
      sum_sq += dists[static_cast<size_t>(i)] * dists[static_cast<size_t>(i)];
      ++inlier_count;
    }
  }

  if (inlier_count < 3) return std::numeric_limits<double>::max();
  return std::sqrt(sum_sq / static_cast<double>(inlier_count));
}

}  // namespace

ICPRefiner::ICPRefiner(const ICPRefinementParams& params) : params_(params) {}

RegistrationResult ICPRefiner::refine(const sensor_msgs::msg::PointCloud2& source_cloud,
                                      const sensor_msgs::msg::PointCloud2& target_cloud,
                                      const RegistrationResult& coarse_result) {
  // Convert PointCloud2 to PCL (full-resolution, kept for RMSE evaluation)
  pcl::PointCloud<pcl::PointXYZ>::Ptr src_full(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr tgt_full(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromROSMsg(source_cloud, *src_full);
  pcl::fromROSMsg(target_cloud, *tgt_full);

  // Too few points to run ICP
  if (src_full->size() < 10 || tgt_full->size() < 10) {
    return coarse_result;
  }

  // Downsample for ICP alignment (not for evaluation)
  pcl::PointCloud<pcl::PointXYZ>::Ptr src = src_full;
  pcl::PointCloud<pcl::PointXYZ>::Ptr tgt = tgt_full;
  if (params_.voxel_leaf_size > 0.0) {
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    const float leaf = static_cast<float>(params_.voxel_leaf_size);

    pcl::PointCloud<pcl::PointXYZ>::Ptr src_ds(new pcl::PointCloud<pcl::PointXYZ>);
    vg.setInputCloud(src_full);
    vg.setLeafSize(leaf, leaf, leaf);
    vg.filter(*src_ds);
    src = src_ds;

    pcl::PointCloud<pcl::PointXYZ>::Ptr tgt_ds(new pcl::PointCloud<pcl::PointXYZ>);
    vg.setInputCloud(tgt_full);
    vg.filter(*tgt_ds);
    tgt = tgt_ds;

    if (src->size() < 10 || tgt->size() < 10) {
      return coarse_result;
    }
  }

  // Build initial guess from coarse transform
  const auto& tr = coarse_result.T_target_from_source.transform.translation;
  const auto& qr = coarse_result.T_target_from_source.transform.rotation;
  Eigen::Quaternionf q(static_cast<float>(qr.w), static_cast<float>(qr.x),
                       static_cast<float>(qr.y), static_cast<float>(qr.z));
  Eigen::Matrix4f initial_guess = Eigen::Matrix4f::Identity();
  initial_guess.block<3, 3>(0, 0) = q.toRotationMatrix();
  initial_guess(0, 3) = static_cast<float>(tr.x);
  initial_guess(1, 3) = static_cast<float>(tr.y);
  initial_guess(2, 3) = static_cast<float>(tr.z);

  // Run Generalized ICP on downsampled clouds
  pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
  icp.setMaximumIterations(params_.max_iterations);
  icp.setMaxCorrespondenceDistance(params_.max_correspondence_distance);
  icp.setTransformationEpsilon(params_.transformation_epsilon);
  icp.setEuclideanFitnessEpsilon(params_.euclidean_fitness_epsilon);
  icp.setInputSource(src);
  icp.setInputTarget(tgt);

  pcl::PointCloud<pcl::PointXYZ> aligned;
  icp.align(aligned, initial_guess);

  if (!icp.hasConverged()) {
    return coarse_result;
  }

  const Eigen::Matrix4f T = icp.getFinalTransformation();

  // Evaluate refined transform using the same NN RMSE metric as backends,
  // on the full-resolution clouds, so comparison is apples-to-apples.
  const double refined_rmse = computeNNRmse(*src_full, *tgt_full, T);

  // Reject if ICP made it worse — keep coarse result
  if (coarse_result.rmse_t_m.has_value() && refined_rmse >= coarse_result.rmse_t_m.value()) {
    RegistrationResult rejected = coarse_result;
    rejected.score = refined_rmse;
    rejected.message = coarse_result.message + " + GICP rejected";
    return rejected;
  }

  // Build refined result
  Eigen::Matrix3f R = T.block<3, 3>(0, 0);
  Eigen::Quaternionf q_out(R);
  q_out.normalize();

  RegistrationResult refined = coarse_result;
  refined.T_target_from_source.transform.translation.x = T(0, 3);
  refined.T_target_from_source.transform.translation.y = T(1, 3);
  refined.T_target_from_source.transform.translation.z = T(2, 3);
  refined.T_target_from_source.transform.rotation.w = q_out.w();
  refined.T_target_from_source.transform.rotation.x = q_out.x();
  refined.T_target_from_source.transform.rotation.y = q_out.y();
  refined.T_target_from_source.transform.rotation.z = q_out.z();

  refined.rmse_t_m = refined_rmse;
  refined.score = refined_rmse;
  refined.message = coarse_result.message + " + GICP refined";

  return refined;
}

}  // namespace mrc
