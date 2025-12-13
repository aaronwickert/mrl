#include <../include/mrl/mrl.h>



MRL::MRL(const KISSMatcherConfig& config) : config_(config) {
  config_ = config;
  reset();
}

std::vector<Eigen::Vector3f> MRL::convert_cloud_to_vec(const pcl::PointCloud<pcl::PointXYZ>& cloud) {
  std::vector<Eigen::Vector3f> vec;
  vec.reserve(cloud.size());
  for (const auto& pt : cloud.points) {
    if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) continue;
    vec.emplace_back(pt.x, pt.y, pt.z);
  }
  return vec;
}

std::pair<std::vector<Eigen::Vector3f>& , std::vector<Eigen::VectorXf>&> MRL::get_fpfh(const pcl::PointCloud<pcl::PointXYZ>::Ptr pcl) {
  clear();
  auto cloud_vec = convert_cloud_to_vec(*pcl);
  auto processInput = [&](const std::vector<Eigen::Vector3f> &input_cloud) {
  if (config_.use_voxel_sampling_) {
    return kiss_matcher::VoxelgridSampling(input_cloud, config_.voxel_size_);
  }
    return input_cloud;
  };

  auto t_init = std::chrono::high_resolution_clock::now();

  preprocessed_cloud_ = processInput(cloud_vec);

  auto t_process = std::chrono::high_resolution_clock::now();

  faster_pfh_->setInputCloud(preprocessed_cloud_);
  faster_pfh_->ComputeFeature(key_points_, descriptors_);

  auto t_extract = std::chrono::high_resolution_clock::now();

  processing_time_ =
      std::chrono::duration_cast<std::chrono::duration<double>>(t_process - t_init).count();
  extraction_time_ =
      std::chrono::duration_cast<std::chrono::duration<double>>(t_extract - t_process).count();

  return {key_points_, descriptors_};
}

kiss_matcher::RegistrationSolution MRL::estimate_transformation() {
  // Implementation of transformation estimation using robin_matching_ and solver_
  // This is a placeholder for the actual implementation.
  auto t_start = std::chrono::high_resolution_clock::now();

  const auto &corr = robin_matching_->establishCorrespondences(
      src_keypoints_, src_descriptors_, tgt_keypoints_, &tgt_descriptors_, config_.robin_mode_, config_.tuple_scale_, config_.use_ratio_test_);
  
  
  src_matched_.resize(corr.size());
  tgt_matched_.resize(corr.size());

  for (size_t i = 0; i < corr.size(); ++i) {
    auto src_idx    = std::get<0>(corr[i]);
    auto dst_idx    = std::get<1>(corr[i]);
    src_matched_[i] = src_keypoints_[src_idx];
    tgt_matched_[i] = tgt_keypoints_[dst_idx];
  }

  auto t_match =  std::chrono::high_resolution_clock::now();

  matching_time_ =
      std::chrono::duration_cast<std::chrono::duration<double>>(t_match - t_start).count();

  return prune_and_solve(src_matched_, tgt_matched_);
}

kiss_matcher::RegistrationSolution MRL::solve(const Eigen::Matrix<double, 3, Eigen::Dynamic> &src_matched,
    const Eigen::Matrix<double, 3, Eigen::Dynamic> &tgt_matched) {
  // In case of too-few matching pairs,
  // Just return invalid solution with the identity matrix
  if (src_matched.cols() < 2) {
    return solver_->getSolution();
  }

  reset_solver();
  std::chrono::steady_clock::time_point t_start = std::chrono::steady_clock::now();
  solver_->solve(src_matched, tgt_matched);
  std::chrono::steady_clock::time_point t_end = std::chrono::steady_clock::now();
  solving_time_ = std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start).count();

  return solver_->getSolution();
}

kiss_matcher::RegistrationSolution MRL::prune_and_solve(const std::vector<Eigen::Vector3f> &src_matched,
                                                const std::vector<Eigen::Vector3f> &tgt_matched) {
  std::vector<std::pair<int, int>> corres, corres_out;
  for (size_t i = 0; i < src_matched.size(); ++i) {
    corres.emplace_back(i, i);
  }
  const auto &pruned_indices =
      robin_matching_->applyOutlierPruning(src_matched, tgt_matched, "max_core");
  size_t num_pruned_corr = pruned_indices.size();

  Eigen::Matrix<double, 3, Eigen::Dynamic> src_eigen(3, num_pruned_corr);
  Eigen::Matrix<double, 3, Eigen::Dynamic> tgt_eigen(3, num_pruned_corr);

  for (size_t i = 0; i < num_pruned_corr; ++i) {
    src_eigen.col(i) = src_matched[pruned_indices[i]].cast<double>();
    tgt_eigen.col(i) = tgt_matched[pruned_indices[i]].cast<double>();
  }
  return solve(src_eigen, tgt_eigen);
}

void MRL::clear() {
  key_points_.clear();
  descriptors_.clear();
  preprocessed_cloud_.clear();
  corr_.clear();
  tgt_descriptors_.clear();
  src_descriptors_.clear();
  tgt_matched_.clear();
  src_matched_.clear(); 
  processing_time_ = -1.0;
  extraction_time_ = -1.0;
  rejection_time_ = -1.0;
  matching_time_ = -1.0;
  solving_time_ = -1.0;
}

void MRL::reset() {
  faster_pfh_ = std::make_unique<kiss_matcher::FasterPFH>(
      config_.normal_radius_, config_.fpfh_radius_, config_.thr_linearity_);
  robin_matching_ = std::make_unique<kiss_matcher::ROBINMatching>(
      config_.robin_noise_bound_, config_.num_max_corr_, config_.tuple_scale_);
}

void MRL::reset_solver() {
  // NOTE(hlim) Please turn on `use_quatro_`
  // when the pitch and roll angles are not dominant in the rotation
  kiss_matcher::RobustRegistrationSolver::Params params;
  params.noise_bound = config_.solver_noise_bound_;

  if (config_.use_quatro_) {
    params.rotation_estimation_algorithm =
        kiss_matcher::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::QUATRO;
  } else {
    params.rotation_estimation_algorithm =
        kiss_matcher::RobustRegistrationSolver::ROTATION_ESTIMATION_ALGORITHM::GNC_TLS;
  }

  solver_ = std::make_unique<kiss_matcher::RobustRegistrationSolver>(params);
}



