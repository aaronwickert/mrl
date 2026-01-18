/**
 * @file mrl.cpp
 * @brief Implementation of the MRL class for point cloud registration
 * @author MRL Team
 *
 * This file implements the MRL class methods for FPFH feature extraction
 * and point cloud registration using the KISS-Matcher library.
 */

#include <../include/mrl/mrl.h>

MRL::MRL(const kiss_matcher::KISSMatcherConfig& config) : config_(config) {
    matcher_ = std::make_unique<kiss_matcher::KISSMatcher>(config_);
    faster_pfh_ = std::make_unique<kiss_matcher::FasterPFH>(
        config_.normal_radius_,
        config_.fpfh_radius_,
        config_.thr_linearity_);
}

std::vector<Eigen::Vector3f> MRL::convert_cloud_to_vec(const pcl::PointCloud<pcl::PointXYZ>& cloud) {
    std::vector<Eigen::Vector3f> result;
    result.reserve(cloud.size());
    for (const auto& point : cloud) {
        if (!std::isnan(point.x) && !std::isnan(point.y) && !std::isnan(point.z)) {
            result.emplace_back(point.x, point.y, point.z);
        }
    }
    return result;
}

std::pair<std::vector<Eigen::Vector3f>, std::vector<Eigen::VectorXf>> MRL::get_fpfh(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
    auto points = convert_cloud_to_vec(*cloud);
    return get_fpfh(points);
}

std::pair<std::vector<Eigen::Vector3f>, std::vector<Eigen::VectorXf>> MRL::get_fpfh(
    const std::vector<Eigen::Vector3f>& points) {

    // Apply voxel downsampling if enabled in configuration
    std::vector<Eigen::Vector3f> processed_cloud;
    if (config_.use_voxel_sampling_) {
        processed_cloud = kiss_matcher::VoxelgridSampling(points, config_.voxel_size_);
    } else {
        processed_cloud = points;
    }

    // Set input cloud and compute FPFH features
    faster_pfh_->setInputCloud(processed_cloud);

    std::vector<Eigen::Vector3f> keypoints;
    std::vector<Eigen::VectorXf> descriptors;
    faster_pfh_->ComputeFeature(keypoints, descriptors);

    return {keypoints, descriptors};
}

FeatureDensityResult MRL::compute_feature_density(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
    auto points = convert_cloud_to_vec(*cloud);
    return compute_feature_density(points);
}

FeatureDensityResult MRL::compute_feature_density(const std::vector<Eigen::Vector3f>& points) {
    FeatureDensityResult result;

    // N_p: Total number of points in the input point cloud
    result.N_p = points.size();

    if (result.N_p == 0) {
        result.N_f = 0;
        result.phi = 0.0;
        return result;
    }

    // Extract features using FPFH
    auto [keypoints, descriptors] = get_fpfh(points);

    // N_f: Number of extracted geometric features (keypoints)
    result.N_f = keypoints.size();

    // φ = N_f / N_p
    result.phi = static_cast<double>(result.N_f) / static_cast<double>(result.N_p);

    return result;
}

kiss_matcher::RegistrationSolution MRL::estimate_transformation(
    const std::vector<Eigen::Vector3f>& src,
    const std::vector<Eigen::Vector3f>& tgt) {
    reset_solver();
    return matcher_->estimate(src, tgt);
}

void MRL::reset() {
    matcher_->reset();
}

void MRL::reset_solver() {
    matcher_->resetSolver();
}
