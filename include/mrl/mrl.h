/**
 * @file mrl.h
 * @brief Core MRL (Multi-Robot Localization) class for point cloud registration
 * @author MRL Team
 *
 * This file contains the MRL class which provides functionality for extracting
 * FPFH (Fast Point Feature Histogram) features from point clouds and estimating
 * transformations between point cloud pairs using the KISS-Matcher library.
 */

#pragma once

#include <Eigen/Core>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/filter.h>
#include <kiss_matcher/KISSMatcher.hpp>
#include <kiss_matcher/FasterPFH.hpp>
#include <kiss_matcher/points/downsampling.hpp>

/**
 * @struct FeatureDensityResult
 * @brief Result structure for the feature-rich environment metric
 *
 * Contains the components of the feature density metric used to quantify
 * the suitability of feature-based registration approaches across different
 * environments.
 *
 * The feature density φ characterizes the geometric informativeness of a scene
 * and is defined as:
 *
 * @f[
 *     \phi = \frac{N_f}{N_p}
 * @f]
 *
 * where:
 * - @f$ N_f @f$ = number of extracted geometric features (keypoints)
 * - @f$ N_p @f$ = total number of points in the input point cloud
 *
 * @par Reference Values
 * As a baseline reference, moderately feature-rich indoor environments
 * (e.g., laboratory settings with walls, furniture, and equipment) typically
 * yield feature densities in a characteristic range that can be used for
 * environment classification.
 */
struct FeatureDensityResult {
    size_t N_f;     ///< Number of extracted geometric features (keypoints)
    size_t N_p;     ///< Total number of points in the input point cloud
    double phi;     ///< Feature density ratio: φ = N_f / N_p

    /**
     * @brief Checks if the result is valid
     * @return true if N_p > 0 (valid computation), false otherwise
     */
    bool valid() const { return N_p > 0; }
};

/**
 * @class MRL
 * @brief Multi-Robot Localization class for point cloud feature extraction and registration
 *
 * The MRL class wraps the KISS-Matcher library to provide a simplified interface
 * for multi-robot localization tasks. It handles:
 * - FPFH feature extraction from point clouds
 * - Point cloud voxel downsampling
 * - Transformation estimation between point cloud pairs
 * - Feature density metric computation for environment characterization
 *
 * @note This class is designed to work with both PCL point clouds and raw Eigen vectors.
 *
 * Example usage:
 * @code
 * kiss_matcher::KISSMatcherConfig config;
 * MRL mrl(config);
 *
 * // Extract features from a point cloud
 * auto [keypoints, descriptors] = mrl.get_fpfh(cloud);
 *
 * // Compute feature density metric
 * auto density = mrl.compute_feature_density(cloud);
 * std::cout << "Feature density φ = " << density.phi << std::endl;
 *
 * // Estimate transformation between two point clouds
 * auto solution = mrl.estimate_transformation(src_points, tgt_points);
 * @endcode
 */
class MRL {
public:
    /**
     * @brief Constructs an MRL instance with the specified configuration
     *
     * Initializes the KISS-Matcher and FasterPFH instances with parameters
     * from the provided configuration.
     *
     * @param config Configuration parameters for KISS-Matcher including:
     *               - voxel_size: Size of voxels for downsampling
     *               - normal_radius: Radius for normal estimation
     *               - fpfh_radius: Radius for FPFH computation
     *               - thr_linearity: Threshold for linearity filtering
     */
    explicit MRL(const kiss_matcher::KISSMatcherConfig& config);

    /**
     * @brief Converts a PCL point cloud to a vector of Eigen 3D points
     *
     * Filters out NaN points during conversion to ensure clean data
     * for downstream processing.
     *
     * @param cloud Input PCL point cloud
     * @return std::vector<Eigen::Vector3f> Vector of valid 3D points
     */
    std::vector<Eigen::Vector3f> convert_cloud_to_vec(const pcl::PointCloud<pcl::PointXYZ>& cloud);

    /**
     * @brief Extracts FPFH features from a PCL point cloud
     *
     * Convenience overload that accepts a PCL point cloud pointer.
     * Internally converts to Eigen vectors before processing.
     *
     * @param cloud Shared pointer to input PCL point cloud
     * @return std::pair containing:
     *         - first: Vector of 3D keypoint positions
     *         - second: Vector of FPFH descriptors (33-dimensional)
     *
     * @see get_fpfh(const std::vector<Eigen::Vector3f>&)
     */
    std::pair<std::vector<Eigen::Vector3f>, std::vector<Eigen::VectorXf>> get_fpfh(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);

    /**
     * @brief Extracts FPFH features from a vector of 3D points
     *
     * Performs the following steps:
     * 1. Optional voxel grid downsampling (if enabled in config)
     * 2. Normal estimation for each point
     * 3. FPFH descriptor computation
     *
     * @param points Input vector of 3D points
     * @return std::pair containing:
     *         - first: Vector of 3D keypoint positions (subset of input after filtering)
     *         - second: Vector of FPFH descriptors (33-dimensional histograms)
     *
     * @note The number of returned keypoints may be less than input points
     *       due to linearity filtering and invalid normal rejection.
     */
    std::pair<std::vector<Eigen::Vector3f>, std::vector<Eigen::VectorXf>> get_fpfh(
        const std::vector<Eigen::Vector3f>& points);

    /**
     * @brief Computes the feature density metric for a PCL point cloud
     *
     * Calculates the feature-rich environment metric φ (phi) which quantifies
     * the suitability of feature-based registration approaches for a given scene.
     *
     * The feature density is defined as:
     * @f[
     *     \phi = \frac{N_f}{N_p}
     * @f]
     *
     * where @f$ N_f @f$ is the number of detected keypoint features and
     * @f$ N_p @f$ is the total number of points in the input cloud.
     *
     * @param cloud Shared pointer to input PCL point cloud
     * @return FeatureDensityResult containing N_f, N_p, and φ
     *
     * @par Interpretation
     * - Higher φ values indicate more geometrically informative scenes
     * - Lower φ values suggest feature-sparse environments where
     *   feature-based registration may be less reliable
     *
     * @see compute_feature_density(const std::vector<Eigen::Vector3f>&)
     */
    FeatureDensityResult compute_feature_density(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);

    /**
     * @brief Computes the feature density metric for a vector of 3D points
     *
     * Calculates the feature-rich environment metric φ (phi) which quantifies
     * the suitability of feature-based registration approaches for a given scene.
     *
     * The feature density is defined as:
     * @f[
     *     \phi = \frac{N_f}{N_p}
     * @f]
     *
     * where @f$ N_f @f$ is the number of detected keypoint features identified
     * by the FPFH feature detection algorithm, and @f$ N_p @f$ is the total
     * number of points in the input point cloud.
     *
     * @param points Input vector of 3D points
     * @return FeatureDensityResult containing:
     *         - N_f: Number of extracted geometric features
     *         - N_p: Total number of input points
     *         - phi: Feature density ratio (N_f / N_p)
     *
     * @par Reference Baseline
     * As a reference baseline, the Team Hector laboratory environment yielded
     * a characteristic feature density, representing a moderately feature-rich
     * indoor setting with structured geometric elements such as walls, furniture,
     * and equipment.
     *
     * @note Uses get_fpfh() internally for feature extraction
     */
    FeatureDensityResult compute_feature_density(const std::vector<Eigen::Vector3f>& points);

    /**
     * @brief Estimates the rigid transformation between two point clouds
     *
     * Uses KISS-Matcher's robust registration pipeline:
     * 1. FPFH feature extraction for both clouds
     * 2. Feature matching with ROBIN correspondence filtering
     * 3. Robust pose estimation using GNC-TLS solver
     *
     * @param src Source point cloud (will be transformed to align with target)
     * @param tgt Target point cloud (reference frame)
     * @return kiss_matcher::RegistrationSolution containing:
     *         - valid: Whether registration succeeded
     *         - rotation: 3x3 rotation matrix
     *         - translation: 3D translation vector
     *
     * @note Automatically resets the solver before estimation.
     */
    kiss_matcher::RegistrationSolution estimate_transformation(
        const std::vector<Eigen::Vector3f>& src,
        const std::vector<Eigen::Vector3f>& tgt);

    /**
     * @brief Resets the internal KISS-Matcher state
     *
     * Clears all cached data including processed clouds, keypoints,
     * and correspondence information.
     */
    void reset();

    /**
     * @brief Resets only the solver state
     *
     * Should be called before each new registration to ensure
     * clean solver state. Called automatically by estimate_transformation().
     */
    void reset_solver();

private:
    /** @brief Configuration parameters for KISS-Matcher */
    kiss_matcher::KISSMatcherConfig config_;

    /** @brief KISS-Matcher instance for full registration pipeline */
    std::unique_ptr<kiss_matcher::KISSMatcher> matcher_;

    /** @brief FasterPFH instance for standalone feature extraction */
    std::unique_ptr<kiss_matcher::FasterPFH> faster_pfh_;
};
