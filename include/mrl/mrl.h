#include <Eigen/Core>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/filter.h>  
#include <kiss_matcher/FasterPFH.hpp>
#include <kiss_matcher/GncSolver.hpp>
#include <kiss_matcher/KISSMatcher.hpp>

struct RegistrationSolution {
  bool valid                  = false;
  Eigen::Vector3d translation = Eigen::Vector3d::Zero();
  Eigen::Matrix3d rotation    = Eigen::Matrix3d::Identity();

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct KISSMatcherConfig {
  bool use_voxel_sampling_ = true;

  // FPFH descriptor params
  float voxel_size_    = 0.3;
  float normal_radius_ = 0.9;  // 2.5 * `voxel_size` - 3.0 * `voxel_size`
  float fpfh_radius_   = 1.5;  // 5.0 * `voxel_size`

  // Graph-theoretic outlier rejection parms
  float thr_linearity_ = 1.0;  // 1.0 means that we won't use linearity-based filtering
  // NOTE(hlim): The final `robin_noise_bound` becomes `voxel_size_` * `robin_noise_bound_gain_`
  float robin_noise_bound_gain_ = 1.0;
  float robin_noise_bound_      = voxel_size_ * robin_noise_bound_gain_;

  // matching params
  // NOTE(hlim): For better usability for map-level registration, I set `true` as a default
  // Enabling `use_ratio_test_` may cause a slight slowdown,
  // and its impact is insignificant at the scan level.
  bool use_ratio_test_    = true;
  std::string robin_mode_ = "max_core";
  float tuple_scale_      = 0.95;
  int num_max_corr_       = 5000;

  // Solver params
  // NOTE(hlim): The final `solver_noise_bound` becomes `voxel_size_` * `solver_noise_bound_gain_`
  float solver_noise_bound_gain_ = 1.0;
  float solver_noise_bound_      = voxel_size_ * solver_noise_bound_gain_;
  bool use_quatro_               = false;

  KISSMatcherConfig(const float voxel_size         = 0.3,
                    const float use_voxel_sampling = true,
                    const float use_quatro         = false,
                    const float thr_linearity      = 1.0,
                    const int num_max_corr         = 5000,
                    // Below params just works in general cases
                    const float normal_r_gain = 3.0,
                    const float fpfh_r_gain   = 5.0,
                    // The smaller, more conservative
                    const float robin_noise_bound_gain     = 1.0,
                    const float solver_noise_bound_gain    = 0.75,
                    const bool enable_noise_bound_clamping = true) {
    if (voxel_size < 5e-3) {
      throw std::runtime_error(
          "Too small voxel size has been given. Please check your voxel size.");
    }

    if (robin_noise_bound_gain < solver_noise_bound_gain) {
      throw std::runtime_error("`solver_noise_bound_gain` (" +
                               std::to_string(solver_noise_bound_gain) +
                               ") should be smaller than or equal to `robin_noise_bound_gain` (" +
                               std::to_string(robin_noise_bound_gain) + ").");
    }

    voxel_size_         = voxel_size;
    use_voxel_sampling_ = use_voxel_sampling;
    use_quatro_         = use_quatro;
    thr_linearity_      = thr_linearity;

    normal_radius_ = normal_r_gain * voxel_size;
    fpfh_radius_   = fpfh_r_gain * voxel_size;

    num_max_corr_            = num_max_corr;
    robin_noise_bound_gain_  = robin_noise_bound_gain;
    solver_noise_bound_gain_ = solver_noise_bound_gain;

    robin_noise_bound_  = voxel_size_ * robin_noise_bound_gain_;
    solver_noise_bound_ = voxel_size_ * solver_noise_bound_gain_;

    if ((robin_noise_bound_ > 1.0) && enable_noise_bound_clamping) {
      std::cout
          << "\033[1;33m[Warning] Too large `robin_noise_bound_` has been set.\n"
          << "Empirically, 1.0 tends to work better for large-scale maps.\n"
          << "If you do not want to clamp these values, disable `enable_noise_clamping`.\n\033[0m";
      robin_noise_bound_ = 1.0;
    }

    if ((solver_noise_bound_ > 1.0) && enable_noise_bound_clamping) {
      std::cout
          << "\033[1;33m[Warning] Too large `solver_noise bound_` has been set.\n"
          << "Empirically, 1.0 tends to work better for large-scale maps.\n"
          << "If you do not want to clamp these values, disable `enable_noise_clamping`.\n\033[0m";
      solver_noise_bound_ = 1.0;
    }
  }
};

class MRL {
    public:
        std::vector<Eigen::Vector3f> convert_cloud_to_vec(const pcl::PointCloud<pcl::PointXYZ>& cloud);
        std::pair<std::vector<Eigen::Vector3f>& , std::vector<Eigen::VectorXf>&> get_fpfh(const pcl::PointCloud<pcl::PointXYZ>::Ptr pcl);
        void reset();
        void clear();
        void reset_solver();
        kiss_matcher::RegistrationSolution estimate_transformation();
        kiss_matcher::RegistrationSolution prune_and_solve(const std::vector<Eigen::Vector3f> &src_matched,
                                                const std::vector<Eigen::Vector3f> &tgt_matched);
        kiss_matcher::RegistrationSolution solve(const Eigen::Matrix<double, 3, Eigen::Dynamic> &src,
                                           const Eigen::Matrix<double, 3, Eigen::Dynamic> &tgt);
                                           
        std::unique_ptr<kiss_matcher::FasterPFH> faster_pfh_;
        std::unique_ptr<kiss_matcher::ROBINMatching> robin_matching_;
        std::unique_ptr<kiss_matcher::RobustRegistrationSolver> solver_;
        std::vector<Eigen::Vector3f> key_points_;
        std::vector<Eigen::VectorXf> descriptors_;
        std::vector<Eigen::Vector3f> src_keypoints_;
        std::vector<Eigen::Vector3f> tgt_keypoints_;
        std::vector<Eigen::Vector3f> preprocessed_cloud_;
        std::vector<Eigen::Vector3f> src_matched_;
        std::vector<Eigen::Vector3f> tgt_matched_;
        KISSMatcherConfig config_;
        double processing_time_;
        double extraction_time_;
        double rejection_time_;
        double matching_time_;
        double solving_time_;
        MRL(const KISSMatcherConfig& config);
        std::vector<std::pair<int, int>> corr_;
        std::vector<Eigen::VectorXf> __attribute__((aligned(EIGEN_MAX_ALIGN_BYTES))) src_descriptors_;
        std::vector<Eigen::VectorXf> __attribute__((aligned(EIGEN_MAX_ALIGN_BYTES))) tgt_descriptors_;
};  