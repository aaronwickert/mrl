#pragma once

#include <optional>
#include <string>
#include <unordered_map>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

#include "mrc/registration_types.hpp"

namespace mrc {

    struct PoseGraphParams {
        // Default noise if backend doesn't provide RMSE.
        double default_sigma_t_m{0.3};     // [m]
        double default_sigma_r_rad{0.2};   // [rad] (~11.5 deg)

        // Clamp noise to avoid overconfidence / underconfidence.
        double min_sigma_t_m{0.10};        // [m]
        double min_sigma_r_rad{0.10};      // [rad] (~5.7 deg)
        double max_sigma_t_m{5.0};         // [m]
        double max_sigma_r_rad{3.14159};   // [rad]

        // If inliers are provided, shrink noise by 1/sqrt(inliers).
        // Disabled by default: point-level inliers are not independent pose
        // measurements, and the RMSE from backends already captures quality.
        bool scale_by_inliers{false};

        // Robustify edges (recommended for real registrations).
        bool use_robust{true};
        double huber_k{1.345};             // standard Huber threshold in whitened units
    };

    class PoseGraphOptimizer {
    public:
        explicit PoseGraphOptimizer(PoseGraphParams params);

        void reset();

        void addAnchor(gtsam::Key key,
                       const gtsam::Pose3& prior,
                       double sigma_t_m,
                       double sigma_r_rad);

        void addRegistrationEdge(gtsam::Key key_a,
                                 gtsam::Key key_b,
                                 const gtsam::Pose3& z_b_from_a,
                                 const RegistrationResult& reg);

        // Provide an initial guess for a node (optional but helps convergence).
        void setInitial(gtsam::Key key, const gtsam::Pose3& x0);

        gtsam::Values optimize();

        // Compute derived sigma values for a registration using the current params.
        static void computeSigmas(const PoseGraphParams& params,
                                  const RegistrationResult& reg,
                                  double& sigma_t_m,
                                  double& sigma_r_rad);

    private:
        PoseGraphParams params_;
        gtsam::NonlinearFactorGraph graph_;
        gtsam::Values initial_;
    };

}  // namespace mrc