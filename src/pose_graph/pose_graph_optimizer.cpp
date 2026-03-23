#include "mrc/pose_graph/pose_graph_optimizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

namespace mrc {
namespace {

static std::string keyStr(gtsam::Key key) {
  gtsam::Symbol sym(key);
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%c%lu", sym.chr(), sym.index());
  return buf;
}

static double clampd(double v, double lo, double hi) {
  return std::max(lo, std::min(v, hi));
}

static double inlierScale(const PoseGraphParams& p, const RegistrationResult& reg) {
  if (!p.scale_by_inliers) return 1.0;
  if (!reg.num_inliers.has_value()) return 1.0;
  const int n = *reg.num_inliers;
  if (n <= 1) return 1.0;
  return 1.0 / std::sqrt(static_cast<double>(n));
}

static gtsam::SharedNoiseModel makeNoiseModel(const PoseGraphParams& p,
                                              double sigma_t_m,
                                              double sigma_r_rad) {
  sigma_t_m = clampd(sigma_t_m, p.min_sigma_t_m, p.max_sigma_t_m);
  sigma_r_rad = clampd(sigma_r_rad, p.min_sigma_r_rad, p.max_sigma_r_rad);

  // GTSAM Pose3 convention: [rot_x rot_y rot_z trans_x trans_y trans_z]
  auto base = gtsam::noiseModel::Diagonal::Sigmas(
      (gtsam::Vector(6) << sigma_r_rad, sigma_r_rad, sigma_r_rad,
                           sigma_t_m,   sigma_t_m,   sigma_t_m).finished());

  if (!p.use_robust) return base;

  auto huber = gtsam::noiseModel::mEstimator::Huber::Create(p.huber_k);
  return gtsam::noiseModel::Robust::Create(huber, base);
}

static void deriveSigmas(const PoseGraphParams& p,
                         const RegistrationResult& reg,
                         double& out_sigma_t_m,
                         double& out_sigma_r_rad) {
  // Start from defaults.
  double sigma_t = p.default_sigma_t_m;
  double sigma_r = p.default_sigma_r_rad;

  // If backend provides RMSEs, use them.
  if (reg.rmse_t_m.has_value()) sigma_t = *reg.rmse_t_m;
  if (reg.rmse_r_rad.has_value()) sigma_r = *reg.rmse_r_rad;

  // Scale by inliers if available.
  const double s = inlierScale(p, reg);
  sigma_t *= s;
  sigma_r *= s;

  // Clamp.
  out_sigma_t_m = clampd(sigma_t, p.min_sigma_t_m, p.max_sigma_t_m);
  out_sigma_r_rad = clampd(sigma_r, p.min_sigma_r_rad, p.max_sigma_r_rad);
}

}  // namespace

PoseGraphOptimizer::PoseGraphOptimizer(PoseGraphParams params)
    : params_(std::move(params)) {}

void PoseGraphOptimizer::reset() {
  graph_.resize(0);
  initial_.clear();
}

void PoseGraphOptimizer::addAnchor(gtsam::Key key,
                                   const gtsam::Pose3& prior,
                                   double sigma_t_m,
                                   double sigma_r_rad) {
  const auto t = prior.translation();
  const auto rpy = prior.rotation().rpy();
  std::fprintf(stderr,
      "[PG] addAnchor key=%s pose=[%.4f %.4f %.4f] rpy=[%.2f %.2f %.2f]deg "
      "sigma_t=%.4fm sigma_r=%.4frad\n",
      keyStr(key).c_str(), t.x(), t.y(), t.z(),
      rpy[0] * 180.0 / M_PI, rpy[1] * 180.0 / M_PI, rpy[2] * 180.0 / M_PI,
      sigma_t_m, sigma_r_rad);
  auto nm = makeNoiseModel(params_, sigma_t_m, sigma_r_rad);
  graph_.add(gtsam::PriorFactor<gtsam::Pose3>(key, prior, nm));
  if (!initial_.exists(key)) initial_.insert(key, prior);
}

void PoseGraphOptimizer::addRegistrationEdge(gtsam::Key key_a,
                                             gtsam::Key key_b,
                                             const gtsam::Pose3& z_b_from_a,
                                             const RegistrationResult& reg) {
  double sigma_t = 0.0;
  double sigma_r = 0.0;
  deriveSigmas(params_, reg, sigma_t, sigma_r);

  const auto t = z_b_from_a.translation();
  const auto rpy = z_b_from_a.rotation().rpy();
  std::fprintf(stderr,
      "[PG] addEdge %s->%s z=[%.4f %.4f %.4f] rpy=[%.2f %.2f %.2f]deg "
      "rmse_t=%s rmse_r=%s inliers=%s corr=%s "
      "sigma_t=%.4fm sigma_r=%.4frad robust=%s\n",
      keyStr(key_a).c_str(), keyStr(key_b).c_str(),
      t.x(), t.y(), t.z(),
      rpy[0] * 180.0 / M_PI, rpy[1] * 180.0 / M_PI, rpy[2] * 180.0 / M_PI,
      reg.rmse_t_m.has_value()
          ? std::to_string(*reg.rmse_t_m).c_str() : "N/A",
      reg.rmse_r_rad.has_value()
          ? std::to_string(*reg.rmse_r_rad).c_str() : "N/A",
      reg.num_inliers.has_value()
          ? std::to_string(*reg.num_inliers).c_str() : "N/A",
      reg.num_correspondences.has_value()
          ? std::to_string(*reg.num_correspondences).c_str() : "N/A",
      sigma_t, sigma_r,
      params_.use_robust ? "huber" : "off");

  auto nm = makeNoiseModel(params_, sigma_t, sigma_r);
  graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(key_a, key_b, z_b_from_a, nm));
}

void PoseGraphOptimizer::computeSigmas(const PoseGraphParams& params,
                                       const RegistrationResult& reg,
                                       double& sigma_t_m,
                                       double& sigma_r_rad) {
  deriveSigmas(params, reg, sigma_t_m, sigma_r_rad);
}

void PoseGraphOptimizer::setInitial(gtsam::Key key, const gtsam::Pose3& x0) {
  if (!initial_.exists(key)) initial_.insert(key, x0);
}

gtsam::Values PoseGraphOptimizer::optimize() {
  if (graph_.size() == 0) {
    throw std::runtime_error("PoseGraphOptimizer: no factors in graph.");
  }

  const double initial_error = graph_.error(initial_);
  std::fprintf(stderr,
      "[PG] optimize: %zu factors, %zu variables, initial_error=%.6f\n",
      graph_.size(), initial_.size(), initial_error);

  gtsam::LevenbergMarquardtParams lm;
  lm.setVerbosityLM("ERROR");
  lm.setMaxIterations(100);

  gtsam::LevenbergMarquardtOptimizer opt(graph_, initial_, lm);
  auto result = opt.optimize();

  const double final_error = graph_.error(result);
  std::fprintf(stderr,
      "[PG] optimize: final_error=%.6f iterations=%d\n",
      final_error, opt.iterations());

  // Log each optimized pose
  for (const auto& kv : result) {
    try {
      const auto pose = result.at<gtsam::Pose3>(kv.key);
      const auto t = pose.translation();
      const auto rpy = pose.rotation().rpy();
      std::fprintf(stderr,
          "[PG] result %s: pos=[%.4f %.4f %.4f] rpy=[%.2f %.2f %.2f]deg\n",
          keyStr(kv.key).c_str(), t.x(), t.y(), t.z(),
          rpy[0] * 180.0 / M_PI, rpy[1] * 180.0 / M_PI,
          rpy[2] * 180.0 / M_PI);
    } catch (...) {
      // Not a Pose3 value, skip
    }
  }

  return result;
}

}  // namespace mrc