#pragma once

#include <sensor_msgs/msg/point_cloud2.hpp>

#include "mrc/registration_types.hpp"

namespace mrc {

struct ICPRefinementParams {
  int max_iterations{50};
  double max_correspondence_distance{1.0};
  double transformation_epsilon{1e-8};
  double euclidean_fitness_epsilon{1e-6};
  double voxel_leaf_size{0.1};
};

class ICPRefiner {
public:
  explicit ICPRefiner(const ICPRefinementParams& params);

  /// Run ICP refinement using coarse_result as initial guess.
  /// On convergence: returns updated RegistrationResult with refined transform.
  /// On failure/divergence: returns coarse_result unchanged (safe fallback).
  RegistrationResult refine(const sensor_msgs::msg::PointCloud2& source_cloud,
                            const sensor_msgs::msg::PointCloud2& target_cloud,
                            const RegistrationResult& coarse_result);

private:
  ICPRefinementParams params_;
};

}  // namespace mrc
