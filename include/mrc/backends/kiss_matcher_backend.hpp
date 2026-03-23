#pragma once

#include "mrc/mrc_registration.hpp"

namespace mrc {

class KissMatcherBackend final : public MRCRegistration {
public:
  std::string name() const override { return "kiss_matcher"; }

  RegistrationResult registerSourceToTarget(
      const RegistrationInput& source,
      const RegistrationInput& target) override;

  void configureFromYaml(const std::string& yaml_text) override;

private:
  uint32_t descriptor_dim_{33};
  float noise_bound_{1.0f};
  int num_max_corr_{5000};
  float tuple_scale_{0.95f};
  float min_inlier_ratio_{0.05f};  // Reject if inliers/correspondences < 5%
};

}  // namespace mrc
