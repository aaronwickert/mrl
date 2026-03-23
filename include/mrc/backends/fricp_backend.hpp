#pragma once

#include "mrc/mrc_registration.hpp"

namespace mrc {

class FRICPBackend final : public MRCRegistration {
public:
  std::string name() const override { return "fricp"; }

  RegistrationResult registerSourceToTarget(
      const RegistrationInput& source,
      const RegistrationInput& target) override;

  void configureFromYaml(const std::string& yaml_text) override;

private:
  double voxel_leaf_{0.2};
  double max_corr_dist_{1.0};
  int max_iter_{30};
};

}  // namespace mrc
