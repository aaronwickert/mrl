#pragma once

#include <memory>
#include <string>

#include "mrc/registration_types.hpp"

namespace mrc {

class MRCRegistration {
public:
  virtual ~MRCRegistration() = default;

  virtual std::string name() const = 0;

  // source -> target
  virtual RegistrationResult registerSourceToTarget(
      const RegistrationInput& source,
      const RegistrationInput& target) = 0;

  virtual void configureFromYaml(const std::string& yaml_text) = 0;
};

using MRCRegistrationPtr = std::shared_ptr<MRCRegistration>;

}  // namespace mrc
