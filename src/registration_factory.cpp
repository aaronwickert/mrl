#include "mrc/registration_factory.hpp"

#include "mrc/backends/fricp_backend.hpp"
#include "mrc/backends/kiss_matcher_backend.hpp"

namespace mrc {

MRCRegistrationPtr createBackend(const std::string& algorithm) {
  if (algorithm == "fricp") {
    return std::make_shared<FRICPBackend>();
  }
  if (algorithm == "kiss_matcher") {
    return std::make_shared<KissMatcherBackend>();
  }
  // Return nullptr for unknown algorithms - caller should check and handle
  return nullptr;
}

std::vector<std::string> availableAlgorithms() {
  return {"fricp", "kiss_matcher"};
}

}  // namespace mrc
