#pragma once

#include <memory>
#include <string>
#include <vector>

#include "mrc/mrc_registration.hpp"

namespace mrc {

// Returns nullptr if algorithm is unknown (caller should check and handle gracefully)
MRCRegistrationPtr createBackend(const std::string& algorithm);

// Returns list of available algorithm names
std::vector<std::string> availableAlgorithms();

}  // namespace mrc
