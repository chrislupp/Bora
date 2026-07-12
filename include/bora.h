#pragma once

#include "bora/kulfan.h"
#include "bora/network.h"
#include "bora/neuralfoil.h"
#include "bora/types.h"

#include <string>

namespace bora {

/// @brief Returns the project version string.
/// @return Version string in "major.minor.patch" format.
std::string GetVersion();

}  // namespace bora
