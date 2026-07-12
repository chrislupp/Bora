#pragma once

#include <string>

namespace projectname {

/// @brief Returns the project version string.
/// @return Version string in "major.minor.patch" format.
std::string GetVersion();

/// @brief Adds two integers.
/// @param a First operand.
/// @param b Second operand.
/// @return Sum of a and b.
int Add(int a, int b);

}  // namespace projectname
