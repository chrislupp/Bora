#pragma once

#include "bora/types.h"

#include <array>
#include <cstddef>
#include <span>
#include <vector>

namespace bora {

/// A single (x, y) airfoil coordinate point.
using Point2 = std::array<double, 2>;

/// @brief Generates airfoil coordinates from Kulfan (CST) parameters.
///
/// Coordinates are returned in Selig format: from the upper surface trailing
/// edge, around the leading edge, to the lower surface trailing edge, using
/// cosine spacing in x/c. The parameterization matches AeroSandbox's
/// get_kulfan_coordinates() with N1=0.5, N2=1.0 and the leading-edge
/// modification term.
///
/// @param parameters Kulfan shape parameters.
/// @param n_points_per_side Number of points per surface.
/// @return Coordinates, (2 * n_points_per_side - 1) points.
std::vector<Point2> GenerateKulfanCoordinates(
    const KulfanParameters& parameters, size_t n_points_per_side = 100);

/// @brief Fits Kulfan (CST) parameters to a set of airfoil coordinates.
///
/// Solves the linear least-squares problem of AeroSandbox's
/// get_kulfan_parameters(method="least_squares"). If the fitted trailing edge
/// thickness is negative, the problem is re-solved with TE thickness fixed to
/// zero. Coordinates must be in Selig format and should be normalized (leading
/// edge at the origin, trailing edge at x=1); no normalization is performed.
///
/// @param coordinates Airfoil coordinates in Selig format.
/// @return Fitted Kulfan parameters (8 weights per side).
/// @throws std::runtime_error if the fit problem is singular.
KulfanParameters FitKulfanParameters(std::span<const Point2> coordinates);

}  // namespace bora
