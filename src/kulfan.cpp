#include "bora/kulfan.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace bora {

namespace {

constexpr size_t kW = kNumCstWeightsPerSide;

/// Binomial coefficients comb(kW - 1, k) for the Bernstein basis of order 7.
std::array<double, kW> BernsteinCoefficients() {
    std::array<double, kW> coefficients{};
    coefficients[0] = 1.0;
    for (size_t k = 1; k < kW; ++k) {
        coefficients[k] = coefficients[k - 1] * static_cast<double>(kW - k) /
                          static_cast<double>(k);
    }
    return coefficients;
}

/// Bernstein basis functions of order kW - 1 evaluated at x.
std::array<double, kW> BernsteinBasis(double x) {
    const auto coefficients = BernsteinCoefficients();
    std::array<double, kW> basis{};
    for (size_t k = 0; k < kW; ++k) {
        basis[k] = coefficients[k] * std::pow(x, static_cast<double>(k)) *
                   std::pow(1.0 - x, static_cast<double>(kW - 1 - k));
    }
    return basis;
}

/// Class function C(x) = x^0.5 * (1 - x)^1.0 for conventional airfoils.
double ClassFunction(double x) {
    return std::sqrt(x) * (1.0 - x);
}

/// Leading-edge modification mode shape: x * (1 - x)^(kW + 0.5).
double LeadingEdgeMode(double x) {
    return x * std::pow(std::max(1.0 - x, 0.0), static_cast<double>(kW) + 0.5);
}

/// Solves the symmetric linear system a * x = b in place via Gaussian
/// elimination with partial pivoting. a is row-major n x n.
std::vector<double> SolveLinearSystem(std::vector<double> a,
                                      std::vector<double> b, size_t n) {
    for (size_t col = 0; col < n; ++col) {
        size_t pivot = col;
        for (size_t row = col + 1; row < n; ++row) {
            if (std::abs(a[row * n + col]) > std::abs(a[pivot * n + col])) {
                pivot = row;
            }
        }
        if (std::abs(a[pivot * n + col]) < 1e-14) {
            throw std::runtime_error("Kulfan fit problem is singular.");
        }
        if (pivot != col) {
            for (size_t j = 0; j < n; ++j) {
                std::swap(a[col * n + j], a[pivot * n + j]);
            }
            std::swap(b[col], b[pivot]);
        }
        for (size_t row = col + 1; row < n; ++row) {
            const double factor = a[row * n + col] / a[col * n + col];
            for (size_t j = col; j < n; ++j) {
                a[row * n + j] -= factor * a[col * n + j];
            }
            b[row] -= factor * b[col];
        }
    }
    std::vector<double> x(n);
    for (size_t row_plus_one = n; row_plus_one > 0; --row_plus_one) {
        const size_t row = row_plus_one - 1;
        double sum = b[row];
        for (size_t j = row + 1; j < n; ++j) {
            sum -= a[row * n + j] * x[j];
        }
        x[row] = sum / a[row * n + row];
    }
    return x;
}

/// Solves min ||A x - b||_2 via the normal equations. a is row-major m x n.
std::vector<double> SolveLeastSquares(const std::vector<double>& a,
                                      const std::vector<double>& b, size_t m,
                                      size_t n) {
    std::vector<double> ata(n * n, 0.0);
    std::vector<double> atb(n, 0.0);
    for (size_t row = 0; row < m; ++row) {
        const double* a_row = &a[row * n];
        for (size_t i = 0; i < n; ++i) {
            atb[i] += a_row[i] * b[row];
            for (size_t j = i; j < n; ++j) {
                ata[i * n + j] += a_row[i] * a_row[j];
            }
        }
    }
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < i; ++j) {
            ata[i * n + j] = ata[j * n + i];
        }
    }
    return SolveLinearSystem(std::move(ata), std::move(atb), n);
}

}  // namespace

std::vector<Point2> GenerateKulfanCoordinates(
    const KulfanParameters& parameters, size_t n_points_per_side) {
    if (n_points_per_side < 2) {
        throw std::runtime_error("n_points_per_side must be at least 2.");
    }

    // Cosine-spaced x/c stations from 0 to 1.
    std::vector<double> x(n_points_per_side);
    for (size_t i = 0; i < n_points_per_side; ++i) {
        const double t =
            static_cast<double>(i) / static_cast<double>(n_points_per_side - 1);
        x[i] = 0.5 * (1.0 - std::cos(std::numbers::pi * t));
    }

    const auto surface_y = [&](double xi, const std::array<double, kW>& weights,
                               double te_sign) {
        const auto basis = BernsteinBasis(xi);
        double shape = 0.0;
        for (size_t k = 0; k < kW; ++k) {
            shape += weights[k] * basis[k];
        }
        return ClassFunction(xi) * shape +
               te_sign * xi * parameters.TE_thickness / 2.0 +
               parameters.leading_edge_weight * LeadingEdgeMode(xi);
    };

    // Selig format: upper surface TE -> LE, then lower surface LE -> TE.
    std::vector<Point2> coordinates;
    coordinates.reserve(2 * n_points_per_side - 1);
    for (size_t i = n_points_per_side; i > 0; --i) {
        const double xi = x[i - 1];
        coordinates.push_back(
            {xi, surface_y(xi, parameters.upper_weights, +1.0)});
    }
    for (size_t i = 1; i < n_points_per_side; ++i) {
        const double xi = x[i];
        coordinates.push_back(
            {xi, surface_y(xi, parameters.lower_weights, -1.0)});
    }
    return coordinates;
}

KulfanParameters FitKulfanParameters(std::span<const Point2> coordinates) {
    const size_t m = coordinates.size();
    if (m < 2 * kW + 2) {
        throw std::runtime_error("Too few coordinates for a Kulfan fit.");
    }

    // The leading edge splits the coordinate list into upper and lower
    // surfaces (Selig ordering: upper surface comes first).
    size_t le_index = 0;
    for (size_t i = 1; i < m; ++i) {
        if (coordinates[i][0] < coordinates[le_index][0]) {
            le_index = i;
        }
    }

    // Unknown ordering matches AeroSandbox: 8 lower weights, 8 upper weights,
    // leading edge weight, TE thickness.
    const size_t n_unknowns = 2 * kW + 2;
    std::vector<double> a(m * n_unknowns, 0.0);
    std::vector<double> b(m);
    for (size_t row = 0; row < m; ++row) {
        const double xi = coordinates[row][0];
        const bool is_upper = row <= le_index;
        const auto basis = BernsteinBasis(xi);
        const double c = ClassFunction(xi);
        double* a_row = &a[row * n_unknowns];
        for (size_t k = 0; k < kW; ++k) {
            a_row[(is_upper ? kW : 0) + k] = c * basis[k];
        }
        a_row[2 * kW] = LeadingEdgeMode(xi);
        a_row[2 * kW + 1] = is_upper ? xi / 2.0 : -xi / 2.0;
        b[row] = coordinates[row][1];
    }

    auto solution = SolveLeastSquares(a, b, m, n_unknowns);

    // A negative fitted TE thickness is non-physical: refit with it fixed to 0.
    if (solution[2 * kW + 1] < 0.0) {
        std::vector<double> a_reduced(m * (n_unknowns - 1));
        for (size_t row = 0; row < m; ++row) {
            for (size_t j = 0; j < n_unknowns - 1; ++j) {
                a_reduced[row * (n_unknowns - 1) + j] = a[row * n_unknowns + j];
            }
        }
        solution = SolveLeastSquares(a_reduced, b, m, n_unknowns - 1);
        solution.push_back(0.0);
    }

    KulfanParameters parameters;
    for (size_t k = 0; k < kW; ++k) {
        parameters.lower_weights[k] = solution[k];
        parameters.upper_weights[k] = solution[kW + k];
    }
    parameters.leading_edge_weight = solution[2 * kW];
    parameters.TE_thickness = solution[2 * kW + 1];
    return parameters;
}

}  // namespace bora
