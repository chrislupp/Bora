/// Gradient-based airfoil optimization example.
///
/// Starting from a NACA 0012, minimizes drag at a fixed lift target
/// (CL = 0.8, Re = 1e6) by shaping the airfoil and trimming alpha, using
/// Bora's analytic gradients with a simple Adam optimizer.
///
/// Constraints are handled with quadratic penalties:
///   - CL must meet the target,
///   - the analysis confidence must stay high, which keeps the optimizer
///     inside the region where the neural network is trustworthy (and, as a
///     side effect, keeps the shape physically reasonable).

#include <cmath>
#include <cstdio>

#include "bora.h"

namespace {

constexpr double kClTarget = 0.8;
constexpr double kReynolds = 1e6;
constexpr double kConfidenceFloor = 0.95;
constexpr double kClPenaltyWeight = 1.0;
constexpr double kConfidencePenaltyWeight = 0.1;

// Design variables: 16 CST weights, leading edge weight, and alpha (18 total,
// mapped onto the corresponding entries of Bora's input vector).
constexpr size_t kNumDesignVars = 18;

size_t DesignToInputIndex(size_t design_index) {
    return design_index < 17 ? design_index  // CST weights and LE weight
                             : bora::input_index::kAlpha;
}

struct Objective {
    double value = 0.0;
    std::array<double, kNumDesignVars> gradient{};
    bora::AeroOutput aero{};
};

Objective Evaluate(const bora::NeuralFoil& model, const bora::AeroInput& input) {
    const bora::AeroResult result = model.AnalyzeWithGradients(input);
    const bora::AeroOutput& aero = result.output;

    const double cl_error = aero.CL - kClTarget;
    const double confidence_deficit = std::max(0.0, kConfidenceFloor - aero.analysis_confidence);

    Objective objective;
    objective.aero = aero;
    objective.value = aero.CD + kClPenaltyWeight * cl_error * cl_error +
                      kConfidencePenaltyWeight * confidence_deficit * confidence_deficit;

    for (size_t i = 0; i < kNumDesignVars; ++i) {
        const size_t k = DesignToInputIndex(i);
        objective.gradient[i] =
            result.gradient.CD[k] + 2.0 * kClPenaltyWeight * cl_error * result.gradient.CL[k] -
            2.0 * kConfidencePenaltyWeight * confidence_deficit *
                result.gradient.analysis_confidence[k];
    }
    return objective;
}

void ApplyDesign(const std::array<double, kNumDesignVars>& design, bora::AeroInput& input) {
    for (size_t i = 0; i < bora::kNumCstWeightsPerSide; ++i) {
        input.kulfan.upper_weights[i] = design[i];
        input.kulfan.lower_weights[i] = design[8 + i];
    }
    input.kulfan.leading_edge_weight = design[16];
    input.alpha = design[17];
}

}  // namespace

int main() {
    const bora::NeuralFoil model(bora::ModelSize::kXLarge);

    // Starting point: NACA 0012 (fitted Kulfan parameters), alpha = 4 deg.
    bora::AeroInput input;
    input.Re = kReynolds;
    input.kulfan.TE_thickness = 0.002548;
    std::array<double, kNumDesignVars> design = {
        0.172884, 0.151563, 0.173763, 0.127681, 0.164818, 0.126376, 0.145898, 0.1392,      // upper
        -0.172884, -0.151563, -0.173763, -0.127681, -0.164818, -0.126376, -0.145898, -0.1392,  // lower
        0.0,  // leading edge weight
        4.0,  // alpha [deg]
    };

    // Adam optimizer state.
    constexpr double kLearningRate = 5e-3;
    constexpr double kBeta1 = 0.9;
    constexpr double kBeta2 = 0.999;
    constexpr double kEpsilon = 1e-9;
    std::array<double, kNumDesignVars> m{};
    std::array<double, kNumDesignVars> v{};

    std::printf("Minimizing CD at CL = %.2f, Re = %.1e (xlarge model)\n\n", kClTarget, kReynolds);
    std::printf("%6s  %10s  %9s  %9s  %8s  %11s\n", "iter", "objective", "CL", "CD", "L/D",
                "confidence");

    ApplyDesign(design, input);
    Objective objective = Evaluate(model, input);
    for (int iteration = 1; iteration <= 500; ++iteration) {
        for (size_t i = 0; i < kNumDesignVars; ++i) {
            m[i] = kBeta1 * m[i] + (1.0 - kBeta1) * objective.gradient[i];
            v[i] = kBeta2 * v[i] + (1.0 - kBeta2) * objective.gradient[i] * objective.gradient[i];
            const double m_hat = m[i] / (1.0 - std::pow(kBeta1, iteration));
            const double v_hat = v[i] / (1.0 - std::pow(kBeta2, iteration));
            design[i] -= kLearningRate * m_hat / (std::sqrt(v_hat) + kEpsilon);
        }
        ApplyDesign(design, input);
        objective = Evaluate(model, input);

        if (iteration % 50 == 0 || iteration == 1) {
            std::printf("%6d  %10.6f  %9.5f  %9.5f  %8.2f  %11.4f\n", iteration, objective.value,
                        objective.aero.CL, objective.aero.CD,
                        objective.aero.CL / objective.aero.CD,
                        objective.aero.analysis_confidence);
        }
    }

    std::printf("\nOptimized airfoil (Kulfan parameters):\n");
    std::printf("  upper_weights = {");
    for (size_t i = 0; i < 8; ++i) {
        std::printf("%s%.6f", i > 0 ? ", " : "", design[i]);
    }
    std::printf("}\n  lower_weights = {");
    for (size_t i = 0; i < 8; ++i) {
        std::printf("%s%.6f", i > 0 ? ", " : "", design[8 + i]);
    }
    std::printf("}\n  leading_edge_weight = %.6f\n  alpha = %.3f deg\n", design[16], design[17]);

    // Export the optimized shape as Selig-format coordinates.
    const auto coordinates = bora::GenerateKulfanCoordinates(input.kulfan);
    const char* out_path = "optimized_airfoil.dat";
    if (FILE* f = std::fopen(out_path, "w")) {
        std::fprintf(f, "Bora optimized airfoil\n");
        for (const auto& point : coordinates) {
            std::fprintf(f, "%.6f %.6f\n", point[0], point[1]);
        }
        std::fclose(f);
        std::printf("\nWrote %s\n", out_path);
    }
    return 0;
}
