/// Basic usage example: analyze an airfoil and print coefficients and
/// selected gradients.

#include <cstdio>

#include "bora.h"

int main() {
    // A NACA-4412-like airfoil in Kulfan (CST) parameterization.
    bora::AeroInput input;
    input.kulfan.upper_weights = {0.24, 0.31, 0.25, 0.35, 0.24, 0.35, 0.28, 0.32};
    input.kulfan.lower_weights = {-0.12, -0.05, -0.13, 0.02, -0.08, 0.06, 0.02, 0.06};
    input.kulfan.leading_edge_weight = 0.02;
    input.kulfan.TE_thickness = 0.0025;
    input.alpha = 3.0;  // deg
    input.Re = 1e6;

    const bora::NeuralFoil model(bora::ModelSize::kXLarge);
    const bora::AeroResult result = model.AnalyzeWithGradients(input);
    const bora::AeroOutput& aero = result.output;

    std::printf("Bora %s (NeuralFoil port), model size '%s'\n\n", bora::GetVersion().c_str(),
                bora::ToString(model.GetModelSize()).c_str());
    std::printf("alpha = %.2f deg, Re = %.2e\n", input.alpha, input.Re);
    std::printf("  CL  = %9.5f\n", aero.CL);
    std::printf("  CD  = %9.5f\n", aero.CD);
    std::printf("  CM  = %9.5f\n", aero.CM);
    std::printf("  L/D = %9.3f\n", aero.CL / aero.CD);
    std::printf("  transition: upper x/c = %.3f, lower x/c = %.3f\n", aero.Top_Xtr, aero.Bot_Xtr);
    std::printf("  analysis confidence = %.4f\n\n", aero.analysis_confidence);

    namespace idx = bora::input_index;
    std::printf("Gradients (analytic, for optimization):\n");
    std::printf("  dCL/dalpha = %9.5f per deg\n", result.gradient.CL[idx::kAlpha]);
    std::printf("  dCM/dalpha = %9.5f per deg\n", result.gradient.CM[idx::kAlpha]);
    std::printf("  dCD/dRe    = %9.3e\n", result.gradient.CD[idx::kRe]);
    std::printf("  dCL/d(upper CST weight 0..7):\n   ");
    for (size_t i = 0; i < bora::kNumCstWeightsPerSide; ++i) {
        std::printf(" %8.4f", result.gradient.CL[idx::kUpperWeights + i]);
    }
    std::printf("\n");
    return 0;
}
