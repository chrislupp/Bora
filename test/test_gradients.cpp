#include <gtest/gtest.h>

#include <cmath>

#include "bora.h"
#include "golden_data.h"

namespace bora::testing {
namespace {

AeroInput InputFromArray(const std::array<double, kNumInputs>& values) {
    AeroInput input;
    for (size_t i = 0; i < kNumCstWeightsPerSide; ++i) {
        input.kulfan.upper_weights[i] = values[input_index::kUpperWeights + i];
        input.kulfan.lower_weights[i] = values[input_index::kLowerWeights + i];
    }
    input.kulfan.leading_edge_weight = values[input_index::kLeadingEdgeWeight];
    input.kulfan.TE_thickness = values[input_index::kTeThickness];
    input.alpha = values[input_index::kAlpha];
    input.Re = values[input_index::kRe];
    input.n_crit = values[input_index::kNCrit];
    input.xtr_upper = values[input_index::kXtrUpper];
    input.xtr_lower = values[input_index::kXtrLower];
    return input;
}

std::array<double, kNumInputs> ArrayFromInput(const AeroInput& input) {
    std::array<double, kNumInputs> values{};
    for (size_t i = 0; i < kNumCstWeightsPerSide; ++i) {
        values[input_index::kUpperWeights + i] = input.kulfan.upper_weights[i];
        values[input_index::kLowerWeights + i] = input.kulfan.lower_weights[i];
    }
    values[input_index::kLeadingEdgeWeight] = input.kulfan.leading_edge_weight;
    values[input_index::kTeThickness] = input.kulfan.TE_thickness;
    values[input_index::kAlpha] = input.alpha;
    values[input_index::kRe] = input.Re;
    values[input_index::kNCrit] = input.n_crit;
    values[input_index::kXtrUpper] = input.xtr_upper;
    values[input_index::kXtrLower] = input.xtr_lower;
    return values;
}

/// Checks one analytic gradient against a central finite difference.
void ExpectGradientNear(double analytic, double numeric, double fd_scale,
                        const std::string& what) {
    // The FD truncation/roundoff error floor scales with the output magnitude.
    const double tolerance = 5e-5 * std::max({std::abs(analytic), std::abs(numeric), fd_scale});
    EXPECT_NEAR(analytic, numeric, tolerance) << what;
}

class GradientTest : public ::testing::TestWithParam<size_t> {};

TEST_P(GradientTest, MatchesFiniteDifferences) {
    const GoldenCase& golden = kGoldenCases[GetParam()];
    SCOPED_TRACE(golden.name);
    const NeuralFoil model(ModelSize::kMedium);

    const AeroInput input = InputFromArray(golden.inputs);
    const AeroResult result = model.AnalyzeWithGradients(input);
    const auto base_values = ArrayFromInput(input);

    for (size_t k = 0; k < kNumInputs; ++k) {
        // Relative step for the large-magnitude Re input; a smaller step for
        // TE thickness, which is scaled by 50 in the latent space and feeds
        // steep exponential decodings; absolute step otherwise.
        double h = 1e-6 * std::max(1.0, std::abs(base_values[k]));
        if (k == input_index::kRe) {
            h = 1e-4 * base_values[k];
        } else if (k == input_index::kTeThickness) {
            h = 1e-7;
        }

        auto perturbed = base_values;
        perturbed[k] = base_values[k] + h;
        const AeroOutput plus = model.Analyze(InputFromArray(perturbed));
        perturbed[k] = base_values[k] - h;
        const AeroOutput minus = model.Analyze(InputFromArray(perturbed));
        const double inv_2h = 1.0 / (2.0 * h);

        const std::string suffix = " d/d(input " + std::to_string(k) + ")";
        ExpectGradientNear(result.gradient.CL[k], (plus.CL - minus.CL) * inv_2h,
                           std::abs(plus.CL) * 1e-7 / h, "CL" + suffix);
        ExpectGradientNear(result.gradient.CD[k], (plus.CD - minus.CD) * inv_2h,
                           std::abs(plus.CD) * 1e-7 / h, "CD" + suffix);
        ExpectGradientNear(result.gradient.CM[k], (plus.CM - minus.CM) * inv_2h,
                           std::abs(plus.CM) * 1e-7 / h, "CM" + suffix);
        ExpectGradientNear(result.gradient.analysis_confidence[k],
                           (plus.analysis_confidence - minus.analysis_confidence) * inv_2h,
                           1e-7 / h, "analysis_confidence" + suffix);
        ExpectGradientNear(result.gradient.Top_Xtr[k], (plus.Top_Xtr - minus.Top_Xtr) * inv_2h,
                           1e-7 / h, "Top_Xtr" + suffix);
        ExpectGradientNear(result.gradient.Bot_Xtr[k], (plus.Bot_Xtr - minus.Bot_Xtr) * inv_2h,
                           1e-7 / h, "Bot_Xtr" + suffix);

        // Spot-check the boundary layer gradients at a few stations.
        for (size_t i : {size_t{0}, size_t{15}, size_t{31}}) {
            ExpectGradientNear(result.gradient.upper.theta[i][k],
                               (plus.upper.theta[i] - minus.upper.theta[i]) * inv_2h,
                               std::abs(plus.upper.theta[i]) * 1e-7 / h,
                               "upper theta " + std::to_string(i) + suffix);
            ExpectGradientNear(result.gradient.lower.H[i][k],
                               (plus.lower.H[i] - minus.lower.H[i]) * inv_2h,
                               std::abs(plus.lower.H[i]) * 1e-7 / h,
                               "lower H " + std::to_string(i) + suffix);
            ExpectGradientNear(result.gradient.upper.ue_over_vinf[i][k],
                               (plus.upper.ue_over_vinf[i] - minus.upper.ue_over_vinf[i]) * inv_2h,
                               1e-7 / h, "upper ue " + std::to_string(i) + suffix);
        }
    }
}

// Golden cases 1, 21, 41, 61 span different airfoils and conditions (model
// size in the golden case is ignored here; gradients use the medium model).
INSTANTIATE_TEST_SUITE_P(SampledCases, GradientTest,
                         ::testing::Values(size_t{1}, size_t{21}, size_t{41}, size_t{61}));

}  // namespace
}  // namespace bora::testing
