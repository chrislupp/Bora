#include <gtest/gtest.h>

#include <cmath>
#include <map>
#include <string>

#include "bora.h"
#include "golden_data.h"

namespace bora::testing {
namespace {

const NeuralFoil& GetModel(const std::string& size_name) {
    static std::map<std::string, NeuralFoil> models;
    auto it = models.find(size_name);
    if (it == models.end()) {
        for (ModelSize size :
             {ModelSize::kXXSmall, ModelSize::kXSmall, ModelSize::kSmall, ModelSize::kMedium,
              ModelSize::kLarge, ModelSize::kXLarge, ModelSize::kXXLarge, ModelSize::kXXXLarge}) {
            if (ToString(size) == size_name) {
                it = models.emplace(size_name, NeuralFoil(size)).first;
                break;
            }
        }
    }
    EXPECT_NE(it, models.end()) << "Unknown model size: " << size_name;
    return it->second;
}

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

/// Flattens an AeroOutput into the golden data ordering.
std::array<double, 198> Flatten(const AeroOutput& output) {
    std::array<double, 198> flat{};
    flat[0] = output.analysis_confidence;
    flat[1] = output.CL;
    flat[2] = output.CD;
    flat[3] = output.CM;
    flat[4] = output.Top_Xtr;
    flat[5] = output.Bot_Xtr;
    size_t pos = 6;
    for (const BoundaryLayerData* surface : {&output.upper, &output.lower}) {
        for (const auto* quantity : {&surface->theta, &surface->H, &surface->ue_over_vinf}) {
            for (double v : *quantity) {
                flat[pos++] = v;
            }
        }
    }
    return flat;
}

TEST(GoldenValues, MatchesPythonNeuralFoil) {
    for (const GoldenCase& golden : kGoldenCases) {
        SCOPED_TRACE(golden.name);
        const NeuralFoil& model = GetModel(golden.model_size);
        const AeroOutput output = model.Analyze(InputFromArray(golden.inputs));
        const auto actual = Flatten(output);

        for (size_t i = 0; i < actual.size(); ++i) {
            const double expected = golden.expected[i];
            const double tolerance = 1e-6 * std::max(std::abs(expected), 1e-8);
            EXPECT_NEAR(actual[i], expected, tolerance) << "output index " << i;
        }
    }
}

TEST(Analyze, GradientVariantReturnsSameValues) {
    const NeuralFoil& model = GetModel("medium");
    const AeroInput input = InputFromArray(kGoldenCases[0].inputs);

    // The two paths may differ in the last bits: the compiler contracts the
    // value and Jacobian loops into different FMA sequences.
    const auto plain = Flatten(model.Analyze(input));
    const auto with_grad = Flatten(model.AnalyzeWithGradients(input).output);
    for (size_t i = 0; i < plain.size(); ++i) {
        EXPECT_NEAR(plain[i], with_grad[i], 1e-12 * std::max(std::abs(plain[i]), 1e-3))
            << "output index " << i;
    }
}

TEST(Analyze, AlphaSymmetryInvariant) {
    // Mirroring the airfoil and negating alpha must exactly negate CL and CM
    // and swap the surfaces, by construction of the fused evaluation.
    const NeuralFoil& model = GetModel("medium");
    AeroInput input = InputFromArray(kGoldenCases[0].inputs);

    AeroInput mirrored = input;
    for (size_t i = 0; i < kNumCstWeightsPerSide; ++i) {
        mirrored.kulfan.upper_weights[i] = -input.kulfan.lower_weights[i];
        mirrored.kulfan.lower_weights[i] = -input.kulfan.upper_weights[i];
    }
    mirrored.kulfan.leading_edge_weight = -input.kulfan.leading_edge_weight;
    mirrored.alpha = -input.alpha;
    std::swap(mirrored.xtr_upper, mirrored.xtr_lower);

    const AeroOutput base = model.Analyze(input);
    const AeroOutput flipped = model.Analyze(mirrored);

    EXPECT_NEAR(flipped.CL, -base.CL, 1e-12);
    EXPECT_NEAR(flipped.CM, -base.CM, 1e-12);
    EXPECT_NEAR(flipped.CD, base.CD, 1e-14);
    EXPECT_NEAR(flipped.analysis_confidence, base.analysis_confidence, 1e-12);
    EXPECT_NEAR(flipped.Top_Xtr, base.Bot_Xtr, 1e-12);
    EXPECT_NEAR(flipped.Bot_Xtr, base.Top_Xtr, 1e-12);
}

TEST(Model, ThrowsOnMissingData) {
    EXPECT_THROW(NeuralFoil(ModelSize::kMedium, "/nonexistent/path"), std::runtime_error);
}

}  // namespace
}  // namespace bora::testing
