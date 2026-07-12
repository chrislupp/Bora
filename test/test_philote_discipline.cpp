/// Tests for the Philote discipline wrapper, exercised directly (no gRPC).

#include "bora/philote_discipline.h"

#include <array>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

namespace {

struct InputSpec {
    const char* name;
    size_t size;
    size_t offset;  ///< Offset into the bora::Gradient vector.
};

constexpr std::array<InputSpec, 9> kInputSpecs{{
    {"upper_weights", bora::kNumCstWeightsPerSide,
     bora::input_index::kUpperWeights},
    {"lower_weights", bora::kNumCstWeightsPerSide,
     bora::input_index::kLowerWeights},
    {"leading_edge_weight", 1, bora::input_index::kLeadingEdgeWeight},
    {"TE_thickness", 1, bora::input_index::kTeThickness},
    {"alpha", 1, bora::input_index::kAlpha},
    {"Re", 1, bora::input_index::kRe},
    {"n_crit", 1, bora::input_index::kNCrit},
    {"xtr_upper", 1, bora::input_index::kXtrUpper},
    {"xtr_lower", 1, bora::input_index::kXtrLower},
}};

/// A NACA-4412-like test case matching examples/example.cpp.
bora::AeroInput TestCase() {
    bora::AeroInput input;
    input.kulfan.upper_weights = {0.24, 0.31, 0.25, 0.35,
                                  0.24, 0.35, 0.28, 0.32};
    input.kulfan.lower_weights = {-0.12, -0.05, -0.13, 0.02,
                                  -0.08, 0.06,  0.02,  0.06};
    input.kulfan.leading_edge_weight = 0.02;
    input.kulfan.TE_thickness = 0.0025;
    input.alpha = 3.0;
    input.Re = 1e6;
    return input;
}

philote::Variables MakeInputs(const bora::AeroInput& input) {
    philote::Variables vars;
    for (const InputSpec& spec : kInputSpecs) {
        vars[spec.name] = philote::Variable(philote::kInput, {spec.size});
    }
    for (size_t i = 0; i < bora::kNumCstWeightsPerSide; ++i) {
        vars["upper_weights"](i) = input.kulfan.upper_weights[i];
        vars["lower_weights"](i) = input.kulfan.lower_weights[i];
    }
    vars["leading_edge_weight"](0) = input.kulfan.leading_edge_weight;
    vars["TE_thickness"](0) = input.kulfan.TE_thickness;
    vars["alpha"](0) = input.alpha;
    vars["Re"](0) = input.Re;
    vars["n_crit"](0) = input.n_crit;
    vars["xtr_upper"](0) = input.xtr_upper;
    vars["xtr_lower"](0) = input.xtr_lower;
    return vars;
}

TEST(PhiloteDiscipline, ComputeMatchesDirectAnalysis) {
    const bora::AeroInput input = TestCase();

    philote::Variables outputs;
    const std::vector<std::string> scalar_outputs = {
        "CL", "CD", "CM", "Top_Xtr", "Bot_Xtr", "analysis_confidence"};
    for (const std::string& name : scalar_outputs) {
        outputs[name] = philote::Variable(philote::kOutput, {1});
    }

    bora::BoraDiscipline discipline;
    discipline.Compute(MakeInputs(input), outputs);

    const bora::NeuralFoil model(bora::ModelSize::kXLarge);
    const bora::AeroOutput expected = model.Analyze(input);

    EXPECT_DOUBLE_EQ(outputs.at("CL")(0), expected.CL);
    EXPECT_DOUBLE_EQ(outputs.at("CD")(0), expected.CD);
    EXPECT_DOUBLE_EQ(outputs.at("CM")(0), expected.CM);
    EXPECT_DOUBLE_EQ(outputs.at("Top_Xtr")(0), expected.Top_Xtr);
    EXPECT_DOUBLE_EQ(outputs.at("Bot_Xtr")(0), expected.Bot_Xtr);
    EXPECT_DOUBLE_EQ(outputs.at("analysis_confidence")(0),
                     expected.analysis_confidence);
}

TEST(PhiloteDiscipline, ComputePartialsMatchesDirectGradients) {
    const bora::AeroInput input = TestCase();

    philote::Partials partials;
    const std::vector<std::string> scalar_outputs = {
        "CL", "CD", "CM", "Top_Xtr", "Bot_Xtr", "analysis_confidence"};
    for (const std::string& output : scalar_outputs) {
        for (const InputSpec& spec : kInputSpecs) {
            partials[{output, spec.name}] =
                philote::Variable(philote::kPartial, {1, spec.size});
        }
    }

    bora::BoraDiscipline discipline;
    discipline.ComputePartials(MakeInputs(input), partials);

    const bora::NeuralFoil model(bora::ModelSize::kXLarge);
    const bora::AeroGradients expected =
        model.AnalyzeWithGradients(input).gradient;

    const std::array<const bora::Gradient*, 6> gradients = {
        &expected.CL,      &expected.CD,      &expected.CM,
        &expected.Top_Xtr, &expected.Bot_Xtr, &expected.analysis_confidence};

    for (size_t k = 0; k < scalar_outputs.size(); ++k) {
        for (const InputSpec& spec : kInputSpecs) {
            const philote::Variable& jac =
                partials.at({scalar_outputs[k], spec.name});
            for (size_t i = 0; i < spec.size; ++i) {
                EXPECT_DOUBLE_EQ(jac(i), (*gradients[k])[spec.offset + i])
                    << scalar_outputs[k] << " wrt " << spec.name << "[" << i
                    << "]";
            }
        }
    }
}

}  // namespace
