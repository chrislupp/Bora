/// Verifies Bora's analytic gradients against jacobians of the Python
/// NeuralFoil computed with CasADi. The reference data lives in
/// test/data/gradient_ref_data.h5 (committed to the repository) and is
/// regenerated with scripts/generate_gradient_ref_data.py only when the
/// reference data needs updating.

#include "bora.h"
#include "ref_data.h"

#include <cmath>
#include <cstddef>
#include <vector>

#include <gtest/gtest.h>

namespace bora::testing {
namespace {

/// Collects pointers to Bora's gradients in the reference jacobian row order:
/// the six scalars, then per surface and quantity the recorded stations.
std::vector<const Gradient*> JacobianRows(const AeroGradients& gradient,
                                          const std::vector<size_t>& stations) {
    std::vector<const Gradient*> rows = {
        &gradient.analysis_confidence,
        &gradient.CL,
        &gradient.CD,
        &gradient.CM,
        &gradient.Top_Xtr,
        &gradient.Bot_Xtr,
    };
    for (const BoundaryLayerGradients* surface :
         {&gradient.upper, &gradient.lower}) {
        for (const auto* quantity :
             {&surface->theta, &surface->H, &surface->ue_over_vinf}) {
            for (size_t station : stations) {
                rows.push_back(&(*quantity)[station]);
            }
        }
    }
    return rows;
}

TEST(ReferenceGradients, MatchesPythonNeuralFoil) {
    const GradientRefData data = LoadGradientRefData();
    ASSERT_GT(data.n_cases, 0U);

    for (size_t c = 0; c < data.n_cases; ++c) {
        SCOPED_TRACE(data.names[c]);
        const NeuralFoil& model = GetModel(data.model_sizes[c]);
        const AeroResult result = model.AnalyzeWithGradients(
            InputFromArray(&data.inputs[c * kNumInputs]));
        const std::vector<const Gradient*> rows =
            JacobianRows(result.gradient, data.bl_stations);

        ASSERT_EQ(rows.size(), data.n_rows);
        const double* expected_case =
            &data.jacobians[c * data.n_rows * kNumInputs];
        for (size_t row = 0; row < rows.size(); ++row) {
            for (size_t k = 0; k < kNumInputs; ++k) {
                const double expected = expected_case[row * kNumInputs + k];
                const double actual = (*rows[row])[k];
                const double tolerance =
                    1e-6 * std::max(std::abs(expected), 1e-8);
                EXPECT_NEAR(actual, expected, tolerance)
                    << "jacobian row " << row << ", input " << k;
            }
        }
    }
}

}  // namespace
}  // namespace bora::testing
