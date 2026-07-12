#include <gtest/gtest.h>

#include <cmath>

#include "bora.h"

namespace bora::testing {
namespace {

KulfanParameters MakeTestParameters() {
    KulfanParameters parameters;
    parameters.upper_weights = {0.18, 0.22, 0.20, 0.19, 0.21, 0.18, 0.17, 0.19};
    parameters.lower_weights = {-0.12, -0.10, -0.08, -0.05, -0.02, 0.00, 0.02, 0.03};
    parameters.leading_edge_weight = 0.05;
    parameters.TE_thickness = 0.002;
    return parameters;
}

TEST(Kulfan, CoordinatesHaveExpectedStructure) {
    const auto coordinates = GenerateKulfanCoordinates(MakeTestParameters(), 100);
    ASSERT_EQ(coordinates.size(), 199u);

    // Selig format: starts and ends at the trailing edge, leading edge in the
    // middle, and the trailing edge gap matches TE_thickness.
    EXPECT_DOUBLE_EQ(coordinates.front()[0], 1.0);
    EXPECT_DOUBLE_EQ(coordinates.back()[0], 1.0);
    EXPECT_DOUBLE_EQ(coordinates[99][0], 0.0);
    EXPECT_DOUBLE_EQ(coordinates[99][1], 0.0);
    EXPECT_NEAR(coordinates.front()[1] - coordinates.back()[1], 0.002, 1e-12);
}

TEST(Kulfan, FitRoundTripRecoversParameters) {
    const KulfanParameters original = MakeTestParameters();
    const auto coordinates = GenerateKulfanCoordinates(original, 150);
    const KulfanParameters fitted = FitKulfanParameters(coordinates);

    for (size_t i = 0; i < kNumCstWeightsPerSide; ++i) {
        EXPECT_NEAR(fitted.upper_weights[i], original.upper_weights[i], 1e-8);
        EXPECT_NEAR(fitted.lower_weights[i], original.lower_weights[i], 1e-8);
    }
    EXPECT_NEAR(fitted.leading_edge_weight, original.leading_edge_weight, 1e-8);
    EXPECT_NEAR(fitted.TE_thickness, original.TE_thickness, 1e-8);
}

TEST(Kulfan, FitClampsNegativeTeThickness) {
    KulfanParameters parameters = MakeTestParameters();
    parameters.TE_thickness = 0.0;
    auto coordinates = GenerateKulfanCoordinates(parameters, 150);
    const KulfanParameters fitted = FitKulfanParameters(coordinates);
    EXPECT_GE(fitted.TE_thickness, 0.0);
}

TEST(Kulfan, BlXPointsAreUniformMidpoints) {
    const auto points = BlXPoints();
    EXPECT_DOUBLE_EQ(points.front(), 1.0 / 64.0);
    EXPECT_DOUBLE_EQ(points.back(), 63.0 / 64.0);
}

}  // namespace
}  // namespace bora::testing
