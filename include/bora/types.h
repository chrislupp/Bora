#pragma once

#include <array>
#include <cstddef>
#include <string>

namespace bora {

/// Number of CST (Kulfan) shape weights per airfoil surface.
inline constexpr size_t kNumCstWeightsPerSide = 8;

/// Number of boundary layer evaluation points per surface (NeuralFoil's
/// Data.N).
inline constexpr size_t kNumBlPoints = 32;

/// Dimension of the network input latent space.
inline constexpr size_t kNumLatentInputs = 25;

/// Dimension of the network output latent space (6 scalars + 6 boundary layer
/// distributions of kNumBlPoints each).
inline constexpr size_t kNumLatentOutputs = 6 + 6 * kNumBlPoints;

/// Number of differentiable user-facing inputs: 16 CST weights, leading edge
/// weight, trailing edge thickness, alpha, Re, n_crit, xtr_upper, xtr_lower.
inline constexpr size_t kNumInputs = 23;

/// Indices of the differentiable inputs within a Gradient array.
namespace input_index {
inline constexpr size_t kUpperWeights = 0;  ///< 8 entries: 0..7
inline constexpr size_t kLowerWeights = 8;  ///< 8 entries: 8..15
inline constexpr size_t kLeadingEdgeWeight = 16;
inline constexpr size_t kTeThickness = 17;
inline constexpr size_t kAlpha = 18;  ///< Angle of attack [deg]
inline constexpr size_t kRe = 19;     ///< Reynolds number
inline constexpr size_t kNCrit = 20;
inline constexpr size_t kXtrUpper = 21;
inline constexpr size_t kXtrLower = 22;
}  // namespace input_index

/// Gradient of a scalar output with respect to the kNumInputs user inputs.
/// Component ordering is given by the constants in bora::input_index.
using Gradient = std::array<double, kNumInputs>;

/// @brief Kulfan (CST) parameterization of an airfoil shape.
///
/// Uses the "CST with leading-edge modification" flavor with 8 weights per
/// side, class function exponents N1=0.5, N2=1.0. Weight ordering is from the
/// leading edge to the trailing edge.
struct KulfanParameters {
    std::array<double, kNumCstWeightsPerSide> upper_weights{};
    std::array<double, kNumCstWeightsPerSide> lower_weights{};
    double leading_edge_weight = 0.0;
    double TE_thickness = 0.0;  ///< Trailing edge thickness, in chord units.
};

/// @brief Inputs describing a single airfoil analysis case.
struct AeroInput {
    KulfanParameters kulfan{};
    double alpha = 0.0;   ///< Angle of attack [deg].
    double Re = 1e6;      ///< Chord-based Reynolds number.
    double n_crit = 9.0;  ///< Critical amplification factor for transition.
    double xtr_upper =
        1.0;  ///< Forced transition location, upper surface (x/c).
    double xtr_lower =
        1.0;  ///< Forced transition location, lower surface (x/c).
};

/// @brief Boundary layer quantities sampled at kNumBlPoints stations along one
/// surface. Station x/c locations are given by BlXPoints().
struct BoundaryLayerData {
    std::array<double, kNumBlPoints>
        theta{};  ///< Momentum thickness (chord-normalized).
    std::array<double, kNumBlPoints> H{};             ///< Shape factor.
    std::array<double, kNumBlPoints> ue_over_vinf{};  ///< Edge velocity ratio.
};

/// @brief Aerodynamic analysis outputs.
struct AeroOutput {
    double analysis_confidence = 0.0;  ///< Network confidence in [0, 1].
    double CL = 0.0;                   ///< Lift coefficient.
    double CD = 0.0;                   ///< Drag coefficient.
    double CM = 0.0;                   ///< Quarter-chord moment coefficient.
    double Top_Xtr = 0.0;  ///< Upper surface transition location (x/c).
    double Bot_Xtr = 0.0;  ///< Lower surface transition location (x/c).
    BoundaryLayerData upper{};
    BoundaryLayerData lower{};
};

/// @brief Gradients of the boundary layer quantities on one surface.
struct BoundaryLayerGradients {
    std::array<Gradient, kNumBlPoints> theta{};
    std::array<Gradient, kNumBlPoints> H{};
    std::array<Gradient, kNumBlPoints> ue_over_vinf{};
};

/// @brief Gradients of all analysis outputs with respect to the user inputs.
struct AeroGradients {
    Gradient analysis_confidence{};
    Gradient CL{};
    Gradient CD{};
    Gradient CM{};
    Gradient Top_Xtr{};
    Gradient Bot_Xtr{};
    BoundaryLayerGradients upper{};
    BoundaryLayerGradients lower{};
};

/// @brief Analysis outputs bundled with their gradients.
struct AeroResult {
    AeroOutput output{};
    AeroGradients gradient{};
};

/// @brief Available NeuralFoil network sizes (speed/accuracy tradeoff).
enum class ModelSize {
    kXXSmall,
    kXSmall,
    kSmall,
    kMedium,
    kLarge,
    kXLarge,  ///< Default in upstream NeuralFoil.
    kXXLarge,
    kXXXLarge,
};

/// @brief Returns the canonical name of a model size (e.g. "xlarge").
/// @param size Model size.
/// @return Lower-case name matching upstream NeuralFoil file naming.
std::string ToString(ModelSize size);

/// @brief Returns the x/c locations of the boundary layer evaluation points.
/// @return Array of kNumBlPoints chordwise stations (uniform midpoints on [0,
/// 1]).
std::array<double, kNumBlPoints> BlXPoints();

}  // namespace bora
