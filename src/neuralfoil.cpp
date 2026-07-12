#include "bora/neuralfoil.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <numbers>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace bora {

namespace {

constexpr size_t kN = kNumBlPoints;
constexpr double kDegToRad = std::numbers::pi / 180.0;

// Matches NeuralFoil's numerically-stable sigmoid: inputs are clipped to
// suppress overflow, with the clip bound derived from the float64 range.
const double kLnEps = std::log(10.0 / std::numeric_limits<double>::max());

double StableSigmoid(double x) {
    x = std::clamp(x, kLnEps, -kLnEps);
    return 1.0 / (1.0 + std::exp(-x));
}

/// Builds the 25-dimensional scaled latent input vector from user inputs.
std::vector<double> BuildLatentInput(const AeroInput& input) {
    std::vector<double> x(kNumLatentInputs);
    for (size_t i = 0; i < kNumCstWeightsPerSide; ++i) {
        x[i] = input.kulfan.upper_weights[i];
        x[8 + i] = input.kulfan.lower_weights[i];
    }
    x[16] = input.kulfan.leading_edge_weight;
    x[17] = input.kulfan.TE_thickness * 50.0;
    const double alpha_rad = input.alpha * kDegToRad;
    x[18] = std::sin(2.0 * alpha_rad);
    x[19] = std::cos(alpha_rad);
    x[20] = 1.0 - x[19] * x[19];
    x[21] = (std::log(input.Re) - 12.5) / 3.5;
    x[22] = (input.n_crit - 9.0) / 4.5;
    x[23] = input.xtr_upper;
    x[24] = input.xtr_lower;
    return x;
}

/// Builds the seed Jacobian d(latent input)/d(user inputs), row-major
/// kNumLatentInputs x kNumInputs.
std::vector<double> BuildSeedJacobian(const AeroInput& input) {
    namespace idx = input_index;
    std::vector<double> jx(kNumLatentInputs * kNumInputs, 0.0);
    auto at = [&jx](size_t row, size_t col) -> double& {
        return jx[row * kNumInputs + col];
    };

    for (size_t i = 0; i < kNumCstWeightsPerSide; ++i) {
        at(i, idx::kUpperWeights + i) = 1.0;
        at(8 + i, idx::kLowerWeights + i) = 1.0;
    }
    at(16, idx::kLeadingEdgeWeight) = 1.0;
    at(17, idx::kTeThickness) = 50.0;
    const double alpha_rad = input.alpha * kDegToRad;
    at(18, idx::kAlpha) = 2.0 * kDegToRad * std::cos(2.0 * alpha_rad);
    at(19, idx::kAlpha) = -kDegToRad * std::sin(alpha_rad);
    at(20, idx::kAlpha) = kDegToRad * std::sin(2.0 * alpha_rad);
    at(21, idx::kRe) = 1.0 / (3.5 * input.Re);
    at(22, idx::kNCrit) = 1.0 / 4.5;
    at(23, idx::kXtrUpper) = 1.0;
    at(24, idx::kXtrLower) = 1.0;
    return jx;
}

/// Applies the alpha-symmetry flip to the latent input: the airfoil is
/// mirrored (upper/lower surfaces swapped and negated) and alpha is negated.
/// The callback receives (dest_index, source_index, sign); the same linear map
/// applies to latent values and to rows of a seed Jacobian.
template <typename Apply>
void ApplyInputFlip(Apply&& apply) {
    for (size_t i = 0; i < 8; ++i) {
        apply(i, 8 + i, -1.0);  // kulfan_upper <- -kulfan_lower
        apply(8 + i, i, -1.0);  // kulfan_lower <- -kulfan_upper
    }
    apply(16, 16, -1.0);  // leading edge weight
    apply(17, 17, 1.0);   // TE thickness
    apply(18, 18, -1.0);  // sin(2a)
    apply(19, 19, 1.0);   // cos(a)
    apply(20, 20, 1.0);   // 1 - cos(a)^2
    apply(21, 21, 1.0);   // scaled ln(Re)
    apply(22, 22, 1.0);   // scaled n_crit
    apply(23, 24, 1.0);   // xtr_upper <- xtr_lower
    apply(24, 23, 1.0);   // xtr_lower <- xtr_upper
}

/// Applies the inverse of the output flip to results of a flipped evaluation,
/// bringing them back to the original airfoil's orientation. The callback
/// receives (dest_index, source_index, sign) and covers every output index.
template <typename Apply>
void ApplyOutputUnflip(Apply&& apply) {
    apply(0, 0, 1.0);   // analysis confidence
    apply(1, 1, -1.0);  // CL
    apply(2, 2, 1.0);   // CD
    apply(3, 3, -1.0);  // CM
    apply(4, 5, 1.0);   // Top_Xtr <- Bot_Xtr
    apply(5, 4, 1.0);   // Bot_Xtr <- Top_Xtr
    for (size_t i = 0; i < 2 * kN; ++i) {
        apply(6 + i, 6 + 3 * kN + i, 1.0);  // upper theta, H <- lower theta, H
        apply(6 + 3 * kN + i, 6 + i, 1.0);  // lower theta, H <- upper theta, H
    }
    for (size_t i = 0; i < kN; ++i) {
        apply(6 + 2 * kN + i, 6 + 5 * kN + i, -1.0);  // upper ue <- -lower ue
        apply(6 + 5 * kN + i, 6 + 2 * kN + i, -1.0);  // lower ue <- -upper ue
    }
}

/// Decodes the fused latent outputs into physical quantities, optionally
/// propagating the latent Jacobian into per-output gradients.
void DecodeOutputs(std::span<const double> y_fused, const AeroInput& input,
                   AeroOutput& output,
                   const std::vector<double>* jy_fused = nullptr,
                   AeroGradients* gradients = nullptr) {
    namespace idx = input_index;
    const auto jac_row = [&](size_t i) -> const double* {
        return &(*jy_fused)[i * kNumInputs];
    };
    const auto scale_row = [&](size_t i, double factor, Gradient& dest) {
        const double* row = jac_row(i);
        for (size_t k = 0; k < kNumInputs; ++k) {
            dest[k] = factor * row[k];
        }
    };

    // Scalar outputs, with the same scaling/squashing as upstream NeuralFoil.
    output.analysis_confidence = StableSigmoid(y_fused[0]);
    output.CL = y_fused[1] / 2.0;
    output.CD = std::exp((y_fused[2] - 2.0) * 2.0);
    output.CM = y_fused[3] / 20.0;
    output.Top_Xtr = std::clamp(y_fused[4], 0.0, 1.0);
    output.Bot_Xtr = std::clamp(y_fused[5], 0.0, 1.0);

    if (gradients != nullptr) {
        const double conf = output.analysis_confidence;
        scale_row(0, conf * (1.0 - conf), gradients->analysis_confidence);
        scale_row(1, 0.5, gradients->CL);
        scale_row(2, 2.0 * output.CD, gradients->CD);
        scale_row(3, 1.0 / 20.0, gradients->CM);
        // Clipped transition locations: zero gradient where the clip is active.
        scale_row(4, (y_fused[4] > 0.0 && y_fused[4] < 1.0) ? 1.0 : 0.0,
                  gradients->Top_Xtr);
        scale_row(5, (y_fused[5] > 0.0 && y_fused[5] < 1.0) ? 1.0 : 0.0,
                  gradients->Bot_Xtr);
    }

    // Boundary layer distributions. Latent layout per surface: theta encoding,
    // then H encoding, then ue/vinf, each kN wide. Upper surface starts at 6,
    // lower surface at 6 + 3 * kN.
    const auto decode_surface = [&](size_t offset, BoundaryLayerData& bl,
                                    BoundaryLayerGradients* bl_grad) {
        for (size_t i = 0; i < kN; ++i) {
            const size_t i_theta = offset + i;
            const size_t i_h = offset + kN + i;
            const size_t i_ue = offset + 2 * kN + i;

            const double ue = y_fused[i_ue];
            const double pow_term = std::pow(10.0, y_fused[i_theta]);
            const double theta = (pow_term - 0.1) / (std::abs(ue) * input.Re);
            const double h = 2.6 * std::exp(y_fused[i_h]);

            bl.ue_over_vinf[i] = ue;
            bl.theta[i] = theta;
            bl.H[i] = h;

            if (bl_grad != nullptr) {
                const double* row_theta = jac_row(i_theta);
                const double* row_h = jac_row(i_h);
                const double* row_ue = jac_row(i_ue);

                const double dtheta_dlatent =
                    std::numbers::ln10 * pow_term / (std::abs(ue) * input.Re);
                const double dtheta_due = -theta / ue;
                for (size_t k = 0; k < kNumInputs; ++k) {
                    bl_grad->theta[i][k] =
                        dtheta_dlatent * row_theta[k] + dtheta_due * row_ue[k];
                    bl_grad->H[i][k] = h * row_h[k];
                    bl_grad->ue_over_vinf[i][k] = row_ue[k];
                }
                // theta also depends on Re directly through the decoding.
                bl_grad->theta[i][idx::kRe] += -theta / input.Re;
            }
        }
    };

    decode_surface(6, output.upper,
                   gradients != nullptr ? &gradients->upper : nullptr);
    decode_surface(6 + 3 * kN, output.lower,
                   gradients != nullptr ? &gradients->lower : nullptr);
}

const char* FileStem(ModelSize size) {
    switch (size) {
        case ModelSize::kXXSmall:
            return "xxsmall";
        case ModelSize::kXSmall:
            return "xsmall";
        case ModelSize::kSmall:
            return "small";
        case ModelSize::kMedium:
            return "medium";
        case ModelSize::kLarge:
            return "large";
        case ModelSize::kXLarge:
            return "xlarge";
        case ModelSize::kXXLarge:
            return "xxlarge";
        case ModelSize::kXXXLarge:
            return "xxxlarge";
    }
    throw std::runtime_error("Invalid ModelSize.");
}

std::filesystem::path ResolveDataDir(
    const std::filesystem::path& explicit_dir) {
    if (!explicit_dir.empty()) {
        return explicit_dir;
    }
    if (const char* env_dir = std::getenv("BORA_DATA_DIR")) {
        return env_dir;
    }
#ifdef BORA_INSTALL_DATA_DIR
    if (std::filesystem::exists(BORA_INSTALL_DATA_DIR)) {
        return BORA_INSTALL_DATA_DIR;
    }
#endif
#ifdef BORA_SOURCE_DATA_DIR
    if (std::filesystem::exists(BORA_SOURCE_DATA_DIR)) {
        return BORA_SOURCE_DATA_DIR;
    }
#endif
    throw std::runtime_error(
        "Cannot locate the Bora weight data directory. Pass it explicitly or "
        "set "
        "the BORA_DATA_DIR environment variable.");
}

Network LoadNetwork(ModelSize size, const std::filesystem::path& data_dir) {
    const auto dir = ResolveDataDir(data_dir);
    return Network::Load(dir / (std::string("nn-") + FileStem(size) + ".h5"));
}

InputDistribution LoadDistribution(const std::filesystem::path& data_dir) {
    const auto dir = ResolveDataDir(data_dir);
    return InputDistribution::Load(dir / "scaled_input_distribution.h5");
}

}  // namespace

std::string ToString(ModelSize size) {
    return FileStem(size);
}

std::array<double, kNumBlPoints> BlXPoints() {
    std::array<double, kNumBlPoints> points{};
    for (size_t i = 0; i < kNumBlPoints; ++i) {
        // Midpoints of a uniform partition of [0, 1] into kNumBlPoints cells.
        points[i] = (2.0 * static_cast<double>(i) + 1.0) / (2.0 * kNumBlPoints);
    }
    return points;
}

NeuralFoil::NeuralFoil(ModelSize size, const std::filesystem::path& data_dir)
    : NeuralFoil(LoadNetwork(size, data_dir), LoadDistribution(data_dir)) {
    size_ = size;
}

NeuralFoil::NeuralFoil(Network network, InputDistribution distribution)
    : network_(std::move(network)), distribution_(std::move(distribution)) {
    if (network_.InputSize() != kNumLatentInputs) {
        throw std::runtime_error("Network input dimension mismatch.");
    }
    if (network_.OutputSize() != kNumLatentOutputs) {
        throw std::runtime_error("Network output dimension mismatch.");
    }
    if (distribution_.Size() != kNumLatentInputs) {
        throw std::runtime_error("Input distribution dimension mismatch.");
    }
}

AeroOutput NeuralFoil::Analyze(const AeroInput& input) const {
    const auto x = BuildLatentInput(input);

    // Flipped input: embeds the alpha-symmetry invariant of airfoil analysis.
    std::vector<double> x_flipped(kNumLatentInputs);
    ApplyInputFlip([&](size_t dest, size_t src, double sign) {
        x_flipped[dest] = sign * x[src];
    });

    std::vector<double> y(kNumLatentOutputs);
    std::vector<double> y_flipped(kNumLatentOutputs);
    network_.Evaluate(x, y);
    network_.Evaluate(x_flipped, y_flipped);

    // Analysis-confidence correction: decays confidence with distance from the
    // training data distribution (this term was baked into training).
    y[0] -= distribution_.SquaredMahalanobis(x) / (2.0 * kNumLatentInputs);
    y_flipped[0] -=
        distribution_.SquaredMahalanobis(x_flipped) / (2.0 * kNumLatentInputs);

    // Un-flip the outputs of the flipped evaluation, then fuse by averaging.
    std::vector<double> y_fused(kNumLatentOutputs);
    ApplyOutputUnflip([&](size_t dest, size_t src, double sign) {
        y_fused[dest] = 0.5 * (y[dest] + sign * y_flipped[src]);
    });

    AeroOutput output;
    DecodeOutputs(y_fused, input, output);
    return output;
}

AeroResult NeuralFoil::AnalyzeWithGradients(const AeroInput& input) const {
    const auto x = BuildLatentInput(input);
    const auto jx = BuildSeedJacobian(input);

    std::vector<double> x_flipped(kNumLatentInputs);
    std::vector<double> jx_flipped(kNumLatentInputs * kNumInputs);
    ApplyInputFlip([&](size_t dest, size_t src, double sign) {
        x_flipped[dest] = sign * x[src];
        for (size_t k = 0; k < kNumInputs; ++k) {
            jx_flipped[dest * kNumInputs + k] = sign * jx[src * kNumInputs + k];
        }
    });

    std::vector<double> y(kNumLatentOutputs);
    std::vector<double> jy(kNumLatentOutputs * kNumInputs);
    std::vector<double> y_flipped(kNumLatentOutputs);
    std::vector<double> jy_flipped(kNumLatentOutputs * kNumInputs);
    network_.EvaluateWithJacobian(x, jx, kNumInputs, y, jy);
    network_.EvaluateWithJacobian(x_flipped, jx_flipped, kNumInputs, y_flipped,
                                  jy_flipped);

    // Analysis-confidence correction and its gradient, for both passes.
    const auto apply_mahalanobis =
        [this](std::span<const double> x_pass, std::span<const double> jx_pass,
               std::vector<double>& y_pass, std::vector<double>& jy_pass) {
            y_pass[0] -= distribution_.SquaredMahalanobis(x_pass) /
                         (2.0 * kNumLatentInputs);
            std::vector<double> grad_latent(kNumLatentInputs);
            distribution_.SquaredMahalanobisGradient(x_pass, grad_latent);
            for (size_t k = 0; k < kNumInputs; ++k) {
                double dot = 0.0;
                for (size_t i = 0; i < kNumLatentInputs; ++i) {
                    dot += grad_latent[i] * jx_pass[i * kNumInputs + k];
                }
                jy_pass[k] -= dot / (2.0 * kNumLatentInputs);
            }
        };
    apply_mahalanobis(x, jx, y, jy);
    apply_mahalanobis(x_flipped, jx_flipped, y_flipped, jy_flipped);

    std::vector<double> y_fused(kNumLatentOutputs);
    std::vector<double> jy_fused(kNumLatentOutputs * kNumInputs);
    ApplyOutputUnflip([&](size_t dest, size_t src, double sign) {
        y_fused[dest] = 0.5 * (y[dest] + sign * y_flipped[src]);
        for (size_t k = 0; k < kNumInputs; ++k) {
            jy_fused[dest * kNumInputs + k] =
                0.5 * (jy[dest * kNumInputs + k] +
                       sign * jy_flipped[src * kNumInputs + k]);
        }
    });

    AeroResult result;
    DecodeOutputs(y_fused, input, result.output, &jy_fused, &result.gradient);
    return result;
}

}  // namespace bora
