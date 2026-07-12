#pragma once

#include "bora/neuralfoil.h"
#include "bora/types.h"

#include <filesystem>
#include <memory>

#include <explicit.h>

namespace bora {

/// @brief Philote explicit discipline exposing NeuralFoil analyses over gRPC.
///
/// Wraps bora::NeuralFoil as a philote::ExplicitDiscipline so that any Philote
/// client (e.g. an OpenMDAO/Philote-Python component) can evaluate airfoil
/// aerodynamics and their analytic gradients remotely.
///
/// Inputs (all differentiable):
///   - upper_weights [8], lower_weights [8]: CST weights, LE to TE
///   - leading_edge_weight, TE_thickness: Kulfan shape modifiers
///   - alpha [deg], Re, n_crit, xtr_upper, xtr_lower: operating condition
///
/// Outputs: CL, CD, CM, Top_Xtr, Bot_Xtr, analysis_confidence, and (when
/// enabled via the include_boundary_layer option) the boundary layer
/// distributions upper_theta, upper_H, upper_ue_over_vinf, lower_theta,
/// lower_H, lower_ue_over_vinf at kNumBlPoints stations each.
///
/// Options:
///   - model_size (string): network size, one of "xxsmall", "xsmall", "small",
///     "medium", "large", "xlarge", "xxlarge", "xxxlarge" (default "xlarge")
///   - data_directory (string): directory containing the .bnn/.bid weight
///     files; empty uses the standard resolution order (BORA_DATA_DIR, then
///     baked-in install/source locations)
///   - include_boundary_layer (bool): expose boundary layer distribution
///     outputs and their partials (default false)
class BoraDiscipline : public philote::ExplicitDiscipline {
public:
    BoraDiscipline() = default;
    ~BoraDiscipline() noexcept = default;

    /// @brief Declares the available discipline options.
    void Initialize() override;

    /// @brief Applies client-provided option values and (re)loads the model.
    /// @param options Option name/value pairs.
    /// @throws std::invalid_argument on an unknown model_size value.
    /// @throws std::runtime_error if the weight files cannot be loaded.
    void SetOptions(const google::protobuf::Struct& options) override;

    /// @brief Declares the analysis inputs and outputs.
    void Setup() override;

    /// @brief Declares the partials of every output with respect to every
    /// input.
    void SetupPartials() override;

    /// @brief Evaluates the aerodynamic outputs.
    /// @param inputs Airfoil shape and operating condition variables.
    /// @param outputs Aerodynamic output variables (filled in place).
    void Compute(const philote::Variables& inputs,
                 philote::Variables& outputs) override;

    /// @brief Evaluates the analytic gradients of every declared output.
    /// @param inputs Airfoil shape and operating condition variables.
    /// @param partials Jacobian blocks (filled in place).
    void ComputePartials(const philote::Variables& inputs,
                         philote::Partials& partials) override;

private:
    /// @brief Returns the analysis model, constructing it with the current
    /// option values on first use.
    const NeuralFoil& Model();

    std::unique_ptr<NeuralFoil> model_;
    ModelSize model_size_ = ModelSize::kXLarge;
    std::filesystem::path data_directory_{};
    bool include_boundary_layer_ = false;
};

}  // namespace bora
