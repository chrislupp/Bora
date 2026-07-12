#pragma once

#include "bora/network.h"
#include "bora/types.h"

#include <filesystem>

namespace bora {

/// @brief NeuralFoil airfoil analysis model.
///
/// A C++ port of NeuralFoil (https://github.com/peterdsharpe/NeuralFoil): a
/// physics-informed neural network surrogate of XFoil. Given an airfoil shape
/// in Kulfan (CST) parameterization and an operating condition, it predicts
/// lift, drag, and moment coefficients, transition locations, and boundary
/// layer distributions — optionally with analytic gradients with respect to
/// all inputs, enabling gradient-based design optimization.
///
/// Instances are immutable after construction and safe to share across
/// threads.
class NeuralFoil {
public:
    /// @brief Constructs an analysis model, loading weights from disk.
    ///
    /// The weight directory is resolved in this order:
    ///   1. The explicit @p data_dir argument, if non-empty.
    ///   2. The BORA_DATA_DIR environment variable, if set.
    ///   3. The installed data directory baked in at build time.
    ///   4. The source-tree data directory baked in at build time.
    ///
    /// @param size Network size (speed/accuracy tradeoff).
    /// @param data_dir Optional directory containing the HDF5 (.h5) weight
    /// files.
    /// @throws std::runtime_error if the weight files cannot be found or
    /// parsed.
    explicit NeuralFoil(ModelSize size = ModelSize::kXLarge,
                        const std::filesystem::path& data_dir = {});

    /// @brief Constructs an analysis model from pre-loaded components.
    /// @param network The network weights (input 25, output 198).
    /// @param distribution The scaled-input training distribution statistics.
    /// @throws std::runtime_error if dimensions are inconsistent.
    NeuralFoil(Network network, InputDistribution distribution);

    /// @brief Analyzes an airfoil at an operating condition.
    /// @param input Airfoil shape and operating condition.
    /// @return Aerodynamic outputs.
    [[nodiscard]] AeroOutput Analyze(const AeroInput& input) const;

    /// @brief Analyzes an airfoil and computes analytic gradients of every
    /// output with respect to all kNumInputs user inputs (CST weights, leading
    /// edge weight, TE thickness, alpha, Re, n_crit, xtr_upper, xtr_lower).
    /// @param input Airfoil shape and operating condition.
    /// @return Aerodynamic outputs and their gradients.
    [[nodiscard]] AeroResult AnalyzeWithGradients(const AeroInput& input) const;

    /// @brief Returns the model size this instance was constructed with (or
    /// kXLarge when constructed from pre-loaded components).
    [[nodiscard]] ModelSize GetModelSize() const { return size_; }

private:
    ModelSize size_ = ModelSize::kXLarge;
    Network network_;
    InputDistribution distribution_;
};

}  // namespace bora
