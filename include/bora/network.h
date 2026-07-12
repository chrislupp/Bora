#pragma once

#include <cstddef>
#include <filesystem>
#include <span>
#include <vector>

namespace bora {

/// @brief A single dense layer: y = W x + b, with W stored row-major.
struct Layer {
    size_t rows = 0;              ///< Output dimension.
    size_t cols = 0;              ///< Input dimension.
    std::vector<double> weights;  ///< rows * cols entries, row-major.
    std::vector<double> biases;   ///< rows entries.
};

/// @brief A multilayer perceptron with swish activations between layers (no
/// activation after the final layer), matching NeuralFoil's architecture.
class Network {
public:
    /// @brief Loads a network from a Bora HDF5 weight file (.h5).
    /// @param path Path to the weight file.
    /// @return The loaded network.
    /// @throws std::runtime_error on I/O or format errors.
    static Network Load(const std::filesystem::path& path);

    /// @brief Parses a network from an in-memory HDF5 file image.
    /// @param data The raw bytes of a network weight .h5 file.
    /// @return The parsed network.
    /// @throws std::runtime_error on format errors.
    static Network FromMemory(std::span<const std::byte> data);

    /// @brief Evaluates the network.
    /// @param x Input vector (InputSize() entries).
    /// @param y Output vector (OutputSize() entries), overwritten.
    void Evaluate(std::span<const double> x, std::span<double> y) const;

    /// @brief Evaluates the network and propagates an input Jacobian.
    ///
    /// Computes y = net(x) and jy = d(net)/dx * jx via forward-mode
    /// differentiation, where jx holds the derivatives of x with respect to
    /// n_seed independent variables.
    ///
    /// @param x Input vector (InputSize() entries).
    /// @param jx Input Jacobian, row-major InputSize() x n_seed.
    /// @param n_seed Number of seed (independent) variables.
    /// @param y Output vector (OutputSize() entries), overwritten.
    /// @param jy Output Jacobian, row-major OutputSize() x n_seed, overwritten.
    void EvaluateWithJacobian(std::span<const double> x,
                              std::span<const double> jx, size_t n_seed,
                              std::span<double> y, std::span<double> jy) const;

    /// @brief Returns the input dimension of the network.
    [[nodiscard]] size_t InputSize() const {
        return layers_.empty() ? 0 : layers_.front().cols;
    }

    /// @brief Returns the output dimension of the network.
    [[nodiscard]] size_t OutputSize() const {
        return layers_.empty() ? 0 : layers_.back().rows;
    }

    /// @brief Returns the layers of the network.
    [[nodiscard]] const std::vector<Layer>& GetLayers() const {
        return layers_;
    }

private:
    std::vector<Layer> layers_;
};

/// @brief Statistics of the training data distribution in the scaled input
/// space, used for the Mahalanobis-distance analysis-confidence correction.
class InputDistribution {
public:
    /// @brief Loads distribution statistics from a Bora HDF5 file (.h5).
    /// @param path Path to the file.
    /// @return The loaded distribution.
    /// @throws std::runtime_error on I/O or format errors.
    static InputDistribution Load(const std::filesystem::path& path);

    /// @brief Parses distribution statistics from an in-memory HDF5 file
    /// image.
    /// @param data The raw bytes of a distribution .h5 file.
    /// @return The parsed distribution.
    /// @throws std::runtime_error on format errors.
    static InputDistribution FromMemory(std::span<const std::byte> data);

    /// @brief Computes the squared Mahalanobis distance of a point from the
    /// training data distribution.
    /// @param x Query point (Size() entries).
    /// @return The squared Mahalanobis distance.
    [[nodiscard]] double SquaredMahalanobis(std::span<const double> x) const;

    /// @brief Computes the gradient of SquaredMahalanobis at x.
    /// @param x Query point (Size() entries).
    /// @param grad Gradient output (Size() entries), overwritten.
    void SquaredMahalanobisGradient(std::span<const double> x,
                                    std::span<double> grad) const;

    /// @brief Returns the dimensionality of the input space.
    [[nodiscard]] size_t Size() const { return mean_.size(); }

private:
    std::vector<double> mean_;  ///< Mean of the scaled inputs.
    std::vector<double>
        inv_cov_;  ///< Inverse covariance, row-major Size() x Size().
};

}  // namespace bora
