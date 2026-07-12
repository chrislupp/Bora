#include "bora/network.h"

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>

static_assert(
    std::endian::native == std::endian::little,
    "Bora weight files are little-endian; big-endian hosts are not supported.");

namespace bora {

namespace {

constexpr char kNetworkMagic[8] = {'B', 'O', 'R', 'A', 'N', 'N', '0', '1'};
constexpr char kDistributionMagic[8] = {'B', 'O', 'R', 'A', 'I', 'D', '0', '1'};

/// Sequential reader over an in-memory blob with bounds checking.
class BlobReader {
public:
    explicit BlobReader(std::span<const std::byte> data) : data_(data) {}

    void ReadRaw(void* dest, size_t n_bytes) {
        if (offset_ + n_bytes > data_.size()) {
            throw std::runtime_error("Bora weight blob is truncated.");
        }
        std::memcpy(dest, data_.data() + offset_, n_bytes);
        offset_ += n_bytes;
    }

    uint32_t ReadU32() {
        uint32_t value = 0;
        ReadRaw(&value, sizeof(value));
        return value;
    }

    void ReadF32Vector(std::vector<double>& dest, size_t count) {
        std::vector<float> buffer(count);
        ReadRaw(buffer.data(), count * sizeof(float));
        dest.resize(count);
        for (size_t i = 0; i < count; ++i) {
            dest[i] = static_cast<double>(buffer[i]);
        }
    }

    void ReadF64Vector(std::vector<double>& dest, size_t count) {
        dest.resize(count);
        ReadRaw(dest.data(), count * sizeof(double));
    }

    void CheckMagic(const char (&magic)[8], const std::string& what) {
        char found[8];
        ReadRaw(found, sizeof(found));
        if (std::memcmp(found, magic, sizeof(found)) != 0) {
            throw std::runtime_error("Not a Bora " + what +
                                     " file (bad magic).");
        }
    }

    [[nodiscard]] bool AtEnd() const { return offset_ == data_.size(); }

private:
    std::span<const std::byte> data_;
    size_t offset_ = 0;
};

std::vector<std::byte> ReadFile(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + path.string());
    }
    const auto size = static_cast<size_t>(file.tellg());
    std::vector<std::byte> data(size);
    file.seekg(0);
    file.read(reinterpret_cast<char*>(data.data()),
              static_cast<std::streamsize>(size));
    if (!file) {
        throw std::runtime_error("Failed to read file: " + path.string());
    }
    return data;
}

/// Dot product with four independent accumulators: breaks the floating-point
/// dependency chain so the loop is throughput-bound instead of latency-bound
/// (the compiler may not reassociate floating-point sums itself).
inline double Dot(const double* __restrict a, const double* __restrict b,
                  size_t n) {
    double s0 = 0.0;
    double s1 = 0.0;
    double s2 = 0.0;
    double s3 = 0.0;
    size_t j = 0;
    for (; j + 4 <= n; j += 4) {
        s0 += a[j] * b[j];
        s1 += a[j + 1] * b[j + 1];
        s2 += a[j + 2] * b[j + 2];
        s3 += a[j + 3] * b[j + 3];
    }
    for (; j < n; ++j) {
        s0 += a[j] * b[j];
    }
    return (s0 + s1) + (s2 + s3);
}

/// swish(x) = x * sigmoid(x), the network's hidden-layer activation.
inline double Swish(double x) {
    return x / (1.0 + std::exp(-x));
}

/// d/dx swish(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
inline double SwishDerivative(double x) {
    const double sig = 1.0 / (1.0 + std::exp(-x));
    return sig * (1.0 + x * (1.0 - sig));
}

}  // namespace

Network Network::Load(const std::filesystem::path& path) {
    const auto data = ReadFile(path);
    return FromMemory(data);
}

Network Network::FromMemory(std::span<const std::byte> data) {
    BlobReader reader(data);
    reader.CheckMagic(kNetworkMagic, "network weight");

    const uint32_t n_layers = reader.ReadU32();
    if (n_layers == 0 || n_layers > 1000) {
        throw std::runtime_error(
            "Bora network file has an implausible layer count: " +
            std::to_string(n_layers));
    }

    Network network;
    network.layers_.resize(n_layers);
    for (auto& layer : network.layers_) {
        layer.rows = reader.ReadU32();
        layer.cols = reader.ReadU32();
        if (layer.rows == 0 || layer.cols == 0) {
            throw std::runtime_error("Bora network file has an empty layer.");
        }
        reader.ReadF32Vector(layer.weights, layer.rows * layer.cols);
        reader.ReadF32Vector(layer.biases, layer.rows);
    }
    if (!reader.AtEnd()) {
        throw std::runtime_error("Bora network file has trailing data.");
    }

    for (size_t i = 1; i < network.layers_.size(); ++i) {
        if (network.layers_[i].cols != network.layers_[i - 1].rows) {
            throw std::runtime_error(
                "Bora network file has inconsistent layer dimensions.");
        }
    }
    return network;
}

void Network::Evaluate(std::span<const double> x, std::span<double> y) const {
    std::vector<double> current(x.begin(), x.end());
    std::vector<double> next;

    for (size_t l = 0; l < layers_.size(); ++l) {
        const Layer& layer = layers_[l];
        const bool is_last = (l == layers_.size() - 1);
        next.assign(layer.rows, 0.0);

        // Restrict-qualified pointers: lets the compiler vectorize the inner
        // products without assuming the input and output buffers alias.
        const double* __restrict in = current.data();
        double* __restrict out = next.data();
        for (size_t i = 0; i < layer.rows; ++i) {
            const double sum =
                layer.biases[i] +
                Dot(&layer.weights[i * layer.cols], in, layer.cols);
            out[i] = is_last ? sum : Swish(sum);
        }
        current.swap(next);
    }

    for (size_t i = 0; i < current.size(); ++i) {
        y[i] = current[i];
    }
}

void Network::EvaluateWithJacobian(std::span<const double> x,
                                   std::span<const double> jx, size_t n_seed,
                                   std::span<double> y,
                                   std::span<double> jy) const {
    // Values and Jacobians are propagated together layer-by-layer. Jacobians
    // are stored row-major: entry (i, k) is d(value_i)/d(seed_k).
    std::vector<double> current(x.begin(), x.end());
    std::vector<double> current_jac(jx.begin(), jx.end());
    std::vector<double> next;
    std::vector<double> next_jac;

    for (size_t l = 0; l < layers_.size(); ++l) {
        const Layer& layer = layers_[l];
        const bool is_last = (l == layers_.size() - 1);
        next.assign(layer.rows, 0.0);
        next_jac.assign(layer.rows * n_seed, 0.0);

        // Restrict-qualified pointers: lets the compiler vectorize the
        // Jacobian accumulation without assuming the buffers alias.
        const double* __restrict in = current.data();
        const double* __restrict in_jac = current_jac.data();
        double* __restrict out = next.data();
        double* __restrict out_jac = next_jac.data();
        for (size_t i = 0; i < layer.rows; ++i) {
            const double* __restrict w_row = &layer.weights[i * layer.cols];
            double* __restrict jac_row = &out_jac[i * n_seed];

            const double sum = layer.biases[i] + Dot(w_row, in, layer.cols);

            // Jacobian rows accumulate two weighted input rows per pass to
            // shorten the dependency chain on each jac_row entry.
            size_t j = 0;
            for (; j + 2 <= layer.cols; j += 2) {
                const double w0 = w_row[j];
                const double w1 = w_row[j + 1];
                const double* __restrict jac_in0 = &in_jac[j * n_seed];
                const double* __restrict jac_in1 = &in_jac[(j + 1) * n_seed];
                for (size_t k = 0; k < n_seed; ++k) {
                    jac_row[k] += w0 * jac_in0[k] + w1 * jac_in1[k];
                }
            }
            for (; j < layer.cols; ++j) {
                const double w = w_row[j];
                const double* __restrict jac_in = &in_jac[j * n_seed];
                for (size_t k = 0; k < n_seed; ++k) {
                    jac_row[k] += w * jac_in[k];
                }
            }

            if (is_last) {
                out[i] = sum;
            } else {
                out[i] = Swish(sum);
                const double d = SwishDerivative(sum);
                for (size_t k = 0; k < n_seed; ++k) {
                    jac_row[k] *= d;
                }
            }
        }
        current.swap(next);
        current_jac.swap(next_jac);
    }

    for (size_t i = 0; i < current.size(); ++i) {
        y[i] = current[i];
    }
    for (size_t i = 0; i < current_jac.size(); ++i) {
        jy[i] = current_jac[i];
    }
}

InputDistribution InputDistribution::Load(const std::filesystem::path& path) {
    const auto data = ReadFile(path);
    return FromMemory(data);
}

InputDistribution InputDistribution::FromMemory(
    std::span<const std::byte> data) {
    BlobReader reader(data);
    reader.CheckMagic(kDistributionMagic, "input distribution");

    const uint32_t n = reader.ReadU32();
    if (n == 0 || n > 10000) {
        throw std::runtime_error(
            "Bora distribution file has an implausible dimension: " +
            std::to_string(n));
    }

    InputDistribution distribution;
    reader.ReadF64Vector(distribution.mean_, n);
    reader.ReadF64Vector(distribution.inv_cov_, static_cast<size_t>(n) * n);
    if (!reader.AtEnd()) {
        throw std::runtime_error("Bora distribution file has trailing data.");
    }
    return distribution;
}

double InputDistribution::SquaredMahalanobis(std::span<const double> x) const {
    const size_t n = mean_.size();
    std::vector<double> centered(n);
    for (size_t i = 0; i < n; ++i) {
        centered[i] = x[i] - mean_[i];
    }

    double distance = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double row_sum = 0.0;
        const double* row = &inv_cov_[i * n];
        for (size_t j = 0; j < n; ++j) {
            row_sum += row[j] * centered[j];
        }
        distance += centered[i] * row_sum;
    }
    return distance;
}

void InputDistribution::SquaredMahalanobisGradient(
    std::span<const double> x, std::span<double> grad) const {
    // d/dx (x - m)^T S (x - m) = 2 S (x - m), with S symmetric.
    const size_t n = mean_.size();
    std::vector<double> centered(n);
    for (size_t i = 0; i < n; ++i) {
        centered[i] = x[i] - mean_[i];
    }
    for (size_t i = 0; i < n; ++i) {
        double row_sum = 0.0;
        const double* row = &inv_cov_[i * n];
        for (size_t j = 0; j < n; ++j) {
            row_sum += row[j] * centered[j];
        }
        grad[i] = 2.0 * row_sum;
    }
}

}  // namespace bora
