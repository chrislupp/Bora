#include "bora/network.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

#include <hdf5.h>

namespace bora {

namespace {

/// Suppresses HDF5's default stderr error printing for the current scope so
/// failures surface only as the exceptions thrown by the loaders.
class ErrorSilencer {
public:
    ErrorSilencer() {
        H5Eget_auto2(H5E_DEFAULT, &handler_, &handler_data_);
        H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
    }
    ~ErrorSilencer() { H5Eset_auto2(H5E_DEFAULT, handler_, handler_data_); }
    ErrorSilencer(const ErrorSilencer&) = delete;
    ErrorSilencer& operator=(const ErrorSilencer&) = delete;

private:
    H5E_auto2_t handler_ = nullptr;
    void* handler_data_ = nullptr;
};

/// Owns an HDF5 identifier and releases it with the given close function.
class Handle {
public:
    Handle(hid_t id, herr_t (*close)(hid_t), const std::string& error)
        : id_(id), close_(close) {
        if (id_ < 0) {
            throw std::runtime_error(error);
        }
    }
    ~Handle() { close_(id_); }
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;

    [[nodiscard]] hid_t Get() const { return id_; }

private:
    hid_t id_;
    herr_t (*close_)(hid_t);
};

Handle OpenFile(const std::filesystem::path& path, const std::string& what) {
    return {H5Fopen(path.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT),
            H5Fclose, "Cannot open " + what + " HDF5 file: " + path.string()};
}

/// Opens an in-memory .h5 blob as a read-only HDF5 file. The image is copied
/// into the HDF5 core driver, so the blob only needs to outlive this call.
Handle OpenFileImage(std::span<const std::byte> data, const std::string& what) {
    Handle fapl(H5Pcreate(H5P_FILE_ACCESS), H5Pclose,
                "Cannot create an HDF5 file-access property list.");
    if (H5Pset_fapl_core(fapl.Get(), 1 << 16, false) < 0 ||
        H5Pset_file_image(
            fapl.Get(),
            const_cast<void*>(static_cast<const void*>(data.data())),
            data.size()) < 0) {
        throw std::runtime_error("Cannot set up an in-memory HDF5 image.");
    }
    return {H5Fopen("bora-in-memory.h5", H5F_ACC_RDONLY, fapl.Get()), H5Fclose,
            "Not a valid " + what + " HDF5 image."};
}

/// Reads an entire dataset as doubles (HDF5 converts from the stored type)
/// and returns its extents in dims.
std::vector<double> ReadDoubleDataset(hid_t location, const char* name,
                                      std::vector<hsize_t>& dims) {
    const std::string what(name);
    Handle dataset(H5Dopen2(location, name, H5P_DEFAULT), H5Dclose,
                   "Bora weight file is missing dataset '" + what + "'.");
    Handle space(H5Dget_space(dataset.Get()), H5Sclose,
                 "Cannot read the dataspace of dataset '" + what + "'.");
    const int n_dims = H5Sget_simple_extent_ndims(space.Get());
    if (n_dims < 0) {
        throw std::runtime_error("Cannot read the rank of dataset '" + what +
                                 "'.");
    }
    dims.resize(static_cast<size_t>(n_dims));
    H5Sget_simple_extent_dims(space.Get(), dims.data(), nullptr);

    size_t total = 1;
    for (const hsize_t dim : dims) {
        total *= dim;
    }
    std::vector<double> values(total);
    if (H5Dread(dataset.Get(), H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                values.data()) < 0) {
        throw std::runtime_error("Failed to read dataset '" + what + "'.");
    }
    return values;
}

/// Reads and validates the /layers/<i>/{weights,biases} network layout.
std::vector<Layer> ReadLayers(hid_t file) {
    Handle layers_group(H5Gopen2(file, "layers", H5P_DEFAULT), H5Gclose,
                        "Bora network file has no 'layers' group.");
    H5G_info_t info;
    if (H5Gget_info(layers_group.Get(), &info) < 0) {
        throw std::runtime_error("Cannot read the 'layers' group.");
    }
    const auto n_layers = static_cast<size_t>(info.nlinks);
    if (n_layers == 0 || n_layers > 1000) {
        throw std::runtime_error(
            "Bora network file has an implausible layer count: " +
            std::to_string(n_layers));
    }

    std::vector<Layer> layers(n_layers);
    for (size_t i = 0; i < n_layers; ++i) {
        const std::string group_name = std::to_string(i);
        Handle layer_group(
            H5Gopen2(layers_group.Get(), group_name.c_str(), H5P_DEFAULT),
            H5Gclose,
            "Bora network file is missing layer group '" + group_name + "'.");
        std::vector<hsize_t> dims;
        layers[i].weights =
            ReadDoubleDataset(layer_group.Get(), "weights", dims);
        if (dims.size() != 2 || dims[0] == 0 || dims[1] == 0) {
            throw std::runtime_error("Layer " + group_name +
                                     " 'weights' must be a non-empty matrix.");
        }
        layers[i].rows = dims[0];
        layers[i].cols = dims[1];
        layers[i].biases = ReadDoubleDataset(layer_group.Get(), "biases", dims);
        if (dims.size() != 1 || dims[0] != layers[i].rows) {
            throw std::runtime_error(
                "Layer " + group_name +
                " 'biases' must be a vector with one entry per weight row.");
        }
    }

    for (size_t i = 1; i < layers.size(); ++i) {
        if (layers[i].cols != layers[i - 1].rows) {
            throw std::runtime_error(
                "Bora network file has inconsistent layer dimensions.");
        }
    }
    return layers;
}

/// The /mean and /inv_cov datasets of an input distribution file.
struct DistributionData {
    std::vector<double> mean;
    std::vector<double> inv_cov;
};

/// Reads and validates the input distribution layout.
DistributionData ReadDistribution(hid_t file) {
    DistributionData data;
    std::vector<hsize_t> dims;
    data.mean = ReadDoubleDataset(file, "mean", dims);
    if (dims.size() != 1 || dims[0] == 0 || dims[0] > 10000) {
        throw std::runtime_error(
            "Bora distribution file has an implausible 'mean' dimension.");
    }
    const size_t n = data.mean.size();
    data.inv_cov = ReadDoubleDataset(file, "inv_cov", dims);
    if (dims.size() != 2 || dims[0] != n || dims[1] != n) {
        throw std::runtime_error(
            "Bora distribution file 'inv_cov' must be a square matrix "
            "matching 'mean'.");
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
    const ErrorSilencer silencer;
    const Handle file = OpenFile(path, "Bora network weight");
    Network network;
    network.layers_ = ReadLayers(file.Get());
    return network;
}

Network Network::FromMemory(std::span<const std::byte> data) {
    const ErrorSilencer silencer;
    const Handle file = OpenFileImage(data, "Bora network weight");
    Network network;
    network.layers_ = ReadLayers(file.Get());
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
    const ErrorSilencer silencer;
    const Handle file = OpenFile(path, "Bora input distribution");
    auto data = ReadDistribution(file.Get());
    InputDistribution distribution;
    distribution.mean_ = std::move(data.mean);
    distribution.inv_cov_ = std::move(data.inv_cov);
    return distribution;
}

InputDistribution InputDistribution::FromMemory(
    std::span<const std::byte> blob) {
    const ErrorSilencer silencer;
    const Handle file = OpenFileImage(blob, "Bora input distribution");
    auto data = ReadDistribution(file.Get());
    InputDistribution distribution;
    distribution.mean_ = std::move(data.mean);
    distribution.inv_cov_ = std::move(data.inv_cov);
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
