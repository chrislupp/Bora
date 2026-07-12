/// Shared loaders for the HDF5 reference data in test/data/ (generated from
/// Python NeuralFoil; see scripts/generate_output_ref_data.py and
/// scripts/generate_gradient_ref_data.py). The files are committed to the
/// repository and only regenerated when the reference data is updated.
#pragma once

#include "bora.h"

#include <cstddef>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <hdf5.h>

namespace bora::testing {

/// Reference outputs loaded from output_ref_data.h5.
struct OutputRefData {
    std::vector<std::string> names;
    std::vector<std::string> model_sizes;
    std::vector<double> inputs;   ///< Row-major [n_cases x kNumInputs].
    std::vector<double> outputs;  ///< Row-major [n_cases x n_outputs].
    size_t n_cases = 0;
    size_t n_outputs = 0;
};

/// Reference jacobians loaded from gradient_ref_data.h5.
struct GradientRefData {
    std::vector<std::string> names;
    std::vector<std::string> model_sizes;
    std::vector<size_t> bl_stations;
    std::vector<double> inputs;     ///< Row-major [n_cases x kNumInputs].
    std::vector<double> jacobians;  ///< Row-major [n_cases x n_rows x 23].
    size_t n_cases = 0;
    size_t n_rows = 0;
};

inline std::vector<std::string> ReadStringDataset(hid_t file,
                                                  const char* name) {
    const hid_t dataset = H5Dopen2(file, name, H5P_DEFAULT);
    EXPECT_GE(dataset, 0) << "missing dataset " << name;
    const hid_t space = H5Dget_space(dataset);
    hsize_t count = 0;
    H5Sget_simple_extent_dims(space, &count, nullptr);
    const hid_t type = H5Dget_type(dataset);
    const size_t width = H5Tget_size(type);

    std::vector<char> buffer(count * width);
    H5Dread(dataset, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer.data());

    std::vector<std::string> values(count);
    for (size_t i = 0; i < count; ++i) {
        const char* start = buffer.data() + i * width;
        values[i] = std::string(start, strnlen(start, width));
    }
    H5Tclose(type);
    H5Sclose(space);
    H5Dclose(dataset);
    return values;
}

inline std::vector<double> ReadDoubleDataset(hid_t file, const char* name,
                                             std::vector<hsize_t>& dims) {
    const hid_t dataset = H5Dopen2(file, name, H5P_DEFAULT);
    EXPECT_GE(dataset, 0) << "missing dataset " << name;
    const hid_t space = H5Dget_space(dataset);
    dims.resize(static_cast<size_t>(H5Sget_simple_extent_ndims(space)));
    H5Sget_simple_extent_dims(space, dims.data(), nullptr);

    size_t total = 1;
    for (const hsize_t dim : dims) {
        total *= dim;
    }
    std::vector<double> values(total);
    H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
            values.data());
    H5Sclose(space);
    H5Dclose(dataset);
    return values;
}

inline OutputRefData LoadOutputRefData() {
    const std::string path =
        std::string(BORA_TEST_DATA_DIR) + "/output_ref_data.h5";
    const hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    EXPECT_GE(file, 0) << "cannot open " << path;

    OutputRefData data;
    data.names = ReadStringDataset(file, "names");
    data.model_sizes = ReadStringDataset(file, "model_sizes");

    std::vector<hsize_t> dims;
    data.inputs = ReadDoubleDataset(file, "inputs", dims);
    EXPECT_EQ(dims.size(), 2U);
    data.n_cases = dims[0];
    EXPECT_EQ(dims[1], kNumInputs);

    data.outputs = ReadDoubleDataset(file, "outputs", dims);
    EXPECT_EQ(dims.size(), 2U);
    EXPECT_EQ(dims[0], data.n_cases);
    data.n_outputs = dims[1];

    H5Fclose(file);

    EXPECT_EQ(data.names.size(), data.n_cases);
    EXPECT_EQ(data.model_sizes.size(), data.n_cases);
    return data;
}

inline GradientRefData LoadGradientRefData() {
    const std::string path =
        std::string(BORA_TEST_DATA_DIR) + "/gradient_ref_data.h5";
    const hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    EXPECT_GE(file, 0) << "cannot open " << path;

    GradientRefData data;
    data.names = ReadStringDataset(file, "names");
    data.model_sizes = ReadStringDataset(file, "model_sizes");

    std::vector<hsize_t> dims;
    const std::vector<double> stations =
        ReadDoubleDataset(file, "bl_stations", dims);
    for (const double station : stations) {
        data.bl_stations.push_back(static_cast<size_t>(station));
    }

    data.inputs = ReadDoubleDataset(file, "inputs", dims);
    EXPECT_EQ(dims.size(), 2U);
    data.n_cases = dims[0];
    EXPECT_EQ(dims[1], kNumInputs);

    data.jacobians = ReadDoubleDataset(file, "jacobians", dims);
    EXPECT_EQ(dims.size(), 3U);
    EXPECT_EQ(dims[0], data.n_cases);
    data.n_rows = dims[1];
    EXPECT_EQ(dims[2], kNumInputs);

    H5Fclose(file);

    EXPECT_EQ(data.names.size(), data.n_cases);
    EXPECT_EQ(data.model_sizes.size(), data.n_cases);
    EXPECT_EQ(data.n_rows, 6 + 2 * 3 * data.bl_stations.size());
    return data;
}

/// Returns a lazily constructed, cached model for a size name ("xlarge", ...).
inline const NeuralFoil& GetModel(const std::string& size_name) {
    static std::map<std::string, NeuralFoil> models;
    auto it = models.find(size_name);
    if (it == models.end()) {
        for (ModelSize size :
             {ModelSize::kXXSmall, ModelSize::kXSmall, ModelSize::kSmall,
              ModelSize::kMedium, ModelSize::kLarge, ModelSize::kXLarge,
              ModelSize::kXXLarge, ModelSize::kXXXLarge}) {
            if (ToString(size) == size_name) {
                it = models.emplace(size_name, NeuralFoil(size)).first;
                break;
            }
        }
    }
    EXPECT_NE(it, models.end()) << "Unknown model size: " << size_name;
    return it->second;
}

/// Builds an AeroInput from a kNumInputs-long row of a reference inputs array.
inline AeroInput InputFromArray(const double* values) {
    AeroInput input;
    for (size_t i = 0; i < kNumCstWeightsPerSide; ++i) {
        input.kulfan.upper_weights[i] = values[input_index::kUpperWeights + i];
        input.kulfan.lower_weights[i] = values[input_index::kLowerWeights + i];
    }
    input.kulfan.leading_edge_weight = values[input_index::kLeadingEdgeWeight];
    input.kulfan.TE_thickness = values[input_index::kTeThickness];
    input.alpha = values[input_index::kAlpha];
    input.Re = values[input_index::kRe];
    input.n_crit = values[input_index::kNCrit];
    input.xtr_upper = values[input_index::kXtrUpper];
    input.xtr_lower = values[input_index::kXtrLower];
    return input;
}

}  // namespace bora::testing
