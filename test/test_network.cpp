#include "bora/network.h"

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace bora::testing {
namespace {

std::vector<std::byte> ReadFileBytes(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    EXPECT_TRUE(file) << "cannot open " << path;
    const auto size = static_cast<size_t>(file.tellg());
    std::vector<std::byte> data(size);
    file.seekg(0);
    file.read(reinterpret_cast<char*>(data.data()),
              static_cast<std::streamsize>(size));
    return data;
}

const std::filesystem::path kWeightsDir = BORA_WEIGHTS_DATA_DIR;

TEST(Network, FromMemoryMatchesLoad) {
    const auto path = kWeightsDir / "nn-xxsmall.h5";
    const Network from_disk = Network::Load(path);
    const Network from_memory = Network::FromMemory(ReadFileBytes(path));

    ASSERT_EQ(from_memory.GetLayers().size(), from_disk.GetLayers().size());
    for (size_t i = 0; i < from_disk.GetLayers().size(); ++i) {
        const Layer& disk = from_disk.GetLayers()[i];
        const Layer& memory = from_memory.GetLayers()[i];
        EXPECT_EQ(memory.rows, disk.rows);
        EXPECT_EQ(memory.cols, disk.cols);
        EXPECT_EQ(memory.weights, disk.weights);
        EXPECT_EQ(memory.biases, disk.biases);
    }
}

TEST(Network, FromMemoryRejectsGarbage) {
    const std::vector<std::byte> garbage(256, std::byte{0x42});
    EXPECT_THROW(Network::FromMemory(garbage), std::runtime_error);
}

TEST(InputDistribution, FromMemoryMatchesLoad) {
    const auto path = kWeightsDir / "scaled_input_distribution.h5";
    const InputDistribution from_disk = InputDistribution::Load(path);
    const InputDistribution from_memory =
        InputDistribution::FromMemory(ReadFileBytes(path));

    ASSERT_EQ(from_memory.Size(), from_disk.Size());
    const std::vector<double> x(from_disk.Size(), 0.5);
    EXPECT_EQ(from_memory.SquaredMahalanobis(x),
              from_disk.SquaredMahalanobis(x));
}

TEST(InputDistribution, LoadRejectsNetworkFile) {
    EXPECT_THROW(InputDistribution::Load(kWeightsDir / "nn-xxsmall.h5"),
                 std::runtime_error);
}

}  // namespace
}  // namespace bora::testing
