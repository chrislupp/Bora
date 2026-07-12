---
sidebar_position: 1
title: "Installation"
---

# Installation

This guide covers building Bora from source. Bora is a C++20 project built with CMake.

## Prerequisites

| Dependency | Minimum Version | Notes |
|------------|----------------|-------|
| CMake | 3.20+ | Build system generator |
| C++ Compiler | GCC 12+ or Clang 16+ | Must support C++20 |
| HDF5 | 1.10+ | C library, loads the network weight data |
| GTest | latest | Unit testing framework |

<!-- TODO: Add any additional dependencies your project requires -->

## Building

Clone the repository and build with CMake:

```bash
# Configure
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build all targets
cmake --build build
```

For a debug build with assertions enabled:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
```

To build a specific target only:

```bash
cmake --build build --target bora
```

## Running Tests

Bora uses Google Test. After building, run the full test suite with:

```bash
ctest --test-dir build
```

To run a specific test executable with verbose output:

```bash
./build/test/BoraTests --gtest_filter="*TestName*"
```

## CMake Options

The following options can be passed during the configure step with `-D`:

| Option | Default | Description |
|--------|---------|-------------|
| `BUILD_TESTS` | `ON` | Build the test suite |
| `BUILD_EXAMPLES` | `ON` | Build example programs |
| `BORA_NATIVE_ARCH` | `ON` | Enable native architecture optimizations (`-march=native`) |

Example with custom options:

```bash
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTS=ON \
  -DBORA_NATIVE_ARCH=OFF
```

## Verifying the Installation

A successful build and test run confirms the installation:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build
```

All tests should pass. If any dependencies are not found, CMake will report an error during the configure step.
